"""
Modified from DETR https://github.com/facebookresearch/detr
"""

import torch
import torch.nn.functional as F
from torch import nn
from models.backbone import build_backbone
from models.video_swin_transformer import build_video_swin_backbone
from models.matcher import build_matcher
from models.segmentation import FPNSpatialDecoder
from models.criterion import SetCriterion
from models.postprocessing import A2DSentencesPostProcess, ReferYoutubeVOSPostProcess, PostProcess, PostProcessSegm
from models.position_encoding import PositionEmbeddingSine1D
from models.wstca import WaveletbasedSpatioTemporalCrossAttention
from models.cfma.memory_encoder import MemoryEncoder, MaskDownSampler, CXBlock, Fuser
from models.cfma.position_encoding import PositionEmbeddingSine
from models.cfma.memory_attention import MemoryAttention, MemoryAttentionLayer
from transformers import RobertaModel, RobertaTokenizerFast
from einops import rearrange, repeat
from misc import NestedTensor, inverse_sigmoid
from .deformable_transformer import build_deforamble_transformer
import math
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# copy module
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FPRFormer(nn.Module):
    """ The main module of the Semantic-Assisted Object Cluster"""
    def __init__(self, config):
        """
        Parameters:
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         FPRFormer can detect in a single image. In our paper we use 20 in all settings.
            mask_kernels_dim: dim of the segmentation kernels and of the feature maps outputted by the spatial decoder.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        if config.backbone in ["video-swin-t", "video-swin-s", "video-swin-b"]:
            self.backbone = build_video_swin_backbone(config)
        elif config.backbone in ["resnet50"]:
            self.backbone = build_backbone(config)

        self.num_feature_levels = config.DeformTransformer['num_feature_levels']
        d_model = config.DeformTransformer['d_model']
        self.num_queries = config.DeformTransformer['num_queries']
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.class_embed = nn.Linear(d_model, config.num_classes)
        self.rel_coord = config.rel_coord

        self.transformer = build_deforamble_transformer(config.DeformTransformer)

        # Preprocessing module, including 2D convolution and group normalization
        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs): # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model),
                ))
                in_channels = d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[-3:][0], d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )])

        # Initialization of box and preprocessing modules
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # Preprocessing module initialization
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Used for calculating auxiliary loss, output by each transformer decoder layer
        num_pred = self.transformer.decoder.num_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None



        self.text_encoder = RobertaModel.from_pretrained(config.text_encoder_type)
        # self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.text_encoder_type)
        self.freeze_text_encoder = config.freeze_text_encoder
        if self.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.text_pos = PositionEmbeddingSine1D(d_model, normalize=True)

        self.query_embed = nn.Embedding(self.num_queries, d_model)
        # self.instance_kernels_head = MLP(d_model, d_model, output_dim=config.mask_kernels_dim, num_layers=3) #set some hyperparameter
        self.spatial_decoder = FPNSpatialDecoder(d_model, 2 * [d_model] + [self.backbone.num_channels[0]], config.mask_kernels_dim)

        mask_kernel_dim = config.mask_kernels_dim
        # Position encoding initialization
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=mask_kernel_dim // 2,
            normalize=True
        )
        # Mask downsampler, not used here
        self.mask_downsampler = MaskDownSampler(
            embed_dim=mask_kernel_dim,
            kernel_size=4,
            stride=4,
            padding=0,
            total_stride=16,
            activation=nn.GELU
        )
        # CXBlock as the basic unit of feature fuser
        cx_block = CXBlock(dim=mask_kernel_dim, kernel_size=7, padding=3)
        # Feature fuser
        self.feature_fuser = Fuser(
            layer=cx_block,
            num_layers=3,
            dim=mask_kernel_dim,
            input_projection=True
        )
        # Memory Encoder
        self.memory_encoder = MemoryEncoder(
            out_dim=d_model,
            mask_downsampler=self.mask_downsampler,
            fuser=self.feature_fuser,
            position_encoding=self.position_encoding,
            in_dim=mask_kernel_dim
        )
        
        # Memory management parameters - Simplified version for RVOS task
        self.num_maskmem = 7  # Default use 7 memory frames (1 current frame + 6 history frames)
        self.mem_dim = mask_kernel_dim
        # Temporal position encoding for memory
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, 256)  # Use 256 dimensions directly instead of mem_dim
        )
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # Object pointer projection layer
        self.obj_ptr_proj = nn.Linear(d_model, self.mem_dim)
        nn.init.xavier_uniform_(self.obj_ptr_proj.weight)
        nn.init.zeros_(self.obj_ptr_proj.bias)
        
        self.vlf = WaveletbasedSpatioTemporalCrossAttention(d_model=d_model, nhead=8)
        # Initialize MemoryAttention module - Replaces VOC module
        # First create MemoryAttentionLayer
        memory_attn_layer = MemoryAttentionLayer(
            activation="relu",  # Activation function type
            cross_attention=nn.MultiheadAttention(d_model, 8, dropout=0.1),  # Cross attention
            d_model=d_model,  # Model dimension
            dim_feedforward=2048,  # Feedforward network dimension
            dropout=0.1,  # Dropout rate
            pos_enc_at_attn=True,  # Add position encoding in self-attention
            pos_enc_at_cross_attn_keys=True,  # Add position encoding in cross-attention keys
            pos_enc_at_cross_attn_queries=True,  # Add position encoding in cross-attention queries
            self_attention=nn.MultiheadAttention(d_model, 8, dropout=0.1),  # Self-attention
        )
        
        # Then create MemoryAttention
        self.memory_attention = MemoryAttention(
            d_model=d_model,
            pos_enc_at_input=True,
            layer=memory_attn_layer,
            num_layers=3,  # Use 3 attention layers
            batch_first=False  # Consistent with VOC, use sequence-first format
        )
        
        # Remove VOC module
        # self.voc = VOC(config.VOC)
        
        # Text preprocessing module
        self.txt_proj = FeatureResizer(
            input_feat_size = self.text_encoder.config.hidden_size,
            output_feat_size = d_model,
            dropout = 0.1,
        )

        # Initialize mask Head
        self.controller_layers = config.controller_layers
        self.in_channels = config.mask_kernels_dim
        self.dynamic_mask_channels = config.dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        # Calculate number of weights and biases
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1) # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(d_model, d_model, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)
        # self.bbox_attention = MHAttentionMap(d_model, d_model, self.transformer.nhead, dropout=0)
        # Define independent controller for second stage and initialize
        self.refined_controller = MLP(d_model, d_model, self.num_gen_params, 3)
        for layer in self.refined_controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)

        self.aux_loss = config.aux_loss

        # Initialize memory storage
        self.memory_outputs = {}

    def reset_memory(self):
        """Reset model memory state
        This method is called at the start of each batch to ensure memory does not interfere between batches.
        This is because different batches may contain different videos or different parts of the same video, with no temporal continuity.
        """
        self.memory_outputs = {}

    def forward_text(self, text_queries, device):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        #with torch.inference_mode(mode=self.freeze_text_encoder):
        encoded_text = self.text_encoder(**tokenized_queries, output_hidden_states=True)
        # Transpose memory because pytorch's attention expects sequence first
        txt_memory = rearrange(encoded_text.last_hidden_state, 'b s c -> s b c')
        txt_memory = self.txt_proj(txt_memory)  # change text embeddings dim to model dim
        text_sentence_feature = encoded_text.pooler_output
        text_sentence_feature = self.txt_proj(text_sentence_feature)
        # text_sentence_feature = None
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        txt_pad_mask = tokenized_queries.attention_mask.ne(1).bool()  # [B, S] #0 for pad
        text_feature = NestedTensor(txt_memory, txt_pad_mask)
        return text_feature, text_sentence_feature


    def forward(self, samples: NestedTensor, valid_indices, text_queries, targets):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: Batched frames of shape [time x batch_size x 3 x H x W]
               - samples.mask: A binary mask of shape [time x batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_cls": The reference prediction logits for all queries.
                                     Shape: [time x batch_size x num_queries x 2]
               - "pred_masks": The mask logits for all queries.
                               Shape: [time x batch_size x num_queries x H_mask x W_mask]
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
                                
            Memory processing logic:
               1. Reset memory at the start of batch to avoid interference
               2. Use history frame memory of current batch to enhance current frame features
               3. Update memory immediately after processing each frame for subsequent frames
               4. This design allows establishing temporal relationships between frames within the same batch
        """
        # Reset memory at the start of each batch
        self.reset_memory()
        
        device = samples.tensors.device
        
        # Get text word feature vectors, text global sentence features
        text_features, text_sentence_feature = self.forward_text(text_queries, device)
        backbone_out, pos = self.backbone(samples) #[backbone_out = [(b t) c h w]] mask: [(b t) h w]
        # keep only the valid frames (frames which are annotated):
        # (for example, in a2d-sentences only the center frame in each window is annotated).
        B = len(text_queries)
        BT = pos[0].shape[0]
        ## prepare for the deformable Transformer
        T = BT // B  # a2d is one but others are not
        # print(f"backbone[0]:{backbone_out[0].tensors.shape}") # [8, 128, 90, 160]
        # if valid_indices is not None:
        #     for layer_out in backbone_out:
        #         layer_out.tensors = layer_out.tensors.index_select(0, valid_indices) #[b*t c h w]
        #         layer_out.mask = layer_out.mask.index_select(0, valid_indices)
        #     for i, p in enumerate(pos):
        #         pos[i] = p.index_select(0, valid_indices) #[bt h w]
        #     samples.mask = samples.mask.index_select(0, valid_indices)
        #     T = 1

        srcs = []
        langs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()  # text_word_feature [l b c]#text_word_mask [B L]
        # text_sentence_feature_fuse = text_sentence_feature.unsqueeze(0) #[1 b C]

        for l, (feat, pos_l) in enumerate(zip(backbone_out[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape
            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=B, t=T)
            mask_l = rearrange(mask, '(b t) h w -> b (t h w)', t=T, b=B)
            pos = rearrange(pos_l, "(b t) c h w -> (t h w) b c", t=T, b=B)
            src_proj_l_new = self.vlf(tgt=src_proj_l,
                                             memory=text_word_features,
                                             num_frames=T,
                                             vision_h=h,
                                             vision_w=w,
                                             memory_key_padding_mask=text_word_masks,
                                             pos=text_pos,
                                             query_pos=None
            )
            src_proj_l_new = rearrange(src_proj_l_new, '(t h w) b c -> (b t) c h w', t=T, h=h, w=w)

            srcs.append(src_proj_l_new)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(backbone_out) - 1):
            _len_srcs = len(backbone_out) - 1 # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](backbone_out[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=B, t=T)
                src = self.vlf(tgt=src,
                                memory=text_word_features,
                                num_frames=T,
                                vision_h=h,
                                vision_w=w,
                                memory_key_padding_mask=text_word_masks,
                                pos=text_pos,
                                query_pos=None
                )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=T, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        query_embeds = self.query_embed.weight #[num_queries, C]
        tgt = torch.zeros_like(query_embeds)
        tgt = repeat(tgt, 'nq c -> b t nq c', b=B, t=T)
        #text_embed = repeat(text_sentence_feature, 'b c -> b t q c', t=T, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                            self.transformer(srcs, tgt, masks, poses, query_embeds)
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]


        layer_outputs = []
        hs = rearrange(hs, 'l (b t) q c -> l t b q c', t=T, b=B)
        
        # Refactor memory processing logic, using frame-by-frame processing
        # Process flow: Process t=0 frame and update memory, then process t=1 frame (can use t=0 memory) and update, and so on
        # This achieves true temporal memory mechanism
        
        # Prepare data needed before memory processing
        memory = [rearrange(mem, '(b t) c h w -> (t b) c h w', b=B, t=T) for mem in memory]
        low_features = [rearrange(mem, '(b t) c h w -> t b c h w', b=B, t=T) for mem in memory]  # Store low-dimensional features
        fpn_first_input = rearrange(backbone_out[0].tensors, '(b t) c h w -> (t b) c h w', b=B, t=T)
        memory.insert(0, fpn_first_input)  # This is the stack process
        decoded_frame_features = self.spatial_decoder(memory[-1], memory[:-1][::-1])
        
        # Rearrange decoded_frame_features to required shape
        high_features = rearrange(decoded_frame_features, '(t b) d h w -> t b d h w', t=T, b=B)  # Store high-dimensional features
        mask_features = rearrange(decoded_frame_features, '(t b) d h w -> b t d h w', t=T, b=B)
        
        # Get original mask - Used only during memory encoding process
        # We will not use this mask as final output
        original_outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])   # [t, b, q, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, 't b q n -> b (t q) n', b=B, t=T)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=B, t=T)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> t b q h w', t=T)
            original_outputs_seg_masks.append(outputs_seg_mask)
        original_masks = torch.stack(original_outputs_seg_masks, dim=0)  # [l t b q h w]
        
        # Get last layer mask, used only for memory encoding
        last_layer_masks = original_masks[-1]  # [t b q h w]
        
        # Create tensor to store processed hidden states for each layer
        hs_processed = torch.zeros_like(hs)  # [l, t, b, q, c]
        self.reset_memory()

        # Frame-by-frame processing - Key improvement: Update memory immediately after processing each frame for subsequent frames
        if valid_indices is not None:
            for t_idx in range(5):
                # Process current time step: Use history memory to enhance features and update memory immediately
                hs_processed[:, t_idx] = self.track_frame(
                    t_idx=t_idx,
                    B=B,
                    T=T,
                    hs=hs,
                    high_features=high_features,
                    last_layer_masks=last_layer_masks
                )
        else:
            for t_idx in range(T):
                # Process current time step: Use history memory to enhance features and update memory immediately
                hs_processed[:, t_idx] = self.track_frame(
                    t_idx=t_idx,
                    B=B,
                    T=T,
                    hs=hs,
                    high_features=high_features,
                    last_layer_masks=last_layer_masks
                )

        # Use processed features for subsequent operations, achieving feature refinement through residual mechanism
        hs_voc = hs_processed + hs

        # Reshape format to adapt to subsequent processing
        hs_voc_for_cls_box = rearrange(hs_voc, 'l t b n c -> l (b t) n c')

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs_voc_for_cls_box.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs_voc_for_cls_box[lvl])
            tmp = self.bbox_embed[lvl](hs_voc_for_cls_box[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_coords.append(outputs_coord)
            outputs_classes.append(outputs_class)

        outputs_coord = torch.stack(outputs_coords)
        outputs_classes = torch.stack(outputs_classes)
        # rearrange
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l t b q n', b=B, t=T)
        outputs_classes = rearrange(outputs_classes, 'l (b t) q n -> l t b q n', b=B, t=T)

        # Recompute mask using enhanced features - This is the main improvement
        outputs_seg_masks = []
        for lvl in range(hs_voc.shape[0]):
            # Generate dynamic_mask_head_params using enhanced features
            enhanced_dynamic_mask_head_params = self.refined_controller(hs_voc[lvl])  # [t, b, q, num_params]
            enhanced_dynamic_mask_head_params = rearrange(enhanced_dynamic_mask_head_params, 't b q n -> b (t q) n', b=B, t=T)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=B, t=T)
            enhanced_outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, enhanced_dynamic_mask_head_params, lvl_references, targets)
            enhanced_outputs_seg_mask = rearrange(enhanced_outputs_seg_mask, 'b (t q) h w -> t b q h w', t=T)
            outputs_seg_masks.append(enhanced_outputs_seg_mask)

        # Mask computed using enhanced features - As final output
        output_masks = torch.stack(outputs_seg_masks, dim=0)  # [l t b q h w]

        # Ensure hs_voc shape is correct
        hs_voc = rearrange(hs_voc, 'l (b t) n c -> l t b n c', b=B, t=T) if hs_voc.ndim == 4 else hs_voc

        hs_voc_reshaped = hs_voc

        for pm, plg, pir, pb in zip(output_masks, hs_voc_reshaped, outputs_classes, outputs_coord):
            plg_reshaped_for_loss = rearrange(plg, 't b q c -> (t b) q c')
            layer_out = {
                'pred_masks': pm,    #[t,b,n,h,w]
                'pred_cls': pir,     #[t b nq K]
                'pred_boxes': pb,    # Add this key to avoid KeyError
                'pred_logit': plg_reshaped_for_loss,   # Add this key to avoid KeyError
                'text_sentence_feature': text_features if 'text_sentence_feature' in locals() or 'text_sentence_feature' in globals() else None
            }
            layer_outputs.append(layer_out)
        out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        if self.aux_loss:
            out['aux_outputs'] = layer_outputs[:-1]  # Previous layers are also used to calculate loss function

        # Add memory_outputs to results
        out['memory_outputs'] = self.memory_outputs

        # If valid_indices exists, filter output to keep only frames corresponding to valid_indices
        if valid_indices is not None:
            # Filter final output layer results
            out['pred_masks'] = out['pred_masks'].index_select(0, valid_indices)  # [t,b,n,h,w] -> [valid_t,b,n,h,w]
            out['pred_cls'] = out['pred_cls'].index_select(0, valid_indices)      # [t,b,n,c] -> [valid_t,b,n,c]
            out['pred_boxes'] = out['pred_boxes'].index_select(0, valid_indices)  # [t,b,n,4] -> [valid_t,b,n,4]
            out['pred_logit'] = out['pred_logit'].index_select(0, valid_indices)  # [t,b,n,c] -> [valid_t,b,n,c]

            # If auxiliary output exists, also need to filter
            if self.aux_loss and 'aux_outputs' in out:
                for i, aux_out in enumerate(out['aux_outputs']):
                    out['aux_outputs'][i]['pred_masks'] = aux_out['pred_masks'].index_select(0, valid_indices)
                    out['aux_outputs'][i]['pred_cls'] = aux_out['pred_cls'].index_select(0, valid_indices)
                    out['aux_outputs'][i]['pred_boxes'] = aux_out['pred_boxes'].index_select(0, valid_indices)
                    out['aux_outputs'][i]['pred_logit'] = aux_out['pred_logit'].index_select(0, valid_indices)
        
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Memory related functions
    def generate_obj_ptr(self, input_embeddings):
        """
        Generate object pointer from input embeddings
        
        Args:
            input_embeddings: Shape [B, C]
            
        Returns:
            obj_ptr: Object pointer with shape [B, C]
        """
        return self.obj_ptr_proj(input_embeddings)
        
    def encode_memory(self, pix_feat, masks):
        """
        Encode memory of current frame
        
        Args:
            pix_feat: Pixel features, shape [B, C, H, W]
            masks: Masks, shape [B, 1, H, W]
            
        Returns:
            memory_dict: Dictionary containing memory features and position encoding
        """
        
        # Directly use memory_encoder to process masks and features
        memory_result = self.memory_encoder(
            pix_feat=pix_feat,
            masks=masks,
            skip_mask_sigmoid=False
        )
        
        return memory_result
        
    def track_frame(self, t_idx, B, T, hs, high_features, last_layer_masks):
        """
        Process frame of a single time step, including:
        1. Use history memory to enhance current frame features
        2. Encode current frame as memory and update memory bank
        
        Args:
            t_idx: Current time step index
            B: Batch size
            T: Total time steps
            hs: Original transformer output features [l, t, b, q, c]
            high_features: High resolution features [t, b, d, h, w]
            last_layer_masks: Last layer masks [t, b, q, h, w]
            
        Returns:
            hs_processed: Features processed by memory
        """
        
        # Create tensor to store processed features
        hs_processed = torch.zeros_like(hs[:, t_idx])  # [l, b, q, c]
        
        # Process each level separately
        for lvl in range(hs.shape[0]):
            # All batches and queries of current layer and current time step
            curr_hs = hs[lvl, t_idx]  # [b, q, c]
            
            # Process all batches in current time step
            for b_idx in range(B):
                # Get all queries of current batch - Refactoring for batch processing here
                batch_hs = curr_hs[b_idx]  # [q, c]
                num_queries = batch_hs.shape[0]
                
                # Create list to store memory and position encoding for all queries
                all_memory_feats = []
                all_memory_pos = []
                has_memory = [False] * num_queries  # Track which queries have history memory
                
                # First, collect history memory for each query
                for q_idx in range(num_queries):
                    # Memory list for each query
                    query_memory_list = []
                    query_memory_pos_list = []
                    
                    # Collect memory for corresponding query from history frames
                    for prev_t in range(max(0, t_idx - self.num_maskmem + 1), t_idx):
                        # Check if there is history memory for this query
                        if prev_t in self.memory_outputs and b_idx in self.memory_outputs[prev_t] and q_idx in self.memory_outputs[prev_t][b_idx]:
                            has_memory[q_idx] = True
                            prev_memory = self.memory_outputs[prev_t][b_idx][q_idx]
                            
                            # Get history memory features
                            mem_feat = prev_memory["maskmem_features"]  # [1, c, h, w]
                            mem_pos = prev_memory["maskmem_pos_enc"]  # [1, c, h, w]

                            # Flatten features to format suitable for attention
                            h, w = mem_feat.shape[2], mem_feat.shape[3]
                            mem_feat_flat = mem_feat.flatten(2).permute(2, 0, 1)  # [h*w, 1, c]
                            mem_pos_flat = mem_pos.flatten(2).permute(2, 0, 1)  # [h*w, 1, c]
                            
                            # Add temporal position encoding
                            t_diff = t_idx - prev_t
                            t_pos_enc = self.maskmem_tpos_enc[t_diff].expand_as(mem_feat_flat)
                            mem_feat_flat = mem_feat_flat + t_pos_enc
                            
                            query_memory_list.append(mem_feat_flat)
                            query_memory_pos_list.append(mem_pos_flat)
                    
                    # Add memory of each query to total list
                    if query_memory_list:  # Ensure there is memory
                        mem_concat = torch.cat(query_memory_list, dim=0)  # [total_hw, 1, c]
                        pos_concat = torch.cat(query_memory_pos_list, dim=0)  # [total_hw, 1, c]
                        all_memory_feats.append(mem_concat)
                        all_memory_pos.append(pos_concat)
                    else:
                        # If no history memory, add empty array as placeholder
                        all_memory_feats.append(None)
                        all_memory_pos.append(None)
                
                # Check if any query has history memory
                if any(has_memory):
                    # Build batch_queries - Only contain queries with history memory
                    active_indices = [i for i, has_mem in enumerate(has_memory) if has_mem]
                    
                    if active_indices:
                        # Extract query features with history memory
                        active_queries = batch_hs[active_indices]  # [active_q, c]
                        active_queries = active_queries.unsqueeze(0)  # [1, active_q, c]
                        
                        # Get position encoding for these queries
                        active_query_pos = torch.zeros_like(active_queries)
                        
                        # Process memory for each active query
                        active_memory_feats = [all_memory_feats[i] for i in active_indices if all_memory_feats[i] is not None]
                        active_memory_pos = [all_memory_pos[i] for i in active_indices if all_memory_pos[i] is not None]
                        
                        if active_memory_feats and active_memory_pos:
                            # Merge memory of different queries - Note dimension change
                            # From [active_q][hw, 1, c] to [hw, active_q, c]
                            
                            # Since feature sizes are same in same batch, process directly without padding
                            num_active_queries = len(active_memory_feats)
                            hw = active_memory_feats[0].shape[0]  # All features have the same hw
                            feat_dim = active_memory_feats[0].shape[2]  # Feature dimension
                            
                            # Create tensor to directly store stacked features
                            stacked_memory_feats = torch.zeros((hw, num_active_queries, feat_dim), 
                                                               dtype=active_memory_feats[0].dtype,
                                                               device=active_memory_feats[0].device)
                            stacked_memory_pos = torch.zeros((hw, num_active_queries, feat_dim), 
                                                             dtype=active_memory_pos[0].dtype,
                                                             device=active_memory_pos[0].device)
                            
                            # Copy features directly to pre-allocated tensor
                            for q_idx, (feat, pos) in enumerate(zip(active_memory_feats, active_memory_pos)):
                                stacked_memory_feats[:, q_idx, :] = feat[:, 0, :]
                                stacked_memory_pos[:, q_idx, :] = pos[:, 0, :]
                            
                            # Batch apply memory_attention
                            attended_hs = self.memory_attention(
                                curr=active_queries,  # [1, active_q, c]
                                memory=stacked_memory_feats,  # [max_hw, active_q, c]
                                curr_pos=active_query_pos,  # [1, active_q, c]
                                memory_pos=stacked_memory_pos  # [max_hw, active_q, c]
                            )
                            
                            # Update processed features of active queries
                            for i, idx in enumerate(active_indices):
                                hs_processed[lvl, b_idx, idx] = attended_hs[0, i]
                
                # For queries without history memory, keep original features
                for q_idx in range(num_queries):
                    if not has_memory[q_idx]:
                        hs_processed[lvl, b_idx, q_idx] = batch_hs[q_idx]
        
        # Create memory dictionary for current time step
        t_memory_dict = {}
        
        # Batch process all batches in current time step
        for b_idx in range(B):
            # Create memory dictionary for current batch
            b_memory_dict = {}
            
            # Get current frame features
            curr_features = high_features[t_idx, b_idx]  # [d h w]
            
            # Get all query masks of current frame
            curr_masks = last_layer_masks[t_idx, b_idx]  # [q h w]
            num_queries = curr_masks.shape[0]
            
            # Process each query mask separately - Since memory encoding needs mask, still need to process one by one here
            for q_idx in range(num_queries):
                # Get current query mask and expand dimension to [1, 1, h, w]
                query_mask = curr_masks[q_idx].unsqueeze(0).unsqueeze(0)
                
                # Use encode_memory method to process masks and features
                memory_result = self.encode_memory(
                    pix_feat=curr_features.unsqueeze(0),
                    masks=query_mask
                )
                
                # Generate object pointer
                # Use memory enhanced features as input for object pointer (last layer)
                input_embedding = hs_processed[-1, b_idx, q_idx]
                obj_ptr = self.generate_obj_ptr(input_embedding.unsqueeze(0)).squeeze(0)
                
                # Store separate memory for each query
                query_memory = {
                    "maskmem_features": memory_result["vision_features"],
                    "maskmem_pos_enc": memory_result["vision_pos_enc"],
                    "obj_ptr": obj_ptr
                }
                
                # Use query index as key
                b_memory_dict[q_idx] = query_memory
            
            # Add current batch memory to time step dictionary
            t_memory_dict[b_idx] = b_memory_dict
        
        # Immediately update memory_outputs dictionary so next time step can use current time step memory
        self.memory_outputs[t_idx] = t_memory_dict
        # Pop unnecessary history memory
        # Calculate index of oldest frame to be evicted
        # self.num_maskmem is the memory window size you set (e.g., 2)
        evict_t_idx = t_idx - self.num_maskmem
        # If this old frame index is valid (>=0) and it is still in our memory dictionary, delete it
        if evict_t_idx >= 0 and evict_t_idx in self.memory_outputs:
            del self.memory_outputs[evict_t_idx]
        # --- End of added code ---
        return hs_processed

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[0][i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                                    locations.reshape(1, 1, 1, h, w, 2) # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4) # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w)

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1) #[hidden_dim ]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        #[input_dim , hidden_dim, hidden_dim, hidden_dim] [hidden_dim hidden_dim hidden_dim output_dim]
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        """
        q the query: [t b n c]
        key: the last memory: [t b c h w]
        """
        q = rearrange(q, 't b nq c -> (t b) nq c')
        k = rearrange(k, 't b c h w -> (t b) c h w')
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights

def build(args):
    device = args.device
    model = FPRFormer(args)
    matcher = build_matcher(args)
    # Loss function balance coefficients
    weight_dict = {'loss_dice': args.dice_loss_coef,
                   'loss_sigmoid_focal': args.sigmoid_focal_loss_coef,
                   'loss_cls': args.class_loss_coef,
                   'loss_bbox': args.box_loss_coef,
                   'loss_giou':args.giou_coef}
    # Auxiliary loss
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.DeformTransformer['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, num_classes=args.num_classes)
    criterion.to(device)

    # Post-processing module
    postprocessor = build_postprocessors(args.dataset_name)

    return model, criterion, postprocessor

def build_postprocessors(dataset_name):
    if dataset_name == 'a2d_sentences' or dataset_name == 'jhmdb_sentences':
        postprocessors = A2DSentencesPostProcess()
    elif dataset_name == 'ref_youtube_vos' or dataset_name == 'joint' or dataset_name == 'mevis':
        postprocessors = ReferYoutubeVOSPostProcess()
        # for coco pretrain postprocessor
    elif "coco" in dataset_name:
        postprocessors = {"bbox": PostProcess(),
                          "segm": PostProcessSegm(threshold=0.5)
                          }
    elif dataset_name == 'davis':
        postprocessors = None
    return postprocessors
