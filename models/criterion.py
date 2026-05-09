"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import torch
from torch import nn
from misc import nested_tensor_from_tensor_list, get_world_size, interpolate, is_dist_avail_and_initialized, box_cxcywh_to_xyxy, generalized_box_iou, _max_by_axis
from .segmentation import dice_loss, sigmoid_focal_loss, sigmoid_focal_loss_refer
from utils import flatten_temporal_batch_dims
import torch.nn.functional as F
from einops import rearrange

class SetCriterion(nn.Module):
    """ This class computes the loss for FPRFormer.
    The process happens in two steps:
        1) we compute the hungarian assignment between the ground-truth and predicted sequences.
        2) we supervise each pair of matched ground-truth / prediction sequences (mask + reference prediction)
    """
    def __init__(self, matcher, weight_dict, eos_coef, num_classes):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the un-referred category
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_classes = num_classes
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        # make sure that only loss functions with non-zero weights are computed:
        losses_to_compute = []
        if weight_dict['loss_dice'] > 0 or weight_dict['loss_sigmoid_focal'] > 0:
            losses_to_compute.append('masks')
        if weight_dict['loss_cls'] > 0:
            losses_to_compute.append('loss_cls')
        if weight_dict['loss_bbox'] > 0 or weight_dict['loss_giou'] > 0:
            losses_to_compute.append("boxes")
        self.losses = losses_to_compute

    def forward(self, outputs, targets, current_epoch):
        aux_outputs_list = outputs.pop('aux_outputs', None)
        # compute the losses for the output of the last decoder layer:
        losses = self.compute_criterion(outputs, targets, losses_to_compute=self.losses, current_epoch=current_epoch)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate decoder layer.
        if aux_outputs_list is not None:
            aux_losses_to_compute = self.losses.copy()
            for i, aux_outputs in enumerate(aux_outputs_list):
                losses_dict = self.compute_criterion(aux_outputs, targets, aux_losses_to_compute, current_epoch)
                losses_dict = {k + f'_{i}': v for k, v in losses_dict.items()}
                losses.update(losses_dict)

        return losses

    def compute_criterion(self, outputs, targets, losses_to_compute, current_epoch):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets, False) #[(tensor, tensor)] batchsize
        # T & B dims are flattened so loss functions can be computed per frame (but with same indices per video).
        # also, indices are repeated so the same indices can be used for frames of the same video.
        T = len(targets) #targets: [({}, {}, {})] training
        # Flatten outputs and targets
        outputs, targets = flatten_temporal_batch_dims(outputs, targets)
        # repeat the indices list T times so the same indices can be used for each video frame
        # Make all frames use the same matching indices
        indices = T * indices

        # Compute the average number of target masks across all nodes, for normalization purposes
        # Count total number of segmentation masks
        num_masks = sum(len(t["masks"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=indices[0][0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in losses_to_compute:
            # Update dictionary and return
            losses.update(self.get_loss(loss, outputs, targets, indices, current_epoch, frame_size=T, num_masks=num_masks))
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_masks, **kwargs):
        src_idx = self._get_src_permutation_idx(indices) #(tensor_batch, tensor_src)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        output_boxes = outputs['pred_boxes'] #[t*b Nq 4]
        output_boxes = output_boxes[src_idx] #[Nq 4]
        tgt_boxes = [t["boxes"] for t in targets]
        max_size = _max_by_axis([list(box.shape) for box in tgt_boxes]) #[o, 4]
        batch_shape = [len(tgt_boxes)] + max_size
        tgt_boxes_new = torch.zeros(size=batch_shape, device=tgt_boxes[0].device, dtype=tgt_boxes[0].dtype)
        for box, pad_box in zip(tgt_boxes, tgt_boxes_new):
            pad_box[: box.shape[0], : box.shape[1]].copy_(box)
        tgt_boxes = tgt_boxes_new[tgt_idx]
        loss_bbox = F.l1_loss(output_boxes, tgt_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_masks
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(output_boxes),
            box_cxcywh_to_xyxy(tgt_boxes)))

        loss_giou = loss_giou.sum() / num_masks

        losses = {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, **kwargs):
        """
        Description: 
        """
        
        assert "pred_masks" in outputs

        # Align outputs and targets
        src_idx = self._get_src_permutation_idx(indices) #(tensor_batch, tensor_src)
        tgt_idx = self._get_tgt_permutation_idx(indices) #[]
        src_masks = outputs["pred_masks"] #[t * b, query, H, W]
        src_masks = src_masks[src_idx] #[instances, H, W] Get model output mask
        masks = [t["masks"] for t in targets] #len batch Get mask for each target
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()#[b, instances, H, W]
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx] #[instances, h, w]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1) #[instances, h*w]

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_sigmoid_focal": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def loss_label(self, outputs, targets, indices, num_masks, **kwargs):
        device = outputs['pred_cls'].device
        frames_size, bs, nq, k = outputs['pred_cls'].shape #[t b, nq, K] K=1 when a2d
        pred_label = rearrange(outputs["pred_cls"], 't b nq k -> b (t nq) k')
        BT = len(targets)
        B = BT // frames_size
        batch_targets = []
        for i in range(B):
            batch_temp = targets[i::B] #t
            b_valid = [t["is_ref_inst_visible"] for t in batch_temp]
            b_label = [t["labels"] for t in batch_temp] #each t only one object
            batch_targets.append({
                "valid": torch.stack(b_valid, dim=0), #[B T]
                "labels": torch.stack(b_label, dim=0),
                'referred_instance_idx': batch_temp[0]['referred_instance_idx']
                }
            )
        valid_indices = []
        valids = [target['valid'] for target in batch_targets] #[b 1]
        
        for id, (valid, (indice_i, indice_j)) in enumerate(zip(valids, indices)): 
            ref_idx = torch.where(indice_j == batch_targets[id]['referred_instance_idx'])[0]
            indice_j = indice_j[ref_idx]
            indice_i = indice_i[ref_idx]
            valid_ind = valid.nonzero().flatten() 
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * frames_size
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices 
        target_classes = torch.full(pred_label.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_label.device) 
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(batch_targets, valid_indices)])
            target_classes[idx] = target_classes_o.squeeze(1)

        target_classes_onehot = torch.zeros([pred_label.shape[0], pred_label.shape[1], pred_label.shape[2] + 1],
                                            dtype=pred_label.dtype, layout=pred_label.layout, device=pred_label.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(pred_label, target_classes_onehot, num_masks, alpha=0.25, gamma=2) * pred_label.shape[1]
        losses = {'loss_cls': loss_ce}
        
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        #tensor([batch_idx1, batch_idx1, batch_idx2, ...])
        src_idx = torch.cat([src for (src, _) in indices])
        #tensor([query_idx1, query_idx1, query_idx2, ...])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _get_query_referred_indices(indices, targets, B):
        """
        extract indices of object queries that where matched with text-referred target objects
        """
        query_referred_indices = []
        batch_idx = []
        for idx, ((query_idxs, target_idxs), target) in enumerate(zip(indices[:B], targets[:B])):
            ref_query_idx = query_idxs[torch.where(target_idxs == target['referred_instance_idx'])[0]]
            # query_referred_indices.append([idx ,ref_query_idx])
            batch_idx.append(torch.tensor(idx))
            query_referred_indices.append(ref_query_idx)
        batch_idx = torch.tensor(batch_idx)
        query_referred_indices = torch.cat(query_referred_indices)
        return batch_idx, query_referred_indices

    # @staticmethod
    # def _get_query_referred_indices_assist(indices, targets):
    #     """
    #     extract indices of object queries that where matched with text-referred target objects
    #     """
    #     query_referred_indices = []
    #     for (query_idxs, target_idxs), target in zip(indices, targets):
    #         ref_query_idx = query_idxs[torch.where(target_idxs == target['referred_instance_idx'])[0]]
    #         query_referred_indices.append(ref_query_idx)
    #     query_referred_indices = torch.cat(query_referred_indices)
    #     return query_referred_indices

    def get_loss(self, loss, outputs, targets, indices, current_epoch, **kwargs):
        loss_map = {
            'masks': self.loss_masks,
            'loss_cls': self.loss_label,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)
