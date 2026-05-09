import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from einops import rearrange
from torch_dwt.functional import dwt3, idwt3


class WaveletbasedSpatioTemporalCrossAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, wave='bior2.2'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.wave = wave

        self.attn_low_st = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.attn_high_st = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.gate_network = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                num_frames: int,
                vision_h: int,
                vision_w: int,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        text_pos, vision_pos = pos, query_pos
        vision_feats, text_feats = tgt, memory

        B, C = vision_feats.shape[1], self.d_model
        T, H, W = num_frames, vision_h, vision_w

        vision_feats_5d = rearrange(vision_feats, '(t h w) b c -> b c t h w', t=T, h=H, w=W)

        coefs = dwt3(vision_feats_5d, self.wave)  # shape: [B, 8, C, Td, Hd, Wd]

        coefs_swapped = rearrange(coefs, 'b d c t h w -> b c d t h w', d=8)
        Yl = coefs_swapped[:, :, 0, ...]
        Yh_bands = coefs_swapped[:, :, 1:, ...]

        _, _, T_d, H_d, W_d = Yl.shape

        q_low_st = rearrange(Yl, 'b c t h w -> (t h w) b c')
        q_high_st = rearrange(Yh_bands, 'b c d t h w -> (d t h w) b c')

        low_pos, high_pos = None, None
        if vision_pos is not None:
            vision_pos_5d = rearrange(vision_pos, '(t h w) b c -> b c t h w', t=T, h=H, w=W)
            downsampled_pos_5d = F.interpolate(vision_pos_5d, size=(T_d, H_d, W_d), mode='trilinear',
                                               align_corners=False)
            low_pos = rearrange(downsampled_pos_5d, 'b c t h w -> (t h w) b c')
            high_pos = low_pos.repeat(7, 1, 1)

        k_text = self.with_pos_embed(text_feats, text_pos)
        v_text = text_feats

        out_low_st = self.attn_low_st(query=self.with_pos_embed(q_low_st, low_pos), key=k_text, value=v_text,
                                      key_padding_mask=memory_key_padding_mask)[0]
        out_high_st = self.attn_high_st(query=self.with_pos_embed(q_high_st, high_pos), key=k_text, value=v_text,
                                        key_padding_mask=memory_key_padding_mask)[0]

        out_low_5d = rearrange(out_low_st, '(t h w) b c -> b c t h w', t=T_d, h=H_d, w=W_d)
        coefs_for_low_recon_swapped = torch.zeros(B, C, 8, T_d, H_d, W_d, device=tgt.device)
        coefs_for_low_recon_swapped[:, :, 0, ...] = out_low_5d
        coefs_for_low_recon = rearrange(coefs_for_low_recon_swapped, 'b c d t h w -> b d c t h w', d=8)
        reconstructed_low_5d = idwt3(coefs_for_low_recon, self.wave)
        reconstructed_low_seq = rearrange(reconstructed_low_5d, 'b c t h w -> (t h w) b c')

        out_high_5d_bands = rearrange(out_high_st, '(d t h w) b c -> b c d t h w', d=7, t=T_d, h=H_d, w=W_d)
        coefs_for_high_recon_swapped = torch.zeros(B, C, 8, T_d, H_d, W_d, device=tgt.device)
        coefs_for_high_recon_swapped[:, :, 1:, ...] = out_high_5d_bands
        coefs_for_high_recon = rearrange(coefs_for_high_recon_swapped, 'b c d t h w -> b d c t h w', d=8)
        reconstructed_high_5d = idwt3(coefs_for_high_recon, self.wave)
        reconstructed_high_seq = rearrange(reconstructed_high_5d, 'b c t h w -> (t h w) b c')

        gate_input = torch.cat([reconstructed_low_seq, reconstructed_high_seq], dim=-1)
        gate = self.gate_network(gate_input)
        tgt2 = reconstructed_low_seq + gate * reconstructed_high_seq

        if tgt2.shape[0] > tgt.shape[0]:
            tgt2 = tgt2[:tgt.shape[0], :, :]
        elif tgt2.shape[0] < tgt.shape[0]:
            pad_size = tgt.shape[0] - tgt2.shape[0]
            tgt2 = F.pad(tgt2, (0, 0, 0, 0, 0, pad_size))

        output = tgt * self.norm(tgt2)

        return output