import torch
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from VideoMamba.videomamba.video_sm.models.videomamba import VisionMamba

class HistoryMamba(VisionMamba):
    def __init__(
            self,
            patch_size=16,
            depth=24,
            embed_dim=192,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            **kwargs
    ):
        super().__init__(
            patch_size=patch_size,
            depth=depth,
            embed_dim=embed_dim,
            fused_add_norm=fused_add_norm,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            **kwargs
        )

        # self.spatial_pool = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.AdaptiveAvgPool1d(1)  # 聚合196个空间token
        # )

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token
        # return hidden_states[:, 0, :]
        print(f"Hidden_States shape: {hidden_states.shape}")
        batch_size, seq_len, hidden_dim = hidden_states[:, 1:, :].shape
        total_frames = T
        patches_per_frame = seq_len // total_frames
        reshaped = hidden_states[:, 1:, :].view(batch_size, total_frames, patches_per_frame, hidden_dim)
        reshaped = reshaped.mean(dim=2)
        return reshaped

    def forward(self, x, inference_params=None):
        # 1. 空间维度聚合
        x_spatial = x.permute(0, 1, 3, 2)  # [B, T, C, S]
        x_pooled = self.spatial_pool(x_spatial)  # [B, T, hidden_dim, 1]
        x_pooled = x_pooled.squeeze(-1)  # [B, 8, hidden_dim]
