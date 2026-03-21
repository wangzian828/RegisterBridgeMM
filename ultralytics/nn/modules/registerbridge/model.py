"""Standalone RegisterBridge YOLO model module scaffold."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import Detect

from .backbone import DualDINOv2RegBackbone
from .bridge import RegisterBridge
from .neck import RegisterBridgeFPN


class RegisterBridgeYOLO(nn.Module):
    """First-pass multimodal feature extractor + YOLO head scaffold.

    Expects a concatenated Dual input with channels [RGB(3), X(Cx)].
    """

    def __init__(
        self,
        nc: int = 80,
        backbone_name: str = "facebook/dinov2-with-registers-base",
        x_channels: int = 3,
        num_register_tokens: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        n_fusion_latents: int = 8,
        prior_rounds: int = 1,
        dropout: float = 0.1,
        multi_scale_layers: tuple[int, ...] = (3, 6, 9, 11),
        local_files_only: bool = False,
        fusion_type: str = "registerbridge",
    ):
        super().__init__()
        self.x_channels = x_channels
        self.fusion_type = fusion_type
        self.backbone = DualDINOv2RegBackbone(
            model_name=backbone_name,
            num_register_tokens=num_register_tokens,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            multi_scale_layers=multi_scale_layers,
            x_channels=x_channels,
            local_files_only=local_files_only,
        )
        backbone_dim = self.backbone.embed_dim
        self.bridge = RegisterBridge(
            embed_dim=backbone_dim,
            fusion_dim=d_model,
            n_heads=n_heads,
            n_points=n_points,
            n_fusion_latents=n_fusion_latents,
            prior_rounds=prior_rounds,
            dropout=dropout,
        ) if fusion_type == "registerbridge" else None
        self.neck = RegisterBridgeFPN(in_dim=backbone_dim, out_dim=d_model, num_outs=3)
        self.detect = Detect(nc=nc, ch=(d_model, d_model, d_model))

    def forward(self, x: torch.Tensor):
        rgb = x[:, :3]
        x_mod = x[:, 3:3 + self.x_channels]
        rgb_out, x_out = self.backbone(rgb, x_mod)
        rgb_patches, rgb_regs, rgb_ms = rgb_out
        x_patches, x_regs, x_ms = x_out
        h_patch = rgb.shape[2] // self.backbone.patch_size
        w_patch = rgb.shape[3] // self.backbone.patch_size
        if self.fusion_type == "registerbridge":
            fused_patches, prior = self.bridge(rgb_patches, x_patches, rgb_regs, x_regs, (h_patch, w_patch))
            fused_ms = list(rgb_ms)
            fused_ms[-1] = fused_patches
            for i in range(len(fused_ms) - 1):
                fused_ms[i] = self.bridge.fuse_shallow(rgb_ms[i], x_ms[i], prior)
        else:
            fused_ms = [0.5 * (r + x) for r, x in zip(rgb_ms, x_ms)]
            fused_ms[-1] = 0.5 * (rgb_patches + x_patches)
        feats = self.neck(fused_ms, (h_patch, w_patch))
        return self.detect(feats)
