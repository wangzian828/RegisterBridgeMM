"""RegisterBridge feature pyramid neck."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegisterBridgeFPN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, num_outs: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)) for _ in range(num_outs)
        ])
        self.p3_upsample = nn.Sequential(nn.ConvTranspose2d(out_dim, out_dim, 2, 2), nn.GroupNorm(32, out_dim), nn.GELU())
        self.p4_conv = nn.Sequential(nn.Conv2d(out_dim, out_dim, 1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.p5_downsample = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, 2, 1), nn.GroupNorm(32, out_dim), nn.GELU())

    def forward(self, multi_scale_tokens: list[torch.Tensor], spatial_shape: tuple[int, int]):
        h, w = spatial_shape
        features = []
        for i in range(self.num_outs):
            idx = len(multi_scale_tokens) - self.num_outs + i
            tokens = multi_scale_tokens[idx]
            b = tokens.shape[0]
            feat = self.lateral_convs[i](tokens).permute(0, 2, 1).view(b, self.out_dim, h, w)
            if i == 0:
                feat = self.p3_upsample(feat)
            elif i == 1:
                feat = self.p4_conv(feat)
            else:
                feat = self.p5_downsample(feat)
            features.append(feat)
        return features
