"""RegisterBridge feature pyramid neck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegisterBridgeFPN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, num_outs: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_outs = num_outs
        if num_outs != 3:
            raise ValueError(f"RegisterBridgeFPN currently expects num_outs=3, got {num_outs}")

        self.token_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)) for _ in range(num_outs)
        ])
        self.lateral_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
            )
            for _ in range(num_outs)
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
            )
            for _ in range(num_outs)
        ])
        self.p3_out = nn.Sequential(
            nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        self.p4_out = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )
        self.p5_out = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
        )

    def _tokens_to_map(self, tokens: torch.Tensor, proj: nn.Module, h: int, w: int) -> torch.Tensor:
        b, n, _ = tokens.shape
        if n != h * w:
            raise RuntimeError(f"RegisterBridgeFPN token mismatch: expected {h * w}, got {n}")
        return proj(tokens).permute(0, 2, 1).reshape(b, self.out_dim, h, w)

    def forward(self, multi_scale_tokens: list[torch.Tensor], spatial_shape: tuple[int, int]):
        h, w = spatial_shape
        selected = multi_scale_tokens[-self.num_outs :]
        c3, c4, c5 = [
            refine(self._tokens_to_map(tokens, proj, h, w))
            for tokens, proj, refine in zip(selected, self.token_proj, self.lateral_refine)
        ]

        p5_base = self.smooth[2](c5)
        p4_base = self.smooth[1](c4 + p5_base)
        p3_base = self.smooth[0](c3 + p4_base)

        p3 = self.p3_out(p3_base)
        p4 = self.p4_out(p4_base)
        p5 = self.p5_out(p5_base)

        target_p4 = (h, w)
        target_p3 = (h * 2, w * 2)
        target_p5 = ((h + 1) // 2, (w + 1) // 2)
        if p3.shape[-2:] != target_p3:
            p3 = F.interpolate(p3, size=target_p3, mode="bilinear", align_corners=False)
        if p4.shape[-2:] != target_p4:
            p4 = F.interpolate(p4, size=target_p4, mode="bilinear", align_corners=False)
        if p5.shape[-2:] != target_p5:
            p5 = F.interpolate(p5, size=target_p5, mode="bilinear", align_corners=False)
        return [p3, p4, p5]
