"""RegisterBridge global-prior multimodal fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PriorPack:
    F: torch.Tensor
    z: torch.Tensor
    gamma_r: torch.Tensor
    gamma_t: torch.Tensor
    tau: torch.Tensor
    o0_r: Optional[torch.Tensor] = None
    o0_t: Optional[torch.Tensor] = None


class AttentivePatchPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        alpha = self.score(self.norm(patches)).softmax(dim=1)
        return (alpha * patches).sum(dim=1, keepdim=True)


class FusionLatentBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln_u = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, f_lat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q = self.ln1(f_lat)
        kv = self.ln_u(u)
        f_lat = f_lat + self.drop(self.cross_attn(q, kv, kv, need_weights=False)[0])
        q2 = self.ln2(f_lat)
        f_lat = f_lat + self.drop(self.self_attn(q2, q2, q2, need_weights=False)[0])
        return f_lat + self.mlp(self.ln3(f_lat))


class RegisterPriorGenerator(nn.Module):
    def __init__(self, embed_dim=768, n_heads=8, n_points=4, n_fusion_latents=8, n_rounds=1, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.patch_pool = AttentivePatchPool(embed_dim)
        self.fusion_latents = nn.Parameter(torch.randn(n_fusion_latents, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([FusionLatentBlock(embed_dim, n_heads, dropout) for _ in range(n_rounds)])
        self.rgb_gate = nn.Linear(embed_dim, embed_dim)
        self.x_gate = nn.Linear(embed_dim, embed_dim)
        self.tau_head = nn.Linear(embed_dim, n_heads)

    def forward(self, rgb_regs, x_regs, rgb_patches, x_patches) -> PriorPack:
        b = rgb_regs.size(0)
        s_r = self.patch_pool(rgb_patches)
        s_x = self.patch_pool(x_patches)
        u = torch.cat([rgb_regs, s_r, x_regs, s_x], dim=1)
        f_lat = self.fusion_latents.unsqueeze(0).expand(b, -1, -1)
        for blk in self.blocks:
            f_lat = blk(f_lat, u)
        z = f_lat.mean(dim=1)
        gamma_r = torch.sigmoid(self.rgb_gate(z)).unsqueeze(1)
        gamma_x = torch.sigmoid(self.x_gate(z)).unsqueeze(1)
        tau = F.softplus(self.tau_head(z)) + 1e-4
        return PriorPack(F=f_lat, z=z, gamma_r=gamma_r, gamma_t=gamma_x, tau=tau)


class PriorConditionedLocalFusion(nn.Module):
    def __init__(self, embed_dim=768, fusion_dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.rgb_ln = nn.LayerNorm(embed_dim)
        self.x_ln = nn.LayerNorm(embed_dim)
        self.seed_proj = nn.Linear(3 * embed_dim, fusion_dim)
        self.z_proj = nn.Linear(embed_dim, fusion_dim)
        self.context_proj = nn.Linear(2 * fusion_dim, fusion_dim)
        self.latents_proj = nn.Linear(embed_dim, fusion_dim)
        self.writeback_ln = nn.LayerNorm(fusion_dim)
        self.writeback_attn = nn.MultiheadAttention(fusion_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_ln = nn.LayerNorm(fusion_dim)
        self.out_proj = nn.Linear(fusion_dim, embed_dim)

    def forward(self, rgb_patches, x_patches, prior: PriorPack, spatial_shape):
        rgb_g = self.rgb_ln(rgb_patches) * prior.gamma_r
        x_g = self.x_ln(x_patches) * prior.gamma_t
        seed = self.seed_proj(torch.cat([rgb_g, x_g, (rgb_g - x_g).abs()], dim=-1))
        z_exp = self.z_proj(prior.z).unsqueeze(1).expand(-1, seed.shape[1], -1)
        p_loc = seed + self.context_proj(torch.cat([seed, z_exp], dim=-1))
        f_ctx = self.latents_proj(prior.F)
        p_f = p_loc + self.writeback_attn(self.writeback_ln(p_loc), f_ctx, f_ctx, need_weights=False)[0]
        return self.out_proj(self.out_ln(p_f))


class RegisterBridge(nn.Module):
    def __init__(self, embed_dim=768, fusion_dim=256, n_heads=8, n_points=4, n_fusion_latents=8, prior_rounds=1, dropout=0.1):
        super().__init__()
        self.prior_generator = RegisterPriorGenerator(embed_dim, n_heads, n_points, n_fusion_latents, prior_rounds, dropout)
        self.local_fusion = PriorConditionedLocalFusion(embed_dim, fusion_dim, n_heads, dropout)

    def forward(self, rgb_patches, x_patches, rgb_regs, x_regs, spatial_shape):
        prior = self.prior_generator(rgb_regs, x_regs, rgb_patches, x_patches)
        fused = self.local_fusion(rgb_patches, x_patches, prior, spatial_shape)
        return fused, prior

    def fuse_shallow(self, rgb_tokens, x_tokens, prior: PriorPack):
        return 0.5 * ((rgb_tokens * prior.gamma_r) + (x_tokens * prior.gamma_t))
