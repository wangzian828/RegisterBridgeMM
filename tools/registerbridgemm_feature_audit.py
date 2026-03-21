"""Audit RegisterBridgeMM feature geometry and scale diversity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Union

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Audit RegisterBridgeMM features")
    parser.add_argument("--model", default="configs/registerbridgemm/registerbridge_yolo_dronevehicle.yaml")
    parser.add_argument("--imgsz", type=int, default=672)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten(1)
    b = b.flatten(1)
    return F.cosine_similarity(a, b, dim=1).mean().item()


def load_model_cfg(model_path: Union[str, Path], backbone: Optional[str], local_files_only: bool):
    model_path = Path(model_path)
    cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    rb_cfg = cfg.setdefault("registerbridge", {})
    if backbone is not None:
        rb_cfg["backbone"] = backbone
    if local_files_only:
        rb_cfg["local_files_only"] = True
    return cfg


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model_cfg = load_model_cfg(args.model, args.backbone, args.local_files_only)
    model = RegisterBridgeDetectionModel(model_cfg, verbose=True).to(device)
    model.eval()

    ch = int(model.yaml.get("channels", 6))
    x = torch.randn(args.batch, ch, args.imgsz, args.imgsz, device=device)

    rb = model.model
    rgb = x[:, :3]
    x_mod = x[:, 3:3 + rb.x_channels]
    rgb_out, x_out = rb.backbone(rgb, x_mod)
    rgb_patches, rgb_regs, rgb_ms = rgb_out
    x_patches, x_regs, x_ms = x_out
    h_patch = rgb.shape[2] // rb.backbone.patch_size
    w_patch = rgb.shape[3] // rb.backbone.patch_size

    if rb.fusion_type == "registerbridge":
        fused_patches, prior = rb.bridge(rgb_patches, x_patches, rgb_regs, x_regs, (h_patch, w_patch))
        fused_ms = list(rgb_ms)
        fused_ms[-1] = fused_patches
        for i in range(len(fused_ms) - 1):
            fused_ms[i] = rb.bridge.fuse_shallow(rgb_ms[i], x_ms[i], prior)
    else:
        fused_ms = [0.5 * (r + y) for r, y in zip(rgb_ms, x_ms)]
        fused_ms[-1] = 0.5 * (rgb_patches + x_patches)

    feats = rb.neck(fused_ms, (h_patch, w_patch))
    rb._update_detect_stride((rgb.shape[2], rgb.shape[3]), feats)

    print(f"input={(args.batch, ch, args.imgsz, args.imgsz)} patch_grid={(h_patch, w_patch)}")
    print(f"detect_stride={rb.detect.stride.tolist()}")
    for i, feat in enumerate(feats, start=3):
        print(
            f"P{i}: shape={tuple(feat.shape)} mean={feat.mean().item():.6f} std={feat.std().item():.6f} "
            f"absmax={feat.abs().max().item():.6f}"
        )

    p3, p4, p5 = feats
    p4_to_p3 = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
    p5_to_p4 = F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
    print(f"cos(P3, up(P4))={cosine_similarity(p3, p4_to_p3):.6f}")
    print(f"cos(P4, up(P5))={cosine_similarity(p4, p5_to_p4):.6f}")


if __name__ == "__main__":
    main()
