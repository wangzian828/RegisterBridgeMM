"""Create a true same-split subset overfit run for RegisterBridgeMM.

This script samples N train images, writes temporary train/val txt files pointing to the
same subset, builds a temporary data yaml, and launches training with those files.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def parse_args():
    parser = argparse.ArgumentParser(description="RegisterBridgeMM true subset overfit")
    parser.add_argument("--model", default="configs/registerbridgemm/registerbridge_yolo_dronevehicle.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=672)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/registerbridgemm")
    parser.add_argument("--name", default="overfit_subset")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fusion-type", choices=["registerbridge", "simple"], default=None)
    parser.add_argument("--rgb-unfreeze-last-n", type=int, default=None)
    parser.add_argument("--x-unfreeze-last-n", type=int, default=None)
    return parser.parse_args()


def build_model_cfg(
    model_path: str,
    output_dir: Path,
    backbone: Optional[str],
    local_files_only: bool,
    fusion_type: Optional[str],
    rgb_unfreeze_last_n: Optional[int],
    x_unfreeze_last_n: Optional[int],
) -> Path:
    cfg_path = Path(model_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    rb_cfg = cfg.setdefault("registerbridge", {})
    if backbone is not None:
        rb_cfg["backbone"] = backbone
    if local_files_only:
        rb_cfg["local_files_only"] = True
    if fusion_type is not None:
        rb_cfg["fusion_type"] = fusion_type
    if rgb_unfreeze_last_n is not None:
        rb_cfg["rgb_unfreeze_last_n"] = rgb_unfreeze_last_n
    if x_unfreeze_last_n is not None:
        rb_cfg["x_unfreeze_last_n"] = x_unfreeze_last_n
    patched_cfg = (output_dir / "model_runtime.yaml").resolve()
    patched_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return patched_cfg


def resolve_entry(root: Path, value: str | list) -> list[Path]:
    if isinstance(value, list):
        files = []
        for v in value:
            files.extend(resolve_entry(root, v))
        return files
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    if p.is_file() and p.suffix.lower() == ".txt":
        lines = [Path(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [x if x.is_absolute() else (p.parent / x).resolve() for x in lines]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([x.resolve() for x in p.rglob("*") if x.suffix.lower() in exts])
    raise FileNotFoundError(f"Unsupported dataset entry: {value}")


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data_cfg.get("path", data_path.parent)).resolve()

    train_files = resolve_entry(root, data_cfg["train"])
    if args.samples >= len(train_files):
        subset = train_files
    else:
        rng = random.Random(args.seed)
        subset = train_files[:]
        rng.shuffle(subset)
        subset = sorted(subset[: args.samples])

    temp_dir = (Path("runs") / "registerbridgemm_subset").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    train_txt = (temp_dir / "train_subset.txt").resolve()
    val_txt = (temp_dir / "val_subset.txt").resolve()
    train_txt.write_text("\n".join(str(p) for p in subset) + "\n", encoding="utf-8")
    val_txt.write_text(train_txt.read_text(encoding="utf-8"), encoding="utf-8")

    subset_cfg = dict(data_cfg)
    subset_cfg["train"] = str(train_txt)
    subset_cfg["val"] = str(val_txt)
    subset_cfg["test"] = str(val_txt)
    subset_yaml = (temp_dir / "subset_data.yaml").resolve()
    subset_yaml.write_text(yaml.safe_dump(subset_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model_cfg = build_model_cfg(
        args.model,
        temp_dir,
        args.backbone,
        args.local_files_only,
        args.fusion_type,
        args.rgb_unfreeze_last_n,
        args.x_unfreeze_last_n,
    )

    if args.imgsz % 28 != 0:
        print(f"Warning: imgsz={args.imgsz} is not divisible by 28; DINO patch/downsample geometry will use fractional YOLO strides.")

    print(f"Subset images: {len(subset)}")
    print(f"Subset yaml: {subset_yaml}")
    print(f"Model yaml: {model_cfg}")
    if args.fusion_type is not None:
        print(f"Fusion type: {args.fusion_type}")
    if args.rgb_unfreeze_last_n is not None or args.x_unfreeze_last_n is not None:
        print(
            f"Unfreeze override: rgb={args.rgb_unfreeze_last_n} x={args.x_unfreeze_last_n}"
        )

    model = RegisterBridgeMM(str(model_cfg), task="detect", verbose=True)
    model.train(
        data=str(subset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        val=True,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
