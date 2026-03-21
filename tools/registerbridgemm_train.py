"""Standard training entry for RegisterBridgeMM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def maybe_launch_ddp() -> None:
    if "LOCAL_RANK" in os.environ:
        return
    device = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--device" and i + 2 <= len(sys.argv[1:]):
            device = sys.argv[i + 2]
            break
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
            break
    if device is None or "," not in str(device):
        return
    nproc = len([x for x in str(device).split(",") if x.strip()])
    port = find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--master_port",
        str(port),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(f"RegisterBridgeMM DDP launch: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RegisterBridgeMM on a full dataset")
    parser.add_argument("--model", default="configs/registerbridgemm/registerbridge_yolo_dronevehicle.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=672)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/registerbridgemm")
    parser.add_argument("--name", default="train_run")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fusion-type", choices=["registerbridge", "simple", "hybrid"], default=None)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    patched_cfg = (output_dir / "model_runtime.yaml").resolve()
    patched_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return patched_cfg


def main():
    maybe_launch_ddp()
    args = parse_args()
    runtime_dir = (Path(args.project) / "_runtime_cfgs").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = build_model_cfg(
        args.model,
        runtime_dir / args.name,
        args.backbone,
        args.local_files_only,
        args.fusion_type,
        args.rgb_unfreeze_last_n,
        args.x_unfreeze_last_n,
    )

    print(f"Model yaml: {model_cfg}")
    if args.fusion_type is not None:
        print(f"Fusion type: {args.fusion_type}")
    if args.rgb_unfreeze_last_n is not None or args.x_unfreeze_last_n is not None:
        print(f"Unfreeze override: rgb={args.rgb_unfreeze_last_n} x={args.x_unfreeze_last_n}")

    model = RegisterBridgeMM(str(model_cfg), task="detect", verbose=True)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
