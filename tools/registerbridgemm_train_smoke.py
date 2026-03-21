"""Minimal training smoke entry for RegisterBridgeMM.

This script keeps all integration outside original project files and provides
an explicit Python-only way to exercise the new family on a real environment.
"""

from pathlib import Path
import sys
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def build_model_cfg(model_path: Path, backbone: Optional[str] = None, local_files_only: bool = False) -> Path:
    cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    rb_cfg = cfg.setdefault("registerbridge", {})
    if backbone is not None:
        rb_cfg["backbone"] = backbone
    if local_files_only:
        rb_cfg["local_files_only"] = True
    out = model_path.parent / "registerbridge_yolo_dronevehicle.runtime.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    model_cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"
    runtime_cfg = build_model_cfg(model_cfg)

    # User should edit this path on the server if needed.
    data_cfg = root / "ultralytics" / "cfg" / "datasets" / "mmdata" / "obbmm.yaml"

    model = RegisterBridgeMM(str(runtime_cfg), task="detect", verbose=True)
    model.train(data=str(data_cfg), epochs=1, imgsz=672, batch=2, workers=0, device=0)


if __name__ == "__main__":
    main()
