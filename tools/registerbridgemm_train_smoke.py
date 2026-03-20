"""Minimal training smoke entry for RegisterBridgeMM.

This script keeps all integration outside original project files and provides
an explicit Python-only way to exercise the new family on a real environment.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def main():
    root = Path(__file__).resolve().parents[1]
    model_cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"

    # User should edit this path on the server if needed.
    data_cfg = root / "ultralytics" / "cfg" / "datasets" / "mmdata" / "obbmm.yaml"

    model = RegisterBridgeMM(str(model_cfg), task="detect", verbose=True)
    model.train(data=str(data_cfg), epochs=1, imgsz=640, batch=2, workers=0, device=0)


if __name__ == "__main__":
    main()
