"""Smoke test for the new RegisterBridgeMM family scaffold."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"
    model = RegisterBridgeMM(str(cfg), task="detect", verbose=False)
    x = torch.zeros(1, 6, 640, 640)
    model.model.eval()
    with torch.no_grad():
        y = model.model(x)
    print(type(y))
    if isinstance(y, tuple):
        print("tuple outputs")
    elif isinstance(y, list):
        print(f"feature levels: {len(y)}")
    else:
        print(getattr(y, "shape", None))


if __name__ == "__main__":
    main()
