"""Minimal validation smoke entry for RegisterBridgeMM."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def main():
    root = Path(__file__).resolve().parents[1]
    model_cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"
    model = RegisterBridgeMM(str(model_cfg), task="detect", verbose=True)
    print("Model instantiated:", model.model.__class__.__name__)
    print("Multimodal config:", getattr(model, "modality_config", {}))


if __name__ == "__main__":
    main()
