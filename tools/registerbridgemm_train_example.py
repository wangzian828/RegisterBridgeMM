"""Example explicit Python entry for RegisterBridgeMM training.

This avoids relying on filename heuristics or modifying the original `ultralytics`
 package exports.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"
    model = RegisterBridgeMM(str(cfg), task="detect", verbose=True)
    print(model)
    print("Task map:", model.task_map)
    print("Underlying model class:", model.model.__class__.__name__)


if __name__ == "__main__":
    main()
