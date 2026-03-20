"""Loss smoke test for RegisterBridgeMM.

Intended for later server-side verification once the environment has torch installed.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ultralytics.models.registerbridgemm.model import RegisterBridgeMM


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"
    model = RegisterBridgeMM(str(cfg), task="detect", verbose=False).model
    model.train()

    bs = 2
    imgs = torch.zeros(bs, 6, 640, 640)
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.int64),
        "cls": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.2, 0.2, 0.1, 0.1],
                [0.6, 0.6, 0.15, 0.15],
            ],
            dtype=torch.float32,
        ),
    }
    loss, items = model.loss(batch)
    print("loss:", loss)
    print("items:", items)


if __name__ == "__main__":
    main()
