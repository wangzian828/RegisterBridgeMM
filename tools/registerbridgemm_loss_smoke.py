"""Minimal loss smoke for RegisterBridgeDetectionModel.

This script intentionally mirrors the known-good interactive construction path,
then adds forward/loss stages one by one.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel


def main():
    cfg_path = Path("configs/registerbridgemm/registerbridge_yolo_dronevehicle.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("[1/7] Load yaml", flush=True)
    print("[2/7] Build task model", flush=True)
    model = RegisterBridgeDetectionModel(cfg, nc=cfg["nc"], ch=cfg["channels"], verbose=False)

    print("[3/7] Switch to train mode", flush=True)
    model.train()

    bs = 2
    print("[4/7] Build dummy batch", flush=True)
    imgs = torch.zeros(bs, cfg["channels"], 256, 256)
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

    print("[5/7] Start forward", flush=True)
    preds = model(batch["img"])
    print("[6/7] Forward done, start loss", flush=True)
    loss, items = model.loss(batch, preds)
    print("[7/7] Loss done", flush=True)
    print("loss:", loss)
    print("items:", items)


if __name__ == "__main__":
    main()
