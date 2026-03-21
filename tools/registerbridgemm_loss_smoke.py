"""Loss smoke test for RegisterBridgeMM.

Intended for later server-side verification once the environment has torch installed.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "registerbridgemm" / "registerbridge_yolo_dronevehicle.yaml"
    print("[1/6] Instantiate family entry", flush=True)
    cfg_dict = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    model = RegisterBridgeDetectionModel(cfg_dict, nc=cfg_dict["nc"], ch=cfg_dict["channels"], verbose=False)
    print("[2/6] Switch to train mode", flush=True)
    model.train()

    bs = 2
    print("[3/6] Build dummy batch", flush=True)
    imgs = torch.zeros(bs, 6, 256, 256)
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
    print("[4/6] Start forward", flush=True)
    preds = model(batch["img"])
    print("[5/6] Forward done, start loss", flush=True)
    loss, items = model.loss(batch, preds)
    print("[6/6] Loss done", flush=True)
    print("loss:", loss)
    print("items:", items)


if __name__ == "__main__":
    main()
