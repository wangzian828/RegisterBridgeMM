"""Sanity-check detection evaluation on a dataset split using GT-as-predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.registerbridgemm.val import RegisterBridgeMMValidator


def parse_args():
    parser = argparse.ArgumentParser(description="GT-as-predictions sanity check for RegisterBridgeMM evaluation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=672)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def build_validator(args):
    overrides = {
        "model": "configs/registerbridgemm/registerbridge_yolo_dronevehicle.yaml",
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "split": args.split,
        "task": "detect",
        "mode": "val",
        "rect": False,
        "plots": False,
        "save_json": False,
        "half": False,
    }
    cfg = get_cfg(overrides=overrides)
    validator = RegisterBridgeMMValidator(args=cfg)
    validator.data = check_det_dataset(args.data)
    validator.device = torch.device("cuda:0" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu")
    validator.stride = 32
    validator.args.split = args.split
    validator.dataloader = validator.get_dataloader(validator.data.get(args.split), args.batch)
    validator.nc = validator.data["nc"]
    validator.names = validator.data["names"]
    validator.metrics.names = validator.names
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.init_metrics(None)
    validator.confusion_matrix = validator.metrics.confusion_matrix
    return validator


def make_gt_predictions(batch):
    preds = []
    batch_idx = batch["batch_idx"].view(-1)
    cls = batch["cls"].view(-1)
    bboxes = batch["bboxes"]
    for i in range(len(batch["im_file"])):
        mask = batch_idx == i
        if mask.any():
            boxes = bboxes[mask].clone()
            labels = cls[mask].float().view(-1, 1)
            conf = torch.ones((boxes.shape[0], 1), device=boxes.device, dtype=boxes.dtype) * 0.999
            preds.append(torch.cat([boxes, conf, labels], dim=1))
        else:
            preds.append(torch.zeros((0, 6), device=bboxes.device, dtype=bboxes.dtype))
    return preds


@torch.no_grad()
def main():
    args = parse_args()
    validator = build_validator(args)
    validator.metrics.clear_stats()

    for batch_i, batch in enumerate(validator.dataloader):
        batch = validator.preprocess(batch)
        preds = make_gt_predictions(batch)
        validator.update_metrics(preds, batch)
        if args.max_batches and (batch_i + 1) >= args.max_batches:
            break

    stats = validator.get_stats()
    validator.print_results()
    print("Sanity stats:")
    for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        print(f"{key}={stats.get(key, 0.0):.6f}")


if __name__ == "__main__":
    main()
