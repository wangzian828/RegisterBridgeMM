"""Compare coarse prediction-target matching diagnostics on val vs test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.registerbridgemm.model import RegisterBridgeMM
from ultralytics.models.registerbridgemm.val import RegisterBridgeMMValidator
from ultralytics.utils.metrics import box_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Compare matching diagnostics across splits")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--imgsz", type=int, default=672)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--modality", default=None)
    return parser.parse_args()


def summarize(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def build_validator(args):
    overrides = {
        "model": args.weights,
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
        "conf": args.conf,
        "modality": args.modality,
    }
    cfg = get_cfg(overrides=overrides)
    validator = RegisterBridgeMMValidator(args=cfg)
    validator.data = check_det_dataset(args.data)
    validator.device = torch.device("cuda:0" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu")
    validator.args.split = args.split
    validator.stride = torch.tensor([32.0])
    validator.dataloader = validator.get_dataloader(validator.data.get(args.split), args.batch)
    validator.names = validator.data["names"]
    validator.nc = validator.data["nc"]
    validator.metrics.names = validator.names
    dummy_model = SimpleNamespace(names=validator.names, stride=torch.tensor([32.0]), pt=True, jit=False, engine=False)
    validator.init_metrics(dummy_model)
    return validator


@torch.no_grad()
def main():
    args = parse_args()
    model = RegisterBridgeMM(args.weights, task="detect", verbose=True)
    validator = build_validator(args)

    best_pred_iou = []
    best_gt_iou = []
    pred_counts = []
    gt_counts = []

    for batch_i, batch in enumerate(validator.dataloader):
        batch = validator.preprocess(batch)
        preds = model.model(batch["img"])
        preds = validator.postprocess(preds)

        for si, pred in enumerate(preds):
            pbatch = validator._prepare_batch(si, batch)
            predn = validator._prepare_pred(pred, pbatch)
            gt_boxes = pbatch["bboxes"]
            pred_boxes = predn["bboxes"]

            pred_counts.append(int(len(pred_boxes)))
            gt_counts.append(int(len(gt_boxes)))

            if len(gt_boxes) == 0:
                best_gt_iou.append(1.0)
                best_pred_iou.extend([0.0] * len(pred_boxes))
                continue
            if len(pred_boxes) == 0:
                best_gt_iou.extend([0.0] * len(gt_boxes))
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            best_pred_iou.extend(ious.max(dim=1).values.cpu().tolist())
            best_gt_iou.extend(ious.max(dim=0).values.cpu().tolist())

        if args.max_batches and (batch_i + 1) >= args.max_batches:
            break

    print(f"split={args.split}")
    print(f"pred_count={summarize(pred_counts)}")
    print(f"gt_count={summarize(gt_counts)}")
    print(f"best_pred_iou={summarize(best_pred_iou)}")
    print(f"best_gt_iou={summarize(best_gt_iou)}")


if __name__ == "__main__":
    main()
