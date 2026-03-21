"""RegisterBridge-specific model build entrypoints.

This file intentionally stays separate from `ultralytics.nn.tasks` so the original
reference project files remain untouched while we scaffold a new family.
"""

from __future__ import annotations

import yaml
from types import SimpleNamespace

import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel
from ultralytics.nn.modules.registerbridge.model import RegisterBridgeYOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights


class _CriterionProxy(nn.Module):
    """Minimal proxy that matches the attributes expected by v8DetectionLoss."""

    def __init__(self, detect_head: nn.Module, args):
        super().__init__()
        self.model = nn.ModuleList([detect_head])
        self.args = args


class RegisterBridgeDetectionModel(BaseModel):
    """Dedicated RegisterBridge detect model using a standalone build path."""

    def __init__(self, cfg, nc=None, ch=None, verbose=True):
        super().__init__()
        print("[RB-Task] init start", flush=True)
        if isinstance(cfg, (str, bytes)):
            with open(cfg, "r", encoding="utf-8") as f:
                self.yaml = yaml.safe_load(f)
        else:
            self.yaml = dict(cfg)
        self.yaml["nc"] = nc or self.yaml.get("nc", 80)
        self.yaml["channels"] = ch or self.yaml.get("channels", 6)
        model_cfg = self.yaml.get("registerbridge", {})
        self.model = RegisterBridgeYOLO(
            nc=self.yaml["nc"],
            backbone_name=model_cfg.get("backbone", "facebook/dinov2-with-registers-base"),
            x_channels=model_cfg.get("x_channels", 3),
            num_register_tokens=model_cfg.get("num_register_tokens", 4),
            lora_rank=model_cfg.get("lora_rank", 8),
            lora_alpha=model_cfg.get("lora_alpha", 16),
            local_files_only=model_cfg.get("local_files_only", False),
            fusion_type=model_cfg.get("fusion_type", "registerbridge"),
            rgb_unfreeze_last_n=model_cfg.get("rgb_unfreeze_last_n", 0),
            x_unfreeze_last_n=model_cfg.get("x_unfreeze_last_n", 0),
            d_model=model_cfg.get("d_model", 256),
            n_heads=model_cfg.get("n_heads", 8),
            n_points=model_cfg.get("n_points", 4),
            n_fusion_latents=model_cfg.get("n_fusion_latents", 8),
            prior_rounds=model_cfg.get("prior_rounds", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )
        print("[RB-Task] model module built", flush=True)
        self.names = {i: f"class_{i}" for i in range(self.yaml["nc"])}
        self.inplace = True
        self.save = []
        self._criterion_model = None
        self.args = getattr(self, "args", SimpleNamespace(box=7.5, cls=0.5, dfl=1.5))
        initialize_weights(self)
        print("[RB-Task] weights initialized", flush=True)
        self._initialize_stride(ch or self.yaml["channels"])
        print("[RB-Task] stride initialized", flush=True)
        self._intended_trainable = {name for name, p in self.named_parameters() if p.requires_grad}

    def _initialize_stride(self, ch):
        m = self.model.detect
        patch = getattr(self.model.backbone, "patch_size", 14)
        m.stride = torch.tensor([patch / 2.0, float(patch), float(patch) * 2.0])
        self.stride = m.stride
        if hasattr(m, "bias_init") and callable(getattr(m, "bias_init")):
            m.bias_init()

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        return self.model(x)

    def _apply(self, fn):
        self = nn.Module._apply(self, fn)
        m = self.model.detect
        if hasattr(m, "stride") and m.stride is not None:
            m.stride = fn(m.stride)
        if hasattr(m, "anchors") and isinstance(m.anchors, torch.Tensor):
            m.anchors = fn(m.anchors)
        if hasattr(m, "strides") and isinstance(m.strides, torch.Tensor):
            m.strides = fn(m.strides)
        return self

    def init_criterion(self):
        if self._criterion_model is None:
            self._criterion_model = _CriterionProxy(self.model.detect, self.args)
        return v8DetectionLoss(self._criterion_model)

    def loss(self, batch, preds=None):
        return super().loss(batch, preds)
