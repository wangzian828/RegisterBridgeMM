"""RegisterBridge multimodal DINOv2 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


class DualDINOv2RegBackbone(nn.Module):
    """Dual-stream DINOv2-with-registers backbone.

    Returns per stream: (patch_tokens, register_tokens, multi_scale_features).
    RGB stream is frozen. X stream is initialized the same way and adapted by LoRA.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-base",
        num_register_tokens: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        multi_scale_layers: tuple[int, ...] = (3, 6, 9, 11),
        x_channels: int = 3,
    ):
        super().__init__()
        self.num_register_tokens = num_register_tokens
        self.multi_scale_layers = multi_scale_layers
        self.x_channels = x_channels

        self.rgb_backbone = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        for p in self.rgb_backbone.parameters():
            p.requires_grad = False

        self.x_backbone = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        for p in self.x_backbone.parameters():
            p.requires_grad = False

        lora_targets = self._resolve_lora_targets(self.x_backbone)
        if lora_targets:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_targets,
                bias="none",
            )
            self.x_backbone = get_peft_model(self.x_backbone, lora_config)

        if x_channels != 3:
            self.x_input_adapter = nn.Conv2d(x_channels, 3, kernel_size=1, bias=True)
            with torch.no_grad():
                self.x_input_adapter.weight.fill_(1.0 / max(x_channels, 1))
                self.x_input_adapter.bias.zero_()
        else:
            self.x_input_adapter = nn.Identity()

        self.rgb_backbone.eval()
        self.rgb_layers = self._resolve_encoder_layers(self.rgb_backbone)
        self.x_layers = self._resolve_encoder_layers(self.x_backbone)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.rgb_backbone.eval()
        return self

    @staticmethod
    def _resolve_lora_targets(model: nn.Module) -> list[str]:
        targets = []
        for name, _ in model.named_modules():
            if any(k in name for k in ("query", "key", "value", "qkv")) and name not in targets:
                targets.append(name)
        return targets

    @staticmethod
    def _iter_backbone_roots(model: nn.Module):
        seen = set()
        queue = [model]
        while queue:
            root = queue.pop(0)
            if root is None or id(root) in seen:
                continue
            seen.add(id(root))
            yield root
            for attr in ("model", "base_model", "module"):
                child = getattr(root, attr, None)
                if child is not None:
                    queue.append(child)

    @classmethod
    def _resolve_encoder_layers(cls, model: nn.Module) -> nn.ModuleList:
        candidate_paths = (("encoder", "layer"), ("encoder", "layers"), ("layer",), ("layers",))
        for root in cls._iter_backbone_roots(model):
            for path in candidate_paths:
                obj = root
                for attr in path:
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if isinstance(obj, nn.ModuleList):
                    return obj
        raise RuntimeError("Could not locate transformer encoder layers for backbone")

    def _extract_features(self, pixel_values: torch.Tensor, backbone: nn.Module, encoder_layers: nn.ModuleList):
        captured = {}
        hooks = []

        def _make_hook(layer_idx: int):
            def _hook(_module, _inputs, output):
                captured[layer_idx] = output[0] if isinstance(output, tuple) else output
            return _hook

        for layer_idx in sorted(set(self.multi_scale_layers)):
            hooks.append(encoder_layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))
        try:
            outputs = backbone(pixel_values=pixel_values, output_hidden_states=False)
        finally:
            for hook in hooks:
                hook.remove()

        last_hidden = outputs.last_hidden_state
        k = self.num_register_tokens
        reg_tokens = last_hidden[:, 1:1 + k, :]
        patch_tokens = last_hidden[:, 1 + k:, :]
        multi_scale = [captured[layer_idx][:, 1 + k:, :] for layer_idx in self.multi_scale_layers]
        return patch_tokens, reg_tokens, multi_scale

    @property
    def embed_dim(self) -> int:
        return self.rgb_backbone.config.hidden_size

    @property
    def patch_size(self) -> int:
        return self.rgb_backbone.config.patch_size

    def forward(self, rgb: torch.Tensor, x: torch.Tensor):
        if x.shape[1] != 3:
            x = self.x_input_adapter(x)
        rgb = (rgb - self.pixel_mean) / self.pixel_std
        x = (x - self.pixel_mean) / self.pixel_std
        with torch.inference_mode():
            rgb_out = self._extract_features(rgb, self.rgb_backbone, self.rgb_layers)
        rgb_out = (rgb_out[0].clone(), rgb_out[1].clone(), [t.clone() for t in rgb_out[2]])
        x_out = self._extract_features(x, self.x_backbone, self.x_layers)
        return rgb_out, x_out
