from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.engine.model import Model

from .predict import RegisterBridgeMMPredictor
from .train import RegisterBridgeMMTrainer
from .val import RegisterBridgeMMValidator


class RegisterBridgeMM(Model):
    """New independent multimodal RegisterBridge family entry."""

    def __init__(self, model: Union[str, Path], task: Optional[str] = "detect", verbose: bool = False):
        self.input_channels = None
        self.modality_config = {}
        self.is_multimodal = True
        super().__init__(model=str(model), task=task or "detect", verbose=verbose)
        self._configure_multimodal_settings()

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel

        return {
            "detect": {
                "model": RegisterBridgeDetectionModel,
                "trainer": RegisterBridgeMMTrainer,
                "validator": RegisterBridgeMMValidator,
                "predictor": RegisterBridgeMMPredictor,
            }
        }

    def _configure_multimodal_settings(self) -> None:
        model_yaml = getattr(self.model, "yaml", None)
        if not isinstance(model_yaml, dict):
            self.input_channels = self.input_channels or 6
            self.modality_config = {
                "models": ["rgb", "ir"],
                "modalities": {"rgb": "images", "ir": "images_ir"},
                "x_channels": 3,
            }
            return

        channels = int(model_yaml.get("channels", 6) or 6)
        x_channels = int(model_yaml.get("Xch", max(1, channels - 3)))
        models = model_yaml.get("modality_used", ["rgb", "ir"])
        modalities = model_yaml.get("modalities", {"rgb": "images", models[-1]: f"images_{models[-1]}"})
        self.input_channels = channels
        self.modality_config = {
            "models": models,
            "modalities": modalities,
            "x_channels": x_channels,
        }
