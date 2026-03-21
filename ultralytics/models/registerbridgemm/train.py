from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel
from ultralytics.utils import RANK
import ultralytics.engine.trainer as trainer_module
from copy import copy
from ultralytics.data.build import build_yolo_dataset
from ultralytics.nn.tasks import de_parallel

from .val import RegisterBridgeMMValidator


class RegisterBridgeMMTrainer(MultiModalDetectionTrainer):
    """Thin trainer adapter for the RegisterBridgeMM detect family."""

    ddp_find_unused_parameters = True

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        trainer_module.check_amp = lambda model: True

    def get_model(self, cfg=None, weights=None, verbose=True):
        channels = self.data.get("channels", 6)
        x_channels = self.data.get("Xch", max(1, channels - 3))
        cfg_dict = dict(cfg) if isinstance(cfg, dict) else None
        if cfg_dict is None and cfg is not None:
            from ultralytics.nn.tasks import yaml_model_load

            cfg_dict = yaml_model_load(cfg)
        if cfg_dict is not None:
            cfg_dict["channels"] = channels
            cfg_dict.setdefault("registerbridge", {})["x_channels"] = x_channels
            cfg = cfg_dict
        model = RegisterBridgeDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=channels,
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def _preserve_registerbridge_freeze(self):
        if self.model is None:
            return
        intended_trainable = getattr(self.model, "_intended_trainable", None)
        if intended_trainable is None:
            return
        preserved = 0
        for name, param in self.model.named_parameters():
            should_train = name in intended_trainable
            if param.requires_grad != should_train:
                param.requires_grad = should_train
                preserved += 1
        if preserved and RANK in {-1, 0}:
            from ultralytics.utils import LOGGER

            LOGGER.info(f"RegisterBridgeMM: restored intended trainability for {preserved} parameters after trainer setup")

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self._preserve_registerbridge_freeze()

    def build_dataset(self, img_path, mode="train", batch=None):
        self.multimodal_config = self._parse_multimodal_config()
        self._validate_modality_compatibility()
        x_modality = [m for m in self.multimodal_config["models"] if m != "rgb"][0]
        x_modality_dir = self.multimodal_config["modalities"][x_modality]
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False if mode == "val" else mode == "val",
            stride=gs,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return RegisterBridgeMMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
