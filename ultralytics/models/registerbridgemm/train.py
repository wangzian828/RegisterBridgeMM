from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel
from ultralytics.utils import RANK
import ultralytics.engine.trainer as trainer_module


class RegisterBridgeMMTrainer(MultiModalDetectionTrainer):
    """Thin trainer adapter for the RegisterBridgeMM detect family."""

    ddp_find_unused_parameters = False

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
