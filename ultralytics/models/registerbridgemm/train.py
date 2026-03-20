from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.nn.tasks_registerbridge import RegisterBridgeDetectionModel
from ultralytics.utils import RANK


class RegisterBridgeMMTrainer(MultiModalDetectionTrainer):
    """Thin trainer adapter for the RegisterBridgeMM detect family."""

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
