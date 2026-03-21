from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.utils import LOGGER


class RegisterBridgeMMValidator(MultiModalDetectionValidator):
    """Thin validator adapter for the RegisterBridgeMM detect family."""

    def build_dataset(self, img_path, mode="val", batch=None):
        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"RegisterBridgeMM验证配置解析完成 - 模态: {self.multimodal_config['models']}")

        modalities = self.multimodal_config["models"]
        modalities_dict = self.multimodal_config["modalities"]
        x_modalities = [m for m in modalities if m != "rgb"]
        x_modality = x_modalities[0] if x_modalities else None
        x_modality_dir = modalities_dict.get(x_modality) if x_modality else None
        stride = self.stride if hasattr(self, "stride") and self.stride else 32

        LOGGER.info(f"构建RegisterBridgeMM验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}, rect=False")
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            modalities=modalities,
        )
