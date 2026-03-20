#这是提供 参考示例的 YOLOMM推理器使用方法
#按必须按照你的需求来更改而不是盲目去使用


from doctest import debug
from ultralytics import YOLOMM
from ultralytics.tools import MultiModalSampler


def sample_rgb_x_sources(dataset_yaml: str, split: str = "val", seed: int | None = None, index: int | None = None):
    sampler = MultiModalSampler(dataset_yaml, split=split, seed=seed)
    if index is not None:
        return sampler.sample_by_index(index)
    return sampler.sample_one()

dataset_yaml = "/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ultralytics/cfg/datasets/mmdata/edgeODMM.yaml"
#模型权重
model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/yolo11s-xch1-lst/weights/best.pt')
rgb_list = '/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png'
x_list = '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png'
sampler = MultiModalSampler(dataset_yaml, split="train", seed=42)
# rgb_list, x_list = sampler.sample_source_list(n=10)
# 测试双模态预测
print("=== 测试双模态预测 ===")
model.predict(rgb_source=rgb_list,
              x_source=x_list,
              project ='ResTest',
              name='test-yolo26-mm-mid-MuSGD_dual_modal',
              save=True,
              exist_ok=True,
              debug=True,
            #   conf=0.001,
            #   iou=0.01
              )

# 测试单模态RGB预测（使用零填充X模态）
print("\n=== 测试单模态RGB预测 ===")
model.predict(rgb_source=rgb_list,
              x_source=None,
              project ='ResTest',
              name='test-yolo26-mm-mid-MuSGD_single_rgb',
              save=True,
              exist_ok=True,
              debug=True,
            #   conf=0.1
              )

# 测试单模态红外预测（使用零填充RGB模态）
print("\n=== 测试单模态红外预测 ===")
model.predict(rgb_source=None,
              x_source=x_list,
              project ='ResTest',
              name='test-yolo26-mm-mid-MuSGD_single_ir',
              save=True,
              exist_ok=True,
              debug=True,
            #   conf=0.1
              )
