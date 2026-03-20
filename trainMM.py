#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

if __name__ == '__main__':
    # model = YOLOMM('yolo11n-mm-mid-YJ.yaml')
    model = YOLOMM('yolo26n-mm-mid.yaml')
    model.train(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
                epochs=2,batch=16,
                scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                loss26=True,  # 测试 loss26 开启
                # modality='X', 模态消融参数 非必要不得开启
                cache=True,
                exist_ok=True,
                project='ResTest',name='test-yolo26-loss26-on')
