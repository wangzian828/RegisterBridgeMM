#这是热力图可视化的方法，请你参考我的使用
#使用方法查看 wiki目录中的vis_usage.md文件
from ultralytics import YOLOMM

model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/Repo/YOLOMM/weights/best.pt')
model.vis(
          rgb_source = '/home/zhizi/work/multimodel/ultralyticmm/00002_rgb.png',  # rgb_source：RGB 输入
          x_source = '/home/zhizi/work/multimodel/ultralyticmm/00002_ir.png',  # x_source：X 模态输入
          method='feature',  #热力图
          layers=[7,15,18,29],  # 使用yaml层
          overlay='rgb',  # 叠加底图：'rgb'|'x'|'dual'，默认'rgb'；仅传X时自动改为'x'
          # modality='X',  # 模态消融：'rgb' 或 'x'；双模态输入时也可强制消融
          alg='gradcam',
        #   split=True,
          save=True,
          project='ResTest/vis',
          name='Vis')
