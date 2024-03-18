import torch
from PIL import Image
from torchvision import transforms

from project.model.CNN import CNN

model = CNN(65)
model.load_state_dict(torch.load('../model.pth'))

# 使用cpu加载模型
model.to("cpu")

# 开启模型评估模式
model.eval()

# 定义预处理操作
data_transform = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
               'X', 'Y', 'Z',
               'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
               'zh_jing',
               'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong', 'zh_shan', 'zh_su',
               'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun', 'zh_zang', 'zh_zhe']

# 加载单张图片
image_path = '甘.png'  # 替换为你的图片路径
image = Image.open(image_path)

# 预处理图像
input_image = data_transform(image).unsqueeze(0)  # 添加一个维度以匹配模型输入要求

# 使用模型进行推断
with torch.no_grad():
    output = model(input_image)
    print('image_path:', image_path)
    print('image_tensor__shape: ', input_image.shape)
    print(f"识别结果为: {labels_name[output.argmax(dim=1).item()]}")
