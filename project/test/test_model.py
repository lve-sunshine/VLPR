import torch
from torchvision import transforms, datasets

# 检查是否有可用的 GPU
from project.model.CNN import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理操作，包括将图像转换为灰度图像
data_transform = transforms.Compose([
    transforms.Grayscale(),  # 将图像转换为灰度图像
    transforms.ToTensor(),  # 将图片转换为Tensor
])

# 加载数据集
dataset = datasets.ImageFolder(root='../../train', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# 加载模型
model = CNN(65)
model.load_state_dict(torch.load('../model.pth'))
model.to(device)


# 测试函数
def test_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            if labels[0] <= 34:
                continue
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')


test_model(model)
