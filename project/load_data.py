import torch
from torchvision import transforms, datasets

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理操作，包括将图像转换为灰度图像
data_transform = transforms.Compose([
    transforms.Grayscale(),  # 将图像转换为灰度图像
    transforms.ToTensor(),  # 将图片转换为Tensor
])

# 加载数据集
dataset = datasets.ImageFolder(root='../train', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_labels):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x



# 创建CNN模型实例
num_labels = 65  # 标签种类数量
model = CNN(num_labels)

import torch
import torch.nn as nn
import torch.optim as optim


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到GPU
model.to(device)


def train_model(model, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')


def test_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')


# 训练模型
train_model(model, criterion, optimizer, num_epochs=5)
torch.save(model.state_dict(), 'model.pth')  # 保存模型参数到文件
# 测试模型
test_model(model)
