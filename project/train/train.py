from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn as nn
from project.model.CNN import CNN

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理操作，包括将图像转换为灰度图像
data_transform = transforms.Compose([
    transforms.Grayscale(),  # 将图像转换为灰度图像
    transforms.ToTensor(),  # 将图片转换为Tensor
])

# 加载数据集
dataset = datasets.ImageFolder(root='../../train', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 创建CNN模型实例
num_labels = 65  # 标签种类数量
model = CNN(num_labels)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到GPU
model.to(device)


def train_model(model, criterion, optimizer, num_epochs=5, log_file='loss_log.txt'):
    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader, 1):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 每隔一定的迭代将损失写入文本文件
                if i % 10 == 0:  # 每10次迭代记录一次
                    iter_loss = running_loss / 10
                    print(f'Epoch {epoch + 1}/{num_epochs}, Iteration {i}, Loss: {iter_loss}')
                    f.write(f'Epoch {epoch + 1}, Iteration {i}, Loss: {iter_loss}\n')
                    running_loss = 0.0

            epoch_loss = running_loss / len(data_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}\n')


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


if __name__ == '__main__':
    # 训练模型
    train_model(model, criterion, optimizer, num_epochs=5)
    torch.save(model.state_dict(), 'model2.pth')  # 保存模型参数到文件
    # 测试模型
    test_model(model)
