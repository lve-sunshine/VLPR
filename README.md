# VLPR
基于CNN实现对单独数字，英文，中国省份简称的65分类识别任务，总体Accuracy: 99.176%，数字+字母Accuracy: 99.529%，省份简称汉字Accuracy: 97.496%。模型分类效果非常好。项目在`master`分支。

## 1 数据集
本项目数据集包含了65个类别，总计16148张20*20大小的三通道图片，所在路径`/train/`,结构如下：
![image](https://github.com/lve-sunshine/VLPR/assets/99074010/b1706b49-5a87-42aa-85ae-fb642375f966)

## 2 数据集加载与预处理
详见`/project/train/train.py`。

### 2.1 数据预处理
本项目原始数据集已处理非常合适，所有图片大小均为20*20，因此只进行灰度转换与张量转换处理。在车牌识别中，颜色的影响不是非常需要，因此将彩色图像转换为灰度图像，灰度图像只包含亮度信息，这样大大减少了数据维度，便于加快训练速度。
```python
from torchvision import transforms
# 定义数据预处理操作，包括将图像转换为灰度图像
data_transform = transforms.Compose([
    transforms.Grayscale(),  # 将图像转换为灰度图像
    transforms.ToTensor(),  # 将图片转换为Tensor
])
```

**注意，若实际使用单张测试需将图片大小转换为20*20，否则请自行计算修改网络结构各层通道数**

### 2.2 数据集加载
```python
from torchvision import datasets
# 加载数据集
dataset = datasets.ImageFolder(root='../../train', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```
本文项目数据集层级明显，具有目录层次结构，因此直接使用`torchvison`的`ImageFolder`数据集类加载。

ImageFolder是PyTorch中的一个数据集类，用于处理图像数据集。它假定数据集的组织形式为以下结构：
```bash
root/class1/image1.jpg
root/class1/image2.jpg
...
root/class2/image1.jpg
root/class2/image2.jpg
...
```
其中，root是数据集的根目录，每个子文件夹（如class1、class2等）包含了同一个类别（label）的图像样本。

在使用ImageFolder时，只需要指定数据集的根目录root，ImageFolder会自动扫描root下的子文件夹，并将每个子文件夹视为一个类别，每个图片文件则被视为该类别的一个样本。这样，ImageFolder会根据文件夹名称自动为每个样本分配类别标签。

通过使用ImageFolder，可以方便地加载和处理符合上述数据组织形式的图像数据集，无需手动处理类别信息，使得数据加载更加简洁和高效。

## 3 构建CNN网络结构
网络结构所在目录`/project/model/CNN.py`。内容如下：
```python
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
```
使用了一个简单的卷积神经网络模型，用于图像分类任务。

该模型的结构如下：

- 第一层卷积层（self.conv1）：输入通道数为1（灰度图像），输出通道数为16，使用3x3的卷积核，步长为1，填充为1。
- ReLU激活函数（self.relu1）：对卷积层的输出进行非线性激活。
- 最大池化层（self.pool1）：使用2x2的池化窗口进行最大池化操作，步长为2。
- 第二层卷积层（self.conv2）：输入通道数为16，输出通道数为32，使用3x3的卷积核，步长为1，填充为1。
- ReLU激活函数（self.relu2）：对卷积层的输出进行非线性激活。
- 最大池化层（self.pool2）：使用2x2的池化窗口进行最大池化操作，步长为2。
- 全连接层1（self.fc1）：将特征展开为一维向量，输入大小为32x5x5（经过两次池化后的尺寸），输出大小为128。
- ReLU激活函数（self.relu3）：对全连接层1的输出进行非线性激活。
- 全连接层2（self.fc2）：输入大小为128，输出大小为标签种类数量（在这里是65）。
- 在前向传播方法（forward）中，输入数据首先经过卷积层1、ReLU激活函数和池化层1，然后再经过卷积层2、ReLU激活函数和池化层2。接着，将特征展开为一维向量，并通过全连接层1和ReLU激活函数得到中间特征表示。最后，通过全连接层2得到最终的分类结果。

## 4 超参数定义
```python
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

对于给定的CNN网络结构，结合已经定义的超参数和模型结构来解释超参数的影响：

- 学习率（lr=0.001）：学习率决定了优化算法在更新模型参数时的步长大小。对于这个CNN模型，学习率的设置会影响参数更新的速度和模型收敛的效果。如果学习率设置过大，可能导致训练不稳定或震荡；如果学习率设置过小，可能导致收敛速度缓慢。由于Adam优化器具有自适应学习率的特性，通常可以较稳定地使用较小的学习率。
- 优化器（Adam）：Adam优化器通过自适应地调整每个参数的学习率，可以更有效地更新模型参数。在这个CNN模型中，使用Adam优化器可以帮助模型更快地收敛，并且能够处理不同参数的学习率需求，从而提高训练效率。
- 损失函数（交叉熵损失函数）：交叉熵损失函数适用于多分类问题，可以衡量模型输出与真实标签之间的差异。对于这个CNN模型，使用交叉熵损失函数可以帮助模型学习正确的类别预测，促进模型在训练过程中更好地优化参数。

通过调整学习率、优化器类型和损失函数来影响模型的训练效果和性能表现。

## 5 模型训练与测试
### 5.1 模型训练
```python
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
```
训练模型的函数`train_model`如上，用于在给定数据集上训练CNN模型。

函数的输入参数包括：
- model：要训练的CNN模型实例。
- criterion：损失函数，用于计算模型预测值和真实标签之间的差异。
- optimizer：优化器，用于更新模型的参数。
- num_epochs：训练的总轮数，默认为5轮。
- log_file：损失记录文件的路径，默认为'loss_log.txt'。

大体思路为：
1. 在函数内部，通过一个循环来遍历每个训练轮数（epoch）。在每个训练轮数中，模型将被设置为训练模式（model.train()），并初始化运行损失（running_loss）为0.0。

2. 然后，通过一个循环遍历数据加载器（data_loader），获取每个批次的输入和标签。将输入和标签移动到设备（GPU或CPU）上，并将优化器的梯度缓存清零（optimizer.zero_grad()）。

3. 接下来，将输入传递给模型进行前向计算（outputs = model(inputs)），并计算预测值与真实标签之间的损失（loss = criterion(outputs, labels)）。然后，通过反向传播计算梯度（loss.backward()）并更新模型的参数（optimizer.step()）。

4. 在每个批次的循环中，将损失值累加到running_loss中，并根据设置的迭代次数（这里是每10次）将损失写入文本文件log_file。当迭代次数达到设定值时，将累计的损失除以迭代次数得到迭代平均损失，并将其打印出来。

5. 在每个训练轮数结束后，将累积的损失除以数据加载器的长度得到该轮的平均损失，并将其打印出来并写入文本文件。

整个训练过程将重复执行num_epochs轮（本项目模型5轮），最终得到训练完毕的模型。

### 5.2 模型训练效果
最终loss为`0.00035873101609251875`,详细损失变化详见`/project/train/loss_log.txt`，可自行可视化损失分析。

### 5.3 测试函数
```python
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
```
本项目直接在训练集测试了训练效果，若需测试测试集请自行修改`data_loader`。

### 5.4 模型测试效果
本项目测试在训练集中：总体Accuracy: 99.176%，数字+字母Accuracy: 99.529%，省份简称汉字Accuracy: 97.496。请自行测试测试集。

### 6 模型使用
本项目中已包含存在路径为`/project/model.pth`的可靠训练模型供直接使用，`/project/useModel`目录下存在单例测试图片与使用该模型进行单例识别的示例，如需实际应用模型可直接参考使用。
```python
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
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin', 'zh_jing',
               'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong', 'zh_shan', 'zh_su',
               'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun', 'zh_zang', 'zh_zhe']

# 加载单张图片
image_path = '鲁.jpg'  # 替换为你的图片路径
image = Image.open(image_path)

# 预处理图像
input_image = data_transform(image).unsqueeze(0)  # 添加一个维度以匹配模型输入要求

# 使用模型进行推断
with torch.no_grad():
    output = model(input_image)
    print(f"识别结果为: {labels_name[output.argmax(dim=1).item()]}")
```

![image](https://github.com/lve-sunshine/VLPR/assets/99074010/8503dc1c-f391-454d-9ed4-8420fb18cf30)


![image](https://github.com/lve-sunshine/VLPR/assets/99074010/edad2af9-0374-4418-80ce-a8d0a998b6a0)


![image](https://github.com/lve-sunshine/VLPR/assets/99074010/6393b3e0-1c1b-4f7d-a154-9827fc3de749)

