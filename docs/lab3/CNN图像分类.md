# 实验任务二：使用CNN来进行图像分类
## CIFAR-10 数据集
本次实验使用CIFAR-10 数据集来进行实验。
CIFAR-10 数据集包含 60,000 张 32×32 像素的彩色图像，
分为 10 个类别，每个类别有 6,000 张图像。
具体类别包括飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。
数据集被分为训练集和测试集，
其中训练集包含 50,000 张图像，测试集包含 10,000 张图像。
## CNN图像分类任务
本次任务要求补全代码中空缺部分，包括实现一个CNN类，以及训练过程代码

```bash
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
```
导入CIFAR-10数据集：
```bash
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载训练集
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False
)
```
定义CNN网络：
```bash
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #此处实现模型结构
        #TODO
        #TODO
        #TODO
        #TODO
        #TODO

    def forward(self, x):
        #此处实现模型前向传播
        #TODO
        #TODO
        #TODO
        
        return x
```
训练函数：
```bash
def train(model, train_loader, test_loader, device):
    num_epochs = 15
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            #实现训练部分，完成反向传播过程
            #TODO
            #TODO
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次损失
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # 每个epoch结束后在测试集上评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                #补全预测部分代码，输出模型正确率
                #TODO
                #TODO
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')
```
```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#创建模型
model = SimpleCNN().to(device)
train(model, trainloader, testloader, device)
```
```bash
def denormalize(tensor):
    # 输入是归一化后的张量 [C, H, W]
    # 反归一化：(tensor * std) + mean
    # 原始归一化参数：mean=0.5, std=0.5
    return tensor * 0.5 + 0.5
```
```bash
data_iter = iter(trainloader)
images, labels = next(data_iter)  # 获取第一个batch

# 反归一化并转换为numpy
img = denormalize(images[0]).numpy()  # 取batch中的第一张
img = np.transpose(img, (1, 2, 0))    # 从(C, H, W)转为(H, W, C)

# 显示图像
plt.imshow(img)
plt.title(f"Label: {trainset.classes[labels[0]]}")
plt.axis('off')
plt.show()
```
## 思考题（可选做）：
在实验二中我们实现了在MINST数据集上进行分类，
使用本节的CNN又该如何实现，结合本节内容以及实验二内容尝试实现