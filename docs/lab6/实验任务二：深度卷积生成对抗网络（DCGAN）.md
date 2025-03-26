# 实验任务二：深度卷积对抗生成网络（DCGAN）

!!! success "目标"

    - 了解 DCGAN 与标准 GAN 的不同之处。
    
    - 掌握 DCGAN 的生成器（Generator）和判别器（Discriminator）设计。
    
    - 使用 PyTorch 搭建并训练 DCGAN 生成 MNIST 手写数字。
    
    - 学习如何优化 GAN 训练以获得更稳定的结果。

## 1. GAN 与 DCGAN 的区别

标准 GAN 主要由全连接层构成，生成器使用全连接网络从随机噪声生成数据，而判别器使用全连接网络对输入数据进行分类。GAN 存在的问题包括：

- 训练不稳定，容易出现模式崩溃（Mode Collapse）。
- 生成的图像质量较低，缺乏空间结构信息。

深度卷积生成对抗网络（DCGAN，Deep Convolutional Generative Adversarial Network）是生成对抗网络（GAN）的一种扩展，它通过使用卷积神经网络（CNN）来实现生成器和判别器的构建。与标准的GAN相比，DCGAN通过引入卷积层来改善图像生成质量，使得生成器能够生成更清晰、更高分辨率的图像。

DCGAN（Deep Convolutional GAN）引入卷积神经网络（CNN）来改进 GAN，使其在生成高分辨率的图像时表现更好。

!!! note "DCGAN 相比普通 GAN 的改进"

    •**卷积层替代全连接层**（提高图像质量）。传统的GAN使用全连接层，而DCGAN将其替换为卷积层。卷积层在处理图像时能够更好地保留图像的空间结构，从而生成更为清晰的图像。
    
    •**使用 BatchNorm**（稳定训练）。
    
    什么是BatchNorm：*Batch Normalization（批归一化）是一种用于**加速训练和稳定梯度**的技术，它的核心思想是**对 mini-batch 内的特征进行归一化**。*
    
    为什么要用BatchNorm：*BatchNorm 使数据的分布更加稳定，从而**防止梯度在深层网络中消失或爆炸**，提高训练的稳定性、由于 BatchNorm 归一化了激活值，使得模型对不同初始权重更加鲁棒，因此可以使用**较大的学习率**，加快收敛速度。*
    
    •**LeakyReLU 代替 ReLU**（防止梯度消失）。
    
    •**Tanh 作为生成器输出激活函数**（适应数据范围）。

如需更加深入的学习，可参考该论文：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](./Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.pdf)

## 2. DCGAN的实现

部分细节可参考实验任务一。

#### 生成器

!!! note "归一化（Batch Normalization）"

     - 加速训练：归一化输入数据，让数据分布更加稳定，提高训练效率。
    	
     - 防止梯度消失或梯度爆炸：避免模型在训练过程中出现数值不稳定的情况。
    	
     - 引入一定的正则化效果：减少模型对特定输入模式的依赖，提高泛化能力。


!!! note "反卷积（Transposed Convolution）"

     - 反卷积（也叫 上采样卷积）用于增大特征图的尺寸，最终生成目标大小的图像。
     
     - 通过学习权重，在空间上扩展特征图，相当于卷积的逆操作。
     
     self.conv1 = nn.Sequential(
     
       nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (7,7) -> (14,14)
        
       nn.BatchNorm2d(64),
        
       nn.ReLU())
       
     - 输入通道数 128，输出通道数 64
    
     - kernel_size=4（4×4 卷积核）
    
     - stride=2（步长 2，扩大一倍）
     
     - padding=1（保持尺寸一致）


```python
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        # 1. 输入层：将 100 维随机噪声投影到 32x32（1024 维）
        #TODO   # 线性变换fc1，将输入噪声扩展到 1024 维

        self.br1 = nn.Sequential(
            #TODO   # 批归一化，加速训练并稳定收敛
            #TODO   # ReLU 激活函数，引入非线性
        )

        # 2. 第二层：将 1024 维数据映射到 128 * 7 * 7 的特征图
        #TODO   # 线性变换fc2，将数据变换为适合卷积层的格式

        self.br2 = nn.Sequential(
            #TODO   # 批归一化
            #TODO   # ReLU 激活函数
        )

        # 3. 反卷积层 1：上采样，输出 64 通道的 14×14 特征图
        self.conv1 = nn.Sequential(
            #TODO   # 反卷积：将 7x7 放大到 14x14, kernel size设置为4, stride设置为2，padding设置为1
            #TODO   # 归一化，稳定训练
            #TODO   # ReLU 激活函数
        )

        # 4. 反卷积层 2：输出 1 通道的 28×28 图像
        self.conv2 = nn.Sequential(
            #TODO   # 反卷积：将 14x14 放大到 28x28
            #TODO   # Sigmoid 激活函数，将输出归一化到 [0,1]
        )

    def forward(self, x):
        x = self.br1(self.fc1(x))  # 通过全连接层，进行 BatchNorm 和 ReLU 激活
        x = self.br2(self.fc2(x))  # 继续通过全连接层，进行 BatchNorm 和 ReLU 激活
        x = x.reshape(-1, 128, 7, 7)  # 变形为适合卷积输入的形状 (batch, 128, 7, 7)
        x = self.conv1(x)  # 反卷积：上采样到 14x14
        output = self.conv2(x)  # 反卷积：上采样到 28x28
        return output  # 返回生成的图像
```

####  判别器

DCGAN 的判别器使用多个卷积层对输入图像进行特征提取，并最终输出真假概率。

在 **DCGAN** 中，判别器是一个 **卷积神经网络（CNN）**，主要有：

- **多个卷积层（Conv2D）**：提取局部特征，如边缘、纹理。

- **LeakyReLU 激活函数**：相比于 ReLU，它可以防止梯度消失问题。

- **最大池化**：降采样，根据kernel_size,和stride降低特征图的尺寸。

- **全连接层（Linear）**：最终映射到 [0,1] 的概率。

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 1. 第一层：输入 1 通道的 28x28 图像，输出 32 通道的特征图，然后通过MaxPool2d降采样
        self.conv1 = nn.Sequential(
            #TODO  # 5x5 卷积核，步长为1
            #TODO   # LeakyReLU，negativ
            # e_slope参数设置为0.1
        )
        self.pl1 = nn.MaxPool2d(2, stride=2)

        # 2. 第二层：输入 32 通道，输出 64 通道特征
        self.conv2 = nn.Sequential(
            #TODO   # 5x5 卷积核，步长 2 继续降采样
            #TODO   # 归一化
            #TODO  # LeakyReLU 激活函数，negative_slope参数设置为0.1
        )
        self.pl2 = nn.MaxPool2d(2, stride=2)

        # 3. 全连接层 1：将 7x7x64 维特征图转换成 1024 维向量
        self.fc1 = nn.Sequential(
            #TODO   # 线性变换，将 64x7x7 映射到 1024 维
            #TODO   # LeakyReLU 激活函数，negative_slope参数设置为0.1
        )

        # 4. 全连接层 2：最终输出真假概率
        self.fc2 = nn.Sequential(
            #TODO   # 线性变换，将 1024 维数据映射到 1 维
            #TODO   # Sigmoid 归一化到 [0,1] 作为概率输出
        )

    def forward(self, x):
        x = self.pl1(self.conv1(x))  # 第一层卷积，降维
        x = self.pl2(self.conv2(x))  # 第二层卷积，降维
        x = x.view(x.shape[0], -1)  # 展平成向量
        x = self.fc1(x)  # 通过全连接层
        output = self.fc2(x)  # 通过最后一层全连接层，输出真假概率
        return output  # 返回判别结果
```

训练过程及数据保存参考实验任务一

```python

def train_discriminator(real_images, D, G, loss_func, optim_D, batch_size, input_dim, device):
    # TODO
    return loss_D.item()

def train_generator(D, G, loss_func, optim_G, batch_size, input_dim, device):
    # TODO
    return loss_G.item()

def main():
    # 设备配置：使用 GPU（如果可用），否则使用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 设定超参数
    input_dim = 100  # 生成器输入的随机噪声向量维度
    batch_size = 128  # 训练时的批量大小
    num_epoch = 10  # 训练的总轮数

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 创建生成器和判别器，并移动到 GPU（如果可用）
    # TODO
    # TODO

    # 定义优化器
    # TODO
    # TODO

    loss_func = nn.BCELoss()

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir='./logs/experiment_dcgan')

    # 开始训练
    for epoch in range(num_epoch):
        total_loss_D, total_loss_G = 0, 0
        for i, (real_images, _) in enumerate(train_loader):
            loss_D = train_discriminator(real_images, D, G, loss_func, optim_D, batch_size, input_dim, device)
            loss_G = train_generator(D, G, loss_func, optim_G, batch_size, input_dim, device)

            total_loss_D += loss_D
            total_loss_G += loss_G

            # 每 100 步打印一次损失
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print(f'Epoch {epoch:02d} | Step {i + 1:04d} / {len(train_loader)} | Loss_D {total_loss_D / (i + 1):.4f} | Loss_G {total_loss_G / (i + 1):.4f}')

        # 记录损失到 TensorBoard
        writer.add_scalar('DCGAN/Loss/Discriminator', total_loss_D / len(train_loader), epoch)
        writer.add_scalar('DCGAN/Loss/Generator', total_loss_G / len(train_loader), epoch)

        # 生成并保存示例图像
        with torch.no_grad():
            noise = torch.randn(64, input_dim, device=device)
            fake_images = G(noise)

            # 记录生成的图像到 TensorBoard
            img_grid = torchvision.utils.make_grid(fake_images, normalize=True)
            writer.add_image('Generated Images', img_grid, epoch)

    writer.close()

if __name__ == '__main__':
    main()
```
