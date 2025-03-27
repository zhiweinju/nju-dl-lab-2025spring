# 实验任务三：WGAN

!!! success "目标"

    - 了解 WGAN 的核心思想及其与传统 GAN 的不同之处。
    
    - 掌握 WGAN 的生成器和判别器设计。
    
    - 使用 PyTorch 搭建并训练 WGAN 生成 MNIST 手写数字。
    
    - 学习如何通过 Wasserstein 距离优化 GAN 模型的训练稳定性。

## **1. WGAN 与传统 GAN 的区别**

WGAN（Wasserstein Generative Adversarial Network）是对传统 GAN 的改进，它通过最小化 Wasserstein 距离来度量生成数据与真实数据分布之间的差异，从而避免了 GAN 在训练过程中可能出现的模式崩溃和梯度消失问题。

!!! note "WGAN 相比传统 GAN 的改进"

    •	**Wasserstein 距离**：传统 GAN 使用 JS Divergence 作为损失函数，可能导致训练不稳定，尤其是在数据分布差异较大时。WGAN 使用 Wasserstein 距离来度量生成数据与真实数据之间的差异，它具有更好的数学性质，尤其在训练过程中能提供更加平滑的损失函数。
    
    •	**权重裁剪**：为保证 Lipschitz 连续性，WGAN 通过裁剪判别器的权重来避免训练过程中的不稳定性。这一方法可以有效避免传统 GAN 中判别器输出过大的问题。
    
    •	**不使用 Sigmoid 激活**：WGAN 的判别器输出没有 Sigmoid 激活函数，因为它不需要将输出限制在 [0,1] 之间。

如需更加深入的学习，可参考该论文：[Wasserstein GAN](./Wasserstein GAN.pdf)

## 2. WGAN的实现
部分细节可参考实验任务一和二。

#### 生成器

WGAN 生成器的设计与 DCGAN 类似，但会使用 Tanh 激活函数将图像的像素值限制在 [-1, 1] 的范围内。


```python
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        # 1. 输入层：将 100 维随机噪声从input_dim投影到 32x32（1024 维）
        #TODO   # 线性变换fc1，将输入噪声扩展到 1024 维

        self.br1 = nn.Sequential(
            #TODO   # 批归一化，加速训练并稳定收敛
            #TODO   # ReLU 激活函数，引入非线性
        )

        # 2. 第二层：将 1024 维数据映射到 128 * 7 * 7 的特征图
        #TODO   # 线性变换，将数据变换为适合卷积层的格式

        self.br2 = nn.Sequential(
            #TODO   # 批归一化
            #TODO   # ReLU 激活函数
        )

        # 3. 反卷积层 1：上采样，输出 64 通道的 14×14 特征图
        self.conv1 = nn.Sequential(
            #TODO   # 反卷积：将 7x7 放大到 14x14
            #TODO   # 归一化，稳定训练
            #TODO   # ReLU 激活函数
        )

        # 4. 反卷积层 2：输出 1 通道的 28×28 图像
        self.conv2 = nn.Sequential(
            #TODO   # 反卷积：将 14x14 放大到 28x28
            #TODO    # WGAN 需要使用 Tanh 激活函数，将输出范围限制在 [-1, 1]
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

!!! warning "判别器的输出"

    请注意！在 WGAN 中，判别器输出的是一个实数，而不是概率，因此不使用 sigmoid 激活函数。

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 1. 第一层：输入 1 通道的 28x28 图像，输出 32 通道的特征图
        self.conv1 = nn.Sequential(
            #TODO  # 5x5 卷积核，步长 2 进行降采样
            #TODO   # 归一化，提高稳定性
            #TODO   # LeakyReLU 避免梯度消失问题
        )

        # 2. 第二层：输入 32 通道，输出 64 通道特征
        self.conv2 = nn.Sequential(
            #TODO   # 5x5 卷积核，步长 2 继续降采样
            #TODO   # 归一化
            #TODO  # LeakyReLU 激活函数
        )

        # 3. 全连接层 1：将 7x7x64 维特征图转换成 1024 维向量
        self.fc1 = nn.Sequential(
            #TODO   # 线性变换，将 64x7x7 映射到 1024 维
            #TODO   # LeakyReLU 激活函数
        )

        # 4. 全连接层 2：最终输出真假概率
        #TODO # 输出一个标量作为判别结果

    def forward(self, x):
        x = self.conv1(x)  # 第一层卷积，降维
        x = self.conv2(x)  # 第二层卷积，降维
        x = x.view(x.shape[0], -1)  # 展平成向量
        x = self.fc1(x)  # 通过全连接层
        output = self.fc2(x)  # 通过最后一层全连接层，输出真假概率
        return output  # 返回判别结果
```

#### 训练过程

**1. 权重裁剪（Weight Clipping）**

在WGAN中，为了满足 **Lipschitz 连续性** 的要求，判别器的权重必须受到限制。WGAN通过使用 **Wasserstein距离** 来衡量生成样本与真实样本之间的差异，但为了保证Wasserstein距离的正确计算，判别器必须是 **Lipschitz 连续的**。判别器的输出不能对输入的微小变化过于敏感。这要求判别器的权重在训练过程中保持稳定。

为了确保判别器满足Lipschitz条件，WGAN采取了 **权重裁剪** 的方法。具体来说，在每次更新判别器的参数时，都会对判别器的权重进行裁剪，确保它们不会过大或过小，限制它们在一个固定的范围内（通常是[-clip_value, clip_value]）。

!!! note "权重裁剪"

    D.parameters() 获取判别器的所有参数（例如卷积层和全连接层的权重）。
    
    示例：for p in D.parameters():
    
    clamp_ 是一个原地操作（in-place operation），将每个参数的值限制在 [-clip_value, clip_value] 范围内。
    
    示例：A.clamp_(-1,1)

**2. 生成器和判别器的损失函数**

在传统的GAN中，生成器和判别器的训练目标是基于 **交叉熵损失（binary cross-entropy）** 来对抗训练，即判别器输出的是一个概率值，生成器的目标是最大化判别器的错误分类概率。而在WGAN中，损失函数采用了 **基于距离的定义**，具体是使用 **Wasserstein距离** 来度量两个分布之间的差异。

Wasserstein Distance的表达式如下：


$$
W(p_r, p_g) = \inf_{\gamma \sim \Pi(p_r, p_g)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]
$$


判别器的损失函数为：


$$
L_D = -\mathbb{E}_{x \sim p_{\text{data}}(x)} [D(x)] + \mathbb{E}_{z \sim p_{z}(z)} [D(G(z))]
$$

生成器的损失函数为：

$$
L_G = - \mathbb{E}_{z\sim p_z(z)}[D(G(z))]
$$


WGAN的损失函数设计目标是最小化生成器和判别器之间的Wasserstein距离，从而保证更稳定的训练过程。WGAN的损失函数并不依赖于概率，而是依赖于判别器的输出值（实数），这使得训练过程更加稳定。

!!! note "损失计算"

    WGAN中，判别器的损失是基于判别器对真实样本和生成样本的评分差异来计算的，对应上述公式，具体计算方法如下：
    
    loss_D = -(torch.mean(real_output) - torch.mean(fake_output))
    
    生成器的目标是尽可能让判别器认为生成的样本是真实的。在WGAN中，生成器的损失计算为：
    
    loss_G = -torch.mean(fake_output)


```python
# =============================== 训练判别器 ===============================
def train_discriminator(real_images, D, G, optim_D, clip_value, batch_size, input_dim, device):
    '''训练判别器'''
    real_images = real_images.to(device)
    real_output = D(real_images)

    noise = torch.randn(batch_size, input_dim, device=device)
    fake_images = G(noise).detach()
    fake_output = D(fake_images)

    #TODO  # 计算 WGAN 判别器损失loss_D

    optim_D.zero_grad()
    loss_D.backward()
    optim_D.step()

    # 对判别器参数进行裁剪
    for p in D.parameters():
        #TODO # 对判别器参数进行裁剪,将参数限制在 [-clip_value, clip_value] 范围

    return loss_D.item()

# =============================== 训练生成器 ===============================
def train_generator(D, G, optim_G, batch_size, input_dim, device):
    '''训练生成器'''
    noise = torch.randn(batch_size, input_dim, device=device)
    fake_images = G(noise)
    fake_output = D(fake_images)

    #TODO  # 计算 WGAN 生成器损失loss_G

    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    return loss_G.item()

```


#### 加载 MNIST 数据集

传统的GAN通常使用[0, 1]范围的图像作为输入，但WGAN要求图像的像素值在 **[-1, 1]** 范围内。这是因为WGAN的生成器输出通常使用 **Tanh** 激活函数，这样可以确保生成图像的像素值在这个范围内，符合生成器的输出要求。此时，输入图像的像素值需要做归一化，使用 (0.5,) 作为均值 (0.5) 和 (0.5,) 作为标准差，确保每个像素的值都被调整到 **[-1, 1]** 之间。

```python
if __name__ == '__main__':
    train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 将数据范围调整到 [-1, 1]
    ]), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

#### 定义优化器

与传统的GAN使用 **Adam** 优化器不同，WGAN推荐使用 **RMSprop** 优化器。这是因为在WGAN中，生成器和判别器的更新不再依赖于动量和自适应学习率（如Adam的做法），而是使用 **RMSprop** 来稳定梯度更新。使用RMSprop能够帮助避免训练过程中参数更新不稳定或过大，从而使训练更加稳定。

```python
#TODO #定义生成器的优化器，使用 RMSprop，学习率的设定请自行测试
#TODO #定义判别器的优化器，使用 RMSprop，学习率的设定请自行测试
```

#### 开始训练

在WGAN中，通常需要在每次生成器训练之前，先训练 **判别器多次**。这种策略有助于使判别器的训练更加稳定，因为WGAN的训练依赖于判别器的稳定性。如果判别器训练不足，它可能无法正确评估生成器的输出，导致训练不稳定。


```python
# =============================== 主函数 ===============================
def main():
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 设定超参数
    input_dim = 100
    batch_size = 128
    num_epoch = 30
    clip_value = 0.01   # 判别器权重裁剪范围，确保满足 Lipschitz 条件

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 创建生成器和判别器，并移动到 GPU（如果可用）
    # TODO
    # TODO

    # 定义优化器
    # TODO
    # TODO

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir='./logs/experiment_wgan')

    # 开始训练
    for epoch in range(num_epoch):
        total_loss_D, total_loss_G = 0, 0
        for i, (real_images, _) in enumerate(train_loader):
            # TODO  # 判别器训练 5 次

            # TODO  # 生成器训练 1 次

            # 每 100 步打印一次损失
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print(f'Epoch {epoch:02d} | Step {i + 1:04d} / {len(train_loader)} | Loss_D {total_loss_D / (i + 1):.4f} | Loss_G {total_loss_G / (i + 1):.4f}')

        # 记录损失到 TensorBoard
        writer.add_scalar('WGAN/Loss/Discriminator', total_loss_D / len(train_loader), epoch)
        writer.add_scalar('WGAN/Loss/Generator', total_loss_G / len(train_loader), epoch)

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

!!! Question "思考题"
    
    思考题1: WGAN与原始GAN的主要区别是什么？为什么WGAN能够改善GAN的训练稳定性？

    思考题2: 对于每个GAN模型（GAN， DCGAN， WGAN），在报告中展示TensorBoard中记录的损失函数变化曲线图和不同epoch时输出的图像（分别添加在epoch总数的0%、25%、50%、75%、100%处输出的图像）；直观分析损失函数的变化趋势和生成图像的变化趋势。

    思考题3: 从生成的图片上直观来看，GAN，DCGAN和WGAN的效果有什么差别？你认为是什么导致了这种差别？

####
