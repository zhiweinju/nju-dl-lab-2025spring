# 实验任务一: Transformer

## **1. Transformer 编码器中 Encoder Layer 的实现**
!!! info "Encoder Layer简介"
    Encoder Layer 是 Transformer 编码器中的基本构建单元，由 多头自注意力机制（Multi-Head Self-Attention） 和 前馈全连接网络（Feed Forward Network） 组成，搭配两次残差连接与 LayerNorm，用于高效建模输入序列的上下文依赖关系和特征表达能力。

这次我们还是使用AG News 数据集进行后续的分类任务，由于读取数据方式的改变，需要重新下载一下数据集。

#### ag数据下载链接：

[https://box.nju.edu.cn/f/a3a2e11167ef4d72a568/?dl=1](https://box.nju.edu.cn/f/a3a2e11167ef4d72a568/?dl=1)

### 预处理

首先导入所需模块：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import pandas as pd
```

数据读取以及预处理

不同于上次，这次我们使用pandas读取数据，相应的代码也有所修改。


```python
# ---------------- 数据读取 ---------------- #
# 这里需要填入train.csv和test.csv实际所在位置
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

tokenizer = get_tokenizer('basic_english')

# ---------------- 构建词表 ---------------- #
def yield_tokens(dataframe):
    for _, row in dataframe.iterrows():
        text = row['Title'] + " " + row['Description']  # 拼接 title 和 description
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_df), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])
pad_idx = vocab['<pad>']

# ---------------- 自定义 Dataset ---------------- #
class AGNewsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['Class Index'] - 1  # 让标签从 0 开始
        text = row['Title'] + " " + row['Description']
        tokens = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        return tokens, label

def collate_batch(batch):
    text_list, label_list = [], []
    for tokens, label in batch:
        text_list.append(tokens)
        label_list.append(label)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
    return text_list, torch.tensor(label_list)

# ---------------- dataloader ---------------- #
train_dataset = AGNewsDataset(train_df)
test_dataset = AGNewsDataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
```

### 位置编码器
在 Transformer 模型中，Self-Attention 是无序的，它无法感知输入序列的「位置信息」，即每个 token 在序列中的先后顺序。

相比之下，RNN（通过时间步）和 CNN（通过局部感受野）天然就有顺序/位置的概念。

因此，Transformer 需要为每个 token embedding 加入「位置信息」，这就是 Positional Encoding 的作用。

论文《Attention is All You Need》中提出了如下的位置编码方式，使用正弦和余弦函数来构造具有不同频率的位置表示。

原始公式：

$$
PE(pos, 2i) = \sin\left( \frac{pos}{10000^{\frac{2i}{d_{model}}}} \right)
$$

$$
PE(pos, 2i+1) = \cos\left( \frac{pos}{10000^{\frac{2i}{d_{model}}}} \right)
$$

其中：

-  $pos$：是序列中的位置$（0, 1, 2, ..., L-1）$
-  $i$：是 embedding 的维度索引$（0 ~ d_{model}/2）$
-  $d_{model}$：是 embedding 的总维度
-  $10000$：是一个控制不同维度的 $sin/cos$ 波动频率的超参数

[//]: # (为什么用 sin/cos？)

[//]: # (-  周期性：sin/cos 本身是周期性函数，有助于编码位置信息。)

[//]: # (-  平滑性：不同位置之间的编码值平滑过渡，利于模型捕捉局部和全局的关系。)

[//]: # (-  相对位置信息：任意两个位置之间的位置差，能通过 sin/cos 的组合被建模。)

公式进一步推导，
原公式中：

$$
\frac{pos}{10000^{\frac{2i}{d_{model}}}}
$$

等价于：

$$
pos \times \frac{1}{10000^{\frac{2i}{d_{model}}}}
$$

进一步展开成指数形式：

$$
= pos \times e^{- \log(10000) \cdot \frac{2i}{d_{model}}}
$$

请参考展开后的指数形式补充完下面的代码：

```python
#这段代码是 Transformer中的位置编码（PositionalEncoding），用于给输入的 token embedding 加入位置信息。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建一个全0的矩阵，shape = (max_len, d_model)
        # 表示每个位置 (0 ~ max_len-1) 对应的 d_model 维位置编码
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引，shape = (max_len, 1)
        # 即 position = [0, 1, 2, ..., max_len-1] 的列向量
        position = torch.arange(0, max_len).unsqueeze(1)

        # TODO 1: 计算 div_term，用于控制不同维度的 sin/cos 频率
        # 要求: 使用 torch.exp() 实现 1 / 10000^(2i/d_model)
        div_term = ...

        # TODO 2: 给偶数维度位置编码赋值
        # 要求: 使用 torch.sin() 完成 position * div_term，赋值给 pe 的偶数列
        pe[:, 0::2] = ...

        # TODO 3: 给奇数维度位置编码赋值
        # 要求: 使用 torch.cos() 完成 position * div_term，赋值给 pe 的奇数列
        pe[:, 1::2] = ...

        # 将 pe 注册为 buffer（不会被训练优化）
        # 并扩展成 (1, max_len, d_model) 方便后续和 batch 做广播
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x 是输入的 embedding，shape = (batch_size, seq_len, d_model)

        # 将对应位置的 pe 加到 x 上
        # self.pe[:, :x.size(1)] shape = (1, seq_len, d_model) 自动广播到 batch_size
        x = x + self.pe[:, :x.size(1)]

        # 返回位置编码后的 embedding
        return x
```
!!! question "思考题"
    思考题1：为什么需要对偶数和奇数维度分别使用 sin 和 cos？

### Multi-Head Self-Attention 模块

下面是多头自注意力的实现，请你按照要求补全代码：


```python
# Multi-Head Self-Attention 的完整实现
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # 保证 d_model 可以被 n_heads 整除，方便分头
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads # 每个 head 的特征维度
        self.n_heads = n_heads

        # 共享一个 Linear 层同时生成 Q, K, V
        self.qkv_linear = nn.Linear(d_model, d_model * 3) # 输出为 [Q; K; V]

        # 输出层，将多头的结果重新映射回 d_model 维度
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # 输入 x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        # 一次性计算 Q、K、V，输出 shape = (batch_size, seq_len, 3 * d_model)
        qkv = self.qkv_linear(x)

        # 切分成 n_heads 个 head，准备 multi-head attention
        # shape 变为 (batch_size, seq_len, n_heads, 3 * d_k)
        qkv = qkv.view(batch_size, seq_len, self.n_heads, 3 * self.d_k)

        # 调整维度顺序，变成 (batch_size, n_heads, seq_len, 3 * d_k)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, 3*d_k)

        # 沿最后一个维度切成 Q, K, V，shape = (batch_size, n_heads, seq_len, d_k)
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, n_heads, seq_len, d_k)

        # TODO 1: 计算 attention scores
        # 要求: 使用缩放点积的方式计算 (Q x K^T)，并除以 sqrt(d_k)
        scores = ...

        # mask 操作，屏蔽掉 padding 部分
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # TODO 2: 计算 attention 权重
        # 要求: 在 seq_len 维度上使用 softmax 归一化 scores
        attn = ...

        # TODO 3: 计算加权求和后的 context
        # 要求: 用 attn 加权 V，得到 context
        context = ...

        # 将多头拼接回去，shape = (batch_size, seq_len, n_heads * d_k) = (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 通过输出层，再映射回原始 d_model 维度
        return self.fc(context)
```

!!! question "思考题"
    思考题2：在 Multi-Head Self-Attention 机制中，为什么我们需要使用多个 attention head？

    思考题3：为什么要用缩放因子 sqrt(d_k)？

### TransformerEncoderLayer

下面的代码实现了 Transformer 编码器中的一个标准 Encoder Layer，包含：

“多头自注意力 + 前馈网络 + 两次残差连接 + 两次 LayerNorm” 的结构，用于对输入序列进行特征建模和上下文信息融合。

请你按照要求补全代码：

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        # 多头自注意力模块，输入输出维度都是 d_model
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)

        # 前馈全连接层，包含两层线性 + ReLU
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # 第一层 LayerNorm，作用在自注意力的残差连接之后
        self.norm1 = nn.LayerNorm(d_model)
        # 第二层 LayerNorm，作用在前馈网络的残差连接之后
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # ------------------ 自注意力块 ------------------ #

        # TODO 1: 计算多头自注意力输出 x2
        x2 = ...

        # TODO 2: 残差连接 + 第一层 LayerNorm
        x = ...

        # ------------------ 前馈神经网络块 ------------------ #

        # TODO 3: 前馈全连接网络（两层 Linear + ReLU）得到 x2
        x2 = ...

        # TODO 4: 残差连接 + 第二层 LayerNorm
        x = ...
 
        
        return x
```

!!! question "思考题"
    思考题4：为什么 Transformer Encoder Layer 中要在 Self-Attention 和 Feed Forward Network 之后都使用残差连接和 LayerNorm？试从“模型训练稳定性”和“特征传递”两个角度进行分析。

## **2. 基于 Transformer Encoder 的文本分类器**

下面，我们实现一个基于 Transformer Encoder 的文本分类器，通过 embedding、位置编码、多层 encoder 处理输入序列，最终使用 mean pooling 和全连接层完成文本的多类别分类任务。

```python
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=256, num_layers=2, num_classes=4):
        super().__init__()

        # 1. 定义词嵌入层（Embedding），输入为词表大小，输出为 d_model 维
        # padding_idx 用于指定 padding token 的索引，避免其被训练
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 2. 定义位置编码器，为 token embedding 添加位置信息
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. 定义多个 TransformerEncoderLayer 叠加起来，num_layers 为层数
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

        # 4. 定义输出分类层，将 encoder 最终输出映射到 num_classes 维度
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)，输入为单词 ID 序列

        # 1. 输入 token ID 通过 Embedding，转成 (batch_size, seq_len, d_model) 的 dense 向量
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # 2. 加入位置编码，增强位置感知能力
        x = self.pos_encoder(x)

        # 3. 创建 padding mask，shape: (batch_size, 1, 1, seq_len)
        # mask = True 代表有效 token，False 代表 padding 位置
        pad_mask = (x.sum(-1) != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

        # 4. 依次通过多层 Encoder，每一层都会使用 pad_mask
        for layer in self.layers:
            x = layer(x, pad_mask)

        # 5. 对时间维度（seq_len）做 mean pooling，聚合所有位置的特征
        out = x.mean(dim=1)  # mean pooling on seq_len

        # 6. 分类输出，映射到类别数
        return self.fc(out)
```

!!! question "思考题"
    思考题5：为什么在 TransformerEncoderClassifier 中，通常会在 Encoder 的输出上做 mean pooling（对 seq_len 取平均）？除了 mean pooling，你能否想到其他可以替代的 pooling 或特征聚合方式？并简要分析它们的优缺点。

下面是模型的训练和测试:

```python
# 使用 split 进行分词
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerEncoderClassifier(len(vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

模型训练部分：

```python
def train_epoch():
    model.train()
    total_loss = 0
    loop = tqdm(train_dataloader, desc="Training", leave=False)
    for text, labels in loop:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 更新tqdm进度条
        loop.set_postfix(loss=loss.item())
    return total_loss / len(train_dataloader)
```

模型测试部分：

```python
def evaluate():
    model.eval()
    correct = 0
    total = 0
    loop = tqdm(test_dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for text, labels in loop:
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


for epoch in range(1, 6):
    loss = train_epoch()
    acc = evaluate()
    print(f'Epoch {epoch}: Loss = {loss:.4f}, Test Acc = {acc:.4f}')
```

!!! question "思考题"
    思考题6：Transformer 相比传统的 RNN/CNN，优势在哪里？为什么 Transformer 更适合处理长文本？