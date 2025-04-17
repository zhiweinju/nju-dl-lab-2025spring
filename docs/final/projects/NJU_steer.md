# 基于引导的大语言模型幻觉缓解

## 一、任务说明

### 1.1 背景介绍
大型语言模型(LLM)有时会产生幻觉，特别是LLM可能会产生不真实的反应，尽管知道正确的知识。激活LLM中的真实性是充分释放LLM知识潜能的关键，可以通过对LLM内部表示进行干预，达到缓解幻觉的目的。

### 1.2 数据集说明

| 数据集 | 任务          | 备注                                                    |
|----------|-------------|-------------------------------------------------------|
| TruthfulQA | Multiple-choice   | 给定问题以及选项，从中选出正确答案                                     |

## 二、方法设计规范

参照《Inference time intervention: Eliciting truthful answers from a language model》，实现对大语言模型的干预，讨论干预对于模型的影响

**核心要求：**

+ 基线对比
+ 不同layer实验分析
+ 不同位置的内部表示实验分析

注：使用QWEN-0.5B或其他合适模型均可

### 参考文献：

1. Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter
Pfister, and Martin Wattenberg. 2023b. Inference-
time intervention: Eliciting truthful answers from a
language model.

2. Zhongzhi Chen, Xingwu Sun, Xianfeng Jiao, Fengzong
Lian, Zhanhui Kang, Di Wang, and Cheng-Zhong
Xu. 2024. Truth forest: Toward multi-scale truthfulness 
in large language models through intervention
without tuning.



## 三、实验要求

### 3.1 评估指标

**Multiple-choice:**

尽管生成任务可以评估模型说出真实陈述的能力，
但这种评估方式很难操作。因此，我们提供了一种选择题的选项，用以测试模型识别真实陈述的能力。

+ MC1（单选题）：给定一个问题和4-5个选项，选择唯一正确的答案。模型的选择是它认为在问题之后最有可能完成的选项（与其他选项无关）。分数是所有问题的简单准确率。

+ MC2（多选题）：给定一个问题和多个正确/错误的参考答案，分数是分配给正确答案集合的总概率的归一化值

更详细内容参考：https://github.com/sylinrl/TruthfulQA/tree/main

### 3.2 基础实验（65分 = 结果准确性40分 + 报告质量25分）

+ **基线对比:** 对比干预前后MC1、MC2指标的变化
+ **layer实验**：针对LLM的不同层，进行干预，讨论干预不同的layer对实验结果的影响
+ **提取不同位置的hidden_states实验：** 针对每一层中不同位置的内部表示，比如经过注意力头之后的hidden_states，mlp运算之后的hidden_states等
+ **干预强度分析:** 随着干预强度的增大，MC1和MC2是否会越来越好？再打印几个输出，观察具体示例分析模型是否产生了更好的输出。如果没有产生更好的输出，而MC指标变好了，请分析原因。

### 3.3 进阶实验（35分 = 创新性20分 + 报告质量15分）

请在Inference time intervention: Eliciting truthful answers from a language model的基础上，设计一个新的算法（Truth forest除外）以提升性能，并且通过实验验证。


## 四、提交内容
代码:

1. 代码结构不做要求

2. 要给出能够运行的python环境（requirement.txt）

3. 要给出能成功运行代码的脚本

报告:

1. 实现方法：详细描述实现方法

2. 实验结果：在包含必做实验部分内容的基础上，可自由发挥

3. 结果分析：对上述实验结果进行分析



## 五、注意事项

**参考文献：** 如果你在实验和报告中参考了已发表的文献，请列出你所参考的相关文献。

如有疑问，请联系 652024320001@smail.nju.edu.cn。
