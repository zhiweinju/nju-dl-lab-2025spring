# 参数高效微调（PEFT）

## 任务背景

近年来，随着大规模预训练模型（如BERT、GPT等）在自然语言处理任务中取得显著成功，参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）逐渐成为热门研究方向。本任务旨在让同学们在掌握Transformer和预训练语言模型的基础上，深入理解并实践主流PEFT方法，通过实验对比它们在特定任务上的性能、效率与适用性。

## 任务目标

- 阅读并理解主流PEFT方法的核心原理与实现机制。
- 选择至少三种PEFT方法进行实现或调用现有开源工具。
- 在统一的下游任务（如文本分类、情感分析、问答等）上进行训练和评估。
- 比较它们的性能（准确率、F1分数等）、参数量、训练时间、显存占用等指标。
- 撰写实验报告，分析各方法的优劣和适用场景。

## 任务内容

### 1. 选题与准备

- 可选任务（任选其一）：
  - 情感分类（如IMDb / SST-2）
  - 新闻分类（如AG News）
  - 句子对匹配任务（如MRPC / QQP）
  - （进阶）问答任务（如SQuAD v1.1）
- 模型基线：建议使用BERT-base或RoBERTa-base

### 2. 方法实现（至少选择三种，需覆盖不同机制类别）：

- **提示工程类**：Prompt Tuning, Prefix Tuning
- **参数注入类**：LoRA, IA^3
- **结构修改类**：Adapter, BitFit

可以使用Hugging Face PEFT库、AdapterHub、或自行实现。

### 3. 实验与评估

- 在相同任务和模型初始化下，分别训练各PEFT方法
- 评估并记录以下指标：
  - 微调后的性能指标（Accuracy / F1-score 等）
  - 训练时间
  - 显存使用
  - 可训练参数比例（与full fine-tuning对比）

## 进阶方向

- 尝试结合多个PEFT方法（如Prompt+LoRA）
- 扩展任务到多语言 / 多任务场景
- 分析在小样本/低资源情境下的PEFT效果
- 可视化每种方法对Transformer层的干预程度（例如梯度、激活）

## 评分标准

| **评分项**                         | **分值** |
| ---------------------------------- | -------- |
| 实现完整性（≥3种方法）             | 30%      |
| 实验设计科学性（控制变量、复现性） | 20%      |
| 结果分析与对比深度                 | 20%      |
| 报告规范性与表达清晰度             | 20%      |
| 创新点（融合方法、可视化等）       | 10%      |

## 参考资源

- HuggingFace PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- AdapterHub: [https://adapterhub.ml/](https://adapterhub.ml/)
- LoRA 原论文: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Prefix Tuning: [https://arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190)
- BitFit: [https://arxiv.org/abs/2106.10199](https://arxiv.org/abs/2106.10199)
- IA^3: [https://arxiv.org/abs/2205.05638](https://arxiv.org/abs/2205.05638)
