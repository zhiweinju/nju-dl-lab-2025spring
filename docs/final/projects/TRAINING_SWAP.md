# 大模型训练过程中的显存换出换入探索

## 一、任务说明
### 1.1 背景
近年来，大模型蓬勃发展，在各个领域的应用越来越广泛。当前普遍认为提高模型能力的最有效的途径之一是增加模型的参数量。然而参数量越来越多，意味着训练模型时占用显存越来越高。在所占用的显存中，一大部分显存用于将前向计算中产生的激活值（activation）保存到反向以计算梯度值。

### 1.2 训练过程中的显存换出换入
为了突破显存的物理容量限制，训练大模模型时可以采用显存换出与换入技术以利用分级内存容纳更大模型的训练。由于激活值在前向的最后一次使用到反向的第一次使用期间存在较长空闲时间，在前向最后一次之后，我们可以将激活值换出到 CPU DRAM 上，在反向第一次使用之前再将其换入回 GPU 显存中，在这期间空出的显存可以用于容纳更大的模型的训练。合理的换出策略可以在有限的显存条件下支持更大规模的模型训练，但也会引入额外的通信开销和计算延迟，可以通过换入换出与计算的并行执行，将换入换出的开销掩藏在计算中。

## 二、技术路线
技术路线一：PyTorch 框架 cpp 侧修改框架代码

1. 寻找合适的换出换入操作下发位置，可尝试方案：
    - 在 PyTorch 框架源码寻找计算算子下发的统一入口
    - 修改 codegen 代码，为每个自动生成的算子加入 hook，以 hook 函数作为下发位置
    - 其他可行方案
2. 在选出的下发位置中插入换入换出操作

技术路线二：Python 侧修改 Megatron 或其他框架代码

1. <font style="color:rgb(38, 38, 38);">寻找合适的换出换入操作下发位置</font>
2. <font style="color:rgb(38, 38, 38);">从 PyTorch 文档中搜寻 Tensor 拷贝接口，作为换入换出接口调用</font>
3. <font style="color:rgb(38, 38, 38);">在选出的下发位置中插入换入换出操作</font>

> 参考文献：
>
> <font style="color:rgb(34, 34, 34);">[1] Rhu, Minsoo, et al. "vDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design." </font>_<font style="color:rgb(34, 34, 34);">2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)</font>_<font style="color:rgb(34, 34, 34);">. IEEE, 2016.</font>
>
> <font style="color:rgb(34, 34, 34);">[2] Peng, Xuan, et al. "Capuchin: Tensor-based gpu memory management for deep learning." </font>_<font style="color:rgb(34, 34, 34);">Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems</font>_<font style="color:rgb(34, 34, 34);">. 2020.</font>
>
> <font style="color:rgb(34, 34, 34);">[3] Yuan, Tailing, et al. "Accelerating the training of large language models using efficient activation rematerialization and optimal hybrid parallelism." </font>_<font style="color:rgb(34, 34, 34);">2024 USENIX Annual Technical Conference (USENIX ATC 24)</font>_<font style="color:rgb(34, 34, 34);">. 2024.</font>
>

## **<font style="color:rgb(38, 38, 38);">三、实验要求</font>**
1. 基础换出换入操作尝试（必选）
    - 选择以上两个技术路线之一，尝试将一个激活值按照其生命周期进行换出换入而不影响训练的正确运行
    - 通过 profiling 工具采集插入换出换入操作后的训练 trace，在其中找到所插入的拷贝算子，判断其是否按预期执行
2. 换入换出操作与计算的掩盖执行优化（可选）
    - 使用单独的流执行换出换入操作，注意需要通过流间同步操作确保换出换入操作与计算算子之间的数据依赖不被打破，确保不影响训练精度
    - 考虑提前触发换入操作，消除依赖被换出梯度的反向算子对换入操作的等待
    - 考虑加入多个换出换入操作，以容纳更大模型的训练

## **<font style="color:rgb(38, 38, 38);">四、提交内容</font>**
1. **<font style="color:rgb(38, 38, 38);">代码</font>**
    - <font style="color:rgb(38, 38, 38);">代码结构不做要求</font>
    - <font style="color:rgb(38, 38, 38);">要给出能够运行的python环境（requirement.txt）</font>
    - <font style="color:rgb(38, 38, 38);">要给出能成功运行代码的脚本</font>
2. **<font style="color:rgb(38, 38, 38);">报告</font>**
    - <font style="color:rgb(38, 38, 38);">实现方法：详细描述实现方法</font>
    - <font style="color:rgb(38, 38, 38);">实验结果：</font>
        * <font style="color:rgb(38, 38, 38);">展示采集包含换出换入操作的 profiling 结果</font>
        * <font style="color:rgb(38, 38, 38);">展示经过优化后单卡下能够训练的最大模型大小，以及相应的训练性能</font>
    - <font style="color:rgb(38, 38, 38);">结果分析：对以上实验结果进行分析</font>

## **<font style="color:rgb(38, 38, 38);">五、注意事项</font>**
**<font style="color:rgb(38, 38, 38);">参考文献：</font>**<font style="color:rgb(38, 38, 38);"> 如果你在实验和报告中参考了已发表的文献，请列出你所参考的相关文献。  
</font><font style="color:rgb(38, 38, 38);">如有疑问，请联系 wzbwangzhibin@gmail.com 或 wangzb@smail.nju.edu.cn。</font>

