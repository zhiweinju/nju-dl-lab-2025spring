# KV cache缓存策略探索

## 一、任务说明

### 1.1 背景要求

基于Transformer的大语言模型（LLM）在推理过程中通过KV Cache缓存历史token的K和V投影，以空间换时间的方式加速自回归生成。在请求生成结束后，这些KV Cache在显存中驻留，未来可能被相同前缀的请求复用来加速生成，减少重计算。

同所有缓存一样，显存容量限制了可缓存的KV序列的大小。在本实验中，你将探索显存中的KV Cache重用机制（也就是前缀缓存，Prefix Cache），设计高效的前缀缓存驱逐策略，在有限显存下，针对相同的trace最大化前缀缓存命中率。

**核心挑战：**

- **显存动态分配：** 运行中的请求（活跃请求）的KV Cache需与prefix cache竞争显存空间
- **前缀匹配优化：** 识别新请求与历史缓存的前缀重叠部分（如系统提示词、共享知识库内容）
- **策略多样性：** 缓存策略有多种选择，需设计多种策略并进行对比

### 1.2 数据集说明

| 数据集类型           | 内容描述                                                                                                                                 | 备注                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **dataset.json** | 包含10,000条请求prompt，每条数据包含：<br>- request_id: 唯一标识符<br>- input_text: 输入文本（含系统提示词和用户query）<br>- output_length: 该请求的输出长度（视为定值） | 用于验证缓存策略        |
| **trace.csv**  | 含10,000条请求request_id和到达时间time    | 用于最终评估，需提交缓存命中结果 |

*注意: 数据集请执行`git clone https://www.modelscope.cn/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split.git`获取，每轮对话仅取第一个回合即可（也就是，一个用户提出的问题 + 一次LLM的回答*

**数据特性：**

- 40%请求包含相同系统提示词前缀
- 75%请求涉及共享知识库（如文学/医疗/法律/工具/文档），模拟真实场景的文本重叠

## 二、方法设计规范

### 2.1 核心策略设计（必选）

需实现至少两种驱逐策略并进行对比：

- **经典算法移植**
    - LRU（Least Recently Used）：优先驱逐最久未使用的缓存块
    - LFU（Least Frequently Used）：统计缓存块被命中次数，淘汰低频块
    - FIFO（First-In-First-Out）：按缓存加载顺序驱逐

- **自定义策略**（需包含以下至少一项）：
    - **Prefix-Aware策略：** XXX
    - **Cost-Benefit模型：** XXX
    - **Hybrid策略：** XXX

### 2.2 可选优化方向

- Swap：将不活跃的KV Cache置换
- 对高频前缀（如系统提示词）进行预缓存

## 三、实验实施要求

### 3.1 评估指标

| 指标类型     | 计算公式                                                                                                                      | 权重  |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- | ----- |
| **缓存命中率**   | HitRate = ∑(HitCount) / ∑(TotalRequest)                                                                                     | 100%   |

<!-- 这里可能需要修改为实际命中长度 -->
### 3.2 必做实验

- **基线对比：** 对比无缓存（ReCompute）和理想缓存（Oracle）的极端情况
- **分布敏感性：** 在高/低前缀重叠率（40% vs 90%）场景下的策略表现差异
<!-- - **消融实验：**  -->

## 四、提交要求

### 4.1 提交内容

**代码包结构示例：**

```
bash
├── strategy/            # 策略实现  
│   ├── lru.py  
│   ├── lfu.py
│   ├── hybrid.py        # 自定义策略示例  
├── evaluator/           # 评估模块  
<!-- 这部分需要补全 -->
├── results/             # 输出结果  
│   ├── test_hits.csv    # 格式：request_id, hit_flag(0/1), hit_ratio(%)  
└── report.pdf           # 实验分析报告
```

**报告内容：**

- 策略设计原理与复杂度分析
- 不同负载下的命中率曲线（如图1）
- 请求prefix cache命中情况

## 5 注意事项

**参考文献：** 如果你在实验和报告中参考了已发表的文献，请列出你所参考的相关文献。

如有疑问，请联系 wzbwangzhibin@gmail.com 或 spli161006@gmail.com。

## 附：示例策略伪代码

```python
# LFU策略示例  
class LFUCache:  
        def __init__(self, capacity):  
                self.capacity = capacity  
                self.cache = {}  # key: request_id, value: {"hit_count": int}  

        def access(self, request_id):  
                if request_id in self.cache:  
                        self.cache[request_id]["hit_count"] += 1  
                        return True  
                return False  

        def evict(self):  
                if sum(v["size"] for v in self.cache.values()) > self.capacity:  
                        # 按 hit_count 升序、size 降序排序  
                        candidates = sorted(self.cache.items(), key=lambda x: (x[1]["hit_count"], -x[1]["size"]))  
                        victim_id = candidates[0][0]  
                        del self.cache[victim_id]
```

