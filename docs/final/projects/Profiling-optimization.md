<h1 id="Title">大模型训练性能Profiling和优化</h1>

<h2 id="AnC0t">一、背景</h2>
随着大模型的流行，越来越多大模型不断涌现，而它们通常拥有海量的参数，同时带来了巨大的训练开销。因此，加速模型训练成为从业者的共识。而要想顺利加速训练，我们必须要对训练过程中的性能瓶颈有所了解，并且尝试选择有效的优化策略。因此，我们需要了解常见的性能profiling工具，并熟练掌握整个性能分析优化过程。  


<h2 id="a18jq">二、任务说明</h2>
<h3 id="Bd6Ni">2.1 Profiling工具介绍</h3>
本次实验基于NVIDIA GPU进行。而<font style="color:rgb(6, 6, 7);">NVIDIA Nsight Systems和Nsight Compute是NVIDIA提供的两款强大的性能分析工具，专为GPU开发和优化而设计。</font>

<h4 id="IwR9I"><font style="color:rgb(6, 6, 7);">NVIDIA Nsight </font><font style="color:rgb(6, 6, 7);">Systems</font></h4>
<font style="color:rgb(6, 6, 7);">NVIDIA Nsight</font><sup><font style="color:rgb(6, 6, 7);">[1]</font></sup><font style="color:rgb(6, 6, 7);">是一个系统级性能分析工具，能够帮助开发人员可视化应用算法，发现优化机会。</font>

+ **<font style="color:rgb(6, 6, 7);">系统级分析</font>**<font style="color:rgb(6, 6, 7);">：提供系统级的性能分析，帮助开发者理解整个应用的性能瓶颈。</font>
+ **<font style="color:rgb(6, 6, 7);">多平台支持</font>**<font style="color:rgb(6, 6, 7);">：支持多种NVIDIA平台，包括大型Tesla多GPU x86服务器、工作站等。</font>
+ **<font style="color:rgb(6, 6, 7);">深度学习框架支持</font>**<font style="color:rgb(6, 6, 7);">：能够为PyTorch和TensorFlow等深度学习框架的行为和负载提供宝贵见解，允许用户调整模型和参数，以提高单个或多个GPU的整体利用率</font><font style="color:rgb(6, 6, 7);">。</font>
+ **<font style="color:rgb(6, 6, 7);">低开销分析</font>**<font style="color:rgb(6, 6, 7);">：以低开销的方式运行，不会对系统性能产生过大影响，确保分析结果的准确性。</font>

<h4 id="PXaFL"><font style="color:rgb(6, 6, 7);">NVIDIA Nsight Compute</font></h4>
<font style="color:rgb(6, 6, 7);">NVIDIA Nsight Compute</font><sup><font style="color:rgb(6, 6, 7);">[2]</font></sup><font style="color:rgb(6, 6, 7);">是专门用于分析和优化CUDA程序性能的工具，主要用于深入分析GPU内核执行的详细性能数据，如寄存器使用、内存带宽、指令执行等。</font>

+ **<font style="color:rgb(6, 6, 7);">内核级分析</font>**<font style="color:rgb(6, 6, 7);">：深入GPU内核，提供详细的性能指标和API调试，帮助开发者定位CUDA内核中的瓶颈。</font>
+ **<font style="color:rgb(6, 6, 7);">引导式分析</font>**<font style="color:rgb(6, 6, 7);">：提供引导式分析，自动检测性能问题，并根据NVIDIA工程师的内置指导提供优化建议。</font>
+ **<font style="color:rgb(6, 6, 7);">源代码关联</font>**<font style="color:rgb(6, 6, 7);">：支持将源代码与详细的指令指标关联起来，帮助开发者快速定位有问题的代码区域。</font>



<font style="color:rgb(6, 6, 7);">Nsight Systems侧重于整个应用程序的系统性能分析，而Nsight Compute专注于GPU内核（算子）的性能分析。通过集成使用，开发者可以同时获得应用程序层面和GPU层面的性能数据，从而更全面地识别和优化性能问题。在实践中，我们通常选择通过Nsight应用程序分析找出执行时间过长的算子，然后再基于Nsight Compute分析其具体原因，再尝试加以优化。</font>

<h3 id="QDRca">2.2 常见的训练性能优化策略</h3>
通过<font style="color:rgb(6, 6, 7);">profiling我们可以定位出实践中耗时最久的算子，同时也可以发现其主要的问题来源。因此接下来需要对它们进行针对性优化。常见的训练性能优化策略包括数据预处理、通信、算子实现等多个方面：</font>

<h4 id="cXTgH"><font style="color:rgb(6, 6, 7);">数据预处理优化</font></h4>

+ **<font style="color:rgb(6, 6, 7);">并行处理</font>**<font style="color:rgb(6, 6, 7);">：通过多线程或进程并行处理数据，加快数据预处理速度。</font>
+ **<font style="color:rgb(6, 6, 7);">数据压缩</font>**<font style="color:rgb(6, 6, 7);">：减少数据存储空间和传输开销，降低预处理时间</font><font style="color:rgb(6, 6, 7);">。</font>
+ **<font style="color:rgb(6, 6, 7);">算法优化</font>**<font style="color:rgb(6, 6, 7);">：使用更高效的算法，减少时间复杂度</font><font style="color:rgb(6, 6, 7);">。</font>
+ **<font style="color:rgb(6, 6, 7);">数据缓存与预取</font>**<font style="color:rgb(6, 6, 7);">：将数据缓存到内存或高速存储设备中，减少数据读取时间</font><font style="color:rgb(6, 6, 7);">。</font>
+ **<font style="color:rgb(6, 6, 7);">数据预处理流水线</font>**<font style="color:rgb(6, 6, 7);">：将数据预处理步骤分解为多个阶段，形成流水线处理，提高整体效率。</font>

<h4 id="e2HyG"><font style="color:rgb(6, 6, 7);">算子优化</font></h4>

+ **<font style="color:rgb(6, 6, 7);">算子融合</font>**<font style="color:rgb(6, 6, 7);">：将多个小算子合并成大算子，减少Kernel Launch开销和访存开销</font><font style="color:rgb(6, 6, 7);">。</font>
+ **<font style="color:rgb(6, 6, 7);">内存优化</font>**<font style="color:rgb(6, 6, 7);">：考虑内存分块等方式，减少全局内存访问，使用共享内存或寄存器来优化数据访问。</font>
+ **<font style="color:rgb(6, 6, 7);">高效的算子实现</font>**<font style="color:rgb(6, 6, 7);">：使用经过优化的算子库或框架，如cuDNN、TensorRT等。</font>
+ **<font style="color:rgb(6, 6, 7);">算子调度：</font>**<font style="color:rgb(6, 6, 7);">通过调度算子执行顺序等，不影响其执行逻辑条件下，降低总的计算时间</font>

<h4 id="qPGMa"><font style="color:rgb(6, 6, 7);">通信优化</font></h4>

+ **<font style="color:rgb(6, 6, 7);">减少通信次数</font>**<font style="color:rgb(6, 6, 7);">：通过合并通信操作、批量传输数据等方式，减少通信次数。</font>
+ **<font style="color:rgb(6, 6, 7);">计算和通信重叠</font>**<font style="color:rgb(6, 6, 7);">：在计算过程中预取下一批数据，同时进行数据传输和计算。</font>
+ **<font style="color:rgb(6, 6, 7);">优化通信拓扑和原语</font>**<font style="color:rgb(6, 6, 7);">：合理设计节点间的通信连接方式、和通信原语实现，减少总通信延迟。</font>

<h4 id="L80mN"><font style="color:rgb(6, 6, 7);">其他优化</font></h4>

+ **<font style="color:rgb(6, 6, 7);">模型优化</font>**<font style="color:rgb(6, 6, 7);">：简化模型结构、减少参数数量、使用量化技术等，降低计算复杂度</font><font style="color:rgb(6, 6, 7);">。</font>
+ **<font style="color:rgb(6, 6, 7);">资源分配</font>**<font style="color:rgb(6, 6, 7);">：合理分配CPU、GPU、通信等资源，避免资源竞争和瓶颈。</font>
+ **<font style="color:rgb(6, 6, 7);">代码优化</font>**<font style="color:rgb(6, 6, 7);">：优化代码结构和算法实现，提高代码效率。</font>



实际训练中，我们可能只会遇到其中的一部分瓶颈，而如何选择优化，需要基于profiling结果和实践经验进行选择，有时候不合适的优化反而会造成更大的性能损失。同时值得注意的是，优化应该不影响训练的收敛性和精度。

<h2 id="jMxTw"><font style="color:rgb(38, 38, 38);">三、实验要求</font></h2>
<h4 id="dvztr">模型训练性能分析与优化</h4>

（1）**训练部署**：给定一个特定的模型配置，你需要在本地进行部署训练（推荐基于Pytorch平台），完成训练并确保精度达标。

（2）**性能Profiling**：基于推荐的profiling工具，对训练的单个iteration进行分析，找出训练中耗时最久的Top 5算子，并基于Nsight compute 对这些算子进行Roofline分析<sup>[3]</sup>，找出其性能不佳的原因。

（3）**性能优化**：根据profiling和分析的结果，你可以自主选择优化，验证优化前后，训练单个iteration的时间是否有改善，优化可以不止一处，效果越佳越好（只和自己的baseline比较）。



<h2 id="BMx6H"><font style="color:rgb(38, 38, 38);">四、提交内容</font></h2>
<font style="color:rgb(38, 38, 38);">最终你需要提交以下材料：</font>

<h4 id="lHTrU"><font style="color:rgb(38, 38, 38);">优化后的代码：</font></h4>

+ **<font style="color:rgb(38, 38, 38);">代码结构不做要求</font>**  
+ **<font style="color:rgb(38, 38, 38);">要给出能够运行的python环境（requirement.txt）</font>**  
+ **<font style="color:rgb(38, 38, 38);">要给出能成功运行代码的脚本</font>**

<h4 id="OGgyt"><font style="color:rgb(38, 38, 38);">实验报告</font></h4>

+ **<font style="color:rgb(38, 38, 38);">实现方法</font>**<font style="color:rgb(38, 38, 38);">：详细描述整个分析和优化过程，重点包括profiling结果的分析、优化策略的选择等</font>  
+ **<font style="color:rgb(38, 38, 38);">实验结果</font>**<font style="color:rgb(38, 38, 38);">：展示 baseline 的 profiling 结果（Top 5算子性能）</font>，<font style="color:rgb(38, 38, 38);">展示Top 5算子的roofline分析结果，以及选择对应优化后的profiling结果（包括训练时间的加速比、新的Top 5算子性能及其roofline分析结果）</font>  
+ **<font style="color:rgb(38, 38, 38);">结果分析</font>**<font style="color:rgb(38, 38, 38);">：对以上实验结果的分析，是否可以继续优化的探讨，以及实验过程中遇到的问题和解决过程</font>

  
<h2 id="sec5"><font style="color:rgb(38, 38, 38);">五、注意事项</font></h2>  
**<font style="color:rgb(38, 38, 38);">参考文献：</font>**<font style="color:rgb(38, 38, 38);"> 如果你在实验和报告中参考了已发表的文献，请列出你所参考的相关文献。 </font>

<font style="color:rgb(38, 38, 38);">如有疑问，请联系wzbwangzhibin@gmail.com 或 yuhangzhou@smail.nju.edu.cn。</font>



参考文献：  
[1] NVIDIA Nsight Systems. [https://docs.nvidia.com/nsight-systems/index.html](https://docs.nvidia.com/nsight-systems/index.html).

[2] NVIDIA Nsight Compute. [https://docs.nvidia.com/nsight-compute/index.html](https://docs.nvidia.com/nsight-compute/index.html).

[3] Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, 52(4), 65-76.  
 

