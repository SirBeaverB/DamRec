# DamRec

Delta-Adam Memory for Streaming Recommendation？

思路：现在有人开始将linear treasformer 应用到推荐系统上。longhorn已经说明ssm等价于在线学习，比如gated delta net中的更新等价于随机梯度下降。现在我的研究思路是，找到linear transformer应用在推荐系统上的实例（linrec、mamba4rec等），找到其中等价于随机梯度下降的部分，然后将之改为更高级的梯度下降方法比如动量法和adam。
Google在LLM预训练领域使用了这个思路，造出来Titan。

DamRec 的核心创新灵感来源于序列建模与最优化理论之间深层的数学等效性。应用于流式推荐系统的线性 Transformer（如GDN）中的自回归状态更新机制，其前向传播过程在本质上可以被解构并等效为一种隐式的模型参数更新步骤。基于这一深刻的联系，DamRec尝试突破原模型基础逻辑的局限，将更高级、收敛性更强的一阶梯度下降算法的数学推导，“等价翻译”并直接融合到线性 Transformer 的算子内部，从而在保持流式推荐所需的高效推理速度的同时，从底层架构上赋予了模型在面对实时非平稳数据流时更卓越的在线学习与记忆演化能力。










## plan

1. 在伯乐框架中实现GDN
- 运行伯乐
- 实现GDN


2. 修改GDN中的公式，实现Dam的基础
3. 进行测试