# DamRec

Delta-Adam Memory for Streaming Recommendation？

思路：现在有人开始将linear treasformer 应用到推荐系统上。longhorn已经说明ssm等价于在线学习，比如gated delta net中的更新等价于随机梯度下降。现在我的研究思路是，找到linear transformer应用在推荐系统上的实例（linrec、mamba4rec等），找到其中等价于随机梯度下降的部分，然后将之改为更高级的梯度下降方法比如动量法和adam。
Google在LLM预训练领域使用了这个思路，造出来Titan。

Gated Delta Networks (GDN) 本质上是一个用 SGD（随机梯度下降）做状态更新的线性 RNN。在此基础上，把 SGD 改成更高级的方法，比如动量法、AdaGrad 等
即，把线性 Transformer 在推荐系统中的状态更新，从“等价于 SGD”升级为“等价于更高级的优化器（如 Adam）”。

DamRec 的核心创新灵感来源于序列建模与最优化理论之间深层的数学等效性。应用于流式推荐系统的线性 Transformer（如GDN）中的自回归状态更新机制，其前向传播过程在本质上可以被解构并等效为一种隐式的模型参数更新步骤。基于这一深刻的联系，DamRec尝试突破原模型基础逻辑的局限，将更高级、收敛性更强的一阶梯度下降算法的数学推导，“等价翻译”并直接融合到线性 Transformer 的算子内部，从而在保持流式推荐所需的高效推理速度的同时，从底层架构上赋予了模型在面对实时非平稳数据流时更卓越的在线学习与记忆演化能力。


可能还需要使用chunk的思想，单数据更新使用一般RNN，填满一个chunk再开始用dam







## plan

1. 在伯乐框架中实现GDN
- **运行伯乐**
- **在伯乐中添加流式数据流** 
- **实现GDN（效果与sasrec差不多，但是速度太慢。尝试接入fla）
(fla: 在不想更新的位置将 k=0、v=0、β=1，使 delta 更新项为 0，从而 S 保持不变。这样无需 FLA 支持 update_mask，把“不更新”编码到输入即可。)**

2. 修改GDN中的公式，实现Dam的基础
3. 进行测试（baseline GDN、linrec、mamba4rec以及其他）(epoch的必要性、SGD\momentum\adm、长短序列等等？)

---

## 流式数据流实现说明 (已完成)

已按参考思路完成 RecBole 流式适配，包含：

### 1. 模型
- **GDN** (`gdn.py`): Gated Delta Networks，SGD 等价的门控 delta 更新
- **DamRec** (`damrec.py`): 继承 GDN，计划将 SGD 等价替换为 Adam 等价（TODO）

### 2. 配置
- `model/GDN.yaml`、`model/DamRec.yaml`
- `shuffle: False`（流式必须）
- `time_decay`、`streaming_mode`、`replay_buffer_size` 等超参

### 3. 流式适配
- **全局时间排序**: `eval_args.order: TO`，`shuffle: False`
- **TBPTT 与记忆驻留**: `forward_with_streaming` 中 `.detach()` 跨 batch 传递状态
- **DamRecTrainer**: 继承 Trainer，每 epoch 开始时重置 streaming state

### FLA 加速（可选）
安装 `flash-linear-attention` 后，GDN 在 GPU 上会自动使用 Chunk 模式加速：
```bash
pip install flash-linear-attention
```
训练与流式模式在 GPU 上均会使用 FLA 的 `chunk_gated_delta_rule`（通过 k=0,v=0,β=1 编码不更新位置，无需 FLA 支持 update_mask）。

### 运行示例
```bash
# GDN（默认 ml-100k，50 epochs）
python run_recbole.py --model=GDN \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming.yaml

# MoRec (动量版 delta rule，SGD→Momentum)
python run_recbole.py --model=MoRec \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_MoRec_streaming.yaml

# DamRec (当前等同 GDN，Adam 升级待实现)
python run_recbole.py --model=DamRec \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_DamRec_streaming.yaml
```

### 流式输入数据：全局时间顺序
使用 `eval_args.group_by: none` + `split: {'RS': [0.8,0.1,0.1]}` + `order: TO`：
- 样本按**全局时间戳**排序（不再按 user 分组）
- 训练时顺序类似：`user_A@t1 → user_B@t2 → user_A@t3 → user_C@t4 ...`

**关于「把所有 user 当成一个 user」**：不是。流式是**一条全局时间线**，每个事件带 user_id；模型为每个 user 维护独立记忆，遇到该 user 时加载并更新其状态。可理解为「一条流，多用户各自记忆」。

### 如何确认流式环境生效
启动时若看到以下日志，说明流式配置正确：
- `[GDN] Streaming mode ON: per-user state persists across batches`
- `[Streaming] shuffle=False + streaming_mode=True: chronological order OK`

若看到 `shuffle=True with streaming_mode=True may break causal order` 警告，说明需在配置中设置 `shuffle: False`。

### 流式实现验证脚本
运行验证脚本检查状态持久化、用户隔离、reset 等逻辑是否正确：
```bash
python scripts/verify_streaming.py
```
通过则输出 `[OK] 所有流式验证通过`。



当前 RecBole 的流程仍是：先在整个 train_data 上训练，再在 valid_data 上评估。要实现严格的 Prequential（在时刻 t 先预测再训练），需要自定义 Trainer，在每批或每条样本上交替执行：

对未知样本做预测（Test）
用该样本的真实标签更新模型（Train）
这属于对 RecBole 训练循环的改造，需要单独实现一个 PrequentialTrainer 或类似组件