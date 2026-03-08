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
(batch size似乎会影响实验效果，待进一步研究)

2. 修改GDN中的公式，实现Dam的基础
- 先实现动量版morec(需要使用chunk级的动量)（后续需要增加不同chunksize的实验)
- 单独实现damrec（调参效果不好，感觉逐元素除法可能存在问题）
- 暂时放弃adam, 尝试继续做更高级的momentum方法，neaterov
- 尝试fro adam, 猜测可能在平稳数据上表现优异，但是不适用于推荐任务（大概）

3. 进行测试（baseline GDN、linrec、mamba4rec以及其他）(epoch的必要性、SGD\momentum\adm、长短序列等等？) ✅ 见「综合测试脚本」
- non streaming测试（参数调优不充分，还有改进空间）


4. ~~实现真正的t2t流式~~ ✅ 已完成（见「Test-Then-Train 流式」）

---
## 快速测试

```
python scripts/run_non_streaming_experiments.py
```
主要注意，chunk size需要小于L，否则全部退化为纯GDN

### 综合测试脚本 (Step 3)
对比 GDN、MoRec、NestRec、DamRec、FroRec、SASRec(baseline)，支持数据集/epoch/序列长度消融：
```bash
python scripts/run_comprehensive_experiments.py
python scripts/run_comprehensive_experiments.py --dataset ml-1m
python scripts/run_comprehensive_experiments.py --epochs 20
python scripts/run_comprehensive_experiments.py --max_seq_len 128
```

### T2T 流式实验脚本 (Step 4)
```bash
python scripts/run_streaming_t2t_experiments.py
python scripts/run_streaming_t2t_experiments.py --dataset ml-100k
python scripts/run_streaming_t2t_experiments.py --model GDN
```



## 流式数据流实现说明 (已完成)

已按参考思路完成 RecBole 流式适配，包含：

### 1. 模型
- **DamRec** (`damrec.py`): **主方法**。将 delta rule 的 SGD 式更新替换为 Adam 式：m=β1*m+(1-β1)*ΔS; v=β2*v+(1-β2)*ΔS²; S += η*m/(√v+ε)
- **MoRec** (`morec.py`): **消融用**。动量版。Token 级 (Python) 或 Chunk 级 (FLA + 宏观动量)
- **NestRec** (`nestrec.py`): **Nesterov 动量**。提前看一步的等效更新方向，纯线性加法不破坏秩一外积结构。Token 级或 Chunk 级 (FLA + 宏观 Nesterov)
- **FroRec** (`frorec.py`): **F-Adam (Frobenius Adam)**。V 降维为标量 [B,H,1,1]，标量除法保留 M 的秩一外积方向，几何安全 + 自适应抗噪。需 FLA+CUDA。
- **GDN** (`gdn.py`): 基线。SGD 等价的门控 delta 更新（delta rule）

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
FLA 加速：`use_fla: True` 时在 GPU 上使用 `chunk_gated_delta_rule`。已通过 `scale=1.0` 与 Python 循环对齐（FLA 默认 `1/sqrt(d_head)` 会导致输出缩小、recall 偏低）。若仍有异常可设 `use_fla: False`。

### 运行示例
```bash
# GDN 非流式（标准训练，recall ~14%）
python run_recbole.py --model=GDN \
  --config_files=recbole/properties/quick_start_config/sequential_GDN.yaml

# GDN 流式（streaming_mode=True，valid recall~12%, test~10%）
python run_recbole.py --model=GDN \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming.yaml

# MoRec 非流式
python run_recbole.py --model=MoRec \
  --config_files=recbole/properties/quick_start_config/sequential_MoRec.yaml

# MoRec 流式 (Chunk 级动量 + FLA 加速，或 Token 级 Python)
python run_recbole.py --model=MoRec \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_MoRec_streaming.yaml

# DamRec 非流式
python run_recbole.py --model=DamRec \
  --config_files=recbole/properties/quick_start_config/sequential_DamRec.yaml

# DamRec 流式 (delta rule + Adam 式更新)
python run_recbole.py --model=DamRec \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_DamRec_streaming.yaml

# NestRec 非流式 (Nesterov 动量版 delta rule)
python run_recbole.py --model=NestRec \
  --config_files=recbole/properties/quick_start_config/sequential_NestRec.yaml

# NestRec 流式 (Chunk 级 Nesterov + FLA 加速)
python run_recbole.py --model=NestRec \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_NestRec_streaming.yaml

# FroRec 非流式 / 流式 (F-Adam，需 FLA+CUDA)
python run_recbole.py --model=FroRec \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_FroRec_streaming.yaml
```

### 高效验证方法
- **快速消融**：追加 `sequential_ablation_fast.yaml`，20 epochs 即可看出趋势
  ```bash
  python run_recbole.py --model=GDN --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming.yaml,recbole/properties/quick_start_config/streaming/sequential_ablation_fast.yaml
  python run_recbole.py --model=MoRec --config_files=recbole/properties/quick_start_config/streaming/sequential_MoRec_streaming.yaml,recbole/properties/quick_start_config/streaming/sequential_ablation_fast.yaml
  ```
- **L=50（默认）**：Chunk 动量在跨 batch（同 user 再访）时生效，MoRec > GDN 已体现
- **L=128 可选**：改 `MAX_ITEM_LIST_LENGTH: 128` 可做序列内多 chunk 实验，但训练更慢

### 自动调参（HyperTuning）
RecBole 支持 Hyperopt（随机/贝叶斯）和 Ray Tune。DamRec 默认约 12 次试验、约 1 小时：

```bash
# 需安装 hyperopt；默认 --max_evals 12 ≈ 1h
python run_hyper.py \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_DamRec_streaming.yaml \
  --params_file=recbole/properties/hyper_tuning/DamRec.hyper \
  --output_file=damrec_hyper.result --tool=Hyperopt
```

`--max_evals 24` 可延长至约 2h；`algo` 在 `run_hyper.py` 中可改为 `bayes`。

### 流式 vs 非流式 recall 差异
- **非流式** (`streaming_mode: False`)：标准训练，recall 可达 ~14%
- **流式** (`streaming_mode: True`)：按时间序、per-user 状态，valid recall~12%、test~10%（use_fla: False + batch 4096）

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

---

## Test-Then-Train 流式（严格 Prequential）

已实现**真正的流式 T2T**：按全局时间轴单次遍历，对每条交互**先预测再训练**，用户状态 S **永不重置**。

### 语义
- **全局时间轴**：所有交互按真实时间戳排序，跨用户交织
- **Test-Then-Train**：若该交互为 test 点 → 先预测并记录 NDCG/Recall → 再算 loss 更新模型 → 最后更新该用户的 S
- **无限长程记忆**：S 伴随用户生命周期演化，不随 epoch 重置

### 运行
```bash
# GDN Test-Then-Train 流式（默认 ml-10m，约 1000 万交互）
python run_recbole.py --model=GDN \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml

# 使用 ml-100k 快速验证
python run_recbole.py --model=GDN --dataset=ml-100k \
  --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml
```

### 配置要点
- `streaming_t2t: True`：启用 T2T 模式
- `streaming_test_ratio: 0.1`：每用户最后 10% 交互作为 test 点
- `epochs: 1`：单次遍历时间轴，无多 epoch
- `enable_amp: False`：避免 Triton 编译问题（可按需开启）

### 与普通流式的区别
| 模式 | 数据顺序 | 评估时机 | S 重置 |
|------|----------|----------|--------|
| 普通流式 | 全局时间序 | 每 epoch 后在 valid/test 上 | 每 epoch 重置 |
| T2T 流式 | 全局时间序 | 每条 test 交互前即时预测 | 永不重置 |