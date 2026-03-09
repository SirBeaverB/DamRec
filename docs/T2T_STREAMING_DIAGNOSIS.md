# T2T 流式动量变体诊断报告

针对「Mo/Nest/Adam/Fro 四个模型 recall@10 完全相同 (0.0129)」的排查结果。

---

## 1. 状态字典是否漏写 M 和 V？

**结论：未发现漏写。**

| 模型 | 存储内容 | 代码位置 |
|------|----------|----------|
| MoRec | `(stored_S, stored_M, new_len, device)` | morec.py:205 |
| NestRec | `(stored_S, stored_M, new_len, device)` | nestrec.py:205 |
| DamRec | `(stored_S, stored_M, stored_V, stored_step, new_len, device)` | damrec.py:230-231 |
| FroRec | `(stored_S, stored_M, stored_V, stored_step, new_len, device)` | frorec.py:208 |

读取时索引正确：`state[0]=S`, `state[1]=M`, `state[2]=V`（Dam/Fro），`state[2]=new_len`（Mo/Nest）。

---

## 2. T2T 初始化时 M/V 维度是否坍塌？

**结论：未发现维度错误。**

- **Mo/Nest**：`M` 与 `S` 同为 `[num_heads, d_h, d_h]`
- **DamRec**：`M`, `V` 与 `S` 同为 `[num_heads, d_h, d_h]`
- **FroRec**：`V` 为 `[num_heads, 1, 1]`（F-Adam 设计），与 `GatedDeltaLayerChunkFroAdam` 一致

新用户初始化均使用 `torch.zeros(..., device=device)`，形状与 layer 期望一致。

---

## 3. 学习率 η 是否传入流式前向？

**结论：Chunk 层正确使用 η，Token 层不使用 η。**

| 层级 | momentum_eta / adam_eta | 实际使用 |
|------|-------------------------|----------|
| **Chunk 层** (use_chunk_*=True) | 在 `__init__` 中传入 | `S = S_end + self.momentum_eta * M` 等 |
| **Token 层** (use_chunk_*=False) | 未传入 | 使用 `gamma_gate`（可学习标量） |

当 `_FLA_AVAILABLE=True` 且 CUDA 可用时，会走 Chunk 路径，η 会生效。  
若 FLA 未安装或非 CUDA，会退回到 Token 层，此时使用 `gamma_gate`，不是固定 η。

---

## 4. 可能的问题来源

### 4.1 T2T 使用通用 Trainer，未用模型专用 Trainer

`streaming_t2t=True` 时，`get_trainer()` 固定返回 `StreamingTestThenTrainTrainer`，不会使用 `MoRecTrainer`、`DamRecTrainer` 等。

专用 Trainer 可能包含：
- 流式状态重置逻辑
- Chunk 级优化相关逻辑

需要确认这些逻辑是否影响 T2T 下的行为。

### 4.2 T2T 数据格式：短序列与 padding

`StreamingTimelineDataLoader` 中，新用户首次交互时：
- `seq = []`（空历史）
- `seq_len = 0`
- 被 pad 到 `min_seq_len=3`，即 3 个 padding 位置
- `item_seq_len = 0`，`last_idx = 0`

此时 `valid_mask` 对 padding 为 False，不会更新状态，但输出仍取自位置 0，可能引入异常行为。

### 4.3 已添加的 Debug 代码

在 `forward_with_streaming` 中已加入两处调试输出：

1. **算子降级检测**（函数开头，每个模型仅打印一次）：
   ```
   [DEBUG] MoRec Layer Type: GatedDeltaLayerChunkMomentum, FLA_AVAILABLE: True, use_chunk_momentum: True
   ```
   - 若显示 `GatedDeltaLayerMomentum`（Token 层）则说明发生 Fallback
   - 若 `use_chunk_momentum: False` 则 FLA 未生效，退化为 Token 层

2. **动量矩阵监测**（更新状态前，仅 uid=0 且前 50 次）：
   ```
   [DEBUG] MoRec uid=0 Step 5, M_mean: 0.001234
   ```
   - 若 M_mean 长期卡在 0.0，说明 ΔS 被 padding/掩码吞掉，动量未点火

**默认关闭**。需开启时，在各模型文件（morec.py、nestrec.py、damrec.py、frorec.py、gdn.py）顶部将 `DEBUG_T2T_STREAMING = False` 改为 `True`，再运行 `python scripts/run_streaming_t2t_experiments_100k.py` 或 `_1m.py`。

---

## 5. 配置中的 η 值

| 模型 | 配置文件 | η 参数 |
|------|----------|--------|
| MoRec | sequential_MoRec_streaming.yaml | momentum_eta: 0.15 |
| NestRec | sequential_NestRec_streaming.yaml | momentum_eta: 0.15 |
| DamRec | sequential_DamRec_streaming.yaml | adam_eta: 0.05 |
| FroRec | sequential_FroRec_streaming.yaml | adam_eta: 0.05 |

上述值均会传入对应 Chunk 层的 `__init__`。
