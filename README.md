# DamRec

Delta-Adam Memory for Streaming Recommendation？

思路：现在有人开始将linear treasformer 应用到推荐系统上。longhorn已经说明ssm等价于在线学习，比如gated delta net中的更新等价于随机梯度下降。现在我的研究思路是，找到linear transformer应用在推荐系统上的实例（linrec、mamba4rec等），找到其中等价于随机梯度下降的部分，然后将之改为更高级的梯度下降方法比如动量法和adam。
Google在LLM预训练领域使用了这个思路，造出来Titan。

Gated Delta Networks (GDN) 本质上是一个用 SGD（随机梯度下降）做状态更新的线性 RNN。在此基础上，把 SGD 改成更高级的方法，比如动量法、AdaGrad 等
即，把线性 Transformer 在推荐系统中的状态更新，从“等价于 SGD”升级为“等价于更高级的优化器（如 Adam）”。

DamRec 的核心创新灵感来源于序列建模与最优化理论之间深层的数学等效性。应用于流式推荐系统的线性 Transformer（如GDN）中的自回归状态更新机制，其前向传播过程在本质上可以被解构并等效为一种隐式的模型参数更新步骤。基于这一深刻的联系，DamRec尝试突破原模型基础逻辑的局限，将更高级、收敛性更强的一阶梯度下降算法的数学推导，“等价翻译”并直接融合到线性 Transformer 的算子内部，从而在保持流式推荐所需的高效推理速度的同时，从底层架构上赋予了模型在面对实时非平稳数据流时更卓越的在线学习与记忆演化能力。


还需要使用chunk的思想，单数据更新使用一般RNN，填满一个chunk再开始用dam




loss目前算的是总值，有些虚高，除batchsize之后是看上去比较正常的值。

独立预训练的结果并不好，


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
- non streaming测试（参数调优不充分，还有改进空间）（GDN和sas结构不一样，对lr的敏感度可能不一样，可以试试调）
- 跑一下其他baseline, 比如sas、bert4rec\linrec、mamba4rec\fmlp-rec、GRU、retrain_sas
在L=64时sas速度挺快，考虑把L拉到很长

4. 测试t2t ✅
直接训练t2t显然效果不行，应该先预训练。已实现
需要先进行预训练，然后再测试为了效率，可以先用1M的前80%数据进行预训练，后面的用于t2t测试
- 写代码 ✅
- 测试（用GDN预训练，然后换不同的几个模型）✅ 词表对齐 + 用户状态导出已实现
- baseline应该怎么测？谁做预训练谁做t2t?应该不能嫁接

---
## 快速测试

### 一、非流式实验（标准训练）

**五模型同时测试**（GDN、MoRec、NestRec、DamRec、FroRec）：

| 命令 | 说明 |
|------|------|
| `python scripts/run_non_streaming_experiments_100k.py` | ml-100k，L=50，五模型 |
| `python scripts/run_non_streaming_experiments_1m.py` | ml-1m，L=128，五模型 |
| `python scripts/run_non_streaming_experiments_1m.py -L 64` | ml-1m，L=64，五模型 |
| `python scripts/run_non_streaming_experiments_1m.py -L 64 --saved` | 同上，并保存 checkpoint |
| `python scripts/run_baseline_experiments_1m.py` | SASRec、LinRec、GRU4Rec、LightSANs（与 non_streaming 同配置） |

**Per-user 历史长度扫描（ml-1m，非流式 leave-one-out）**：将每用户交互截断为按时间**最近 N 条**（默认 N=10/50/100），跑 **GDN / DamRec（CLI 名 Adam）/ FroRec（Fro）**；默认**双卡并行**（`--n_gpus 2`）。**跑完后终端即打印汇总表**，并写出结果文件。

与下一节「**二、T2T 流式**」**不是同一设定**：这里**不做** 80/20 时间划分、**不**开 `streaming_t2t` / prequential 流式评估；而是 RecBole 常规 **非流式**训练（`streaming_mode: False`，可 shuffle），每 user **留一法**（倒数第 2 条 valid、最后 1 条 test）。截断 N 仅表示在**离线建表**时每位用户只保留最近 N 条交互，用来研究「历史变短」对非流式指标的影响；若要做**流式**短历史，需另改数据与 `streaming_*` 配置，本脚本未覆盖。

| 命令 | 说明 |
|------|------|
| `python scripts/run_per_user_history_sweep_1m.py` | 上表实验一次性跑完；得到 `experiment_results/per_user_hist_sweep_L{L}_{timestamp}.txt` 与同名 `.csv`（含 TEST/VALID 指标与相对 GDN 的 Δ%；默认 L=64） |
| `python scripts/run_per_user_history_sweep_1m.py --n_gpus 1` | 单卡顺序 |
| `python scripts/run_per_user_history_sweep_1m.py -L 64 --Ns 10,50 --models GDN,Adam` | 自定义 N 与模型子集（逗号分隔） |

中间过程与子进程日志在 `experiment_results/per_user_hist_sweep_L{L}_{timestamp}/` 下（`00_manifest.txt`、`orchestrate.log`、各 `{Model}_N{N}.log`）。

---

### 二、T2T 流式实验（80/20 划分，推荐）

**数据划分**：ml-1m 按时间戳前 80% 预训练、后 20% 流式测试。脚本自动完成词表对齐、用户状态导出、预训练历史注入。

**最简路径**：
- 有 ckp 且已有 `user_states.pt`：`python scripts/run_t2t_from_ckp_unified.py --ckp <ckp目录> -L 64`
- 有 ckp 但无 `user_states.pt`：`python scripts/run_t2t_from_ckp_unified.py --ckp <ckp目录> --dump_state -L 64`（先导出再跑五模型）

#### 场景 A：从零开始（首次运行）

```bash
# Step 1：准备 80/20 划分数据（只需执行一次）
python scripts/prepare_ml1m_80_20_split.py

# Step 2：预训练 + 状态导出 + T2T，一气呵成
python scripts/run_pretrain_t2t_1m.py --mode full
```
输出：`saved/pretrain_t2t_1m/` 下 GDN/MoRec/NestRec/DamRec/FroRec 的 checkpoint 和 `user_states.pt`，以及 `experiment_results/pretrain_t2t_1m_*.txt`。

#### 场景 B：已有 checkpoint，直接跑 T2T

```bash
# 指定 checkpoint 目录或 .pth 文件，自动词表对齐、自动加载 user_states.pt（若存在）
python scripts/run_t2t_from_ckp_unified.py --ckp saved/gdn_pretrain_t2t_unified_L64 -L 64
```
输出：五模型 T2T 表格 → `experiment_results/t2t_from_ckp_L64_t2t_*.txt`

#### 场景 C：已有 ckp、无 user_states，一次性跑五模型（含嫁接）

**推荐**：一条命令完成状态导出 + 五模型 T2T（GDN 同模型加载 user_states；MoRec/NestRec/DamRec/FroRec 嫁接，M/V 从 0 吸收流式）。

```bash
python scripts/run_t2t_from_ckp_unified.py --ckp saved/gdn_pretrain_t2t_unified_L64 --dump_state -L 64
```
- `--dump_state`：若目录下无 `user_states.pt` 则先导出，再依次跑 GDN、MoRec、NestRec、DamRec、FroRec
- 输出：`experiment_results/t2t_from_ckp_L64_t2t_*.txt`（五模型表格）

#### 场景 D：单模型 T2T（含嫁接）

```bash
# 同模型
python scripts/run_pretrain_t2t_1m.py --mode t2t --ckp saved/pretrain_t2t_1m/GDN-xxx.pth

# 嫁接：用 GDN 权重热启动 DamRec
python scripts/run_pretrain_t2t_1m.py --mode t2t --ckp saved/GDN-xxx.pth --model DamRec
```

#### 场景 E：Zero-Shot 体检（仅评估，不训练）

```bash
# 用于排查预训练权重是否加载成功
python scripts/run_t2t_from_ckp_unified.py --ckp saved/gdn_pretrain_t2t_unified_L64 --zero_shot -L 64
```

#### 场景 F：GDN 预训练 + 五模型 T2T 表格（一步到位）

```bash
python scripts/run_gdn_pretrain_t2t_unified.py
# 或跳过预训练，用已有 ckp
python scripts/run_gdn_pretrain_t2t_unified.py --ckp saved/GDN-xxx.pth
# 指定 L=64
python scripts/run_gdn_pretrain_t2t_unified.py -L 64
```

#### 场景 G：每模型独立预训练 + 独立 state_dump + 流式 T2T（**双卡并行，推荐用于正式对比**）

与场景 C/F 的区别：
- **不嫁接**：五模型各自完整预训练 150 epoch，MoRec/NestRec/DamRec/FroRec 的独有参数（如 DamRec 的 V_r/V_k 投影层）不再是随机初始化。
- **强制 dump_state**：每个模型预训练完立刻导出**自己**的 `user_states_{MODEL}.pt`；T2T 阶段 per-user 内部状态 (S, M, V) 以预训练末态为起点，不从零吸收。
- **双 GPU 并行**：每个模型独占一张卡跑完整流水线；5 个模型在 2 卡上排两波，总耗时约 2.5~3h（L=64）。

```bash
# 前置：首次运行需先划分数据
python scripts/prepare_ml1m_80_20_split.py

# 标准命令（默认 2 GPU, L=64, pretrain 150 ep）
python scripts/run_per_model_pretrain_t2t_1m.py

# 只跑部分模型（快速验证）
python scripts/run_per_model_pretrain_t2t_1m.py --models Adam,Fro

# 单卡顺序跑
python scripts/run_per_model_pretrain_t2t_1m.py --n_gpus 1

# smoke test（调小 epochs 验证管线）
python scripts/run_per_model_pretrain_t2t_1m.py --models GDN --epochs 5

# 断点续跑：ckp 已有，只重跑 T2T（调流式 lr 扫参用）
python scripts/run_per_model_pretrain_t2t_1m.py --skip_pretrain --t2t_lr 5e-5

# 断点续跑：ckp + user_states 都已有，只重新评估
python scripts/run_per_model_pretrain_t2t_1m.py --skip_pretrain --skip_dump

# 单模型调试（worker 模式，直接锁 GPU、输出到终端）
CUDA_VISIBLE_DEVICES=0 python scripts/run_per_model_pretrain_t2t_1m.py \
    --worker --model Adam --out_json /tmp/adam.json --show_progress
```

**产物**：
- `saved/per_model_pretrain_L64/{MODEL}-*.pth`（5 个 checkpoint）
- `saved/per_model_pretrain_L64/user_states_{MODEL}.pt`（5 份 per-user 状态）
- `experiment_results/per_model_streaming_L64_{timestamp}.txt` / `.csv`（汇总表 + 含抬头说明）
- `experiment_results/per_model_streaming_L64_{timestamp}/{MODEL}.log`（单模型日志）
- `experiment_results/per_model_streaming_L64_{timestamp}/{MODEL}.json`（单模型结果 JSON）

**注意**：
- 单模型占用显存约 3GB（DamRec 最高），双卡并行各自独立，不会相互挤占。
- `--show_progress` 在双卡并行下会导致 tqdm 日志穿插，调试时再开。
- 输出的 TXT 为单 seed feasibility 结果；正式论文需 3~5 seed 批量聚合（CSV 列已预留 `seed` 字段）。

##### 场景 G 数据如何被使用（数据流详解）

**Step 0：数据集划分（一次性离线做）**
`scripts/prepare_ml1m_80_20_split.py` 把 `ml-1m.inter` 按 `timestamp` 全局排序，切成：

| 子集文件 | 内容 | 用途 |
|---|---|---|
| `ml-1m-pretrain.inter` | 前 80% 真实交互 + 占位行 | Step 1 预训练 & Step 2 state dump |
| `ml-1m-t2t.inter` | 后 20% 真实交互 + 占位行 | Step 3 流式 T2T（含 valid/test） |

> 占位行（rating=0）只是为了让两子集的 user/item 词表完全对齐（避免 RecBole 的 `pd.factorize` 在两子集里给同一 token 分配不同内部 ID），不进入任何真实训练信号。

**Step 1：独立预训练（`ml-1m-pretrain`）**
- 训练集：前 80%（普通 RecBole 非流式训练，leave-one-out 划分 valid/test）
- `streaming_mode=False`，150 epochs（受 `stopping_step=10` 早停）
- 五个模型各自训练 → 5 个独立 checkpoint：`saved/per_model_pretrain_L64/{MODEL}-*.pth`
- **关键：不嫁接**。MoRec/NestRec/DamRec/FroRec 的独有参数（如 DamRec 的 `V_r/V_k` 投影层）在这一步被**实际训练**，不再是随机初始化。

**Step 2：State Dump（流式预热）**
- 读刚存的 checkpoint，按 `(user_id, timestamp)` 严格排序遍历**整个** `ml-1m-pretrain`，对每个 user 做一次 `forward_with_streaming`，捕获该 user 在"预训练末态"时的 per-user 内部状态（S, M, V 等），dump 成 `saved/per_model_pretrain_L64/user_states_{MODEL}.pt`。
- **关键：每个模型导出自己的状态**，不共用。DamRec 的 V 就用 DamRec 模型 replay 得到；F-Adam 的标量 V 就用 FroRec 得到。
- 这一步不算模型效果，只是把预训练学到的 per-user 历史"物化"出来，供 Step 3 作为起点。

**Step 3：流式 T2T（`ml-1m-t2t`）**
启动时三份东西先加载进内存：

| 加载内容 | 来源 | 作用 |
|---|---|---|
| `model.load_state_dict(ckp)` | Step 1 的 `.pth` | 模型 trainable 参数 |
| `model._streaming_state = load(user_states_{MODEL}.pt)` | Step 2 的 `.pt` | per-user S/M/V 内部状态 |
| `initial_user_history = read(ml-1m-pretrain.inter)` | 前 80% 原始数据 | per-user 历史 item 序列 |

然后构建 timeline：对 `ml-1m-t2t` 里的全部 (user, item) 按 `timestamp` 全局排序；对每个 user 在 t2t 内的交互，标其**尾部 10%** 为 test 点（`streaming_test_ratio=0.1`）。

进入流式循环（`epochs=1`，按时间顺序单次扫过）。每 batch（256 条交互）做：

```
for batch in timeline:                              # 严格时间顺序，不 shuffle
    if batch 里有 is_test=True 的样本:
        model.eval()
        scores = model.full_sort_predict(batch)    # 用当前 user_history 预测
        记录这些 test 点的 Recall/NDCG/MRR @10
        model.train()

    # 不管 test 与否，整个 batch 都参与 loss.backward()
    # （prequential 协议：预测完就用真实反馈更新模型）
    loss = model.calculate_loss(batch)
    loss.backward(); optimizer.step()              # t2t_lr=1e-4

    # batch 处理完后：
    # 1) 每条交互 append 到对应 user 的 user_history（下次看到此 user 序列更长）
    # 2) _streaming_state[uid] 已在 forward 里被更新（持续演化，永不重置）
```

**Step 4：评估汇总**
timeline 扫完 = epoch 结束 = T2T 结束。把循环中记录的所有 test 点的 Recall@10/NDCG@10/MRR@10 全局聚合 → 一份 test 结果。

**单用户 A 实际经历的完整旅程（假设有 100 条完整交互，L=64）：**

```
Step 1 预训练：在 a₁..a₈₀ 上训练模型参数（leave-one-out 划分）
Step 2 dump ：replay a₁..a₈₀ → 保存 A 的末态 (S, M, V)

Step 3 T2T 启动：
  user_history[A]      = [a₁, a₂, ..., a₈₀]          ← 从 pretrain 加载
  _streaming_state[A]  = (S, M, V) 预训练末态         ← 从 dump 加载

Step 3 流式迭代（A 的 a₈₁..a₁₀₀ 散布在全局时间轴上）:
  遇到 (A, a₈₁) 非 test：
    - 模型用 history 最近 64 项 [a₁₇..a₈₀] 做 forward
    - loss 反传，更新模型参数；_streaming_state[A] 演化
    - user_history[A] ← [a₁, ..., a₈₁]
  ...（中间非 test 点略）...
  遇到 (A, a₉₉) 标记 TEST（尾部 10%）：
    - 先 predict → 记 recall@10（a₉₉ 是否 ∈ top10?）
    - 再 loss 反传；状态继续演化
    - user_history[A] ← [..., a₉₉]
  遇到 (A, a₁₀₀) 标记 TEST：同上

Step 4：所有 user 所有 test 点聚合 → Recall/NDCG/MRR @10
```

**量级估算**：ml-1m 共 6040 users × 平均 165 交互。t2t 每 user 平均 33 条，尾部 10% ≈ 3 个 test 点。总 test 点 ≈ 18K。

**结果对标参考（Recall@10）：**

| 基线 | 预期值 | 说明 |
|---|---|---|
| 纯随机（10/3700） | 0.0027 | 10 个 item 猜中正确 |
| Popularity（推荐最热门） | 0.01 ~ 0.03 | 常见 baseline 上限 |
| 非流式 offline（leave-one-out） | 0.28 ~ 0.31 | 全量训练上界 |
| Streaming 合理区间 | **0.05 ~ 0.15** | 任务固有比 offline 低 2~5× |

**如果跑完 Recall@10 ~ 0.01**：说明模型基本没学到用户个性化（与 popularity 持平），管线仍有病；**如果 ≥ 0.05**：Fix 1+2 起效，方法之间差距有望拉开，可以进入论文实验阶段。

历史数据：嫁接 + 不 dump_state 的旧管线观测到 0.011（即与 popularity 持平）——正是场景 G 要解决的问题。

**本场景与场景 C/F 的核心区别**：

| 维度 | 场景 C/F（嫁接式） | **场景 G（独立式）** |
|---|---|---|
| 预训练 | 只预训练 GDN | **五模型各自预训练** |
| 独有参数（如 V_r/V_k） | 随机初始化，流式阶段用 780 步 + lr=1e-4 几乎没训 | **预训练 150 epoch 完整训练** |
| Per-user 内部 V | 从 0 开始吸收流式（每 user ~33 步） | **load 预训练末态，直接使用** |
| 适用场景 | 快速对比 / GDN 嫁接能力评估 | **正式论文主实验** |

#### 场景 H：Popularity 基线（流式 T2T 评估的地板线）

**用途**：判断场景 G 跑出的五模型 Recall@10 ≈ 0.01 究竟是"模型有效但分辨度低"还是"完全没学到用户个性化"。
**特点**：纯 CPU、无训练、< 30 秒出结果；test 点定义与场景 G 完全一致。

```bash
# 默认（test_ratio=0.1，与场景 G 一致）
python scripts/run_popularity_baseline_streaming.py

# 调 test_ratio（看不同密度下基线如何变化）
python scripts/run_popularity_baseline_streaming.py --test_ratio 0.05
python scripts/run_popularity_baseline_streaming.py --test_ratio 0.3
```

**两个基线**：

| 基线 | 定义 |
|---|---|
| **POP_global** | 所有 user 共享同一个 top-10 推荐 = pretrain 全局 item 频次 top-10 |
| **POP_user**   | 推荐该 user 在 pretrain 历史中频次 top-10 item，不足补 POP_global |

> 注意 next-item 预测的特殊性：POP_user 推荐"该用户已经看过的"，但 test 标签是"用户没见过的下一项"，所以 POP_user 通常**比 POP_global 还低**。这与传统评分预测任务相反。

**ml-1m 80/20 上的实测基线（test_ratio=0.1, 19449 个 test 点, 1783 个 t2t user）**：

| 基线 | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 纯随机（10 / 3662 items） | 0.0027 | - | - |
| **POP_user**   | 0.0067 | 0.0032 | 0.0022 |
| **POP_global** | **0.0105** | 0.0050 | 0.0033 |

**判读规则**（对照 `per_model_streaming_L*.txt` 里的 5 模型 Recall@10）：

| 五模型 Recall@10 区间 | 含义 |
|---|---|
| < POP_global (0.011) | **严重病理**：模型连"无脑推热门"都不如 |
| ≈ POP_global (~0.011) | **没学到用户个性化**：和场景 G 当前结果（0.010-0.011）就是这个状态 |
| ≥ POP_user (0.007) 但 < POP_global | 学到了流行偏好，没学到用户级信号 |
| ≥ 1.5 × POP_global (~0.016) | 真正学到了用户级信号，方法有效 |

**对当前 ml-1m streaming 实验的诊断结论**：场景 G 的五模型 Recall@10 ≈ POP_global → 在该 setup 下方法不可区分，应当转向数据稀疏 regime 实验（参见 `scripts/run_non_streaming_experiments_1m.py` + 改 `dataset = "ml-1m-t2t"`，或换 Amazon Beauty 等稀疏数据集）。

**输出**：
- `experiment_results/popularity_baseline_streaming_L{L}_{ts}.txt`（带数据规模 + 判读指南）
- `experiment_results/popularity_baseline_streaming_L{L}_{ts}.csv`（两行：POP_global / POP_user）

---

### 三、全量 ml-1m 流式（非 80/20，用于对比）

```bash
python scripts/run_streaming_t2t_experiments_1m.py
```

### T2T 机制说明（自动完成，无需手动）

| 机制 | 作用 |
|------|------|
| **词表对齐** | RecBole 按首次出现分配 ID，pretrain/t2t 会错位。脚本自动用预训练词表覆盖 T2T。 |
| **用户状态库** | 导出 `user_states.pt`（S, M, V），T2T 时加载，老用户不再从零开始。`--mode full` 默认导出；`--dump_state` 从已有 ckp 导出。 |
| **预训练历史** | 用前 80% 的 item 序列初始化 user_history，避免 T2T 仅含 20% 导致序列过短。 |

### 常用参数

| 参数 | 说明 |
|------|------|
| `-L` / `--max_seq_len` | 序列长度，如 `-L 64` |
| `--dump_state` | 先导出 user_states.pt 再 T2T（若已存在则跳过） |
| `--zero_shot` | 仅评估不训练，排查预训练权重是否加载成功 |
| `--model` | 嫁接时指定目标模型，如 `--model DamRec` |
| `--t2t_lr` | 流式学习率，默认 0.0001 |

### GDN 嫁接

用 GDN 权重热启动 MoRec/NestRec/DamRec/FroRec：`--ckp saved/GDN-xxx.pth --model DamRec`。嫁接时 M/V 从 0 吸收流式，不加载 user_states.pt。

### 非流式 L / Chunk 配置说明
- **L (MAX_ITEM_LIST_LENGTH)**：序列长度，默认 50。ml-100k 用 50 即可；**ml-1m 用户历史更长，建议 L=128**。`run_non_streaming_experiments` 在 ml-1m 下已自动设为 128。
- **CHUNK_SIZE**：在 `layers.py` 中硬编码为 16。MoRec/NestRec/DamRec/FroRec 的 Chunk 级更新按 16 个 token 为一组做宏观动量/Adam。
- **约束**：`CHUNK_SIZE < L`，否则整序列只有 1 个 chunk，宏观更新退化为纯 GDN。当前 L=50→3 chunks，L=128→8 chunks，均满足。
- **FLA 依赖**：Chunk 级模式需 `pip install flash-linear-attention` + CUDA，否则 DamRec/MoRec/NestRec 退化为 Token 级（仍可跑，但更慢）。

### 综合测试脚本 (Step 3)
对比 GDN、MoRec、NestRec、DamRec、FroRec、SASRec(baseline)，支持数据集/epoch/序列长度消融：
```bash
python scripts/run_comprehensive_experiments.py
python scripts/run_comprehensive_experiments.py --dataset ml-1m
python scripts/run_comprehensive_experiments.py --epochs 20
python scripts/run_comprehensive_experiments.py --max_seq_len 128
```

### T2T 流式实验脚本 (Step 4)
- **80/20 划分**（推荐）：见上方「二、T2T 流式实验」
- **全量数据集**：`python scripts/run_streaming_t2t_experiments_100k.py`（ml-100k）、`python scripts/run_streaming_t2t_experiments_1m.py`（ml-1m）

### 输出文件命名
结果保存在 `experiment_results/` 下，按脚本区分：
| 脚本 | 输出文件 |
|------|----------|
| non_streaming_100k | `non_streaming_100k_{timestamp}.txt` / `.csv` |
| non_streaming_1m | `non_streaming_1m_L{L}_{timestamp}.txt` / `.csv`（L=64 或 128） |
| baseline_1m | `baseline_1m_L{L}_{timestamp}.txt` / `.csv`（SASRec、GRU4Rec、LightSANs） |
| streaming_t2t_100k | `streaming_t2t_100k_{timestamp}.txt` / `.csv` |
| streaming_t2t_1m | `streaming_t2t_1m_{timestamp}.txt` / `.csv` |
| run_streaming_t2t_experiments.py | `streaming_t2t_{dataset}_{timestamp}.txt` |
| pretrain_t2t_1m | `pretrain_t2t_1m_{timestamp}.txt`（80% 预训练 + 20% T2T） |
| gdn_pretrain_t2t_unified | `gdn_pretrain_t2t_unified_{timestamp}.txt` / `.csv`（GDN 预训练 + 五模型 T2T 表格） |
| t2t_from_ckp_unified | `t2t_from_ckp_L{L}_t2t_{timestamp}.txt` / `.csv`（五模型 T2T） |
| t2t_from_ckp_unified (zero_shot) | `t2t_from_ckp_L{L}_zero_shot_{timestamp}.txt` / `.csv`（仅评估不训练） |

---

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

### DamRec 专用 sweep 脚本

DamRec FLA 路径在论文公式之外加了两类经验性 clip：(1) `s_r/s_k` 的上下界 `damrec_scale_max/min`；(2) `g = log(α)` 前的下界 clamp `damrec_log_clip_min`（默认 `1e-4`，没有 clip 时 α 极小会下溢导致 recall 极差）。两个参数都已接出到 yaml + CLI，分别有专用 sweep 脚本。

#### 1）`damrec_log_clip_min` sweep

默认扫 4 个值 `{1e-2, 1e-3, 1e-4, 1e-6}`，双卡并行（每张卡串行 2 个值）。配置：ml-1m + L=64 + EPOCHS=150。

```bash
# 默认 4 值（gpu0: 1e-2,1e-3 ；gpu1: 1e-4,1e-6）
bash scripts/run_damrec_log_clip_search.sh

# 改 EPOCHS / L
EPOCHS=200 L=64 bash scripts/run_damrec_log_clip_search.sh

# 自定义候选值与 tag（数量需为偶数）
CLIP_LIST="1e-2 1e-4 1e-5 1e-6" TAG_LIST="c1em2 c1em4 c1em5 c1em6" \
  bash scripts/run_damrec_log_clip_search.sh

# 单值快速试跑（不走 sweep 脚本，直接 CLI）
python scripts/run_non_streaming_experiments_1m.py -L 64 --models DamRec --damrec-log-clip-min 1e-3
```

输出 → `logs/damrec_log_clip_<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `00_manifest.txt` | 启动信息 + 每组 clip 参数 + 退出码 |
| `00_summary.txt` | **所有 tag 按 clip 升序的对比表**（valid/test recall@10、ndcg@10、训练时长、显存）|
| `<TAG>_gpu{0,1}.log` | Python stdout/stderr 全程 |
| `<TAG>_recbole.log` | RecBole 内部结构化日志 |
| `<TAG>_cmd.txt` | 复现命令 |
| `result_<TAG>/non_streaming_1m_L{L}_*.txt/.csv` | 该 tag 的原始结果表（每组隔离避免并发冲突）|

`00_summary.txt` 示例（4 个值跑完后）：

```
tag          clip       v_recall@10  v_ndcg@10    t_recall@10  t_ndcg@10    time_sec   mem_GB
------------ ---------- ------------ ------------ ------------ ------------ ---------- --------
c1em6        1e-6       0.xxxx       0.xxxx       0.xxxx       0.xxxx       xxx.xx     x.xx
c1em4        1e-4       0.xxxx       0.xxxx       0.xxxx       0.xxxx       xxx.xx     x.xx
c1em3        1e-3       0.xxxx       0.xxxx       0.xxxx       0.xxxx       xxx.xx     x.xx
c1em2        1e-2       0.xxxx       0.xxxx       0.xxxx       0.xxxx       xxx.xx     x.xx
```

#### 2）`damrec_scale_max/min` sweep

```bash
# 演示版双卡并行（2 组）
bash scripts/run_damrec_scale_search.sh

# 双卡分波多组（默认 4 组）
bash scripts/run_damrec_scale_search_4h_dual_gpu.sh

# 自定义组合
SMAX_LIST="2.0 3.0" SMIN_LIST="0.5 0.3" TAG_LIST="s2p0_m0p5 s3p0_m0p3" \
  bash scripts/run_damrec_scale_search_4h_dual_gpu.sh

# 单组 CLI
python scripts/run_non_streaming_experiments_1m.py -L 64 --models DamRec --damrec-scale-max 3 --damrec-scale-min 0.3
```

输出 → `logs/damrec_scale_<时间戳>/` 或 `logs/damrec_scale_4h_<时间戳>/`，结构与 log_clip sweep 一致。

#### 共用 CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--damrec-log-clip-min C` | yaml 1e-4 | `log(α)` 前下界 clamp，过小会下溢、过大会扼杀梯度 |
| `--damrec-scale-max S`   | yaml 2.0 | FLA 路径 `s_r/s_k` 上界 |
| `--damrec-scale-min S`   | yaml `1/max` | FLA 路径 `s_r/s_k` 下界 |
| `-L / --max_seq_len`     | 128       | 序列长度，sweep 默认 64 |
| `--epochs N`             | 150       | 上界，实际由早停决定 |

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

### 评估设置验证脚本
检查评估是否为**全排序 (Full-sort)** 以及是否存在 **Target Leakage**（未来信息穿越）：
```bash
python scripts/verify_eval_setup.py --dataset=ml-100k
# 指定配置
python scripts/verify_eval_setup.py --dataset=ml-1m --config_files="recbole/properties/quick_start_config/sequential_DamRec.yaml"
```
输出：`eval_args.mode`（应为 full）、DataLoader 类型（应为 FullSortEvalDataLoader）、随机抽查 `item_seq[-1] ≠ pos_item` 是否成立、候选物品总数。

---

## Test-Then-Train 流式（严格 Prequential）

已实现**真正的流式 T2T**：按全局时间轴单次遍历，对每条交互**先预测再训练**，用户状态 S **永不重置**。

### 语义
- **全局时间轴**：所有交互按真实时间戳排序，跨用户交织
- **Test-Then-Train**：若该交互为 test 点 → 先预测并记录 NDCG/Recall → 再算 loss 更新模型 → 最后更新该用户的 S
- **无限长程记忆**：S 伴随用户生命周期演化，不随 epoch 重置

### 推荐：80/20 划分（ml-1m）
**首选**：使用 `run_pretrain_t2t_1m.py` 或 `run_t2t_from_ckp_unified.py`，自动完成词表对齐、用户状态导出、预训练历史注入。见上方「快速测试」T2T 部分。

### 全量数据集（run_recbole.py）
公共配置：`recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml`

```bash
# ml-100k
python run_recbole.py --model=GDN --dataset=ml-100k --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml

# ml-1m
python run_recbole.py --model=GDN --dataset=ml-1m --config_files=recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml
```

### 配置要点
- `streaming_t2t: True`：启用 T2T 模式
- `streaming_test_ratio: 0.1`：每用户最后 10% 交互作为 test 点
- `epochs: 1`：单次遍历时间轴，无多 epoch

### 与普通流式的区别
| 模式 | 数据顺序 | 评估时机 | S 重置 |
|------|----------|----------|--------|
| 普通流式 | 全局时间序 | 每 epoch 后在 valid/test 上 | 每 epoch 重置 |
| T2T 流式 | 全局时间序 | 每条 test 交互前即时预测 | 永不重置 |