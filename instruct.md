# DamRec (Delta-Adam Memory Recommendation) 算法实现指南

## 1. 概述

DamRec 在 GDN（Gated Delta Networks）的门控线性递归框架上，为隐状态更新引入**基于梯度二阶矩的自适应预条件机制**，使每个特征维度的更新步长根据其历史梯度波动尺度自适应缩放。

核心思想：**前向推断 = 自适应预条件在线优化**。

---

## 2. 符号约定

| 符号 | 含义 |
|---|---|
| `d` | 特征/隐状态维度 |
| `H` | 多头数量，每头维度 `d_h = d / H` |
| `L` | 序列长度 |
| `C` | 分块大小（chunk size） |
| `⊙` | 逐元素乘积（Hadamard 积） |
| `⊘` | 逐元素除法 |
| `√·`, `+ε` 等 | 默认均为逐元素操作 |
| `S_t ∈ R^{d×d}` | 隐状态矩阵（多头下为 `S_t^{(h)} ∈ R^{d_h × d_h}`） |

---

## 3. 输入投影

对时间步 `t` 的输入 `x_t ∈ R^d`，通过线性投影生成键、值、查询：

```
k_t = W_k · x_t          (W_k ∈ R^{d×d}, 无偏置)
v_t = W_v · x_t          (W_v ∈ R^{d×d}, 无偏置)
q_t = W_q · x_t          (W_q ∈ R^{d×d}, 用于输出读取)
```

多头：将 `k_t, v_t, q_t` 沿特征维度均匀切分为 `H` 组，每组 `d_h` 维。后续所有操作在每个头内独立执行。

---

## 4. 门控信号

```
α_t = σ(W_α · x_t + b_α)     ∈ (0, 1)    遗忘门控（标量）
β_t = σ(W_β · x_t + b_β)     ∈ (0, 1)    输入门控（标量）
```

- `σ` 为 sigmoid 激活函数（论文标注"注意最终实现是否为 sigmoid"，但推导全程按 sigmoid）
- **初始化**：`b_α` 初始化为正值（如 `1.0`），使训练初期偏向保留历史信息
- 门控为**标量**，作用于隐状态矩阵的所有元素；逐维度自适应的职责完全交给预条件子

---

## 5. 核心状态更新流程（单步递归）

### 5.1 过渡状态（遗忘门衰减）

```
S̃_{t-1} = α_t · S_{t-1}                                    ... (19)
```

### 5.2 计算重构误差与一阶梯度

局部损失：
```
ℓ_t = (1/2) · ‖S̃_{t-1} · k_t - v_t‖²₂                    ... (21)
```

残差向量：
```
r_t = S̃_{t-1} · k_t - v_t = α_t · S_{t-1} · k_t - v_t      (d 维向量)
```

一阶梯度更新信号（秩一矩阵）：
```
G_t = r_t · k_t^⊤                                           ... (22)
```
> **关键性质**：`G_t` 是秩一矩阵（两个 d 维向量的外积），后续用于空间压缩。

### 5.3 二阶矩估计（EMA）

维护二阶统计量 `V_t ∈ R^{d×d}`：
```
V_t = ρ · V_{t-1} + (1 - ρ) · (G_t ⊙ G_t)                  ... (23)
```
- `ρ ∈ [0, 1)` 为衰减因子，典型值 `0.99`
- 初始化 `V_0 = 0`
- **stop-gradient**：`V_t` 的更新不参与反向传播计算图

### 5.4 偏差修正

```
V̂_t = V_t / (1 - ρ^t)                                      ... (24)
```

### 5.5 构建预条件子

```
P_t = √(V̂_t + ε)                                           ... (25)
```
- `ε` 为数值平滑常数，典型值 `1e-8`
- 开方、加法均为逐元素操作

### 5.6 自适应梯度

```
G̃_t = G_t ⊘ P_t                                            ... (26)
```

### 5.7 状态更新（最终公式）

```
S_t = α_t · S_{t-1} + β_t · ((v_t - α_t · S_{t-1} · k_t) · k_t^⊤ ⊘ P_t)
```

等价展开形式：
```
S_t = α_t · S_{t-1} - β_t · (G_t ⊘ P_t)                    ... (27)/(28)
```

完整闭式：
```
S_t = α_t · S_{t-1} + β_t · ((v_t - α_t · S_{t-1} · k_t) · k_t^⊤ ⊘ (√(V̂_t) + ε))   ... (29)
```

### 5.8 输出

```
o_t = S_t · q_t                                              (或 S_{t-1} · q_t，取决于实现)
```

---

## 6. 秩一分解压缩（空间 O(d²) → O(d)）

### 6.1 原理

由于 `G_t = r_t · k_t^⊤`，其逐元素平方可分解：
```
G_t ⊙ G_t = (r_t · k_t^⊤) ⊙ (r_t · k_t^⊤) = (r_t ⊙ r_t) · (k_t ⊙ k_t)^⊤   ... (30)
```

### 6.2 分解后的二阶矩追踪

只需维护两个 `d` 维向量（而非 `d×d` 矩阵）：
```
V_t^{(r)} = ρ · V_{t-1}^{(r)} + (1 - ρ) · (r_t ⊙ r_t)     ... (31)
V_t^{(k)} = ρ · V_{t-1}^{(k)} + (1 - ρ) · (k_t ⊙ k_t)     ... (32)
```
初始化：`V_0^{(r)} = 0`, `V_0^{(k)} = 0`

### 6.3 重构预条件子

近似重构（注意：这是近似，非精确等价）：
```
V̂_t ≈ V_t^{(r)} · (V_t^{(k)})^⊤ / (1 - ρ^t)              ... (35)
```

预条件子：
```
P_t = √(V_t^{(r)} · (V_t^{(k)})^⊤ / (1 - ρ^t) + ε)        ... (36)
```

> **实现提示**：利用广播（broadcasting）机制，`V_t^{(r)}` 为列向量 `(d,1)`，`V_t^{(k)}` 为行向量 `(1,d)`，外积自动广播得到 `d×d` 矩阵，无需显式构造后存储。

### 6.4 多头下的空间复杂度

每个头维护：`S_t^{(h)} ∈ R^{d_h × d_h}` + `V_t^{(r,h)}, V_t^{(k,h)} ∈ R^{d_h}`

总空间：`O(H · d_h² + H · 2d_h) = O(d²/H + 2d)`，二阶矩额外开销仅 `O(2d)`。

---

## 7. 分块并行计算（训练阶段）

将长度 `L` 的序列划分为 `L/C` 个大小为 `C` 的块。

### 7.1 块内并行（补充：预条件子吸收与 FLA 对接）

在每个块内，**固定预条件子**为块边界值 `P_{nC}`：

```
S_t^{(intra)} = α_t · S_{t-1}^{(intra)} + β_t · ((v_t - α_t · S_{t-1}^{(intra)} · k_t) · k_t^⊤ ⊘ P_{nC})   ... (37)
```

#### 关键问题

FLA 的 `chunk_gated_delta_rule` 接口中 `beta` 是**标量**（每个时间步一个数），而 `β_t ⊘ P_{nC}` 是一个 `d×d` 矩阵。标量接口无法直接接受矩阵形状的系数。

#### 解决方案：利用 P 的秩一结构做向量级吸收

由于 `P_{nC}` 通过秩一分解近似为两个向量的外积再开方：

```
P_{nC} ≈ √(V_{nC}^{(r)} · (V_{nC}^{(k)})^⊤ / (1 - ρ^{nC}) + ε)
```

定义两个逐维度缩放向量（在每个块开始时预计算，形状均为 `(d_h,)`）：

```
s^{(r)} = 1 / (√(V_{nC}^{(r)} / (1 - ρ^{nC})) + √ε)      ... (d_h,)
s^{(k)} = 1 / (√(V_{nC}^{(k)} / (1 - ρ^{nC})) + √ε)      ... (d_h,)
```

其中 `√ε` 的选择需满足 `s^{(r)}_i · s^{(k)}_j ≈ 1 / P_{nC}[i,j]`。
令 `ε_r = ε_k = √ε`，则 `(√a_i + √ε)(√b_j + √ε) ≈ √(a_i·b_j) + ε`（当 `a_i·b_j >> ε` 时近似良好）。

#### 吸收后的等价变形

将预条件缩放吸收进更新信号的两个向量中：

```
原始更新:  β_t · (v_t - α_t · S · k_t) · k_t^⊤ ⊘ P_{nC}
         = β_t · (-r_t) · k_t^⊤ ⊘ P_{nC}
```

利用秩一结构：`(-r_t · k_t^⊤) ⊘ P_{nC} ≈ (-r_t ⊙ s^{(r)}) · (k_t ⊙ s^{(k)})^⊤`

因此，定义缩放后的键和值向量：

```
k̃_t = k_t ⊙ s^{(k)}          ... 块内每个时间步的 k 预乘缩放向量
ṽ_t = v_t ⊙ s^{(r)}          ... 块内每个时间步的 v 预乘缩放向量
```

> **注意**：`ṽ_t` 是对原始 `v_t` 做缩放，而非对残差 `r_t` 做缩放。
> Delta Rule 中 `r_t = S·k_t - v_t`，残差和 `v_t` 在同一个维度空间中。
> 对 `v_t` 和 `k_t` 分别做缩放后，GDN 的标准 Delta Rule 递归会自动产生缩放后的残差。

同时，**隐状态也需要对应变换**。令 `S̃ = diag(s^{(r)}) · S · diag(s^{(k)})`，则缩放后的递归在 `S̃` 空间中等价于标准 GDN：

```
S̃_t = α_t · S̃_{t-1} + β_t · (ṽ_t - α_t · S̃_{t-1} · k̃_t) · k̃_t^⊤
```

这与标准 GDN 形式完全一致，`β_t` 回到**标量**，可直接调用 FLA。

块处理完毕后，将隐状态变换回原空间：`S = diag(1/s^{(r)}) · S̃ · diag(1/s^{(k)})`。

#### 完整的块内 FLA 调用流程

```python
def chunk_with_fla(chunk_k, chunk_v, chunk_q, chunk_alpha, chunk_beta,
                   S_prev, V_r, V_k, rho, t_start, eps=1e-8):
    """
    chunk_k:     (C, d_h)  块内所有时间步的键向量
    chunk_v:     (C, d_h)  块内所有时间步的值向量
    chunk_q:     (C, d_h)  块内所有时间步的查询向量
    chunk_alpha: (C,)      块内所有时间步的遗忘门（标量）
    chunk_beta:  (C,)      块内所有时间步的输入门（标量）
    S_prev:      (d_h, d_h)  前一块末尾的隐状态
    V_r:         (d_h,)    残差二阶矩向量
    V_k:         (d_h,)    键二阶矩向量
    rho:         float     EMA 衰减因子
    t_start:     int       当前块起始时间步（用于偏差修正）
    eps:         float     数值平滑常数
    """
    sqrt_eps = sqrt(eps)
    C = chunk_k.shape[0]
    
    # === Step 1: 预计算块边界处的缩放向量 ===
    bias_corr = 1.0 - rho ** t_start
    s_r = 1.0 / (sqrt(V_r / bias_corr) + sqrt_eps)   # (d_h,)
    s_k = 1.0 / (sqrt(V_k / bias_corr) + sqrt_eps)   # (d_h,)
    
    # === Step 2: 缩放键和值 ===
    chunk_k_scaled = chunk_k * s_k[None, :]           # (C, d_h)
    chunk_v_scaled = chunk_v * s_r[None, :]           # (C, d_h)
    
    # === Step 3: 将隐状态变换到缩放空间 ===
    S_scaled = s_r[:, None] * S_prev * s_k[None, :]   # (d_h, d_h)
    
    # === Step 4: 调用 FLA 的标准 GDN 内核 ===
    # beta 为标量，形式与标准 GDN 完全一致
    S_scaled_out, chunk_outputs_scaled = fla_chunk_gated_delta_rule(
        k=chunk_k_scaled,         # (C, d_h)
        v=chunk_v_scaled,         # (C, d_h)
        q=chunk_q,                # (C, d_h) — 见下方查询处理说明
        g=log(chunk_alpha),       # (C,) FLA 用 log 域表示衰减
        beta=chunk_beta,          # (C,) 标量输入门
        initial_state=S_scaled,   # (d_h, d_h)
    )
    
    # === Step 5: 将隐状态变换回原空间 ===
    S_out = (1.0 / s_r)[:, None] * S_scaled_out * (1.0 / s_k)[None, :]
    
    # === Step 6: 块间二阶矩更新（stop-gradient） ===
    with torch.no_grad():
        # 在原空间近似计算残差（用 S_prev 近似整个块内的 S）
        # r_τ ≈ α_τ · S_prev · k_τ - v_τ
        all_r = (chunk_alpha[:, None] *
                 (S_prev @ chunk_k.T).T - chunk_v)        # (C, d_h)
        
        # 按公式 (38)(39) 带 ρ^{C-τ} 权重更新
        weights = rho ** torch.arange(C-1, -1, -1)        # [ρ^{C-1}, ..., ρ^0]
        V_r_new = (rho**C * V_r +
                   (1 - rho) * (weights[:, None] * (all_r ** 2)).sum(dim=0))
        V_k_new = (rho**C * V_k +
                   (1 - rho) * (weights[:, None] * (chunk_k ** 2)).sum(dim=0))
    
    return S_out, chunk_outputs_scaled, V_r_new, V_k_new
```

#### 查询向量的处理

输出 `o_t = S_t · q_t`。在缩放空间中 `S̃ = diag(s_r) · S · diag(s_k)`，所以：

```
o_t = S_t · q_t
    = diag(1/s_r) · S̃_t · diag(1/s_k) · q_t
```

有两种处理方案（根据 FLA 的 API 选择）：

- **方案 A**：传入 `q_adjusted = diag(1/s_k) · q_t`，FLA 输出后再乘 `diag(1/s_r)`
- **方案 B**：传入原始 `q_t`，在 FLA 输出后统一做 `o = diag(1/s_r) · diag(1/s_k) · o_raw`

建议根据 FLA 版本的具体接口选择方案。

#### ε 的分拆说明

将全局 `ε` 分拆为 `√ε · √ε` 会引入轻微的近似误差：

```
精确:   1/P[i,j] = 1 / (√(a_i · b_j / c) + ε)
近似:   s_r[i] · s_k[j] = 1 / ((√(a_i/c) + √ε) · (√(b_j/c) + √ε))
```

当 `a_i · b_j >> ε · c` 时（即对应维度的梯度波动不是极端微小），两者非常接近。
对于 `ε = 1e-8` 的典型取值，该条件几乎总是满足。
在极端稀疏的维度上，两种处理方式都会将步长约束在 `O(1/ε)` 附近，行为差异可忽略。



### 7.2 块间递归

第 `n` 个块接收前一块的输出 `S_{nC}` 及 `V_{nC}^{(r)}, V_{nC}^{(k)}` 作为初始条件，串行执行完整更新。

### 7.3 二阶矩的块级更新

在第 `n` 个块处理完毕后，批量更新二阶矩：

```
V_{(n+1)C}^{(r)} = ρ^C · V_{nC}^{(r)} + (1-ρ) · Σ_{τ=nC+1}^{(n+1)C} ρ^{(n+1)C-τ} · (r_τ ⊙ r_τ)   ... (38)

V_{(n+1)C}^{(k)} = ρ^C · V_{nC}^{(k)} + (1-ρ) · Σ_{τ=nC+1}^{(n+1)C} ρ^{(n+1)C-τ} · (k_τ ⊙ k_τ)   ... (39)
```

求和项可通过向量化操作在 GPU 上高效并行。

### 7.4 推断模式

推断时退化为逐步递归（`C = 1`），完整二阶自适应机制在每个时间步生效。

---

## 8. 隐状态正则化

对隐状态矩阵 `S_t` 施加 **GroupNorm**（论文中标注"这个正则化选择 group norm 还是 layernorm 还是没有还得等待实验验证"）。

目的：在无界流式推断中防止隐状态幅值因递归累积而漂移。

---

## 9. 初始化汇总

| 变量 | 初始化值 |
|---|---|
| `S_0` | **零矩阵** |
| `V_0` (或 `V_0^{(r)}, V_0^{(k)}`) | **零向量** |
| `b_α`（遗忘门偏置） | **正值**（如 `1.0`） |
| `W_k, W_v, W_q` 等 | 标准初始化（如 Xavier） |

---

## 10. 超参数

| 超参数 | 含义 | 典型搜索范围 |
|---|---|---|
| `ρ` | 二阶矩 EMA 衰减因子 | `{0.9, 0.95, 0.99, 0.995, 0.999}` |
| `C` | 分块大小 | `{16, 32, 64, 128, 256}` |
| `ε` | 数值平滑常数 | `1e-8` |
| `d` | 隐状态维度 | `{64, 128, 256}` |
| `H` | 多头数 | 取决于 `d` |
| 层数 | 堆叠层数 | `{1, 2}` |

---

## 11. 完整单步伪代码

```python
def damrec_step(x_t, S_prev, V_r_prev, V_k_prev, t, params):
    """
    x_t:       当前输入 (d,)
    S_prev:    前一步隐状态 (d, d)  [多头下为 (H, d_h, d_h)]
    V_r_prev:  残差二阶矩向量 (d,)  [多头下为 (H, d_h)]
    V_k_prev:  键二阶矩向量 (d,)    [多头下为 (H, d_h)]
    t:         当前时间步（从 1 开始，用于偏差修正）
    params:    模型参数
    """
    # --- 门控信号 ---
    alpha_t = sigmoid(W_alpha @ x_t + b_alpha)   # 标量遗忘门
    beta_t  = sigmoid(W_beta  @ x_t + b_beta)    # 标量输入门

    # --- 键值投影 ---
    k_t = W_k @ x_t                               # (d,)
    v_t = W_v @ x_t                               # (d,)
    q_t = W_q @ x_t                               # (d,) 用于输出

    # --- [多头] 切分为 H 组, 以下按单头示意 ---

    # --- Step 1: 过渡状态 ---
    S_tilde = alpha_t * S_prev                     # (d, d)

    # --- Step 2: 残差与一阶梯度 ---
    r_t = S_tilde @ k_t - v_t                      # (d,)
    # G_t = r_t @ k_t^T  (秩一, 不需显式构造)

    # --- Step 3: 更新分解后的二阶矩 ---
    V_r_t = rho * V_r_prev + (1 - rho) * (r_t * r_t)   # (d,) 逐元素
    V_k_t = rho * V_k_prev + (1 - rho) * (k_t * k_t)   # (d,) 逐元素

    # --- Step 4: 偏差修正 + 预条件子 ---
    bias_correction = 1.0 - rho ** t
    # P_t 通过广播计算，形状 (d, d)
    # P_t[i,j] = sqrt(V_r_t[i] * V_k_t[j] / bias_correction + epsilon)

    # --- Step 5: 自适应状态更新 ---
    # 更新信号: (v_t - alpha_t * S_prev @ k_t) @ k_t^T = -r_t @ k_t^T
    # 注意符号: r_t = alpha_t * S_prev @ k_t - v_t, 所以 v_t - ... = -r_t
    update = (-r_t)[:, None] * k_t[None, :]         # (d, d) 秩一外积
    update_scaled = update / P_t                     # (d, d) 逐元素除

    S_t = alpha_t * S_prev + beta_t * update_scaled  # (d, d)

    # --- Step 6: 正则化 ---
    S_t = group_norm(S_t)                            # 可选

    # --- Step 7: 输出 ---
    o_t = S_t @ q_t                                  # (d,)

    # --- [多头] 拼接各头输出 ---

    return o_t, S_t, V_r_t, V_k_t
```

---

## 12. 训练阶段分块并行伪代码

```python
def damrec_chunked_forward(X, C, params):
    """
    X: 输入序列 (L, d)
    C: 块大小
    """
    L = X.shape[0]
    num_chunks = L // C

    # 初始化
    S = zeros(d, d)
    V_r = zeros(d)
    V_k = zeros(d)

    all_outputs = []

    for n in range(num_chunks):
        chunk = X[n*C : (n+1)*C]  # (C, d)

        # 计算块边界处的预条件子（固定用于整个块内）
        bias_corr = 1.0 - rho ** (n * C + 1)  # 近似
        P_boundary = sqrt(V_r[:, None] * V_k[None, :] / bias_corr + eps)  # (d, d)

        # --- 块内并行递归 ---
        # 对 chunk 内 C 个时间步，用固定 P_boundary 执行门控线性递归
        # 可用并行前缀扫描加速
        S, chunk_outputs, all_r, all_k = chunk_parallel_recurrence(
            chunk, S, P_boundary, params
        )

        all_outputs.append(chunk_outputs)

        # --- 块边界处更新二阶矩 ---
        # 利用块内所有 r_τ, k_τ 批量更新
        for tau in range(C):
            decay = rho ** (C - tau)
            # 累积到 V_r, V_k（可向量化）
        V_r = rho**C * V_r + (1-rho) * weighted_sum(all_r)
        V_k = rho**C * V_k + (1-rho) * weighted_sum(all_k)

    return concat(all_outputs)
```

---

## 13. 复杂度汇总

| 模式 | 时间复杂度 | 空间复杂度 |
|---|---|---|
| **流式推断（单步）** | `O(d²)` | `O(d²)` （与序列长度 L 无关） |
| **训练（全序列）** | `O(L · d²)` | `O(d²)` + 序列中间状态 |

二阶自适应机制的额外开销：时间 `O(d)`，空间 `O(d)`，不改变渐近复杂度。

---

## 14. 与 GDN 基线的关键区别

| 组件 | GDN (一阶基线) | DamRec |
|---|---|---|
| 状态更新 | `S_t = α·S_{t-1} + β·(v-α·S·k)·k^⊤` | 同左，但对更新信号除以预条件子 `P_t` |
| 步长缩放 | 标量 `β_t` 对所有维度统一 | `β_t / P_t` 实现逐维度差异化 |
| 额外状态 | 无 | `V_r, V_k` 各 `d` 维向量 |
| 训练并行 | 标准并行前缀扫描 | 分块：块内固定预条件 + 块间更新 |

---

## 15. 注意事项

1. **stop-gradient**：二阶矩 `V_t` 的更新以 detach/stop_gradient 方式执行，不参与反向传播。
2. **秩一分解是近似**：`V_r · V_k^⊤` ≠ 真实的 `V_t`，差异源于"和的外积"与"外积的和"的交叉项。当 `ρ` 接近 1 时近似较好。
3. **偏差修正只做一次**：对重构后的矩阵 `V_r · V_k^⊤` 统一除以 `(1 - ρ^t)`，不要对两个因子分别修正再相乘（否则分母变 `(1-ρ^t)²`，冷启动期过度放大）。
4. **块内预条件子近似误差**：在 `ρ ≥ 0.99, C ≤ 128` 时，`(1-ρ)·C ≤ 1.28`，误差可忽略。
5. **门控与预条件子职责分离**：`α_t, β_t` 负责全局时序节奏，`P_t` 负责逐维度自适应，避免功能冗余。

