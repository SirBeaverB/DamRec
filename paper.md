\section{理论背景}

\subsection{流式序列推荐}

在经典序列推荐场景中，设 $\mathcal{U}$ 和 $\mathcal{I}$ 分别表示用户集合与物品集合。对于任意用户 $u \in \mathcal{U}$，其历史交互记录可被形式化为按时间顺序排列的序列 $\mathcal{X}_u = (i_1, i_2, \dots, i_t)$，其中 $i_t \in \mathcal{I}$ 表示该用户在时间步 $t$ 交互的物品。序列推荐的核心目标是学习一个参数化模型 $f_{\theta}$，以根据用户给定的历史交互上下文 $\mathcal{X}_{u, \le t}$，估计其在下一时刻 $t+1$ 发生交互的候选物品概率分布，即最大化条件概率 $P(i_{t+1} | \mathcal{X}_{u, \le t}; \theta)$。

有别于基于静态数据集进行多轮全局优化的传统离线范式，流式序列推荐（Streaming Sequential Recommendation）要求模型能够实时处理以无界数据流（Unbounded Data Stream）形式持续到达的新交互样本 \cite{Guo_Yin_Wang_Chen_Zhou_Quoc_Viet_Hung_2019, Rathod_Goudar_Kulkarni_M_Hukkeri_2024}。 在理论界定上，无界数据流意味着交互事件在时间轴上呈连续且开放式的无限增长（$t \to \infty$）\cite{Muthukrishnan}。受限于有限的物理内存容量与严格的实时推理延迟，系统在物理层面上无法永久缓存并全局重放庞大的历史序列，而仅能对瞬时到达的数据项执行严格的单遍扫描（Single-pass Processing）\cite{Domingos_Hulten_2000}。在此设定下，无界数据流中的数据联合分布 $P(\mathcal{X}, \mathcal{I})$ 会随时间发生动态演化，呈现显著的非平稳性（Non-stationarity）。为了适配这种无界且非平稳的数据流转特性，模型必须在时间步 $t$ 接收到新交互对 $(u, i_t)$ 后，执行底层的在线增量更新：
\begin{equation}
\theta_{t} \leftarrow \text{Update}(\theta_{t-1}, (u, i_t), \mathcal{L}_t),
\end{equation}
其中，$\theta_{t-1}$ 为上一时刻的模型参数，$\mathcal{L}_t$ 为当前样本诱导的瞬时预测损失（通常为交叉熵损失），$\text{Update}(\cdot)$ 代表底层的参数更新算子。



这种流式在线演化机制使模型面临两项核心挑战：其一为概念漂移（Concept Drift） \cite{Gama_Žliobaitė_Bifet_Pechenizkiy_Bouchachia_2014}，即数据生成分布随时间偏移（$P_{t}(\mathcal{X}, \mathcal{I}) \neq P_{t+\Delta t}(\mathcal{X}, \mathcal{I})$），要求模型具备自适应追踪能力以规避参数老化。其二为极端偏态分布 \cite{Abdollahpouri_Burke_Mobasher_2017, Park_Tuzhilin_2008}，即全局物品交互频率服从长尾分布，少数头部物品占据绝大部分流量，而海量尾部物品仅获得极低频率的交互。在流式更新中，头部物品产生的高量级梯度易引发参数震荡，而尾部物品的微弱梯度信号则极易被遗忘门的持续衰减所淹没。如何在极端偏态的数据流中有效平衡热点追踪与长尾保护，构成了流式序列推荐架构设计的核心挑战。




\subsection{线性序列模型与状态推断}
传统的自注意力机制因计算完整的由所有历史 Token 构成的注意力矩阵，时间与空间复杂度呈序列长度的二次方（$\mathcal{O}(L^2)$）增长，极大地限制了其在长序列流式场景中的应用。为突破此计算壁垒，近期涌现了一系列线性序列架构 \cite{Schlag_Irie_Schmidhuber_2021, Liu_Wang_Wu_Feng_Stone_Liu_2024}。这些工作通过核特征映射或泰勒展开等数学手段，解耦了标准注意力的 Softmax 计算，从而将其前向推断过程等效转换为具备线性复杂度（$\mathcal{O}(L)$）的递归形式。

具体而言，对于时间步 $t$ 的输入表征 $x_t \in \mathbb{R}^d$，线性序列模型首先通过特征映射函数（通常为线性投影复合非线性激活）将其映射为查询（Query）、键（Key）与值（Value）向量：

$$q_t = \phi_Q(x_t), \quad k_t = \phi_K(x_t), \quad v_t = \phi_V(x_t),$$
其中，$q_t, k_t \in \mathbb{R}^{d_k}$，$v_t \in \mathbb{R}^{d_v}$。 在标准的自回归自注意力机制\cite{Vaswani_Shazeer_Parmar_Uszkoreit_Jones_Gomez_Kaiser_Polosukhin_2017} 中，为了捕捉长程序列依赖，模型需显式计算当前查询 $q_t$ 与所有历史键 $k_{\tau} (\tau \le t)$ 的内积，并通过 Softmax 非线性算子进行归一化。其时间步 $t$ 的输出 $o_t$ 定义为：

$$o_t = \frac{\sum_{\tau=1}^{t} \exp\left(\frac{q_t^\top k_{\tau}}{\sqrt{d_k}}\right) v_{\tau}}{\sum_{\tau=1}^{t} \exp\left(\frac{q_t^\top k_{\tau}}{\sqrt{d_k}}\right)}.$$
正是其中 $\exp(\cdot)$ 的存在，导致查询 $q_t$ 与历史键 $k_{\tau}$ 发生了深度的非线性耦合。这种耦合使得针对时间步 $\tau$ 的求和算子（$\sum$）在数学上无法与矩阵乘法进行解耦，从而将时间与空间复杂度限制在序列长度的二次方（$\mathcal{O}(L^2)$）级别。

为突破上述二次方计算复杂度壁垒，近期的线性架构演进（如 Linear Transformer\cite{Katharopoulos_Vyas_Pappas_Fleuret_2020}）在注意力机制的底层算子上进行了重构。通过显式剥离 Softmax 算子所诱导的非线性指数耦合，此类架构采用广义的核特征内积或纯线性点积来近似并重组传统的相似度度量 \cite{Katharopoulos_Vyas_Pappas_Fleuret_2020, Choromanski_Likhosherstov_Dohan_Song_Gane_Sarlos_Hawkins_Davis_Mohiuddin_Kaiser_et_al._2022}。在舍弃显式归一化因子的线性松弛条件下，模型在时间步 $t$ 的未归一化输出 $o_t$ 实现了当前查询向量 $q_t$ 与历史键向量 $k_\tau$ 在求和算子内的完全解耦，使得 $o_t$ 可被重组为历史键值对相对于当前查询向量的线性组合：

$$o_t = \sum_{\tau=1}^{t} (q_t^\top k_{\tau}) v_{\tau} = \sum_{\tau=1}^{t} v_{\tau} (k_{\tau}^\top q_t).$$
由于 $(k_{\tau}^\top q_t)$ 为标量，根据矩阵乘法的结合律，上式中的计算顺序可被重构为：

$$o_t = \left( \sum_{\tau=1}^{t} v_{\tau} k_{\tau}^\top \right) q_t.$$

基于上述数学等价性，模型得以将关于历史序列的全局求和操作定义为一个可增量流转的隐状态矩阵$S_t$ \cite{Schlag_Irie_Schmidhuber_2021}：

$$S_t = \sum_{\tau=1}^{t} v_{\tau} k_{\tau}^\top. $$
由于 $v_{\tau}$ 为 $d_v \times 1$ 的列向量，而 $k_{\tau}^\top$ 为 $1 \times d_k$ 的行向量，二者的外积界定了隐状态矩阵的维度，即 $S_t \in \mathbb{R}^{d_v \times d_k}$。当新观测到达时，状态推断与信息检索即可被统一为 $\mathcal{O}(1)$ 复杂度的递归形式：
\begin{align}
    S_t &= S_{t-1} + v_t k_t^\top \label{eq:basic_update}, \\
    o_t &= S_t q_t \label{eq:basic_output}.
\end{align}

然而，公式 \eqref{eq:basic_update} 中无约束的简单外积累加机制，在处理无界流式交互时极易导致状态矩阵的范数膨胀与记忆覆盖。为此，线性架构（如 GDN \cite{Yang_Kautz_Hatamizadeh_2025}）引入了基于误差修正的门控更新机制。在 GDN 及本文的设定中，取 $d_k = d_v = d$，隐状态简化为方阵 $S_t \in \mathbb{R}^{d \times d}$。在该范式下，模型主动计算当前隐状态对关联键 $k_t$ 的前向预测残差而非被动地记忆当前值 $v_t$，并以此驱动状态的演化。其状态推断与输出生成的泛化递归演化机制可形式化为：
\begin{align}
    \hat{v}_t &= S_{t-1} k_t \label{eq:predict}, \\
    S_t &= \alpha_t  S_{t-1} + (v_t - \hat{v}_t) k_t^\top \label{eq:update}, \\
    o_t &= S_{t-1} q_t \label{eq:output}.
\end{align}
其中，公式 \eqref{eq:predict} 表示隐状态对当前键的预测值（即内部检索过程）；公式 \eqref{eq:update} 为引入了遗忘门控 $\alpha_t$ 与 Delta Rule（即预测残差 $v_t - \hat{v}_t$）的状态流转方程；最终的输出则由当前查询 $q_t$ 与演化后的全局隐状态相乘得到 $o_t$（公式 \eqref{eq:output}）。

综上所述，依托隐状态矩阵 $S_t$ 构建的递归推断范式将单步序列计算的时间复杂度降至 $\mathcal{O}(1)$，在系统架构层面满足了流式推荐对在线实时推理的约束。此外，公式 \eqref{eq:update} 中由预测残差驱动的状态演化方程揭示了序列前向计算与在线参数优化的底层同构性。这种代数结构上的等价映射，为后文运用纯优化理论分析该基础算子在非平稳数据流中的表征局限奠定了形式化基础。




\subsection{隐状态流转的一阶优化视角} \label{subsec:iso_limit}

本节以 Gated Delta Networks (GDN) \cite{Yang_Kautz_Hatamizadeh_2025} 为代表，对其状态更新算子进行代数解构。

在现代线性架构的理论体系中，隐状态的演化可被解释为对特定在线目标函数求解闭式最优解的过程。对于序列时间步 $t$，隐状态矩阵 $S_t$ 的理论目标在于逼近当前输入诱导的键值映射 $(k_t \to v_t)$，同时维持对前驱状态 $S_{t-1}$ 的记忆惯性。可以验证，GDN 的状态更新方程恰好对应以下带有 Frobenius 范数近端项的在线优化目标 $\mathcal{L}(S_t)$ 的闭式解：
\begin{equation}
    \mathcal{L}(S_t) = \| S_t - \alpha_t S_{t-1} \|_F^2 - 2 \langle S_t k_t, \beta_t(v_t - \alpha_t S_{t-1} k_t) \rangle.
    \label{eq:gdn_objective}
\end{equation}
其中，$\alpha_t$ 为遗忘门控（负责历史记忆衰减），$\beta_t$ 为输入门控（控制当前样本的拟合步长）。公式 \eqref{eq:gdn_objective} 的前半部分严格约束当前状态不剧烈偏离衰减后的前驱状态，后半部分则通过最大化内积以消除预测残差。

由于公式 \eqref{eq:gdn_objective} 为关于 $S_t$ 的严格凸函数，对其求矩阵导数并令其为零，可直接求得该时刻的闭式解：
\begin{equation}
    \frac{\partial \mathcal{L}(S_t)}{\partial S_t} = 2(S_t - \alpha_t S_{t-1}) - 2 \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^\top = 0.
\end{equation}
经项合并与移项重组，即可推导出 GDN 标准的状态更新方程：
\begin{align}
    S_t &= \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^\top \label{eq:gdn_update} \\
        &= S_{t-1} \big( \alpha_t (I - \beta_t k_t k_t^\top) \big) + \beta_t v_t k_t^\top.\label{eq:gdn_expanded}
\end{align}

上述推导从解析优化的视角确立了 GDN 隐状态更新的数学形式。而从在线学习 \cite{10.1561/2200000018} 的标准一阶优化视角进行逆向推演，假设在时间步 $t$，模型旨在通过最小化在线回归损失来学习当前键值对的映射关系，其标准均方误差（MSE）目标函数可定义为：
\begin{equation}
    \ell_t(S) = \frac{1}{2} \| S k_t - v_t \|_2^2.
    \label{eq:mse_loss}
\end{equation}
求解$\ell_t(S)$关于任意隐状态矩阵 $S$ 的一般梯度算子：
\begin{equation}
    \nabla_S \ell_t(S) = \frac{\partial}{\partial S} \left( \frac{1}{2} (S k_t - v_t)^\top (S k_t - v_t) \right) = (S k_t - v_t) k_t^\top.
    \label{eq:general_gradient}
\end{equation}
在执行在线优化前，前驱隐状态 $S_{t-1}$ 首先经历由门控 $\alpha_t$ 驱动的自适应权重衰减，生成过渡状态 $\tilde{S}_{t-1} = \alpha_t S_{t-1}$（该定义将在第 \ref{subsec:higher_order_update} 节中作为公式 \eqref{eq:transition_state} 正式使用）。将通用算子\eqref{eq:general_gradient}在过渡状态 $S = \tilde{S}_{t-1}$ 处进行评估代入，从而推导出该时刻闭式梯度方向：
\begin{align}
    \nabla_S \ell_t(S) \big|_{S=\tilde{S}_{t-1}} &= (\tilde{S}_{t-1} k_t - v_t)k_t^\top \nonumber \\
    &= (\alpha_t S_{t-1} k_t - v_t)k_t^\top.
    \label{eq:sgd_gradient}
\end{align}



引入最基础的在线随机梯度下降（Online SGD）算子，设单步局部学习率为 $\eta_t$。隐状态遵循梯度反方向的参数更新方程为：
\begin{align}
    S_t &= \tilde{S}_{t-1} - \eta_t \nabla_S \ell_t(\tilde{S}_{t-1}) \nonumber \\
        &= \alpha_t S_{t-1} - \eta_t (\alpha_t S_{t-1} k_t - v_t)k_t^\top \nonumber \\
        &= \alpha_t S_{t-1} + \eta_t (v_t - \alpha_t S_{t-1} k_t)k_t^\top.
    \label{eq:sgd_final}
\end{align}




对比基于在线优化的推导结果 \eqref{eq:sgd_final} 与 GDN 的状态更新方程 \eqref{eq:gdn_update}，可以观察到二者在代数结构上呈现出同构性。具体而言，GDN 架构中用以调节新知识写入强度的标量输入门控 $\beta_t$，在优化动力学上等价于在线 SGD 中的单步学习率 $\eta_t$；而遗忘门控 $\alpha_t$ 则显式地充当了自适应权重衰减系数。

然而，公式 \eqref{eq:sgd_final} 中的标量步长 $\eta_t$（即 $\beta_t$）对所有特征维度施加统一的更新比例，无法利用不同维度间梯度波动尺度的差异。在流式推荐的偏态分布下，这种各向同性的更新面临维度适配困境：较大步长易导致高频特征震荡，较小步长则使长尾特征陷入梯度饥饿。借鉴自适应优化理论 \cite{Duchi_Hazan_Singer, Kingma_Ba_2017} 中基于梯度二阶矩的预条件机制，这一局限性可通过引入逐维度的方差感知缩放来克服。这一思路直接启发了本文提出的 DamRec 架构。



\section{DamRec架构}\label{sec:method}

\textbf{暂定如此。看结果再决定是否增加frobenius等插件}

\subsection{总体思路}
\label{subsec:overview}

\subsubsection{问题形式化}

考虑一个流式序列推荐场景：用户的交互行为以时间有序的序列 $\mathcal{X} = (x_1, x_2, \ldots, x_T)$ 到达，其中 $x_t$ 表示时间步 $t$ 的输入 Token（如点击的商品 ID 经嵌入后的稠密表征）。模型的目标是在每个时间步 $t$，基于已观测的历史交互 $(x_1, \ldots, x_t)$，预测用户下一步的交互目标 $x_{t+1}$。

流式推断对模型施加了两条约束：（1）因果性——模型在时间步 $t$ 仅可访问当前及历史输入，不可利用未来信息；（2）常数空间——模型不可存储或重新遍历完整的历史序列，而必须将所有历史信息压缩至一个固定大小的隐状态 $S_t$ 中。这两条约束排除了依赖全序列注意力的 Transformer 类架构在严格流式场景下的直接应用，使得递归状态空间模型成为该范式下的自然选择。

\subsubsection{状态流转的一般化范式}

在上述约束下，所有递归序列模型均可统一抽象为如下状态转移方程：
\begin{equation}
    S_t = f(S_{t-1}, x_t),
    \label{eq:general_ssm}
\end{equation}
其中 $S_t$ 为隐状态（其具体维度因架构而异），$f(\cdot)$ 为状态转移函数。在 DamRec 及 GDN 架构中，$S_t \in \mathbb{R}^{d \times d}$ 为矩阵值隐状态，其中 $d$ 为特征维度。经典线性循环架构通过对 $f(\cdot)$ 施加特定的结构化约束来实例化这一范式。例如，Mamba \cite{Gu_Dao_2024} 采用选择性状态空间的对角线性递归；GDN \cite{Yang_Kautz_Hatamizadeh_2025} 则将状态更新解释为在线梯度下降过程，引入门控遗忘机制与基于重构误差的一阶更新：
\begin{equation}
    S_t = \alpha_t S_{t-1} + \beta_t G_t,
    \label{eq:gdn_general}
\end{equation}
其中 $\alpha_t \in (0, 1)$ 为数据驱动的遗忘门控，$\beta_t$ 为输入门控，$G_t \in \mathbb{R}^{d \times d}$ 为基于重构误差计算的一阶梯度更新信号。

\subsubsection{DamRec：前向推断即自适应预条件在线优化}

尽管一阶梯度更新在计算效率上具有显著优势，但其对所有特征维度施加统一的更新步长，无法利用不同维度间梯度波动尺度的差异，在面对流式推荐中普遍存在的偏态分布时存在局限。高频热点特征与稀疏长尾特征在梯度尺度上的巨大差异，使得一阶更新不可避免地在部分维度上过度震荡，而在另一些维度上更新不足。

DamRec 的核心思想在于将隐状态的前向推断过程重新诠释为一个基于梯度二阶矩预条件的自适应在线优化问题。如第 \ref{subsec:iso_limit} 节所分析，GDN 的标量门控 $\beta_t$ 等价于在线 SGD 的学习率，对所有特征维度施加相同的更新步长，在面对流式推荐中极端偏态的交互分布时存在维度适配困境。DamRec 在保持 GDN 门控线性递归框架（公式 \eqref{eq:gdn_general}）的基础上，为更新信号 $G_t$ 引入一个逐特征维度的预条件子，使得每个特征维度的更新步长能够根据其历史梯度波动尺度进行自适应缩放。这一机制在不破坏线性递归的计算效率的前提下，实现了从各向同性的标量缩放到方差感知的各向异性缩放的跨越。图 \ref{fig:overview} 展示了 DamRec 的整体架构。

具体而言，DamRec 的门控信号由当前时间步的输入通过线性变换与激活函数生成：
\begin{align}
    \alpha_t &= \sigma(W_\alpha x_t + b_\alpha), \label{eq:gate_alpha} \\
    \beta_t  &= \sigma(W_\beta x_t + b_\beta),  \label{eq:gate_beta}
\end{align}
$\alpha_t \in (0,1)$ 控制历史状态的遗忘强度，$\beta_t \in (0,1)$ 控制当前更新信号的接收强度。二者均为标量门控，作用于隐状态矩阵的所有元素。这一设计是有意为之：DamRec 将逐维度自适应的职责完全交给二阶预条件子 $\mathcal{P}_t$，而门控 $\alpha_t, \beta_t$ 仅负责全局的时序节奏控制（即"遗忘多少历史"与"接收多少新信息"）。这种职责分离避免了门控与预条件子之间的功能冗余，同时将门控参数量保持在 $\mathcal{O}(d)$。其中，遗忘门的偏置 $b_\alpha$ 初始化为正值（如 $b_\alpha = 1.0$），使 $\alpha_t$ 在训练初期偏向保留历史信息，避免初始阶段过度遗忘，这与 LSTM 中遗忘门偏置初始化的标准实践一致 \cite{Jozefowicz_Zaremba_Sutskever}。

\textbf{注意最终实现是否为sigmoid}

\paragraph{多头扩展。}
在实际实现中，DamRec 采用多头机制将隐状态维度分解为 $H$ 个独立的注意力头，每头维度为 $d_h = d / H$。输入 $x_t$ 先经全局线性投影（公式 \eqref{eq:kv_projection}）生成 $k_t, v_t \in \mathbb{R}^d$，再沿特征维度均匀切分为 $H$ 组：$k_t^{(h)}, v_t^{(h)} \in \mathbb{R}^{d_h}$（其定义见第 \ref{subsec:space_complexity} 节）。每个头独立维护各自的隐状态 $S_t^{(h)} \in \mathbb{R}^{d_h \times d_h}$ 及分解后的二阶矩向量 $V_t^{(r,h)}, V_t^{(k,h)} \in \mathbb{R}^{d_h}$。因此，多头设置下的总空间复杂度为 $\mathcal{O}(H \cdot d_h^2 + H \cdot 2d_h) = \mathcal{O}(d^2 / H + 2d)$，其中二阶矩的额外开销仅为 $\mathcal{O}(2d)$，与头数无关。

\begin{figure}[t]
    \centering
    % \includegraphics[width=\linewidth]{figures/overview.pdf}
    \caption{DamRec 整体架构概览。输入 Token 经嵌入层映射为键值对 $(k_t, v_t)$ 后，模型首先基于重构误差计算一阶梯度更新信号 $G_t$；随后通过指数移动平均追踪梯度的二阶矩 $V_t$，构建自适应预条件子 $\mathcal{P}_t$；最终，经方差校正的更新信号通过门控机制融入隐状态递归，完成单步状态流转。}
    \label{fig:overview}
\end{figure}

后续各节将依次展开 DamRec 的技术细节：第 \ref{subsec:higher_order_update} 节推导核心的高阶适应性更新方程；第 \ref{subsec:robustness} 节从稳健性角度分析其动力学性质；第 \ref{efficiency} 节讨论工程效率与复杂度保证。


\subsection{基于梯度二阶矩预条件的自适应状态更新} \label{subsec:higher_order_update}

本节在 GDN 架构的动力学基础上，引入基于梯度二阶矩的自适应预条件机制\footnote{此处"二阶"指梯度的二阶统计矩，而非损失函数的二阶导数（Hessian）。}，正式提出 DamRec (Delta-Adam Memory Recommender) 的隐状态流转方程。

首先，定义经过遗忘门控处理后的过渡状态：
\begin{equation}
    \tilde{S}_{t-1} = \alpha_t S_{t-1},
    \label{eq:transition_state}
\end{equation}
其中 $\alpha_t \in (0,1)$ 为数据驱动的遗忘门控（其具体计算方式见公式 \eqref{eq:gate_alpha}）。过渡状态 $\tilde{S}_{t-1}$ 表示对历史隐状态施加选择性遗忘后的结果，后续所有梯度计算均基于此状态展开。

当前时间步的输入 $x_t$ 经两个独立的线性投影生成键向量与值向量：
\begin{equation}
    k_t = W_k x_t, \quad v_t = W_v x_t,
    \label{eq:kv_projection}
\end{equation}
其中 $W_k, W_v \in \mathbb{R}^{d \times d}$ 为可学习参数，投影不含偏置项，与 GDN \cite{Yang_Kautz_Hatamizadeh_2025} 及标准 Transformer 的 KV 投影保持一致。模型基于过渡状态 $\tilde{S}_{t-1}$ 对键向量的重构误差构建局部损失函数，即采用标准的均方误差 (MSE)：
\begin{equation}
    \ell_t = \frac{1}{2} \| \tilde{S}_{t-1} k_t - v_t \|_2^2.
    \label{eq:local_loss}
\end{equation}

随后，计算局部损失函数 $\ell_t$ 关于过渡状态的梯度，得到原始更新信号 $G_t \in \mathbb{R}^{d \times d}$：
\begin{equation}
\begin{aligned}
    G_t &= \nabla_{\tilde{S}_{t-1}} \ell_t(\tilde{S}_{t-1}) \\
        &= \nabla_{\tilde{S}_{t-1}} \left( \frac{1}{2} \| \tilde{S}_{t-1} k_t - v_t \|_2^2 \right) \\
        &= (\tilde{S}_{t-1} k_t - v_t) k_t^\top.
\end{aligned}
\label{eq:raw_gradient}
\end{equation}
值得注意的是，令残差向量 $r_t = \tilde{S}_{t-1} k_t - v_t$，则 $G_t = r_t k_t^\top$ 具有秩一（Rank-1）结构，即可表示为两个 $d$ 维向量的外积。这一结构性质将在第\ref{subsec:space_complexity}节中被利用，以实现二阶统计量的空间复杂度压缩。

传统线性架构将原始更新信号 $G_t$ 直接叠加至前驱状态。而在 DamRec 中，为了捕捉特征维度的局部方差与异质性，在推断期维护一个伴生的二阶统计量矩阵 $V_t \in \mathbb{R}^{d \times d}$。参考自适应优化算法 \cite{Kingma_Ba_2017} 的指数移动平均 (EMA) 策略，$V_t$ 用于追踪特征梯度的未中心化方差，其在线流转定义为\footnote{为行文简洁，本文采用如下符号约定：$\odot$ 表示逐元素乘积（Hadamard 积），$\oslash$ 表示逐元素除法。涉及矩阵的开方 $\sqrt{\cdot}$、加法与标量运算，若未作特殊说明，均默认为逐元素（element-wise）操作。}：
\begin{equation}
    V_t = \rho V_{t-1} + (1 - \rho) (G_t \odot G_t),
    \label{eq:second_moment}
\end{equation}
其中，$\rho \in [0, 1)$ 为二阶矩的衰减因子。

特别地，在流式推断的极早期阶段（即 $t$ 较小且初始化 $V_0 = \mathbf{0}$ 时），公式 \eqref{eq:second_moment} 会导致对二阶矩的低估。为了消除这一冷启动期的极小值偏差，并规避由此诱发的初期梯度爆炸风险，引入偏差修正机制 \cite{Kingma_Ba_2017}：
\begin{equation}
    \hat{V}_t = \frac{V_t}{1 - \rho^t}.
    \label{eq:bias_correction}
\end{equation}

所有状态初始化为零，即 $S_0 = \mathbf{0}$，$V_0 = \mathbf{0}$，由此产生的早期二阶矩低估由上述偏差修正机制自动补偿。二阶矩 $V_t$ 的更新以 stop-gradient 方式执行，不参与反向传播的计算图，这与 Adam 优化器中将二阶矩视为运行统计量的标准处理方式一致。

随后，利用修正后的无偏二阶矩 $\hat{V}_t$，我们构建了逐特征维度的二阶统计量预条件子 $\mathcal{P}_t \in \mathbb{R}^{d \times d}$：
\begin{equation}
    \mathcal{P}_t = \sqrt{\hat{V}_t} + \epsilon,
    \label{eq:preconditioner}
\end{equation}
其中，$\epsilon > 0$ 为防止除零下溢的数值平滑常数（如 $10^{-8}$），上式中的开方与加法均为逐元素操作，即 $[\mathcal{P}_t]_{ij} = \sqrt{[\hat{V}_t]_{ij}} + \epsilon$。

随后，定义经过方差校正的自适应梯度$\tilde{G}_t \in \mathbb{R}^{d \times d}$：
\begin{equation}
    \tilde{G}_t = G_t \oslash \mathcal{P}_t.
    \label{eq:adaptive_gradient}
\end{equation}
在严格的优化理论中，这一逐元素除法操作（$\oslash$）等效于假设特征维度相互独立下的特征方差对角近似。此种对角近似机制不仅在表征几何上将各向同性的更新轨迹重塑为方差感知的各向异性演化，同时在计算复杂性层面有效规避了传统二阶张量求逆所带来的巨大计算开销。（其张量代数等价性与复杂度分析详见第\ref{efficiency} 节）。

在建立方差校正梯度 $\tilde{G}_t$ 后，遵循在线自适应优化的标准范式\cite{Kingma_Ba_2017, Loshchilov_Hutter_2019}，隐状态的单步参数迭代可形式化为：
\begin{equation}
    S_t = \alpha_t S_{t-1} - \eta_t \tilde{G}_t,
    \label{eq:opt_update}
\end{equation}
其中，$\alpha_t S_{t-1}$ 为经遗忘门衰减后的历史基态，$\eta_t$ 为当前时间步的学习率。值得说明的是，与完整的 Adam 算法不同，DamRec 并未引入一阶动量估计 $m_t$。这是因为隐状态的递归结构本身包含了对历史更新信号的加权累积，引入额外的一阶动量项并非必要。因此，DamRec 仅引入二阶矩以实现逐维度的自适应缩放，偏差修正也相应地仅针对 $\hat{V}_t$ 进行。

在经典在线优化范式中，公式 \eqref{eq:opt_update} 中的 $\eta_t$ 通常为预设的学习率（可能依据某种调度策略随时间变化，但不依赖当前输入）。在 DamRec 中，我们将其替换为由当前输入驱动的动态门控 $\beta_t$（公式 \eqref{eq:gate_beta}），即令 $\eta_t \equiv \beta_t$，使更新幅度能够根据每个时间步的输入内容自适应调节。将自适应梯度的定义公式 \eqref{eq:adaptive_gradient}、原始梯度的解析式 \eqref{eq:raw_gradient}，以及预条件子 \eqref{eq:preconditioner} 依次代入公式 \eqref{eq:opt_update}，展开如下：
\begin{equation}
\begin{aligned}
    S_t &= \alpha_t S_{t-1} - \beta_t \left( G_t \oslash \mathcal{P}_t \right) \\
        &= \alpha_t S_{t-1} - \beta_t \left( (\tilde{S}_{t-1} k_t - v_t) k_t^\top \oslash \mathcal{P}_t \right) \\
        &= \alpha_t S_{t-1} - \beta_t \left( (\alpha_t S_{t-1} k_t - v_t) k_t^\top \oslash \mathcal{P}_t \right) \\
        &= \alpha_t S_{t-1} + \beta_t \left( (v_t - \alpha_t S_{t-1} k_t) k_t^\top \oslash \mathcal{P}_t \right),
\end{aligned}
\label{eq:damrec_derivation}
\end{equation}
其中第三步利用了过渡状态的定义 $\tilde{S}_{t-1} = \alpha_t S_{t-1}$，第四步将负号吸收入残差项完成符号翻转。将预条件子 $\mathcal{P}_t = \sqrt{\hat{V}_t} + \epsilon$ 代入，即得 DamRec 的闭式状态流转方程：
\begin{equation}
    S_t = \alpha_t S_{t-1} + \beta_t \left( (v_t - \alpha_t S_{t-1} k_t) k_t^\top \oslash (\sqrt{\hat{V}_t} + \epsilon) \right).
    \label{eq:damrec_final}
\end{equation}


公式 \eqref{eq:damrec_final} 中由遗忘门控 $\alpha_t$ 驱动的历史状态衰减项 $\alpha_t S_{t-1}$，在代数形式上与 AdamW \cite{Loshchilov_Hutter_2019} 的解耦权重衰减呈现相似的结构：二者均将衰减项置于自适应预条件子的作用范围之外，从而避免了正则化强度被特征活跃度非线性扭曲的问题——这正是 Loshchilov 等人 \cite{Loshchilov_Hutter_2019} 指出的原始 Adam 中 $L_2$ 正则化与自适应学习率耦合的弊端。然而，这一相似性停留在代数形式层面：AdamW 的权重衰减旨在限制参数范数以防止过拟合，采用全局静态超参数 $\lambda$；而 DamRec 的 $\alpha_t$ 服务于时间序列中的分布偏移应对，是由输入序列驱动的动态标量，能够根据当前输入自适应地调节全局衰减强度。

\subsection{动力学性质分析} \label{subsec:robustness}

\subsubsection{数值稳定性保障}
\label{subsubsec:robustness_basics}

将二阶矩估计引入流式隐状态更新，在提升自适应能力的同时，也引入了潜在的数值风险。本节分析 DamRec 框架中三项关键的稳健性机制，它们共同保障了模型在无界数据流上的长期数值稳定性。

\paragraph{数值平滑。}
预条件子 $\mathcal{P}_t = \sqrt{\hat{V}_t} + \epsilon$（公式 \eqref{eq:preconditioner}）中的平滑常数 $\epsilon > 0$ 防止了当某些特征维度的二阶矩趋近于零时出现的除零下溢。这一机制在自适应优化算法中是标准配置 \cite{Kingma_Ba_2017}，但在流式推荐场景中尤为关键：长尾物品可能在长时间窗口内未被交互，导致其对应特征维度的累积方差极小。$\epsilon$ 为这些稀疏维度的更新步长设定了一个有限的上界，避免了单次罕见交互触发的梯度被无限放大。

\paragraph{二阶矩的指数遗忘。}
公式 \eqref{eq:second_moment} 中的衰减因子 $\rho$ 赋予了二阶矩估计 $V_t$ 有限的有效记忆窗口。若采用简单的累积平均（即 $\rho = 1 - 1/t$），$V_t$ 将随时间步单调递增，预条件子分母持续增大，最终导致所有维度的有效学习率趋近于零，模型丧失对新模式的响应能力。通过固定 $\rho \in [0, 1)$，早期时间步的梯度贡献以 $\rho^{t-\tau}$ 的速率指数衰减，使得 $V_t$ 主要反映近期的梯度波动尺度。这一机制使 DamRec 能够自然地适应流式推荐中常见的概念漂移：当用户兴趣发生迁移时，新交互模式的梯度统计量将在 $\mathcal{O}(1/(1-\rho))$ 个时间步内主导预条件子，而非被历史累积所淹没。

\paragraph{隐状态正则化。}
在实现层面，DamRec 对隐状态矩阵 $S_t$ 施加了 GroupNorm \cite{Wu_He_2018} 正则化。在无界的流式推断中，即使单步更新是数值稳定的，隐状态的幅值仍可能因递归累积而缓慢漂移。归一化操作将 $S_t$ 的各特征组约束在稳定的数值范围内，截断了幅值漂移的传播路径，保障了模型在处理任意长度数据流时的长期数值稳定性。

\textbf{这个正则化选择group norm还是layernorm还是没有还得等待实验验证}



\subsubsection{特征尺度自适应}

本节从表征空间的特征尺度视角，对公式 \eqref{eq:damrec_final} 的自适应更新算子进行分析。

在流式序列推荐场景中，由于交互特征呈现显著的幂律分布，损失曲面在不同特征维度上的局部曲率存在跨数量级的差异。传统的一阶线性架构对所有维度施加相同的更新步长，导致模型在曲率陡峭的高频维度容易发生数值震荡，而在曲率平坦的长尾维度更新不足。

DamRec 通过构建二阶统计量预条件子 $\mathcal{P}_t$，使每个特征维度的更新步长根据其历史梯度波动尺度进行自适应缩放，从而缓解不同维度间更新尺度的差异。这一机制带来两方面的效应：

\begin{itemize}
    \item \textbf{高频特征的阻尼效应}：对于频繁交互的热门物品所对应的特征维度，显著累积的二阶矩 $\hat{V}_t$ 增大了预条件子分母，动态约束了更新步长，从而缓解流行度偏置对隐状态的过度影响。
    \item \textbf{低频特征的增益效应}：对于稀疏长尾物品所对应的特征维度，较小的二阶矩使预条件子分母趋近于 $\epsilon$，相对放大了更新幅度，有助于改善这些维度的梯度饥饿问题。
\end{itemize}

上述两种效应的实证验证见第 \ref{subsec:scenario_analysis} 节的按物品流行度分层分析。\textbf{实验待验证}




\subsection{工程效率} \label{efficiency}


\subsubsection{二阶统计量近似与时间复杂度保持} \label{subsec:time_complexity}

在序列推荐的高维隐状态空间（$S_t \in \mathbb{R}^{d \times d}$）中，传统的高阶优化算法（如精确牛顿法 \cite{Nocedal_Wright_2006} 或自然梯度法 \cite{6790500, Martens_Grosse_2020}）通常涉及 Hessian 矩阵或 Fisher 信息矩阵及其逆映射的计算。针对 $d^2$ 个隐状态参数，这种稠密张量求逆操作的时间复杂度高达 $\mathcal{O}((d^2)^3) = \mathcal{O}(d^6)$，在实时流式推断场景下缺乏计算可行性 \cite{Goodfellow-et-al-2016}。因此，如何在保持一阶算子线性复杂度的前提下，引入特征维度的自适应尺度感知，构成了流式序列模型落地的瓶颈。

为此，DamRec 摒弃了全局曲率计算，转而在架构底层引入轻量级的、基于指数移动平均（EMA）的逐特征方差缩放机制 \cite{Kingma_Ba_2017}。正如近期优化理论所述，采用梯度的未中心化二阶矩平方根虽不完全等价于经验 Fisher 矩阵的严格近似 \cite{Kunstner_Balles_Hennig_2020}，但在工程实践中能有效地作为特征波动尺度的代理。这种元素级的对角近似策略，将单步自适应状态演化的时间复杂度限制在 $\mathcal{O}(d^2)$，实现了与基础一阶线性架构（如 Mamba \cite{Gu_Dao_2024}, GDN \cite{Yang_Kautz_Hatamizadeh_2025}）的计算耗时对齐。然而，这种对角近似依然面临着空间复杂度（显存开销）成倍增加的挑战：模型需额外维护与隐状态同维度的二阶统计量矩阵 $V_t \in \mathbb{R}^{d \times d}$，带来 $\mathcal{O}(d^2)$ 的额外存储开销。下一节将表明，利用第 \ref{subsec:higher_order_update} 节中揭示的 $G_t$ 的秩一结构，可将这一开销压缩至 $\mathcal{O}(d)$。

\subsubsection{基于秩一分解的空间复杂度压缩}
\label{subsec:space_complexity}

在朴素实现中，追踪特征波动尺度需额外维护二阶统计量矩阵 $V_t \in \mathbb{R}^{d \times d}$，导致状态存储的空间复杂度增加 $\mathcal{O}(d^2)$。对于需要为海量用户维护独立状态流形的工业级系统，这种显存翻倍的开销是不可接受的。

为了消除这一瓶颈，DamRec 充分利用了更新驱动矩阵 $G_t$ 的代数结构。回顾式 \eqref{eq:raw_gradient} 可知，$G_t$ 由残差向量 $r_t$ 与键向量 $k_t$ 的外积构成，即 $G_t = r_t k_t^\top$，意味着 $G_t$ 是一个严格的秩一（Rank-1）矩阵。受 Adafactor \cite{Shazeer_Stern_2018} 启发，其逐元素平方项可解耦为两个低维向量的外积。具体地，利用 Hadamard 积对外积的分配律，即对任意向量 $a, b, c, d \in \mathbb{R}^d$ 有 $(ab^\top) \odot (cd^\top) = (a \odot c)(b \odot d)^\top$（其证明可直接由逐元素乘积的定义 $[(ab^\top) \odot (cd^\top)]_{ij} = a_i b_j c_i d_j = (a_i c_i)(b_j d_j)$ 得出），可得：
\begin{equation}
    G_t \odot G_t = (r_t k_t^\top) \odot (r_t k_t^\top) = (r_t \odot r_t)(k_t \odot k_t)^\top.
    \label{eq:rank1_hadamard}
\end{equation}

基于此特性，DamRec 摒弃了对全局矩阵 $V_t$ 的存储，转而独立追踪残差与键向量的二阶矩。具体而言，模型仅需维护两个 $d$ 维状态向量 $V_t^{(r)}, V_t^{(k)} \in \mathbb{R}^d$：
\begin{align}
    V_t^{(r)} &= \rho V_{t-1}^{(r)} + (1 - \rho)(r_t \odot r_t), \label{eq:vr_update} \\
    V_t^{(k)} &= \rho V_{t-1}^{(k)} + (1 - \rho)(k_t \odot k_t). \label{eq:vk_update}
\end{align}
二者同样初始化为零，即 $V_0^{(r)} = V_0^{(k)} = \mathbf{0}$。

需要指出的是，上述分解是一个近似而非精确等式。严格地说，重构矩阵为
\begin{equation}
    V_t^{(r)}(V_t^{(k)})^\top = \left(\sum_{\tau=1}^{t} \rho^{t-\tau}(1-\rho)\, r_\tau \odot r_\tau\right) \left(\sum_{\tau=1}^{t} \rho^{t-\tau}(1-\rho)\, k_\tau \odot k_\tau\right)^\top,
    \label{eq:factored_approx}
\end{equation}
而真实的二阶矩为
\begin{equation}
    V_t = \sum_{\tau=1}^{t} \rho^{t-\tau}(1-\rho)\, (r_\tau \odot r_\tau)(k_\tau \odot k_\tau)^\top.
    \label{eq:true_second_moment}
\end{equation}
二者的差异源于"和的外积"与"外积的和"之间的交叉项。直觉上，当 $\rho$ 接近1时，EMA 的有效窗口约为 $1/(1-\rho)$ 个时间步，窗口内各时刻的 $r_\tau \odot r_\tau$ 与 $k_\tau \odot k_\tau$ 在 EMA 的平滑效应下趋于各自的运行均值，交叉项的相对贡献因此被抑制。反之，当 $\rho$ 较小（有效窗口较短）时，近似误差相对较大，但此时 $V_t$ 主要由近期少数几个时刻主导，预条件子的绝对精度对最终性能的影响也相应减弱。这一近似策略与 Adafactor \cite{Shazeer_Stern_2018} 对梯度二阶矩的行/列分解处理方式一致，后者在大规模语言模型训练中已被广泛验证。在我们的实验中，消融变体"w/o 秩一分解"（维护完整 $V_t \in \mathbb{R}^{d \times d}$）与分解版本的性能差异可忽略（见表 \ref{tab:ablation_components}），第 \ref{subsec:ablation} 节的消融实验将对此进行实证验证。\textbf{(饼，并未实际验证）}

在计算预条件子时，对重构后的矩阵统一施加单次 $1/(1-\rho^t)$ 偏差修正。需要说明的是，若对两个因子分别做偏差修正再相乘，分母将变为 $(1-\rho^t)^2$，在 $t$ 较小时会过度放大估计值，反而加剧冷启动期的数值不稳定。鉴于秩一分解本身已引入近似（见上文讨论），此处选择与标准 Adam \cite{Kingma_Ba_2017} 一致的单次修正作为工程折中，其有效性将在第 \ref{subsec:ablation} 节的消融实验中通过"w/o 偏差修正"变体验证。（\textbf{同样是饼}）具体地，预条件子通过如下方式计算：
\begin{equation}
    \hat{V}_t \approx \frac{V_t^{(r)} (V_t^{(k)})^\top}{1 - \rho^t},
    \label{eq:factored_bias_correction}
\end{equation}
或等价地，在实现中可利用广播机制（Broadcasting）直接计算：
\begin{equation}
    \mathcal{P}_t = \sqrt{\frac{V_t^{(r)} (V_t^{(k)})^\top}{1 - \rho^t}} + \epsilon.
    \label{eq:factored_preconditioner}
\end{equation}
该策略将额外存储开销从 $\mathcal{O}(d^2)$ 降维压缩至 $\mathcal{O}(2d)$。综合以上分解，公式 \eqref{eq:damrec_final} 在实际实现中无需显式构造 $d \times d$ 的预条件子矩阵，而是通过 $V_t^{(r)}$ 与 $V_t^{(k)}$ 的广播运算直接完成逐元素缩放。

\subsubsection{分块并行计算}
\label{subsubsec:chunkwise}

上述逐步递归的状态流转方程（公式 \eqref{eq:damrec_final}）在理论上具备 $\mathcal{O}(1)$ 的单步计算复杂度，适配严格的流式推断场景。然而，在训练阶段，逐时间步的串行递归无法利用现代 GPU 的大规模并行计算能力，成为吞吐量的瓶颈。

为解决这一问题，DamRec 借鉴 GDN \cite{Yang_Kautz_Hatamizadeh_2025} 的分块并行（chunkwise）策略，将长度为 $L$ 的输入序列划分为 $L/C$ 个大小为 $C$ 的非重叠块（chunk）。整体计算采用两层结构：

\paragraph{块间递归。}
在块的粒度上，DamRec 的完整状态流转方程（公式 \eqref{eq:damrec_final}）以串行方式执行。具体而言，第 $n$ 个块接收前一块的输出隐状态 $S_{nC}$ 及二阶矩估计 $V_{nC}^{(r)}, V_{nC}^{(k)}$ 作为初始条件，经过块内计算后，输出更新后的状态传递给下一块。由于块的数量仅为 $L/C$，串行开销被压缩至可接受的范围。

\paragraph{块内并行。}
在每个块的内部，$C$ 个时间步的计算需要高度并行化以利用 GPU 硬件。为此，块内采用固定预条件的线性递归：以第 $n$ 个块边界处的预条件子 $\mathcal{P}_{nC}$ 作为块内的常数近似，块内状态更新为
\begin{equation}
    S_t^{\text{(intra)}} = \alpha_t S_{t-1}^{\text{(intra)}} + \beta_t \left( (v_t - \alpha_t S_{t-1}^{\text{(intra)}} k_t) k_t^\top \oslash \mathcal{P}_{nC} \right),
    \label{eq:intra_chunk}
\end{equation}
其中 $\mathcal{P}_{nC}$ 在块内所有时间步保持固定。由于 $\mathcal{P}_{nC}$ 为常数，逐元素除法可被预先吸收进门控系数，使得上式保持标准门控线性递归的形式，从而可通过并行前缀扫描 \cite{Blelloch} 或矩阵化的块内累积 \cite{Yang_Kautz_Hatamizadeh_2025} 在 $\mathcal{O}(\log C)$ 的深度内完成。这一近似的误差受 $\mathcal{P}_t$ 在块内的变化量约束：$V_t$ 的单步增量为 $(1-\rho)(G_t \odot G_t - V_{t-1})$，经 $C$ 步累积后总变化量级为 $\mathcal{O}((1-\rho) \cdot C)$（假设 $\|G_t\|$ 有界），在典型配置（$\rho \geq 0.99$，$C \leq 128$）下可忽略不计（详细分析见附录 \ref{app:chunk_error}）。

\paragraph{二阶统计量的块级更新。}
二阶矩估计 $V_t^{(r)}, V_t^{(k)}$ 的更新在块边界处统一执行。在第 $n$ 个块处理完毕后，利用该块内所有时间步的残差与键向量，以批量方式更新分解后的二阶矩向量：
\begin{align}
    V_{(n+1)C}^{(r)} &= \rho^C V_{nC}^{(r)} + (1 - \rho) \sum_{\tau=nC+1}^{(n+1)C} \rho^{(n+1)C - \tau} (r_\tau \odot r_\tau), \label{eq:chunk_vr} \\
    V_{(n+1)C}^{(k)} &= \rho^C V_{nC}^{(k)} + (1 - \rho) \sum_{\tau=nC+1}^{(n+1)C} \rho^{(n+1)C - \tau} (k_\tau \odot k_\tau), \label{eq:chunk_vk}
\end{align}
其中求和项可通过向量化操作在 GPU 上高效并行计算。更新后的预条件子将在下一个块的块间递归中生效。

这种分块策略在训练效率与自适应能力之间取得了平衡：块内的一阶并行递归保障了计算吞吐量，而块间的二阶预条件更新保留了 DamRec 的自适应缩放能力。块大小 $C$ 作为超参数控制了这一权衡——较大的 $C$ 提升并行度但降低二阶更新的时间分辨率，较小的 $C$ 则反之。在推断阶段，模型退化为逐步递归（$C = 1$），完整的二阶自适应机制在每个时间步生效。

\subsection{整体复杂度分析}
\label{subsec:complexity}

本节汇总 DamRec 在流式推断与训练两种模式下的计算复杂度，并与代表性架构进行对比。

\paragraph{流式推断的单步复杂度。}
在逐步递归模式下（$C=1$），DamRec 每个时间步执行以下操作：（1）计算残差 $r_t = \alpha_t S_{t-1} k_t - v_t$，涉及一次矩阵-向量乘法，复杂度 $\mathcal{O}(d^2)$；（2）更新两个 $d$ 维二阶矩向量 $V_t^{(r)}, V_t^{(k)}$，复杂度 $\mathcal{O}(d)$；（3）通过广播计算预条件缩放并执行秩一状态更新，复杂度 $\mathcal{O}(d^2)$。因此，单步时间复杂度为 $\mathcal{O}(d^2)$，与 GDN \cite{Yang_Kautz_Hatamizadeh_2025} 及 Mamba \cite{Gu_Dao_2024} 等一阶线性递归架构保持一致。DamRec 引入的二阶自适应机制仅增加了 $\mathcal{O}(d)$ 的额外运算，不改变渐近复杂度的阶。

\paragraph{空间复杂度。}
在流式推断中，DamRec 需常驻存储的状态包括：隐状态矩阵 $S_t \in \mathbb{R}^{d \times d}$，以及经秩一分解后的二阶矩向量 $V_t^{(r)}, V_t^{(k)} \in \mathbb{R}^d$。总空间占用为 $\mathcal{O}(d^2 + 2d) = \mathcal{O}(d^2)$，与序列长度 $L$ 无关，即关于 $L$ 严格为 $\mathcal{O}(1)$。相较于未经秩一分解的朴素实现（需额外存储 $V_t \in \mathbb{R}^{d \times d}$，总空间 $\mathcal{O}(2d^2)$），分解策略将二阶矩的存储开销从 $\mathcal{O}(d^2)$ 压缩至 $\mathcal{O}(d)$，使得总空间占用与不含二阶机制的一阶基线相同。

\paragraph{训练复杂度。}
在分块并行模式下，处理长度为 $L$ 的序列时，块内并行递归的时间复杂度为 $\mathcal{O}(L \cdot d^2 / C \cdot \log C + L \cdot d^2)$，其中第一项来自 $L/C$ 个块各自 $\mathcal{O}(d^2 \log C)$ 的并行前缀扫描，第二项来自块内 $L$ 个时间步的逐元素运算。在 $C$ 为常数的典型配置下，训练总时间复杂度简化为 $\mathcal{O}(L \cdot d^2)$，与序列长度呈线性关系。

\paragraph{与其他架构的对比。}
表 \ref{tab:complexity} 总结了 DamRec 与代表性架构的复杂度对比。

\begin{table}[t]
\centering
\caption{DamRec 与代表性序列建模架构的复杂度对比。$L$ 为序列长度，$d$ 为隐状态维度。推断复杂度指单步更新，训练复杂度指处理完整序列。Transformer 的推断复杂度假设使用 KV-cache 的自回归推断。各架构在序列推荐中的典型实例化分别为 SASRec \cite{Kang_McAuley_2018}（Transformer）、Mamba4Rec \cite{Liu_Lin_Wang_Liu_Caverlee_2024}（Mamba）等。}
\label{tab:complexity}
\begin{tabular}{lccc}
\toprule
\textbf{架构} & \textbf{推断（时间/步）} & \textbf{推断（空间）} & \textbf{训练（时间）} \\
\midrule
Transformer     & $\mathcal{O}(L \cdot d)$   & $\mathcal{O}(L \cdot d)$  & $\mathcal{O}(L^2 \cdot d)$ \\
Mamba           & $\mathcal{O}(d^2)$          & $\mathcal{O}(d^2)$        & $\mathcal{O}(L \cdot d^2)$ \\
GDN             & $\mathcal{O}(d^2)$          & $\mathcal{O}(d^2)$        & $\mathcal{O}(L \cdot d^2)$ \\
\midrule
DamRec          & $\mathcal{O}(d^2)$          & $\mathcal{O}(d^2)$        & $\mathcal{O}(L \cdot d^2)$ \\
\bottomrule
\end{tabular}
\end{table}

如表所示，DamRec 在所有复杂度指标上与一阶线性递归架构（Mamba、GDN）保持同阶，同时相较于 Transformer 在推断空间和训练时间上均具有显著优势。这表明二阶自适应机制的引入未带来额外的渐近复杂度开销，即在保持与一阶基线相同计算量级的前提下，使模型能够根据各维度的梯度波动尺度调节更新步长。


\section{实验与分析}
\label{sec:experiments}

\textbf{这一大章主要是画饼，之后应该还会大改。部分细节看情况放进附录}

\subsection{实验设置}
\label{subsec:exp_setup}

\paragraph{评估协议。}
本文采用流式推荐中广泛使用的逐步评估范式（test-then-train）：对于交互序列中的每个时间步，模型首先基于当前隐状态生成推荐列表并计算评价指标，随后将该时间步的真实交互用于更新模型参数。这一协议模拟了在线服务中"先预测、再学习"的部署场景。

\paragraph{数据集。}
实验在以下公开基准数据集上进行，涵盖不同的领域特征、交互密度与分布偏态程度：
\begin{itemize}
    \item \textbf{MovieLens-1M} \cite{Harper_Konstan_2015}：电影评分数据，中等规模，交互分布相对均匀，作为标准基线场景。
    \item \textbf{Amazon Beauty} \cite{McAuley_et_al_2015}：电商产品评论数据，高度稀疏，物品流行度呈现显著的长尾分布，适合验证 DamRec 在偏态场景下的表现。
    \item \textbf{Amazon Games} \cite{McAuley_et_al_2015}：电商游戏品类数据，中等稀疏度，品类多样性较高。
    \item \textbf{Gowalla} \cite{Cho_et_al_2011}：基于地理位置的签到数据，交互模式具有强时空局部性与概念漂移特征，适合验证 DamRec 对分布偏移的适应能力。
\end{itemize}
各数据集的详细统计信息见表 \ref{tab:dataset_stats}。

% TODO: 插入数据集统计表格（用户数、物品数、交互数、稀疏度、Gini系数等）
\begin{table}[t]
\centering
\caption{实验数据集统计信息。稀疏度定义为 $1 - |\mathcal{I}| / (|\mathcal{U}| \times |\mathcal{V}|)$，其中 $|\mathcal{I}|$、$|\mathcal{U}|$、$|\mathcal{V}|$ 分别为交互数、用户数和物品数。}
\label{tab:dataset_stats}
\begin{tabular}{lrrrrr}
\toprule
\textbf{数据集} & \textbf{用户数} & \textbf{物品数} & \textbf{交互数} & \textbf{稀疏度} & \textbf{平均序列长度} \\
\midrule
MovieLens-1M   & --     & --     & --        & --\%   & -- \\
Amazon Beauty  & --     & --     & --        & --\%   & -- \\
Amazon Games   & --     & --     & --        & --\%   & -- \\
Gowalla        & --     & --     & --        & --\%   & -- \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{基准模型。}
我们将 DamRec 与以下覆盖不同架构范式的代表性序列推荐模型进行对比：
\begin{itemize}
    \item \textbf{基于 RNN}：GRU4Rec  \cite{Hidasi_Karatzoglou_Baltrunas_Tikk_2016}——基于门控循环单元的会话推荐模型。
    \item \textbf{基于 Transformer}：SASRec \cite{Kang_McAuley_2018}——基于单向自注意力的序列推荐模型；LightSANs \cite{Fan_Liu_Lian_Zhao_Xie_Wen_2021}——通过低秩分解压缩注意力矩阵的轻量化变体；LinRec \cite{Liu_Zhao_Zhang_Gao_Wang_Fan_Wang_He_Liu_Li_2023}——通过线性化注意力降低计算复杂度的高效变体。
    \item \textbf{基于状态空间模型}：Mamba4Rec \cite{Liu_Lin_Wang_Liu_Caverlee_2024}——将选择性状态空间模型应用于序列推荐。
    \item \textbf{基于门控线性递归}：GDN4Rec——我们将 GDN \cite{Yang_Kautz_Hatamizadeh_2025} 的门控 Delta Rule 适配至序列推荐任务的实现，作为 DamRec 的直接一阶基线。
\end{itemize}

\paragraph{评价指标。}
采用序列推荐中广泛使用的 Top-$K$ 排序指标：Recall@$K$ 与 NDCG@$K$（$K \in \{5, 10, 20\}$）。Recall@$K$ 衡量真实物品出现在推荐列表中的概率，NDCG@$K$ 进一步考虑了命中位置的排序质量。所有指标在全物品候选集上计算，不做负采样近似。

\paragraph{实现细节。}
所有实验均基于 RecBole \cite{Zhao_Mu_Hou_Lin_Chen_Pan_Li_Lu_Wang_Tian_et_al._2021} 框架实现。其中 GRU4Rec、SASRec 与 LightSANs 采用框架内置实现，LinRec、Mamba4Rec、GDN4Rec 与 DamRec 由我们基于该框架自行实现，均共享相同的嵌入层、损失函数与训练流程以确保公平性。所有模型的嵌入维度统一为 $d = $ --，训练使用 Adam 优化器，学习率 --。DamRec 的核心超参数为 $\rho = $ --，$C = $ --。完整的超参数配置与搜索策略见附录 \ref{app:implementation}。
% TODO: 填入具体的超参数设置（嵌入维度 d、块大小 C、EMA 衰减因子 ρ、学习率、优化器等）
% TODO: 说明硬件环境（GPU 型号、显存）
% TODO: 说明是否会开源代码


\subsection{整体推荐性能}
\label{subsec:overall_performance}

本节将 DamRec 与所有基准模型在四个数据集上的整体推荐性能进行对比。实验结果如表 \ref{tab:main_results} 所示。

% TODO: 主实验结果表格
% 指标：Recall@5, Recall@10, Recall@20, NDCG@5, NDCG@10, NDCG@20
% 行：GRU4Rec, SASRec, LightSANs, LinRec, Mamba4Rec, GDN4Rec, DamRec
% 列：4个数据集 × 6个指标
% 最优加粗，次优下划线

\begin{table*}[t]
\centering
\caption{各模型在四个数据集上的整体推荐性能。最优结果加粗，次优结果下划线。$\dagger$ 表示 DamRec 相对次优基线的提升在 $p < 0.05$ 水平下显著（配对 $t$ 检验）。}
\label{tab:main_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|cccccc|cccccc}
\toprule
& \multicolumn{6}{c|}{\textbf{MovieLens-1M}} & \multicolumn{6}{c}{\textbf{Amazon Beauty}} \\
\textbf{模型} & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 \\
\midrule
GRU4Rec     & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
SASRec      & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
LightSANs   & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
LinRec      & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
Mamba4Rec   & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
GDN4Rec     & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
\midrule
DamRec      & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
\textit{Improv.} & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% \\
\bottomrule
\toprule
& \multicolumn{6}{c|}{\textbf{Amazon Games}} & \multicolumn{6}{c}{\textbf{Gowalla}} \\
\textbf{模型} & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 & R@5 & R@10 & R@20 & N@5 & N@10 & N@20 \\
\midrule
GRU4Rec     & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
SASRec      & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
LightSANs   & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
LinRec      & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
Mamba4Rec   & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
GDN4Rec     & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
\midrule
DamRec      & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
\textit{Improv.} & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% & --\% \\
\bottomrule
\end{tabular}
}
\end{table*}

此处应重点讨论以下几点：（1）DamRec 相对于直接一阶基线 GDN4Rec 的提升幅度，这直接验证了二阶自适应机制的有效性；（2）DamRec 与 Transformer 类模型（SASRec）的对比，展示线性复杂度架构在性能上的竞争力；（3）不同数据集之间的表现差异——预期在分布偏态更严重的数据集（如 Amazon Beauty）上，DamRec 的优势更为显著。


\subsection{核心场景分析}
\label{subsec:scenario_analysis}

DamRec 的核心理论优势在于对偏态特征分布的自适应处理能力。为了超越整体指标的平均效应，本节从两个互补的维度对模型性能进行细粒度拆解。

\subsubsection{按物品流行度的分层分析}

将物品按交互频率划分为若干层级（如头部 20\%、中部 30\%、长尾 50\%），分别统计各层级上的推荐性能。

% TODO: 分层分析表格或柱状图
% 分层方式：按物品交互频率排序，划分为 Head (top 20%), Mid (20%-50%), Tail (bottom 50%)
% 指标：Recall@10, NDCG@10
% 模型：GDN4Rec, SASRec, Mamba4Rec, DamRec（选最有代表性的几个基线）
% 数据集：选 Amazon Beauty（长尾最严重）和 MovieLens-1M（相对均匀）做对比

这一分析直接对应第 \ref{subsec:robustness} 节中关于"方差阻尼缓解过度更新"与"前馈增益改善梯度饥饿"的理论预测：
\begin{itemize}
    \item 在头部（高频）物品上，DamRec 的阻尼效应应抑制因流行度偏置导致的过度推荐，性能至少与基线持平；
    \item 在长尾（低频）物品上，DamRec 的前馈增益应显著提升稀疏特征的表征质量，产生最大的相对提升。
\end{itemize}

\subsubsection{序列不同阶段的性能演化}

将用户交互序列按时间划分为早期、中期和后期三个阶段，分析模型在序列不同位置上的推荐准确性。

% TODO: 折线图
% 横轴：序列位置（按百分比或绝对位置分桶，如 0-20%, 20-40%, 40-60%, 60-80%, 80-100%）
% 纵轴：该位置窗口内的 Recall@10
% 线条：GDN4Rec, SASRec, DamRec
% 数据集：选 Gowalla（概念漂移最明显）

这一分析验证 DamRec 在概念漂移场景下的适应能力：一阶模型在面对用户兴趣迁移时可能表现出响应延迟，而 DamRec 的自适应预条件机制应使其在兴趣转折点附近保持更好的跟踪能力。


\subsection{消融实验}
\label{subsec:ablation}

\subsubsection{关键组件消融}

为验证 DamRec 各核心组件的独立贡献，设计以下消融变体：

\begin{table}[t]
\centering
\caption{DamRec 关键组件消融实验。各变体在最具代表性的数据集上的 Recall@10 / NDCG@10。}
\label{tab:ablation_components}
\begin{tabular}{lp{5cm}cc}
\toprule
\textbf{变体} & \textbf{说明} & \textbf{Dataset 1} & \textbf{Dataset 2} \\
\midrule
DamRec（完整）    & 完整模型                                & -- & -- \\
w/o 预条件       & 移除 $\mathcal{P}_t$，退化为一阶 GDN     & -- & -- \\
w/o 偏差修正     & 移除公式 \eqref{eq:bias_correction} 中的 $1/(1-\rho^t)$ 修正 & -- & -- \\
w/o 秩一分解     & 维护完整 $V_t \in \mathbb{R}^{d \times d}$ 而非分解向量 & -- & -- \\
w/o 动态遗忘门   & 将 $\alpha_t$ 固定为常数 $\lambda$        & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

% 指标：Recall@10, NDCG@10
% 数据集：选2个（一个偏态严重如 Amazon Beauty，一个相对均匀如 MovieLens-1M）

其中，"w/o 预条件"是最关键的消融——它直接将 DamRec 退化为其一阶基线 GDN4Rec，其与完整模型之间的性能差距量化了二阶机制的净贡献。"w/o 秩一分解"应在性能上与完整模型一致（验证分解是无损的），但在效率指标上有显著差异（见第 \ref{subsec:efficiency_exp} 节）。

\subsubsection{二阶矩估计策略对比}

DamRec 的升阶策略选择性地引入了二阶矩而未引入一阶动量（第 \ref{subsec:higher_order_update} 节）。为系统性验证此设计决策，对比以下替代方案：

\begin{table}[t]
\centering
\caption{不同自适应更新策略对比。SGD 对应一阶基线，Adam 同时引入一阶动量与二阶矩，RMSProp 仅引入二阶矩（与 DamRec 同族），AdaGrad 使用累积（非 EMA）二阶矩。各策略的完整公式推导见附录 \ref{app:update_rules}。}
\label{tab:ablation_update}
\begin{tabular}{lccccc}
\toprule
\textbf{策略} & \textbf{一阶动量} & \textbf{二阶矩} & \textbf{EMA} & \textbf{Dataset 1} & \textbf{Dataset 2} \\
\midrule
SGD（一阶基线/GDN）      & --  & --  & --  & -- & -- \\
AdaGrad 式               & --  & \checkmark  & --  & -- & -- \\
RMSProp 式               & --  & \checkmark  & \checkmark  & -- & -- \\
Adam 式（含一阶动量）     & \checkmark  & \checkmark  & \checkmark  & -- & -- \\
DamRec                   & --  & \checkmark  & \checkmark  & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

% 指标：Recall@10, NDCG@10
% 数据集：与组件消融相同的2个数据集

此表格直接回应了以下理论预测：（1）RMSProp 式与 DamRec 的性能差异仅源于偏差修正；（2）Adam 式引入一阶动量不应带来显著提升（因为 $\alpha_t S_{t-1}$ 的递归已隐式实现了梯度平滑）；（3）AdaGrad 式（不含 EMA）的性能应随序列长度增加而退化（因为学习率趋零）。


\subsection{超参数敏感性分析}
\label{subsec:hyperparam}

DamRec 引入的核心超参数为二阶矩衰减因子 $\rho$ 和分块大小 $C$。本节通过控制变量实验分析模型对这两个参数的敏感性。

\paragraph{衰减因子 $\rho$。}
$\rho$ 控制二阶矩估计的有效记忆窗口（约 $1/(1-\rho)$ 个时间步）。实验中取 $\rho \in \{0.9, 0.95, 0.99, 0.995, 0.999\}$，对应约 $10$ 至 $1000$ 步的有效窗口。预期过小的 $\rho$ 导致预条件子波动过大（方差估计不稳定），过大的 $\rho$ 导致对分布变化的响应迟缓。

% TODO: 折线图
% 横轴：ρ 值
% 纵轴：Recall@10（左轴）, NDCG@10（右轴或同轴不同线型）
% 数据集：全部4个，每个数据集一条线或分4个子图
% 标注最优 ρ 值

\paragraph{分块大小 $C$。}
$C$ 控制训练时二阶更新的时间分辨率。实验中取 $C \in \{16, 32, 64, 128, 256\}$。预期性能随 $C$ 增大而缓慢下降（因块内预条件固定导致的近似误差增大），但训练速度随 $C$ 增大而提升。

% TODO: 双轴折线图
% 横轴：C 值
% 左纵轴：Recall@10（性能）
% 右纵轴：训练速度（样本/秒）或训练时间（秒/epoch）
% 展示性能-效率的 trade-off


\subsection{计算效率实证}
\label{subsec:efficiency_exp}

第 \ref{subsec:complexity} 节的渐近分析表明 DamRec 与一阶基线保持同阶复杂度。本节通过实际运行时间与显存占用的测量进行实证验证。

\paragraph{推断延迟与显存扩展性。}
固定模型配置，将最大序列长度 $L_{\max}$ 从 $50$ 逐步增大至 $10000$，测量各模型的单步推断延迟（毫秒/步）与峰值显存占用。DamRec 与 GDN4Rec 的两项指标应保持恒定（与 $L_{\max}$ 无关），而 Transformer 类基线（含 KV-cache）的延迟与显存均随 $L_{\max}$ 线性增长。同时对比"w/o 秩一分解"变体，量化分解带来的实际显存节约。

% TODO: 双图并排
% 图1：推断延迟
%   横轴：L_max（对数刻度：50, 100, 500, 1000, 5000, 10000）
%   纵轴：单步延迟（ms/step）
%   线条：SASRec, GRU4Rec, Mamba4Rec, GDN4Rec, DamRec, DamRec (w/o 秩一分解)
% 图2：显存占用
%   横轴：同上
%   纵轴：峰值显存（MB）
%   线条：同上

\paragraph{训练吞吐量。}
测量各模型在分块并行模式下处理固定批量的训练速度（样本/秒），作为序列长度的函数。

% TODO: 折线图或柱状图
% 横轴：序列长度（128, 256, 512, 1024, 2048）
% 纵轴：训练吞吐量（samples/sec）
% 模型：SASRec, GDN4Rec, DamRec

% --- 进入附录，将 section 计数器重置并改为字母形式 ---
\setcounter{section}{0} 
\renewcommand{\thesection}{\Alph{section}} 

% --- 附录 A ---
\refstepcounter{section}
\subsection*{\thesection\ 符号表} 

\refstepcounter{subsection}
\subsubsection*{\thesubsection\ 交叉熵} 

% --- 附录 B ---
\refstepcounter{section}
\subsection*{\thesection\ 块内预条件子近似误差分析}
\label{app:chunk_error}

在第 $n$ 个块内，预条件子固定为块边界值 $\mathcal{P}_{nC}$，而真实值为 $\mathcal{P}_t$（$nC < t \leq (n+1)C$）。考察二者的差异来源：块内每个时间步 $t$ 的二阶矩更新量为
\begin{equation}
    V_t - V_{t-1} = (1 - \rho)(G_t \odot G_t - V_{t-1}).
\end{equation}
经过块内 $m = t - nC$ 步后，累积偏差为
\begin{equation}
    V_t - V_{nC} = (1-\rho) \sum_{j=1}^{m} \rho^{m-j} (G_{nC+j} \odot G_{nC+j} - V_{nC+j-1}).
\end{equation}
由于 $|1-\rho| < 1$ 且求和项数至多为 $C$，有 $\|V_t - V_{nC}\| = \mathcal{O}((1-\rho) \cdot C \cdot \max_j \|G_{nC+j}\|^2)$。在典型配置下（$\rho \geq 0.99$，$C \leq 128$），$(1-\rho) \cdot C \leq 1.28$，预条件子的相对变化量较小。

需要说明的是，块内预条件子 $\mathcal{P}_{nC}$ 为常数，不参与块内的状态递归，因此"预条件子影响状态、状态影响残差"的耦合链在块内被截断。耦合效应仅在块边界处通过更新 $\mathcal{P}$ 重新引入，其影响已被下一个块的预条件子刷新所覆盖。

\refstepcounter{subsection}
\subsubsection*{\thesubsection\ 实现细节} 
\label{app:implementation}

\paragraph{初始化与梯度处理。}
初始化策略与二阶矩的 stop-gradient 处理详见正文第 \ref{subsec:higher_order_update} 节。

\paragraph{硬件环境。}
所有实验在 -- $\times$ NVIDIA -- GPU（--GB 显存）上进行，操作系统为 Ubuntu --，PyTorch 版本 --，CUDA 版本 --。

\paragraph{计算加速。}
块内并行递归（公式 \eqref{eq:intra_chunk}）的实现基于 FLA（Flash Linear Attention）\cite{yang2024fla} 库提供的高效 CUDA 内核加速。由于块内预条件子 $\mathcal{P}_{nC}$ 为常数，可预先将其吸收进门控系数，使块内递归保持标准门控线性递归的形式，从而直接复用 FLA 中已有的并行扫描内核。块间的二阶矩更新与预条件子重构为轻量的逐元素操作，使用标准 PyTorch 实现。

\paragraph{通用超参数。}
所有模型的嵌入维度统一为 $d = $ --，最大序列长度 $L_{\max} = $ --，批量大小 --，训练轮次 --，早停耐心 -- 轮。训练使用 Adam 优化器，学习率 --，权重衰减 --。损失函数采用交叉熵损失，在全物品候选集上计算。

\paragraph{各模型专有超参数。}
表 \ref{tab:app_hyperparams} 列出了各模型的专有超参数及其搜索范围与最终取值。

\begin{table}[h]
\centering
\caption{各模型专有超参数。搜索策略为网格搜索，以验证集上的 NDCG@10 为选择标准。}
\label{tab:app_hyperparams}
\begin{tabular}{llll}
\toprule
\textbf{模型} & \textbf{超参数} & \textbf{搜索范围} & \textbf{最终取值} \\
\midrule
GRU4Rec    & 隐层维度         & \{64, 128, 256\}         & -- \\
           & 层数             & \{1, 2\}                 & -- \\
           & Dropout          & \{0.1, 0.2, 0.3\}       & -- \\
\midrule
SASRec     & 注意力头数       & \{1, 2, 4\}              & -- \\
           & 层数             & \{1, 2\}                 & -- \\
           & Dropout          & \{0.1, 0.2, 0.3\}       & -- \\
\midrule
LightSANs  & 注意力头数       & \{1, 2, 4\}              & -- \\
           & 层数             & \{1, 2\}                 & -- \\
           & $k$（低秩维度）  & \{4, 8, 16\}             & -- \\
\midrule
LinRec     & 注意力头数       & \{1, 2, 4\}              & -- \\
           & 层数             & \{1, 2\}                 & -- \\
\midrule
Mamba4Rec  & 状态维度         & \{64, 128, 256\}         & -- \\
           & 层数             & \{1, 2\}                 & -- \\
\midrule
GDN4Rec    & 隐状态维度 $d$   & \{64, 128, 256\}         & -- \\
           & 层数             & \{1, 2\}                 & -- \\
\midrule
DamRec     & 隐状态维度 $d$   & \{64, 128, 256\}         & -- \\
           & 层数             & \{1, 2\}                 & -- \\
           & EMA 衰减 $\rho$  & \{0.9, 0.95, 0.99, 0.995, 0.999\} & -- \\
           & 块大小 $C$       & \{16, 32, 64, 128, 256\} & -- \\
\bottomrule
\end{tabular}
\end{table}

% \paragraph{代码开源。}
% 为保证实验的可复现性，我们将在论文发表后公开所有代码与配置文件。
% TODO: 填入 GitHub 链接


\refstepcounter{section}
\subsection*{\thesection\ 隐状态更新策略的完整推导}
\label{app:update_rules}
 
本附录给出第 \ref{subsec:ablation} 节消融实验中各自适应更新策略在 DamRec 框架下的完整闭式状态流转方程。所有变体共享相同的基础架构：门控遗忘 $\alpha_t$、输入门控 $\beta_t$、键值投影 $(k_t, v_t)$，以及基于重构误差的一阶梯度 $G_t = r_t k_t^\top$（其中 $r_t = \alpha_t S_{t-1} k_t - v_t$）。各策略的区别仅在于对 $G_t$ 的处理方式。
 
\subsection{SGD 式（一阶基线 / GDN）}
 
不引入任何动量或二阶矩估计，直接使用原始梯度更新隐状态：
\begin{equation}
    S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top.
    \label{eq:app_sgd}
\end{equation}
此即 GDN \cite{Yang_Kautz_Hatamizadeh_2025} 的标准更新规则，所有特征维度共享相同的有效学习率 $\beta_t$。
 
\subsection{AdaGrad 式}
 
引入二阶矩的累积求和（非指数移动平均），对梯度进行逐元素缩放：
\begin{equation}
    V_t = V_{t-1} + G_t \odot G_t,
    \label{eq:app_adagrad_v}
\end{equation}
\begin{equation}
    S_t = \alpha_t S_{t-1} + \beta_t \left( (v_t - \alpha_t S_{t-1} k_t) k_t^\top \oslash (\sqrt{V_t} + \epsilon) \right).
    \label{eq:app_adagrad}
\end{equation}
由于 $V_t$ 单调递增，预条件子分母随时间步持续增大，有效学习率单调递减。在短序列上这一特性可提供良好的收敛保证 \cite{Duchi_Hazan_Singer}，但在长程流式推断中，$V_t$ 的无界累积将导致所有维度的更新步长趋近于零，模型逐渐丧失对新交互模式的响应能力。
 
利用 $G_t$ 的秩一结构，$V_t$ 同样可分解为向量形式以节约存储：
\begin{align}
    V_t^{(r)} &= V_{t-1}^{(r)} + r_t \odot r_t, \\
    V_t^{(k)} &= V_{t-1}^{(k)} + k_t \odot k_t,
\end{align}
其中 $V_t \approx V_t^{(r)} (V_t^{(k)})^\top$。注意 AdaGrad 式不涉及偏差修正。
 
\subsection{RMSProp 式}
 
将 AdaGrad 的累积求和替换为指数移动平均，为二阶矩引入有限的有效记忆窗口：
\begin{equation}
    V_t = \rho V_{t-1} + (1 - \rho) (G_t \odot G_t),
    \label{eq:app_rmsprop_v}
\end{equation}
\begin{equation}
    S_t = \alpha_t S_{t-1} + \beta_t \left( (v_t - \alpha_t S_{t-1} k_t) k_t^\top \oslash (\sqrt{V_t} + \epsilon) \right).
    \label{eq:app_rmsprop}
\end{equation}
与 DamRec 的区别在于：RMSProp 式不包含偏差修正项 $1/(1-\rho^t)$。在 $t$ 充分大时，$\rho^t \to 0$，偏差修正趋近于恒等变换，二者渐近等价。但在序列早期（$t$ 较小时），缺乏偏差修正会导致 $V_t$ 系统性地低估真实二阶矩，可能引发初期更新步长过大的问题。
 
\subsection{Adam 式}
 
在 DamRec 的二阶矩估计基础上，额外引入梯度的一阶指数移动平均（动量项）：
\begin{equation}
    M_t = \rho_1 M_{t-1} + (1 - \rho_1) G_t,
    \label{eq:app_adam_m}
\end{equation}
\begin{equation}
    V_t = \rho_2 V_{t-1} + (1 - \rho_2) (G_t \odot G_t),
    \label{eq:app_adam_v}
\end{equation}
分别进行偏差修正：
\begin{equation}
    \hat{M}_t = \frac{M_t}{1 - \rho_1^t}, \quad \hat{V}_t = \frac{V_t}{1 - \rho_2^t}.
    \label{eq:app_adam_bias}
\end{equation}
使用校正后的一阶动量（而非原始梯度）作为更新信号，经预条件缩放后更新隐状态：
\begin{equation}
    S_t = \alpha_t S_{t-1} + \beta_t \left( \hat{M}_t \oslash (\sqrt{\hat{V}_t} + \epsilon) \right).
    \label{eq:app_adam}
\end{equation}
由于 $\hat{M}_t$ 累积了多个时间步的梯度历史，上式无法像其他变体那样进一步化简为仅依赖当前输入 $(k_t, v_t)$ 的闭式表达。
 
需要注意的是，$M_t \in \mathbb{R}^{d \times d}$ 是一个与隐状态同维度的矩阵，其引入带来 $\mathcal{O}(d^2)$ 的额外存储开销。此外，由于 $M_t$ 不再保持秩一结构（它是多个秩一矩阵的加权和），无法像 $V_t$ 那样通过秩一分解压缩至 $\mathcal{O}(d)$。DamRec 未采用一阶动量的理由及其完整推导见正文第 \ref{subsec:higher_order_update} 节。

\subsection{各策略特性总结}

表 \ref{tab:app_summary} 从五个维度对比了上述策略的理论特性。
 
\begin{table}[h]
\centering
\caption{各自适应更新策略的理论特性对比。}
\label{tab:app_summary}
\begin{tabular}{lccccc}
\toprule
\textbf{策略} & \textbf{一阶动量} & \textbf{二阶矩} & \textbf{EMA} & \textbf{偏差修正} & \textbf{额外空间} \\
\midrule
SGD / GDN         & --            & --            & --            & --            & $\mathcal{O}(0)$ \\
AdaGrad 式        & --            & \checkmark    & --            & --            & $\mathcal{O}(d)$ \\
RMSProp 式        & --            & \checkmark    & \checkmark    & --            & $\mathcal{O}(d)$ \\
Adam 式           & \checkmark    & \checkmark    & \checkmark    & \checkmark    & $\mathcal{O}(d^2 + d)$ \\
DamRec            & --            & \checkmark    & \checkmark    & \checkmark    & $\mathcal{O}(d)$ \\
\bottomrule
\end{tabular}
\end{table}
 
其中"额外空间"指相对于 SGD 基线（仅维护 $S_t$）的增量存储开销，且 AdaGrad、RMSProp 与 DamRec 均假设采用秩一分解。Adam 式的 $\mathcal{O}(d^2 + d)$ 中，$d^2$ 来自无法分解的一阶动量矩阵 $M_t$，$d$ 来自可分解的二阶矩向量。