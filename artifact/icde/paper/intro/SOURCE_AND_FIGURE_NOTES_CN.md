# 引言主源与首页问题图核查

核查日期：2026-11-05。已实际打开以下四篇完整 PDF，阅读物理第 1–2 页的引言，并逐一查看第一页渲染图；另核对与本文定位相关的方法／实验段落。页码以下均为 PDF 物理页码。首页图建议是本次设计判断，不是引用论文的实验结论。

## 1. 四个主源

| 主源与准确书目信息 | 本次核查位置 | 引言／首页图观察与采用边界 |
|---|---|---|
| **GRACE: Alleviating Reconstruction Cost in Dynamic Graph Processing Systems**；Hongru Gao、Shuhao Zhang、Xiaofei Liao、Hai Jin；ICDE 2026。[作者 PDF](https://me.sage.org.ai/contents/research_papers/2026/2026_grace_icde_2026.pdf)；[作者单位录用记录](https://grid.hust.edu.cn/info/1053/2982.htm) | pp.1–2 §I、Fig.1；p.3 §II-B；方法 §III | p.1 右栏顶端用成本占比图指出重建开销，再拆分具体挑战。配色为深浅暖橙，重建部分另加斜线。它研究 PMA–CSR 存储更新，不是 GPU 跨调用保留。可借鉴“一个现象→明确困难”，不能借用它的开销比例作为本文动机数据。 |
| **An Efficient Memoization Engine for Concurrent Graph Query Processing（KGraph）**；Sen Gao、Shengliang Lu、Shixuan Sun、Yuchen Li、Bingsheng He；ICDE 2025，pp.1015–1028；[DOI 10.1109/ICDE65448.2025.00081](https://doi.org/10.1109/ICDE65448.2025.00081)；[作者 PDF](https://gaosen.org/resources/papers/KGraph.pdf)；[合作者机构元数据](https://ink.library.smu.edu.sg/sis_research/10407/) | pp.1–2 §I、Fig.1–2；p.2 §II；p.9 §VI-A | p.1 右上以出租车请求的日内分布建立真实需求；不同日期用绿、黄、橙柱区分。随后讨论重复计算与 memoization 开销。其并发同函数查询及结果复用不等于本文多图、多函数和版本维护；其计时排除磁盘读取与分区。只借鉴具体任务开场，不把本文受控轨迹说成真实业务日志。 |
| **White-Box Micro-Adaptive Query Processing**；Jack Pearce、Hubert Mohr-Daurat、Holger Pirk；ICDE 2025，pp.2880–2893；[DOI 10.1109/ICDE65448.2025.00216](https://doi.org/10.1109/ICDE65448.2025.00216)；[Imperial 作者稿 PDF](https://spiral.imperial.ac.uk/server/api/core/bitstreams/6d7e9f53-4ebc-49df-8996-eda224d2f2c7/content)；[作者机构出版记录](https://profiles.imperial.ac.uk/pirk/publications?id=950944&noscript=noscript&respub-action=citation.html) | pp.1–2 §I、Fig.1、§III；pp.8–9 §VII 和实验设置 | p.1 固定选择率，改变数据有序性，以黑色实线／虚线呈现实现选择的变化；无需彩色也清楚。引言由未被常见统计量捕捉的特征进入运行时决策。本文借鉴“固定表面条件、暴露隐藏状态”的问题呈现，不把其 CPU 硬件计数器方法或 TPC-H 数字写成本系统结果。 |
| **Efficient Multi-GPU Graph Processing with Remote Work Stealing（GUM）**；Ke Meng、Liang Geng、Xue Li、Qian Tao、Wenyuan Yu、Jingren Zhou；ICDE 2023，pp.191–204；[DOI 10.1109/ICDE55515.2023.00022](https://doi.org/10.1109/ICDE55515.2023.00022)；[作者团队 PDF](https://graphscope.io/publication/gum.pdf) | pp.1–2 §I、Fig.1；§III–IV；§VI | p.1 右上以一次 SSSP 的八卡时间线呈现两种停顿，青绿为计算、浅色为停顿，两个编号直接对应后文困难。可借鉴“一张具体图支持两个相连问题”。其 FSteal／OSteal 针对执行中负载与长尾，不能推导本文的版本维护或回收收益；NVLink 机器结论也不能直接外推到当前硬件。 |

White-Box 的作者、题名、ICDE 页码和 DOI 已由 Imperial 作者出版记录交叉核对；PDF 来自该校 Spiral 仓库，不是用搜索摘要代替原文。GRACE 本稿以用户指定作者 PDF 及作者单位记录为主源，不依赖第三方 DOI 搜索摘要。以上论文的性能数字均未挪入本文初稿。

补充定位主源：[Shared Arrangements: practical inter-query sharing for streaming dataflows](https://www.vldb.org/pvldb/vol13/p1793-mcsherry.pdf)，Frank McSherry、Andrea Lattuada、Malte Schwarzkopf、Timothy Roscoe，PVLDB 13(10):1793–1806，2020，DOI [10.14778/3401960.3401974](https://doi.org/10.14778/3401960.3401974)。本次打开原 PDF 并核对 §I–II：共享索引状态、多个查询／读者和跨查询状态维护已有明确先例。此处仅用于收紧新颖性陈述，不扩大首页图的表达任务。

## 2. 对当前首页图方案的判断

**建议保留“逻辑撤销量不等于真实物理释放量”这个单一焦点。** 这能让读者先看到一个错误容量判断，再理解为什么普通对象大小不足以指导保留。上方一次权重更新形成两个版本，可以自然连接 P3 的维护合法性；图面无需同时解释 D-Maintain 和 C-Retain 的内部步骤。这样与四篇参考的首页处理一致：先展示问题，方法结构放后文。

推荐两个 panel，使用相同物理对象标识和颜色：

- **(a) Shared physical state after a weight update.** 两个逻辑版本 $G^{0}$、$G^{1}$ 指向同一个拓扑分配 $T=8$，各有私有权重 $W^{0}=1$、$W^{1}=1$；独立图 $H=6$。只画一份物理 $T$，两个引用箭头，避免图形本身暗示复制。更新箭头仅说明权重版本变化，不表示持续时间，也不声称所有权重均需复制。
- **(b) Logical size versus actual release.** 在随后可回收边界，列出同一组候选动作的两种计数，并检查是否足以填补 6 个单位的容量缺口。三个候选并列，不预选“最优”，不把可行等同于性能更优。

| 候选撤销集合 | 被撤视图名义大小之和 | 实际释放 | 剩余物理占用 | 是否满足预算 10 |
|---|---:|---:|---:|---|
| $\{G^{0}\}$ | 9 | 1 | 15 | 否 |
| $\{G^{0},G^{1}\}$ | 18 | 10 | 6 | 是 |
| $\{H\}$ | 6 | 6 | 10 | 是 |

初始物理占用为 $8+1+1+6=16$；逐视图名义大小相加为 $9+9+6=24$。表内是集合和的精确算术，不是实验。第二候选中实际 10 也不能由两个单独撤销收益相加得到，因为共享 $T$ 只有在两侧均撤销后才真正释放。不能将上表再画成似乎来自实测的系统加速柱状图。

**必须写出的合法性条件：** 回收 panel 的全部持有者都已画出，候选 view 已允许撤销，没有活动读者、外部 pin 或未画出的 catalog holder。若 $G^{0}$ 仍被活动旧读者使用，撤销它不是合法候选。图 (a) 可以说明更新时保护旧版本；图 (b) 必须写明随后到达回收边界，不能把这两个时刻混为一谈。对于实际代码，catalog 也需要进入允许撤销的完整闭包后才可能释放共享分配。

## 3. 图面与配色建议

- 85 mm 单栏版优先纵向排列两个 panel；175 mm 版可横排。正文以单栏缩小后仍可读为验收条件，图内最终字号建议 8–9 pt，名称直接贴近对象。
- 控制在三类语义色：共享拓扑用青绿 `#2A7F7F`；旧／新权重用同一暖橙的浅深变化 `#F3CFB4` / `#C97743`，并保留 $0/1$ 上标；独立图与文字辅助用灰 `#777777` / `#303030`。这些色号是本次建议值，不声称从参考图精确提取。
- “逻辑大小”用浅灰或空心表示，“实际释放”用深色或数值强调。即使灰度打印，也应能通过填充／轮廓／标注分辨；不要只依赖红绿表示成功失败。
- 数值只标存储单位与精确求和，不使用秒、MiB、加速比或实验系统名作横轴。容量条件用短标签“resident 16; budget 10; release needed 6”。
- 参考 GUM 的编号可以只标两个观察：“unchanged topology is shared”与“one view removed, shared storage remains”。不要追加算法模块、控制器、成本模型、回收伪代码等方法框架。
- 8+1+1+6 比 40/40/48 字节更便于首页阅读。后者若来自手工小图仍是构造例，不会自动变成性能证据；只有在需要解释 CSR 类型宽度时才值得换成精确字节图，而那更适合后文或补充材料。

建议英文图注：

> **Fig. 1. Logical view removal does not determine physical memory release.** In this constructed example, two versions share topology storage and own separate weights. At a later reclamation boundary, all shown views are evictable and no unshown holder or active reader remains. Removing one version releases only its private weights; releasing the shared topology requires removing both holders. Values are illustrative storage units, not measured GPU memory or performance. Both sufficient reclamation choices are shown without assuming either is optimal for future requests.

## 4. 文稿主线与评估边界

P1 给请求序列；P2 定位执行、动态存储与计算结果复用；P3 把小更新与旧引用合法性分开；P4 用精确反例解释共享回收；P5 将维护和保留连接到相同身份／持有关系；P6 交代如何用完整任务验证，并留下正式数据占位。不要将 P6 写成已经达成 90% 领先、所有函数多卡单调加速或 D/C 完整时间正交互。

当前实现边界以[实验范围核对](../EXPERIMENT_SCOPE_RECONCILIATION_20261105_CN.md)和[冻结协议](../DATASET_AND_EXPERIMENT_FREEZE_20261104_CN.md)为准：固定拓扑权重维护、FP64 COW／lease、紧凑静态路线、请求完成边界容量和实际计时需要分别表述。构造图证明的是物理共享的记账反例，不证明当前策略的平均性能，更不能代替 E3–E5 的实测因果证据。

## 5. 本次核查留档

四篇原 PDF、前两页渲染和逐页文字保存在临时资料目录，不复制进论文源目录：

`INTERNAL_EVIDENCE_ROOT/intro_20261105/primary_sources/`

下载 URL、文件字节数、页数和 SHA-256 记录于该目录 `source_manifest.json`。四篇均为 14 页。网页获取 KGraph 时遇到文件大小限制，已直接从同一作者 URL 下载 13,058,976 字节 PDF，完成本地逐页读取和第一页视觉核查；没有以摘要替代全文。

| 文件 | SHA-256 |
|---|---|
| `grace.pdf` | `7bf694bad8b9147f04b46315d112523ba189af771e950f9d3c24a773afff15ce` |
| `kgraph.pdf` | `c888d8663c2a3421d840a8a0ac99d9c16e3146b05b9e7b62aa4646f6c0f389f3` |
| `white_box.pdf` | `29bf025d42aa1ff0d8741451c7373e3283c15da924805c607fd4eacb9fb04e4f` |
| `gum.pdf` | `7a193326e961763fd2ed4e2d57afcac0af5f1f7974945d7530a1c7eef86b8976` |

## 已生成的第一页问题图

推荐采用 [85 mm 单栏 PDF](first_page_problem_85mm.svg)；若第一页上方允许跨栏，可采用 [175 mm 双栏 PDF](first_page_problem_175mm.svg)。两版都提供同名 SVG 和 300 dpi PNG，数据来自 [明确构造的所有权模型](first_page_example.json)，由 [Python 脚本](render_first_page.py) 计算并绘制。[计算回执](example_calculation.json) 和 [文件来源记录](figure_provenance.json) 保存完整输入、代码与产物 SHA-256。图内所有数值是示例存储单位，不是实测显存。

**中文图注草稿。** 图 1：共享状态使逻辑撤销量与物理释放量分离。(a) 一次权重更新后的两个保留版本共享大小为 8 的拓扑，各自持有大小为 1 的权重，另一个图独占大小为 6 的状态。(b) 在随后允许回收的边界，要将物理占用从 16 降至 10，必须释放至少 6 个单位。仅撤销 G⁰ 虽移除 9 个单位的逻辑引用，却只释放 1；联合撤销 G⁰、G¹ 或撤销 H 才能满足此容量要求。构造例列出了全部持有者；此时没有活动读者、外部固定引用或额外工作区，所有视图均可合法撤销。图形面积不表示大小。两种可行撤销集合的后续恢复代价取决于请求需求，图中不宣称某一种总是最优。

**English caption draft.** Shared state separates logical removal from physical reclamation. (a) Two retained versions share 8 units of topology and each owns 1 unit of weights; H owns 6 units. (b) Reducing physical occupancy from 16 to 10 requires reclaiming 6 units. Removing G⁰ drops 9 referenced units but frees only 1. Removing both versions or H meets the capacity requirement. This is a constructed example at a later reclamation boundary: all holders are shown and evictable, with no active readers, external pins, or additional workspaces. Box areas are not to scale. Feasibility alone does not determine the best choice for subsequent requests.

单栏 LaTeX 插入位置（需论文模板实际编译后微调浮动位置）：

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/first_page_problem_85mm.pdf}
  \caption{Shared state separates logical removal from physical reclamation.
  Removing $G^0$ drops 9 referenced units but frees only 1, below the required 6.
  This constructed example shows all holders at an idle reclamation boundary;
  all views are evictable and box areas are not to scale.}
  \label{fig:ownership-gap}
\end{figure}
```

复现命令：`python render_first_page.py --input first_page_example.json --output rendered`。依赖 Python 与 Matplotlib；不使用 GPU，不读取实验耗时，不生成模拟性能曲线。脚本检查全部视图子集的释放量，防止重复计算共享分配。生成后已查看两版 PNG，修正了初版双栏标题裁切及阈值文字重叠。

方法框架图和 D/C 机制图继续保留外部绘制占位；这张图只承担首页问题说明。
