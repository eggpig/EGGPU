# 引言与首页图：主源、设计和计时

修订：2026-11-06。当前方案固定为两个 panel：左侧四次 SSSP 请求，右侧为初始 G⁰ 已就绪后两臂的实际连续会话时间。主窗口依据用户 ready 后计时要求，在第二臂结果返回前固定，未按胜负选择。旧版共享分配构造例仅保留在 `../previous`；不再作为当前首页图。机制图与框架图仍留给协作者。

## 1. 本轮实际复核的原论文

本轮重新读取两篇完整 PDF 的引言和实验定义，并查看第一页渲染，不以搜索摘要或旧笔记替代原文。

| 原论文 | 实际阅读位置 | 借鉴与边界 |
|---|---|---|
| Meng et al., **Efficient Multi-GPU Graph Processing with Remote Work Stealing**, ICDE 2023, pp. 191–204；[作者 PDF](https://graphscope.io/publication/gum.pdf)，[DOI](https://doi.org/10.1109/ICDE55515.2023.00022) | §I、Fig. 1；§VI-A/B、Fig. 6 | 先交代并行执行进展，再用 Gunrock 在 webbase 上的一次 SSSP 八卡时间线，解释动态负载不均和长尾。其图并非宽泛框架图。本文借鉴“具体现象→两个挑战”的推进；不借用它的时间或加速比。 |
| Gao et al., **GRACE: Alleviating Reconstruction Cost in Dynamic Graph Processing Systems**, ICDE 2026；[作者 PDF](https://me.sage.org.ai/contents/research_papers/2026/2026_grace_icde_2026.pdf)，[DOI](https://doi.org/10.1109/ICDE65706.2026.00075) | §I、Fig. 1；§V-A Evaluation Method | 先说明 PMA–CSR 兼顾更新与计算，再用实际更新时间分解暴露重建开销，最后拆分复制、遍历和重平衡挑战。本文保留其“已有进展→剩余代价”的结构，不挪用其占比或将 CPU 拓扑重建视为本文 GPU 权重维护的同一实验。 |

**原论文计时不能直接替换本文计时。** GUM §VI-A 明确排除 I/O；§VI-B 将执行分为 GPU kernel、设备间通信及等待、序列化和辅助开销。GRACE Fig. 1 的分母是整体图更新时间；§V-A 将随机打乱的静态图边按每批一千万条插入初始空图，另行评估图计算。两者均不等于下述四请求连续会话。本文必须自己测量；不能从 kernel time 推导总会话时间，也不能把 GUM 的通信桶或 GRACE 的重建桶直接套到本系统。

本轮读取的缓存位于：`validation-scratch/intro_20261105/primary_sources/`。下载 URL 与校验和已保存在该目录 `source_manifest.json`；`gum.pdf` SHA-256 为 `7a193326e961763fd2ed4e2d57afcac0af5f1f7974945d7530a1c7eef86b8976`，`grace.pdf` 为 `7bf694bad8b9147f04b46315d112523ba189af771e950f9d3c24a773afff15ce`。

## 2. 首页实验：固定四次请求，不按结果选场景

执行入口为 first_page_trace.py（内部资产未随本增量分发：`../../../experiments/campaign_20261104/first_pass_20261106/timing/first_page_trace.py`）。只有主代理执行 GPU。本图使用冻结输入中的 IN（indochina-2004）与 UK（uk-2002）、八张已授权 GPU、同一源版本、同一数值合同；固定轨迹如下。

| 顺序 | 请求 | 需要呈现的问题 |
|---|---|---|
| 1 | `IN@0 / SSSP`，记为 $G^0$ | 准备首个完整请求的输入状态 |
| 2 | 按冻结 packet 提交固定拓扑权重更新；`IN@1 / SSSP`，记为 $G^1$ | 旧拓扑仍有效，新权重需要进入设备表示 |
| 3 | `UK@0 / SSSP`，记为 $H$ | 不同图的保留状态竞争受管显存预算 |
| 4 | `IN@1 / SSSP`，记为回访 $G^1$ | 前次保留与释放影响本次恢复 |

每张图的源点固定为输入描述中的第 0 个源点；三次 IN 请求使用同一源点。两臂均返回完整距离向量，使用同一参考检查，并检查 Session 关闭后输出仍保持有效。先执行与计时分离的每图状态大小 pilot，以第 $d$ 张卡上单图 SSSP 视图的真实分配并集 $V_{G,d}$、$V_{H,d}$ 决定预算：

$$B_d=\left\lfloor\max(V_{G,d},V_{H,d})+0.25\min(V_{G,d},V_{H,d})\right\rfloor.$$

预算公式不读取耗时；pilot 不是性能重复。`0.25` 是预先固定的预算参数，不是物理 GPU 容量比例。对两臂均检查 H 请求结束、回收前确有超预算设备，且回收确实释放字节；若没有观察到该现象，本图不得声称展示了预算压力。

| 图中名称 | 实际实现 | 解释 |
|---|---|---|
| Full upload + LRU | `persistent_full + closure_lru` | 强对照：保留主机内存中的版本提交，对变化设备状态采用完整上传；LRU 仍按分配闭包计算真实释放量并执行合法回收。不是全量主机重建、不是逻辑大小错误记账、不是关闭全部优化。 |
| EGGPU | `lease_sparse + closure` | 启用依赖维护路径与基于需求的闭包保留；与对照共用执行后端、请求、预算和完整输出要求。 |

这是一组策略配对动机实验，不是 E1 的外部系统比较，也不是 D/C 四臂因果消融。若总时间没有改善，应保留原值；维护字节下降不能覆盖更慢的总时间。一次观测只记为 n=1；补第二次时保留原观测，不择优。

## 3. 实际计时边界与允许的分解

右侧主柱采用实际连续区间 `arms[*].commit.interval_ns[0]` → `arms[*].events[-1].complete_query_interval_ns[1]`，即初始 G⁰ 的首次完整请求已经结束，从权重更新开始，到 G¹ 回访的完整 query 返回。它包含权重提交、之后三次完整查询、延迟设备维护、查询结束后的保留决策和区间内记录开销。图中称 **Session time**，纵轴为 `Time (s)`。输入准备、初始 Session 创建、G⁰ 的首次查询、大小 pilot、参考校验、输出归档和关闭不在主窗口中。该窗口在第二臂结果返回前按用户 ready 后计时要求固定，直接用起止时间戳计算，不由 kernel time 推断，也不需要新一轮 GPU 执行。

原始 `arms[*].total_seconds` 是从公开 Session 创建前至第四次 query 后记录完成的四请求总时间，包含初始输入装载与验证；它与以下全部原始阶段继续保存在原始回执和 first_page_measured_input.json，作为补充口径，不删除或覆盖。

**本次实际观测（n=1，统一源码 71ea）。** 两臂完整输出、关闭后的独立所有权和 H 阶段预算压力均通过。主窗口为 **8.938643417 s 对 7.899269512 s**，本次 EGGPU 少用时约 **11.6%**。四请求完整口径为 21.794597326 s 对 21.716015897 s，差异很小；不混淆这两个分母，也不据一次观测声称稳定收益。图 1 使用预先冻结的初始状态就绪后窗口。

| 实际区间 | Full upload + LRU (s) | EGGPU (s) | 主窗口内 |
|---|---:|---:|---|
| Session 创建 | 11.056276585 | 11.160787329 | 否 |
| 首次 G⁰ 完整查询 | 1.799542721 | 2.655525675 | 否 |
| 权重提交 | 3.772154957 | 3.028497473 | 是 |
| 更新后 G¹ 完整查询 | 1.513837334 | 1.658387371 | 是 |
| H 完整查询 | 2.993007885 | 2.812657190 | 是 |
| G¹ 回访完整查询 | 0.659519187 | 0.399666644 | 是 |
| **主窗口连续时间** | 8.938643417 | 7.899269512 | **是** |
| 四请求完整会话总时间（补充口径） | 21.794597326 | 21.716015897 | 包含初始准备 |

图 1(b) 堆叠的是以下三个互斥墙钟分类，而不是累加嵌套 CUDA event 或各卡时间。

| 主窗口内互斥分类 | Full upload + LRU (s) | EGGPU (s) |
|---|---:|---:|
| Host preparation：提交＋后续 host_plan | 7.140406591 | 5.882518681 |
| Device execution：后续 cooperative_gang | 1.472921808 | 1.748599334 |
| Other：剩余墙钟时间 | 0.325315018 | 0.268151497 |
| **主机准备占比** | **79.9%** | **74.5%** |

`cooperative_gang` 包含上传、设备计算、通信和结果回传，不是纯 kernel time。三个后续请求均核对“所有 host phase 之和 = native_outer_seconds”，且不超出所属完整 query 扣除保留后的区间。Other 包含其余原生阶段、Python 层处理、保留和连续区间记录。完整阶段来源绑定在 first_page_measured_input.json。

实际改动已在两臂同样生效：G¹ 更新涉及 19,160 个位置，native_delta 保留确切父版本及修改包检查，省去对全部 191,606,827 个权重的再次合法性验证；完整权重重建及内容哈希仍需 O(m) 工作。不可变拓扑共享 920,469,020 字节，实测父拓扑复制量为 0。两臂使用相同实现改动，因此这些优化的前后收益不能直接解释为 D/C 策略差异。

旧 b4 版本的主时间 **17.174507245 / 20.776297179 s** 及负结果继续保存在 `first_page_trace_receipt.json`，没有删除。它与当前统一源码不同，不能当作第二次重复平均，也不能把不同版本中较快的单项拼接。当前准备占比仍只能说明这一受控实例；普遍性与 D/C 各自效果由后续独立矩阵验证。

原始回执保留的阶段如下，名称必须保持边界含义：

| 回执字段 | 对应实际区间 | 不能称为 |
|---|---|---|
| `setup_update_seconds` | 公开 Session 创建＋主机权重提交两个实测区间 | 所有 GPU 维护时间 |
| `query_excluding_retention_seconds` | 每次 query 起止内，扣除实际观察到的 `finish_query` 区间；包含延迟设备维护、算法执行和完整独立结果 | 纯计算／纯 kernel time |
| `retention_seconds` | 原始 `finish_query` 调用的实际墙钟区间 | 全部内存管理时间 |
| `bookkeeping_seconds` | 连续会话时长减去以上不相交实测区间 | 未经测量的模拟开销 |

当前首页使用上表三个主窗口互斥阶段的堆叠柱。下表四请求原始阶段仅供补充，不能直接堆叠进较短主窗口；不在图中加入长段解释文字。不能把重建耗时或数据传输耗时从事件数量、字节数或外部论文比例中反推。

## 4. 图面与数据准入

- **(a) Request sequence：** 四个 SSSP 请求按时间向下排列，箭头仅标 `Weight update`、`Switch graph`、`Revisit`。左侧计时括号从权重更新开始到最后一次返回；初始 G⁰ 不重复标注计时术语，边界在图注说明。底部只标 `G: IN; H: UK`。左图没有构造性能值。
- **(b) Session time：** 两根柱，纵轴 `Time (s)`，横轴 `Full upload + LRU` 与 `EGGPU`。主机准备用蓝色、设备执行用金色及斜线，其余用浅灰及点纹；颜色编码阶段，系统名称由横轴区分。无真实回执时，只显示 `Measurement pending`，没有柱、数值或虚拟刻度。
- 提供 85 mm 单栏与 175 mm 两栏矢量 PDF、SVG，以及 300 dpi PNG。175 mm 采用左右两个 panel；85 mm 采用上下两个 panel，保持字号可读。图中文字以短标签为主，方法流程与框架细节放后文。

[render_first_page.py](render_first_page.py) 直接接受原始 `eggpu.first-page.four-request.v1` 回执。绘图前要求：状态 `PASS_FIRST_PAGE_TRACE`，`gpu_executed=true`，两个完整配对臂，八卡身份，两臂正确性通过，H 阶段压力与实际释放均已记录，同一预算和四请求顺序，有限正总时间、主窗口起止，以及阶段和、请求区间与连续区间的一致性。图旁来源 JSON 记录输入和生成器哈希、实际值与所有输出文件哈希。它不自行提交 GPU 作业，也不按时间挑选数据。

当前图已直接绑定[真实回执](first_page_measured_input.json)，渲染来源见 [figure_provenance.json](figure_provenance.json)。first_page_trace_spec.json（内部资产未随本增量分发：`first_page_trace_spec.json`） 仅是保留的待测输入模板，不是当前图的数据源。若真实回执不完整，脚本报错，不自动降格为看似有效的时间柱。

复现命令（需已有 matplotlib 环境）：

```bash
python3 -B render_first_page.py --input first_page_measured_input.json --output /your/output/first-page
```

## 5. 图注与正文接入

**当前中文图注。** 图 1：连续图分析中的状态维护与回访。(a) IN（G）与 UK（H）上的四次 SSSP，包括权重更新和回访。(b) 初始 G⁰ 请求完成后，从权重更新到最后完整结果的实际连续时间；八卡、相同逐卡保留预算、一次配对观测。计时包含维护与完整输出；LRU 使用分配闭包感知回收。

**当前英文图注。** Continuous graph analysis under a limited retention budget. (a) Four SSSP requests on IN ($G$) and UK ($H$), including a weight update and a revisit. (b) Measured time from the weight update through the final result, after the initial G⁰ request completes, on eight GPUs with equal per-device budgets (one paired observation). Timing includes maintenance and complete results; LRU uses allocation-aware reclamation.

图注中的 one paired observation 只适用于 n=1，不使用误差条。正文可写该序列的实际秒数；不把这一个配对点写成所有图或所有算法的普遍规律。初始输入准备、首次 G⁰ 查询、参考校验和关闭阶段不在主时钟内，详细边界在实验设置明确。

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/first_page_problem_85mm.pdf}
  \caption{Continuous graph analysis under a limited retention budget.
    (a) Four SSSP requests include a weight update and a revisit.
    (b) Time from the weight update to the final result, after the initial $G^0$
    request completes (eight GPUs; one paired observation). Timing includes
    maintenance and complete results; LRU uses allocation-aware reclamation.}
  \label{fig:continuous-session}
\end{figure}
```

当前图由真实数据生成；本公开包分发 SVG、标量输入及绘图源码，PDF/PNG 可按下述命令生成。上一版正文和构造图仅保存在 `../previous`，其示例数值不进入当前实验输入。
