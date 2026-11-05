# EGGPU ICDE：实验章节与图表呈现方案

更新：2026-11-04。本文件组织当前论文的实验问题、比较对象和呈现方式；**设计不是实测结果**。七图、数值语义、输入和计时合同以 [冻结协议](DATASET_AND_EXPERIMENT_FREEZE_20261104_CN.md) 与 [数据注册表](DATASET_REGISTRY_20261104.json) 为准。本轮执行状态、最新结果入口和替换关系由 `../CURRENT_DELIVERY_CN.md` 统一维护，机器采样及执行清单位于 `../experiments/campaign_20261104/setup.json`、`campaign.json`；这些执行文件未生成或字段未核实时，正文保持待填，不能抄旧报告的机器或数字。

## 1. 实验章节要回答的一个问题

**在相同输入语义与资源约束下，EGGPU 能否更快地正确完成连续图分析任务；减少的时间分别来自高效执行、合法维护和合适的状态保留中的哪一部分？**

论文的因果链按以下顺序展开：E1 先证明系统有竞争力，E2 说明增加资源何时有效；E3/E4 分别隔离 D/C，E5 检验两者的联系；E6 证明扩展函数能够改善一个存在真实数据依赖的完整任务。单卡基础执行不能被较弱的多卡后端 P1 替代，单次内核快也不能替代完整任务快。

每个实验正文固定四步：**问题与控制变量 → 主要观察 → 用日志指标解释机制 → 适用边界或反例**。结果未出时只写前一步和待验证问题。结果出后优先讲相对最强适用外部实现的实测差距，不能只讲相对无优化版本的收益。

## 2. 从五篇论文采纳什么

下列定位来自论文实际实验节与图表；物理页码从 PDF 第一页起计数。各论文自己的数据、计时和参数不直接变成本文的测量规则。

| 论文与已核位置 | 展示组织及值得采用的做法 | 本文怎样使用 |
|---|---|---|
| [GUM，ICDE 2023](https://graphscope.io/publication/gum.pdf)，§VI，物理第8–12页；表III、图6–10、表IV | 总体精确时间用表；扩展与拆解用图；先问整体、扩展，再查负载和优化成本。图7使用颜色与标记，图8直接展示每卡工作/等待。 | E1 用表承载完整矩阵，E2 用曲线与开销分解；保留不单调点和路网边界。不能借其 NVLink 平台结果推断本机 RWS 可用。 |
| [GRACE，ICDE 2026](https://me.sage.org.ai/contents/research_papers/2026/2026_grace_icde_2026.pdf)，§V，物理第8–12页；图8–14 | 分开呈现重建、整体更新、后续计算；图9拆解时间，图13消融，图14扫参数。浅色加纹理区分基线与增强版本。 | E3 不能只报补丁字节，必须报提交到结果；E5 用强替代的四格消融。拓扑重建与本文属性维护的语义区别保留。 |
| [HyTGraph，ICDE 2023](https://arxiv.org/pdf/2208.14935)，§VII，物理第9–12页；表V、图7–10 | 全体时间矩阵后，用执行路径解释何时选不同传输；模块收益、规模和设备差异各自回答问题。方法图采用浅蓝、浅绿、浅橙与黑色边框。 | E3/E4 选一条有日志的轨迹解释决策，再用扫描证明普遍性；不把机制图画得漂亮当作测量。总体失败或落后格子照常显示。 |
| [KGraph，ICDE 2025](https://gaosen.org/resources/papers/KGraph.pdf)，§VI，物理第9–12页；表II、图12–18；[作者代码](https://github.com/Gawssin/KGraph) | 表II覆盖应用/图，参数图连接分区、命中与时间，图16区分记忆构建与查询，图18展示真实案例。正文按 finding 推进，但仍解释收益差异。 | E4 增加共同 SSSP 状态复用面板，计入 memoization；E6 展示真正的前后依赖。KGraph 为 CPU 结果复用系统，其 PPR 是随机游走任务，不能冒充全局 PR 或 GPU 物理预算策略。 |
| [C²graph，ICDE 2026](https://wzgcs.github.io/papers/32_ICDE_2026.pdf)，§VI，物理第8–13页；图5–8、表II/III | 内存与时间分图，单查询与多查询分开，准备成本用表，迁移时机用曲线解释。 | E1 分首次、已准备与任务前缀；E2/E4 不混淆延迟、吞吐、容量；压缩准备必须进入本文完整任务边界。 |

这些论文的实验部分通常跨约三至五个物理页面，但有的页包含算法、讨论或参考文献起始，不能把跨页数当作净实验篇幅。**本文建议预留约 3.5–4 页**（下面是写作预算，不是 ICDE 的页数规定），把次要参数图与完整来源记录放入 artifact。KGraph 的实际源码和完整正文已核对；正式版与完整技术报告的图编号、补充位置有差异，最终引用时绑定使用的 PDF。

## 3. 章节结构与信息量

| 正文部分 | 建议面积 | 必须讲清楚 | 主展示 |
|---|---:|---|---|
| Setup、图集、比较合同 | 0.5页 | 机器、输入、实现、语义、首次/热调用边界 | Setup 紧凑表、七图表；完整基线覆盖矩阵放入 artifact |
| E1 Overall Performance | 0.8–1.0页 | 相同资源与质量下实际位置；准备成本如何摊销 | 单卡/多卡分块时间表；16请求累计时间小图 |
| E2 Multi-GPU Scalability | 0.4–0.5页 | 固定后端扩展；资源增多为何有时无益 | 四算法小分面曲线，代表性时间分解 |
| E3 Dependency-Aware Maintenance | 0.5页 | 哪些变化需要维护，少维护是否影响完整时间 | 更新比例曲线与维护字节面板 |
| E4 Closure-Aware Retention | 0.5页 | 共享物理对象如何影响释放与回访成本 | 预算曲线、真实分配/释放字节；共同状态复用副面板 |
| E5 Joint Effects | 0.3–0.4页 | D/C 是否存在完整时间交互，证据是否稳定 | 四组配对时间；交互区间或传输量小面板 |
| E6 Application and Function Coverage | 0.5页 | 扩展函数对完整依赖任务的真实贡献 | 阶段堆叠图与最终输出一致性简表 |

保留两个方法图位置：**Fig. M1（机制示例，双栏宽）** 放在 D/C 概念引入附近；**Fig. M2（框架概览，双栏宽）** 放在系统设计开头。它们由用户委托画图者制作，目前不生成替代图、不把占位图当作结果。最终编号随排版确定，实验文件采用 `e1_overall`、`e2_scaling` 等稳定名字，不依赖容易变化的 Figure 数字。

## 4. 比较对象的纳入规则

**论文说实现过的功能首先进入源码查证清单，不直接记成不支持。** 对每个系统依次记录：作者论文的功能 → 对应仓库/提交/入口 → 能否编译和运行 → 与本文的语义是否一致 → 本机正确性是否通过 → 进入哪个实验。没有及时找到入口是“待查证”，环境失败是“构建失败”，精度证据缺失是“待质量资格”，三者都不等于算法不支持。

使用作者提供的实现、合法参数与正常优化编译配置；只做构建兼容、格式接入、输出导出和共同合同所需修补，保存差异。不替作者重新设计内核或新增性能优化，也不关掉作者已有缓存、CPU 协作、压缩或批处理来制造弱基线。若接口没有合法更新能力，先核实应如何失效并恢复状态，再报告其实际代价。

| 比较角色 | 本轮须核验/接入的系统 | 适用位置 | 不应做的比较 |
|---|---|---|---|
| 通用单卡/异构执行 | cuGraph、Gunrock、Grapin、CGgraph、HyTGraph；SYgraph 按实际函数进入 | E1 共同语义格子；E3/E4 的完整任务外部参照 | 未实现的 PR 不用 PPR 顶替；FP32 结果未满足统一质量前，不与 FP64 主表直接合并 |
| 加权遍历和多查询协作 | C²graph-P / PM | E1 单源 SSSP 使用 P；E4-S 同图不同源完整批任务使用 PM | 两模式独立资格；不能用 PM 批时间除以批长填 E1，不能把其 PPR/Adsorption 数字当全局 PR；首次压缩计入完整任务 |
| 多 GPU 执行 | GUM、cuGraph 的真正多卡路线；Gunrock/Groute 按作者多卡实现与本机资格决定 | E1 多卡分块、E2 共同后端强扩展 | 单卡实现不进入 P>1 曲线；CPU fallback 不标成正常 GPU 执行；不同机器数字不拼在同一行 |
| 查询结果复用 | KGraph；其作者对照 ForkGraph/Glign 作为必要时的复用机制参照 | E4-S；与 D/C 讨论衔接 | 不直接当作 GPU 物理闭包淘汰算法，不强塞进属性 patch 微基准 |
| 动态存储/重建 | GRACE 及其 PPCSR/Terrace/VCSR 接入 | Related Work、语义边界；若另有共同更新任务才做性能对照 | 当前固定拓扑权重任务不能冒充拓扑插删；不能为“都有更新”强行列同一速度表 |
| 扩展函数与独立参考 | 已有 CuPy + direct pylibcugraph 组合、igraph、其他确有同义函数的作者实现 | E6 共同阶段、完整组合；数值参考 | 不只挑慢 CPU 代表全部外部实现；不把拼接组合包装成原作者独立系统 |

**当前 PR 资格修正。** cuGraph SG 旧适配器的 `epsilon=1e-9/n` 与实装 C++ 的全图 L1 停止式不符；新的共同协议使用 `epsilon=1e-9`、FP64、2000 次上限，外部残差/质量门槛不变。旧 cuGraph PR 和包含它的16请求轨迹保留原始证据，但不得进入比较或胜率，必须新 epoch 重测。US 的旧非收敛不算性能胜利。这属于修正实验调用协议，不属于额外优化 baseline；多卡 cuGraph 已直接使用全局 `1e-9`，应按其独立资格与迭代上限记录，不能把 SG 修改机械套用到已有正确路线。

GUM 必须按真实执行路线标识：原生远程工作窃取、作者标准路径及 PR FP64 兼容变体分别资格和计时。当前标准路径的六图通过不能作为原生 RWS 可用的证明；原生路径在本机若因明确 P2P 限制不能运行，应列平台限制和实际失败阶段，不把它计为 EGGPU 胜出，也不悄悄换成标准路径仍沿用相同名称。

候选清单不等于全部已经跑通。最终图表只展示适用实验的系统，但所有候选的结论、源码与失败原因在[当前能力矩阵](BASELINE_CAPABILITY_MATRIX_20261104_CN.md)及其[机器可读记录](BASELINE_CAPABILITY_MATRIX_20261104.json)可查。只有语义不相交或实验问题不相交才排除；性能慢不构成排除理由。

### 状态复用基线要怎样进入 D/C 论证

内部消融回答“这个模块是否起作用”，外部共同任务回答“已有复用系统是否也能很好地完成任务”，两者都需要。E3 保留合理的复用加全量维护作为强替代；E4 保留共享组 LRU 和真实物理占用判断；再通过 E4-S 将 KGraph 与 C²graph-PM 等现成复用能力放进共同语义比较。不能拿只会完全重建的内部版本冒充所有已有状态复用研究。

## 5. 六项实验的具体实施与图表

### E1：整体性能、单卡竞争力与准备成本

七图固定为 **US、EP、LJ、OR、IN、UK、TW**，四函数 BFS、SSSP、WCC、PR。图类型统一 RN/SN/WG，主表可按 `US | EP LJ OR TW | IN UK` 分组；所有图表采用同一顺序。曲线上的数值型横轴才按数值排序，不把无序数据集名称连成一条暗示连续性的折线。

分别测 T_first 与 T_ready，BFS/SSSP 三源等权归并；另外测 `[BFS,SSSP,WCC,PR]×4` 的16请求静态任务，显示前缀1/4/16及完整累计曲线。首次任务用一个连续主计时器，从共同 host CSR ready 开始，包含系统专有准备、启动、上传直至完整 host 输出；布局时间与原生时间的分解和式不能代替该端到端主计时。准备后任务仍然重新计算并返回完整结果，重复 BFS 等状态须正确复位。C²graph-P 的三个主源分别独立初始化图，不能以 PM 的批时间折算共同单源延迟。具体边界沿用冻结协议。

**主表优先于几十组柱子。** 每行是算法×系统，列是七图，一卡和目标多卡资源分成明确表块。正文主表给完整时间，另一计时边界放正文紧凑配对表或 artifact 源表；不能只给加速比而无法恢复绝对量。统一单位可用秒，保留三位有效数字；毫秒级值以同单位小数显示，不能每格换单位。真正最快且正确的已测均值加粗；EGGPU 行可以浅灰底便于定位，但不预先加粗所有 EGGPU 数字。表注注明重复数、误差位置与缺失原因。

累计时间图：横轴 `Requests`，纵轴 `Cumulative time (s)`；最多选 US、IN、TW 三个预先固定代表面板，七图完整数据仍在 artifact。线表示真实已完成前缀，不拟合不存在的中间请求时间。首次与热调用各自分析，观察窗口内未交叉就报告未交叉。

**90% 目标的统计口径要先固定。** 单卡每种计时边界有28个图×算法计划格，三源不扩大分母；28格全部有合法共同结果时，至少26格领先才超过90%。T_first、T_ready 的胜率分别给出，不把冷、热、内核、多源和扩展函数重复混成一个更好看的百分比。若有未完成/未资格格，既报告“已配对格中的领先数/配对数”，也报告“领先数/28计划格”和完成覆盖率。没有外部合格结果的格子不能自动判胜；EGGPU 自己失败的计划格保留。均值领先与差异区间跨零的近似持平另作说明。

### E2：固定后端的强扩展

US、OR、IN、UK、TW，四算法，P=1/2/4/8；只有实际获授权且可用的资源点才能进入新测。图、权重、源和正确性要求固定，P 改变只按同一规则产生分区。EGGPU 的同一分区后端 P1 是强扩展分母，E1 的优化单卡路线可作为单独参考点，不代入 T1。

主图横轴 `GPUs`，纵轴 `Speedup`，定义 `T1/TP`，带理想线；四函数分别成小面板，数据集用固定颜色/标记。若曲线过密，主文展示预先指定的 US/IN/TW，OR/UK 放同一附图，不能按结果挑漂亮曲线。另给绝对 `Time (s)` 或表格，避免高比值掩盖低绝对性能。P1 无法完成时不造分母，显示最小可完成资源及绝对时间。

对 US 与 IN 的代表配置给 `Compute / Transfer / Sync / Other` 关键路径拆解，记录逐卡负载和通信轮数；异步重叠阶段不能简单相加。外部基线只有真实支持该卡数并满足质量时才进入对应曲线。资源扩展带来的容量提升和固定工作加速分别表述。

### E3：D-Maintain 的合法维护与代价

冻结主协议为 US/IN/OR、FP64 权重、`ρ={0,10⁻⁶,10⁻⁴,10⁻²,0.1,1}`，均匀/行聚集与旧读者 none/per_update 分面；ρ=0/1 不重复等价的局部性输入。每条轨迹20次提交、80次 `[BFS,SSSP,WCC,WPR]` 查询，另有相同的四次初始查询。当前 `state_v4` 的 E3 主比较仅 **Reuse+Full（D0C1）与 D-Maintain（D1C1）**，C固定且不施加容量预算；相同执行器、布局、所有权保护和初始状态。P1覆盖三图，P8复核IN/OR。D0的 `persistent_full` 复用未变拓扑，对新的未命中权重身份执行全量上传；ρ=0仍可命中，不能称为每次无条件重建整图。旧方案的独立Rebuild第三臂当前未实现，不绘制其结果。

两个主面板：横轴 `Updated edges (%)`（定义为变化存储位置占 A，且与无向配对规则一致），纵轴分别 `Workflow time (s)` 与 `Transferred data (GiB)`。ρ=0 是独立左端类别/小面板，不把0放进对数坐标；ρ>0使用明确对数刻度。每面板只画上述两种变体；局部性和 lease 用分面，不用十几种颜色混在一张图。

记录提交/哈希、主机复制、视图维护、真实 H2D/D2D、算法及输出；将 packet 载荷与实际总传输分开。当前精确 packet 限 SSSP，WPR 仍可能有主机差分成本，不能将后者画成同样的常数维护。ρ=0 的管理开销、ρ=1 的退化及旧读者引发的 COW 都是必要结果。无权PR权重不变性已有独立 `e3_unweighted_control` 入口与CPU测试，GPU资格及正式测量待完成。该控制在US/IN/OR上作base PR加20次权重提交后的PR，共21次真实调用，D0C1/D1C1、C固定、只允许none；现有weighted-view租约在纯无权PR轨迹不适用。它不改变80查询主trace，也不能在未测前写成已有控制结果。多卡分区集中轴同样仍缺明确输入协议，不能将普通行聚集冒称已覆盖。

### E4：C-Retain 的保留与状态复用

主机制实验使用 US/OR/IN/UK 固定池、32次访问和128查询，比较同一请求多集合的 grouped/interleaved/permuted 三种顺序；不更新，D固定为D1。当前 `state_v4` 主比较仅 **closure-LRU（D1C0）与 C-Retain（D1C1）**，都按真实物理占用和相同lease约束作决策。预算取冻结 β=0/0.25/0.5/0.75/1；每个当前源码/图池/GPU配置先独立运行实际物理字节pilot，以逐卡 `floor(B_min+β(B_all−B_min))` 冻结字节预算，禁止按速度选预算。旧稿的 No-retention、view-LRU、logical-size demand 未作为本轮已实现臂，不绘制它们的结果。

主图横轴 `Memory budget (GiB)`；多卡不均匀预算时使用 `Budget fraction` 并在表注给 β 与逐卡字节向量的定义，不能用所有卡显存相加掩盖某张卡溢出。纵轴 `Workflow time (s)`；第二面板使用已实际记录的 `New shared arrays (GiB)` 与 `Freed memory (GiB)`。新共享数组字节属于分配证据，不等于所有重载/H2D；当前没有全拷贝计数时，不能标成 `Reloaded data`。β=1全驻留与β=0.25的once无回访控制都保留；once为4次访问、16查询，单独展示，不和128查询主项混合。另用一个真实日志片段连接“撤销视图—实际释放数组—下一次重载”，帮助读者理解命中率相近为何仍可能不同。

**E4-S 共同状态复用副面板（本轮已纳入设计，待执行资格化）：** 在相同静态图上的不同源 SSSP 批任务，比较 EGGPU、KGraph、C²graph-PM，以及已接入的适用强静态系统。仍使用 US/OR/IN/UK，不增加新图；批长128为主点，32/512作敏感性，源点按冻结的拓扑散列规则扩展成不重复列表。所有请求都要返回全距离，不使用预存答案。允许作者已有 memoization，但其构建与有效性维护进入总时间；CPU 核数和主机内存上限一致，标明 CPU-only 与 CPU+GPU 的资源构成。横轴 `Queries`，纵轴 `Total time (s)`，准备与查询另作分解。该面板不计入 E1 单卡28格或90%目标的分母。

E4-S 是“已有状态复用系统如何完成共同任务”的比较，不是 C 的驱逐因果实验。KGraph 的主机结果缓存不能伪装成 GPU 闭包缓存；没有相同 GPU 驻留合同的系统不进入 β 曲线。若作者实现无法导出共同输出，应补正常输出接口并计时，不能直接将论文吞吐填入本机图。该副面板实际批长和运行状态以本轮 campaign 的正式记录为准，设计中未测的格子不得填历史数据。

### E5：D/C 联合干预

IN/UK/OR 固定交错轨迹、96请求；主点ρ=10⁻⁴、β=0.25，ρ=0和β=1为负控制。D0为Reuse+Full、C0为closure-LRU，比较D0C0、D1C0、D0C1、D1C1；其他开关和输入完全相同。

横轴只写四个组合，纵轴 `Workflow time (s)`；图例/图注解释D0/C0。采用配对重复，误差条表示同一统计合同的95%CI。下方画真实 `Weight H2D (GiB)`（full-weight + patch，dense-lease 子集不重复加）与 `COW D2D (GiB)`；当前未完整记录全部 CUDA 拷贝，不能生成 `Total H2D` 曲线或推断它为零。交互 `I=T11−T10−T01+T00` 单独给估计值和区间；时间越小越好，因此负交互表示超出加和的节省。

沿日志解释 C 是否保留了 D 后续可合法维护的对象。字节下降、时间下降和时间交互是三个不同结论。已有历史联合结果是维护载荷下降而完整时间交互区间跨零；新正式测量完成前不得将它改写成已证实的时间协同。

### E6：扩展函数为什么有用

EP-U、LJ-U 主任务：**全图 Effective Size + Constraint → 排名取32候选 → 候选精确 WF Closeness → 选一个节点 → 该点 BFS + 全图 WCC → 输出**。OR 保留预注册压力面板。候选和 winner 必须在本次计时内由前序结果产生，所有变体返回相同候选、winner、距离和连通划分。

按冻结 `e6_worker` 的四组 structural/closeness/bfs/wcc 分别做单组覆盖消融：完整EGGPU（none）加四种仅将其中一组回退的配置，共五个单卡EGGPU配置；不是两扩展组的2×2。另列三种已接入的外部共同任务路线，各自独立资格。11项tiny资格由四种卡数的完整EGGPU、四个单组回退和三个外部路线构成，不能当作11种正式性能结果。外部用其现成函数与正常图对象复用，真实交接/格式成本计入并单列解释。

主图横轴 `Configuration`（完整EGGPU、四个单组回退与合格外部路线），纵轴 `Application time (s)`；按 `Structural holes / Ranking / Closeness / BFS+WCC / Handoff` 堆叠。返回和初始准备如占比大，可明确增加段，不能藏到“其他”。第一次和热应用分面，每次重算候选；短表列最终决策一致性及各阶段是否完成。

多卡补图逐阶段注明结构洞为整图复制+ego分工、Closeness为整图复制+source分工、尾部既定GPU路线。它证明功能组合或并行工作分配的价值，不证明这些扩展函数都能用聚合显存容纳超单卡图。BC、LCC、K-core、SCC、MST 等未进入主任务的函数在能力矩阵保留，不借 E6 声称全部完成大图多卡验证。

## 6. 统一绘图规范

### 哪些信息用图，哪些用表

- **用表：** 硬件/版本、七图规模、完整算法×图×系统数值、功能与适用域、失败原因、精确正确性指标。读者需要查一个具体格子时用表。
- **用图：** GPU数、预算、更新比例、请求数等有顺序自变量的变化；时间组成；受控模块差异；配对交互区间。读者需要理解趋势或机制时用图。
- **不重复：** 同一完整数值矩阵不再画一遍密密麻麻的柱状图；主图展示趋势，artifact提供其源表。机制图不是实验图，系统框架图不承载性能数字。

### 颜色与几何

采用白底、黑/深灰轴与细边框。以参考图的浅蓝、浅绿、浅橙为视觉起点，下面色号是本文自己的统一设计，而不是声称从论文精确取样。

| 编码对象 | 颜色建议 | 兼容灰度的补充 |
|---|---|---|
| EGGPU 完整系统 | 深蓝 `#356A9A`，填充浅蓝 `#B8D3E7` | 实线、实心圆、清晰边框 |
| 主要外部系统 | 橙 `#C88646`、绿 `#6B8E67`、紫 `#8C79A6`、灰 `#8A8A8A` | 分别三角/方块/菱形/叉；柱形使用不同稀疏纹理 |
| 维护机制 D | 浅蓝 `#B8D3E7` | 固定斜线纹理 |
| 保留机制 C | 浅绿 `#BED4AA` | 固定点纹理 |
| 传输/复制代价 | 浅橙 `#E9BE96` | 稀疏反斜线 |
| 其他开销/未优化对照 | 浅灰 `#D3D3D3` | 空心或横线 |

系统比较图中颜色固定映射系统；时间拆解图中颜色固定映射阶段。两种图的图例明确，不能在同一张图让蓝色既代表 EGGPU 又代表传输。D/C 消融按四组合稳定编码，完整组合保持深蓝；颜色不足时优先分面，不无限加色。

出版宽度按约85mm单栏、175mm双栏准备（最终以实际模板为准）；最终缩放后标签约8–9pt、图例约8pt、线宽约1–1.3pt。输出矢量 PDF/SVG；PNG只作预览。零起点柱图不截断，时间跨多数量级可用明确的对数轴；避免双纵轴、3D、渐变、每根柱子写长数字。误差条与点不能被粗线遮挡，公共图例放整组上方，不重复占每个分面的面积。

### 坐标和标题

图内不放长标题，不写“Experimental Results of…”；面板标签只写 `(a) BFS`、`(b) SSSP` 或 `(a) IN`。横轴标题使用 `GPUs`、`Requests`、`Queries`、`Memory budget (GiB)`、`Updated edges (%)`；纵轴使用 `Time (s)`、`Speedup`、`Transferred data (GiB)`、`Peak memory (GiB)`。单位只在轴上放一次；算法、卡数、固定参数、首次/热调用定义放图注或面板分组，不塞进每一个刻度。

比值必须在图注写清分子分母；`Normalized time` 要指明归一化到哪个控制组，`Speedup` 要指明相对哪一条路线。外部最快系统可能随格子变化时，直接称“fastest applicable external baseline”，并保留对应身份表，不能给这条包络线取一个仿佛真实系统的名字。

## 7. 图注与表注模板

图注的顺序为：**回答什么问题/测什么 → 本图必要的固定条件 → 统计或归一化定义**。机制结论写正文，不在图注提前宣布全面胜出。下列模板中的变量均由真实记录填入。

- **E1表注：** “Complete-task time on seven graphs. Single-GPU and multi-GPU results use the same input and output-quality contract. Values are means of ten independent runs; dispersion and source-level results are provided in the artifact. Bold marks the lowest qualified mean in each case. OOM, TO, NS, and QF denote out-of-memory, timeout, unsupported semantics, and quality failure.” 实际还在待核资格时用 `Pending`，不能误用 QF。
- **E2图注：** “Strong scaling of the fixed partitioned backend. Speedup is relative to the same backend on one GPU; the dashed line denotes ideal scaling. Error bars show 95% confidence intervals.” 多条系统线各自以本系统P1为分母；不同系统间性能用绝对时间补充。
- **E3图注：** “Effect of update density on complete workflow time and transferred bytes. Full weight maintenance and dependency-aware maintenance share the executor and version/reader semantics; retention policy is fixed. Zero updates are shown separately from the logarithmic density axis.”
- **E4图注：** “Effect of retained-memory budget on workflow time and measured shared-array allocation/release. Policies process the same request sequence and enforce the same physical-memory accounting. The full-residency endpoint measures policy overhead when eviction is unnecessary.”
- **E5图注：** “Joint effect of dependency-aware maintenance and closure-aware retention under the same paired workload. D0 uses reuse with full maintenance; C0 uses closure-LRU. Interactions require ten source-bound, explicitly matched four-arm execution blocks; repeat indices alone do not establish pairing.”
- **E6图注：** “Contribution of extended function groups to the dependent application. Stacks include computation and cross-implementation handoff. Candidate selection and final outputs are recomputed in every run.”

状态缩写须在表注解释一次；只在确实未实现相同语义时用NS。OOM、索引范围不足、超时、构建失败、输出不合格和未完成资格分别记录。表里不使用一个破折号覆盖所有原因。

## 8. Setup 与 artifact 记录要求

论文 Setup 用一段紧凑正文加一张表，机器实采 JSON 是唯一来源。当前已核快照为 `setup.json` 的 `observed_utc=2026-11-04T12:49:49.958543+00:00`（北京时间20:49）：

| 已核项目 | 当前机器 |
|---|---|
| GPU | 8 × NVIDIA GeForce RTX 4090 D，每卡24,564 MiB显存 |
| CPU | 2 × Intel Xeon Platinum 8474C，每socket 48核、每核2线程；共96物理核、192逻辑线程 |
| 容器资源 | CPU配额等效128逻辑CPU时间；内存上限640 GiB。配额不等于固定128物理核绑定，实际作业线程/亲和性另存 |
| 主机内存 | `/proc/meminfo` 总量1,056,479,508 KiB，约1007.54 GiB；不能以此替代容器640 GiB上限 |
| 软件 | Ubuntu 22.04.5 LTS；NVIDIA驱动580.105.08；本机nvcc工具链13.0.88，各基线实际编译/运行环境仍逐项绑定 |
| 互联 | topo记录为PCIe NODE/PHB及跨NUMA SYS连接，未显示NVLink；实际P2P和通信路线须以运行探测为准 |

此次探测时8卡均为空闲，属于资源快照，不是整个实验过程的独占证明。以下字段继续随每次正式作业绑定：

| 类别 | 本轮必须保存的字段 |
|---|---|
| CPU/内存 | 型号、socket/物理核/线程、实际开放核数、NUMA绑定、物理RAM、作业前MemAvailable、内存上限 |
| GPU | 型号、卡数、每卡显存、物理与逻辑编号映射、UUID、驱动、功耗/时钟策略；报告中无需公开不必要的机器身份 |
| 互联 | `nvidia-smi topo -m`、PCIe/NVLink、实际P2P能力、NCCL/MPI通信路径；不能由卡型猜测 |
| 软件 | OS/内核、编译器、CUDA runtime/toolkit、MPI/NCCL、Python、cuGraph和各基线版本、编译选项、环境路径/锁定清单 |
| 代码 | EGGPU统一release/commit、脏树差异哈希、DSO哈希、调度能力策略；各基线作者commit与本地兼容补丁 |
| 输入 | 来源URL、原文件校验和、规范化规则与计数、ID映射、方向/投影、权重/源点/分区/轨迹哈希 |
| 运行 | 完整命令、环境差异、种子、计时边界、十次原始记录、作业顺序、开始/结束、退出码、资源快照、正确性结果 |

GPU实验串行调度，作业前核实空闲显存和真实主机内存余量；不抢占其他用户任务。若换机器，单独记录campaign和硬件，不将不同平台的均值拼成同一主表。所需资源点暂不可用时保留未完成状态，不能以历史数字填补“最新”表。

artifact 至少提供：环境与构建、七图获取和校验、确定性规范化、资格检查、E1–E6逐项运行命令、强基线适用矩阵、原始结果到图表的脚本、预计资源与时间、失败状态解释。测试和小图 smoke 路径应与正式任务共享同一入口，但 smoke 通过不能冒充主实验完成。不在仓库放凭据、大数据集或不能再分发的第三方内容；第三方源码按许可与固定commit获取。

**输出管理统一为当前成功版本和上一个成功版本。** 每个实验/图表拥有稳定路径与唯一源绑定，重复生成同内容不再另存日期副本；新运行只有完成正确性和数据完整性检查后才能替换当前入口。原始十次重复属于一次运行的证据，不能为了“只保留最新”压成一个均值；历史引用仍需保留的证据用内容寻址或明确存档，不继续作为当前图表入口。未完成新campaign不能导致最后一套可追溯结果被误删；所有旧结果即使保留也必须标明历史身份。

最终论文主张由新结果决定：单卡领先目标指导优化，落后格子指导瓶颈分析；数据集、基线、源点、质量和分母不能随胜负改变。尚未达到目标时继续优化或如实缩小结论，不通过删负例、混源码、换计时边界或使用不合格输出来满足百分比。
# 显存容量边界补充（2026-11-05）

显存不足是独立的可执行性结果，必须保留并展示。E1 的时间比较和容量面板回答不同问题：前者只统计完成且通过数值验证的完整任务；后者说明固定七图从小到大时，各系统在哪一阶段仍能完成、何时达到容量边界。容量证据可以来自本轮源码/输入绑定的资格运行，但必须标明该阶段，不能把一次资格耗时当正式均值。

E1 增加一张紧凑的容量矩阵或表，按规范化存储 arcs 数递增排列固定七图。行是适用的系统与函数，列是图；单卡与八卡分组。单元格用 Complete、OOM、Timeout、Index limit、Failed 和 Pending 等互斥状态；OOM 用浅红色与斜线同时编码，避免只凭颜色辨认。时间表中的 OOM 不填零，不产生无穷大加速比，也不补成一次胜利。失败必须绑定原始结果、源码 epoch 和完整异常日志，并注明发生于 graph construction、upload、workspace allocation 或 execution 等实际阶段；没有足够证据时保留 Failed，不把一般 GPU error 判成 OOM。

显存列仅填真实测量。完整周期采样的最大值称为 **Sampled peak VRAM (GiB)**；分配器提供的高水位另写 **Allocator peak (GiB)**。二者不混用，未采集填“—”。多卡同时报告最繁忙卡的峰值与设备数，必要时在补充表列每卡值；累计显存和最大单卡显存不能互相代替。发生分配失败不意味着已用显存恰好等于卡容量，不将 OOM 点画在任意的容量线处。

E4 的 β 是 **managed retention budget** 的归一化预算，不是人为修改24GiB物理显存上限。不能将预算下的状态管理收益写成解决单图物理显存不足。E4 另用工作集压力面板解释 C-Retain：所有单次任务在最小必要工作集可容纳的条件下，多个图和多个函数的派生状态合计超过驻留预算，固定 US/OR/IN/UK 池与调用顺序，扫描事前定义的预算比例。主轴保持 **Budget fraction** 和 **Workflow time (s)**；配套表列真实每卡预算、完整任务完成状态、驱逐/重新创建的派生状态字节和实测显存。不要把“新共享数组分配字节”改名为全部重载流量。低于物理字节 pilot 给出的最小工作集的配置是 Infeasible，不通过隐藏所需内存或改变图来强行完成。

两类边界必须分开解释：**单个任务的工作集能否装入显存**，以及 **多图/多函数状态不能同时驻留时能否完成连续任务**。C-Retain 针对后者，不宣称能够处理任意单图的 out-of-core 情况；D-Maintain 维护版本所需的复制峰值也需要实际测量。E6 的多卡实现复制整图、分摊 ego/source 工作，不能用来证明分区系统的聚合显存容量。外部基线只纳入其实际支持的资源和任务边界，OOM 既不隐去，也不作为改变数据集或绕过正确性门禁的理由。

建议容量图注： “Execution feasibility on the fixed graph suite, ordered by normalized stored arcs. OOM denotes an observed device-memory allocation failure in the source-bound execution; the failed stage is reported in the accompanying table. Qualification failures establish feasibility boundaries and are excluded from formal time averages. Sampled peaks are reported only when the complete sampling record is available.”
