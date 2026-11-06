# Related Work：按优化前提组织的正文与参考

本稿提供两部分的行文脉络、可继续修改的中文正文，以及每部分十篇 CCF A/B 文献。**二十篇是供取舍的文献池，正文按问题组织，不必逐篇点名。** 正文中的 `[1.1–1.4]` 等编号对应本节参考表，定稿时可替换为正式引用。cuGraph 等软件背景单列。

## 1. 整体故事

**先解释一个图任务怎样执行得更快，再解释连续请求之间什么可以复用，最后落到共享状态在有限显存下怎样维护和释放。**

两部分分别回答两个问题：第一部分关注执行和数据供给，第二部分关注状态的有效性与保留。它们通过同一个具体场景相接：对图 $G$ 提交权重更新后，无权遍历仍可使用已有拓扑，最短路径需要新权重；旧请求可能仍持有旧版本，随后到来的图 $H$ 又要求腾出空间。此时，系统既要判断哪些内容仍然有效，也要判断哪些存储真正能够释放。

“按假设分组”在这里指梳理各类工作的主要优化前提与决策范围。写作顺序采用 **前提 → 已解决的问题 → 连续请求引出的下一层问题 → 本文定位**，无需将这些前提断言为每篇论文的共同限制。建议保留 **GPU-Accelerated Graph Processing** 和 **Graph State Maintenance and Reuse** 两个正式标题；两部分共约半页至四分之三页，最终随模板调整。下面的提纲和参考表用于组织材料，只有“正文草稿”进入论文。

## 2. 第一部分：GPU-Accelerated Graph Processing

### 行文脉络

| 论证台阶 | 优化前提 | 这一段怎样讲 | 怎样过渡 |
|---|---|---|---|
| 给定任务的执行 | 图表示和分析任务已确定，主要瓶颈来自不规则并行、访存和工作分配 | 将 frontier、负载均衡、异步通信与远程工作窃取放在同一条执行主线上；通用接口作为应用入口补充 | 计算组织之外，设备容量和主机—设备带宽还决定数据能否及时到达 |
| 受容量约束的数据供给 | 活跃集合及访问方式能够指导数据移动 | 围绕按需访问、活跃子图、混合传输与 CPU–GPU 协作，说明“何时移动哪些数据” | 多次迭代、查询和更新还会重复使用相同数据 |
| 持续工作中的复用 | 查询间或更新前后存在可利用的共同工作 | 用 C²graph 的多查询协作、Grapin 的增量计算与热子图管理承接前文 | 函数、版本和图回访共同变化时，还需识别输入依赖与共享分配的持有关系 |

### 正文草稿

GPU 图处理研究通常以给定的图表示和分析任务为优化单元，重点提高不规则计算的并行效率。基于活跃顶点或边的执行抽象、邻接工作负载均衡和高效访存，使不同图算法能够利用设备并行性；异步通信和远程工作窃取进一步缓解多 GPU 分区间的负载不均。[1.1–1.4] cuGraph 及其高层接口则将 GPU 算法提供给现有分析生态，支持应用组合多个分析函数。[软件 S1] 当图数据超过设备容量时，优化重点转向数据供给：按需访问主机数据、提取活跃子图、选择显式或隐式传输，以及 CPU–GPU 协作，分别减少无效传输和设备等待。[1.5–1.8]

持续处理中的复用已成为这一方向的重要组成部分。驻留子图可以跨迭代使用，多查询可以共享压缩表示并协调异构资源，流式图处理也能结合增量求解与热子图管理减少重复访问。[1.8–1.10] 跨函数、版本和图回访的连续请求，使输入状态的有效性与物理共享同时进入决策。已有执行与传输机制为这些请求提供计算基础，EGGPU 则围绕内容依赖和物理持有关系，决定当前请求需要维护什么，以及完成后能够保留或释放什么。

### 十篇文献及各自的写作用途

| 编号 | 论文、出版位置与 CCF | 用来支撑哪一句 | 必须保留的边界 |
|---|---|---|---|
| 1.1 | **Gunrock** — [Gunrock: A High-Performance Graph Processing Library on the GPU](https://escholarship.org/uc/item/6xz7z9k0)；PPoPP 2016，**A**；[DOI](https://doi.org/10.1145/2851141.2851145) | frontier、advance/filter及负载均衡，提供高层GPU图执行抽象。 给定输入后的执行效率；不是整个连续会话状态策略。 | 原VLDB条目是TOPC2017，当前十篇明确选PPoPP2016，不挪用等级。 |
| 1.2 | **Groute** — [Groute: An Asynchronous Multi-GPU Programming Model for Irregular Computations](https://cs.rochester.edu/~sree/papers/groute-ppopp17.pdf)；PPoPP 2017，**A**；[DOI](https://doi.org/10.1145/3018743.3018756) | 异步通信、分布式工作列表与collective支撑不规则多GPU计算。 从单卡并行转向多卡协调；可讨论执行中消息/工作管理。 | 不得把异步多GPU运行时写成跨图版本维护或任意图缓存策略。 |
| 1.3 | **GUM** — [Efficient Multi-GPU Graph Processing with Remote Work Stealing](https://graphscope.io/publication/gum.pdf)；ICDE 2023，**A**；[DOI](https://doi.org/10.1109/ICDE55515.2023.00022) | 用远程工作窃取缓解多GPU图执行中的负载不均和长尾。 引出给定任务执行中的利用率问题与本机多卡基线。 | ICDE2023正式研究论文；作者属于工业界不等于Industry Track。NVLink结果不代表本机PCIe的RWS能力。 |
| 1.4 | **SYgraph** — [SYgraph: A Portable Heterogeneous Graph Analytics Framework for GPUs](https://antonio-decaro.github.io/assets/pdf/ICPP25.pdf)；ICPP 2025，**B**；[DOI](https://doi.org/10.1145/3754598.3754615) | SYCL框架、两层bitmap及负载平衡，兼顾跨厂商可移植性。 为通用接口及设备执行抽象提供较新证据。 | 使用ICPP2025正式10页论文；不使用IWOCL海报凑A/B，也不把future multi-GPU当已实现。 |
| 1.5 | **EMOGI** — [EMOGI: Efficient Memory-access for Out-of-memory Graph-traversal in GPUs](https://www.vldb.org/pvldb/vol14/p114-min.pdf)；PVLDB 14(2):114–127 / VLDB 2021，**A**；[DOI](https://doi.org/10.14778/3425879.3425883) | 合并并对齐直接访问主机图数据的cache-line请求，提高PCIe带宽利用。 设备容量不足时的数据供给路径，不等于所有GPU状态始终驻留。 | PDF Reference Format写2021，DOI出版元数据2020-10，分别记录；不把两种年份算两篇。 |
| 1.6 | **Subway** — [Subway: Minimizing Data Transfer during Out-of-GPU-Memory Graph Processing](https://www.cs.ucr.edu/~anode001/eurosys20.pdf)；EuroSys 2020，**A**；[DOI](https://doi.org/10.1145/3342195.3387537) | 构建活跃子图并异步处理，减少超显存图中的无用传输。 与EMOGI并列说明显式压缩活跃集和隐式按需访问的取舍。 | 已有驻留分区重用；不能笼统写为所有旧系统每次都传整图。 |
| 1.7 | **HyTGraph** — [HyTGraph: GPU-Accelerated Graph Processing with Hybrid Transfer Management](https://arxiv.org/pdf/2208.14935)；ICDE 2023，**A**；[DOI](https://doi.org/10.1109/ICDE55515.2023.00049) | 按迭代工作负载在显式与隐式传输间选择，结合任务合并和贡献驱动调度。 输入相同仍需按运行时活跃性选择数据移动路径。 | 2022 arXiv与ICDE2023不是两篇；TPDS2025扩展版新增vertex-centric caching须承认。 |
| 1.8 | **CGgraph** — [CGgraph: An Ultra-fast Graph Processing System on Modern Commodity CPU-GPU Co-processor](https://www.vldb.org/pvldb/vol17/p1405-yuan.pdf)；PVLDB 17(6):1405–1417 / VLDB 2024，**A**；[DOI](https://doi.org/10.14778/3648160.3648179) | 提取可跨迭代复用的GPU子图，按需分配CPU–GPU工作。 展示数据驻留与异构协作已有结合，不能把重用子图本身称首次。 | 不是同名CGCGraph；CPU协作应保留且记录资源，不能人为关掉。 |
| 1.9 | **C²graph** — [C²graph: A Compression-Collaboration Algorithm for CPU-GPU Hybrid Weighted Graph Traversals](https://wzgcs.github.io/papers/32_ICDE_2026.pdf)；ICDE 2026，**A**；[DOI](https://doi.org/10.1109/ICDE65706.2026.00225) | 联合拓扑/边权压缩与消息补偿，再按多查询计算需求做CPU–GPU协作。 单查询压缩与同图不同源批任务分开，作为E1与E4-S不同角色。 | P 与 PM 模式分开；原文 runtime 不含压缩。本轮主表采用准备完成后的完整查询时间，首次执行的补充时间另含压缩，两种区间分别保留。 |
| 1.10 | **Grapin** — [Efficient Graph Data Access for Out-of-Memory GPU Streaming Graph Processing](https://www.vldb.org/pvldb/vol18/p3854-wang.pdf)；PVLDB 18(11):3854–3867 / VLDB 2025，**A**；[DOI](https://doi.org/10.14778/3749646.3749659) | GPU依赖驱动增量求解与细粒度热子图管理，减少动态长期计算的重复访问。 必须正面承认更新与GPU缓存已共同研究；区别落到多函数输入状态身份与物理释放合同。 | 它维护查询结果，我们D当前维护输入准备状态且算法完整重算；不能用“连续计算”本身声称新颖性。 |

上述文献可以按“4 + 4 + 2”选引，正文无需严格按表格顺序点名。特别补充：HyTGraph 的 [TPDS 2025 扩展版](https://neuzhangyf.github.io/assets/pdf/paper-2025-tpds-towards.pdf) 已加入 vertex-centric graph caching；当前十篇以用户指定的 ICDE 2023 版本计一篇，扩展版作为能力和新颖性补充，不能说 HyTGraph 从未考虑缓存。它也明确将多 GPU 扩展列为未来工作，不能因此给它补一条没有作者实现的多卡曲线。

## 3. 第二部分：Graph State Maintenance and Reuse

### 行文脉络

| 论证台阶 | 优化前提 | 这一段怎样讲 | 怎样过渡 |
|---|---|---|---|
| 局部变化与存储组织 | 更新代价与变化传播到的存储范围有关 | 从动态邻接布局、局部重建和顺序扫描讲起，说明减少数据搬移的已有办法 | 更新更便宜之后，仍需界定新旧读者分别可使用哪些状态 |
| 版本与依赖界定复用 | 版本、计算依赖或共享索引的身份能够说明复用是否合法 | 区分图快照、计算结果片段和共享索引；承认已有多读者生命周期管理 | 合法状态可能多于可用内存，必须继续判断保留价值 |
| 有限容量下的保留 | 后续需求与恢复工作影响缓存价值 | 先讲依赖感知缓存和整体缓存规划，再引入视图共享物理数组的情况 | 对象的保留价值与撤销后实际释放的字节数，需要分别判断并共同决策 |

### 正文草稿

当更新成本主要来自存储布局的调整时，动态图系统通过分层邻接容器、局部重建和顺序扫描，在更新效率与读取局部性之间作出权衡。[2.1–2.3] 例如，同一批逻辑更新在不同布局中可能引起不同范围的数据搬移，限制重建范围因而能够减少后续分析的准备工作。EGGPU 关注这一问题在 GPU 输入状态中的延伸：固定拓扑的权重变化可能影响数值视图及其派生状态，而与权重无关的拓扑表示仍可继续使用。

版本和依赖信息为进一步复用提供了依据。不可变快照和稳定视图使更新与分析可以并存；计算依赖支持对已有结果进行增量修正，查询间 memoization 复用局部计算，共享索引则将同一物理状态提供给多个查询。[2.4–2.8] 这些工作分别明确了图存储、计算结果或索引的有效性与生命周期。EGGPU 在此基础上管理分析函数所需的输入状态：将内容版本、表示布局和属性依赖对应到物理分配，在满足旧读者访问要求的前提下选择复用、修复或复制；分析算子随后仍生成完整结果。

有限内存使合法复用进一步成为保留决策。依赖感知缓存利用消费者需求判断对象价值，整体缓存规划还将重新计算与恢复成本纳入选择。[2.9–2.10] GPU 图视图之间的物理共享又带来一个具体约束：两个版本共用拓扑数组时，撤销其中一个视图并不会释放仍被另一个版本持有的拓扑。因此，面向视图的价值排序还需要与分配层面的实际释放量对应。EGGPU 通过 D-Maintain 维护当前请求的合法输入，通过 C-Retain 在请求完成后依据完整持有关系检查可释放空间，并结合历史需求选择保留状态。两者的联系在于，维护决定哪些分配继续共享，保留决定后续请求需要恢复哪些状态。

### 十篇文献及各自的写作用途

| 编号 | 论文、出版位置与 CCF | 用来支撑哪一句 | 必须保留的边界 |
|---|---|---|---|
| 2.1 | **GRACE** — [GRACE: Alleviating Reconstruction Cost in Dynamic Graph Processing Systems](https://me.sage.org.ai/contents/research_papers/2026/2026_grace_icde_2026.pdf)；ICDE 2026，**A**；[DOI](https://doi.org/10.1109/ICDE65706.2026.00075) | 针对PMA/CSR动态图重建，依据图结构安排空间与迁移，借助cousin区间限制重平衡；以插件形式接入PPCSR、Terrace、VCSR。 与D共同关心变化传播到物理布局后的准备成本；它优化拓扑插入造成的布局重建，我们当前核心是固定拓扑下FP64权重版本、函数依赖视图的维护及GPU驻留。 | 不能把GRACE说成只优化计算、不处理更新；也不能宣称我们已覆盖其拓扑插删能力。 |
| 2.2 | **Terrace** — [Terrace: A Hierarchical Graph Container for Skewed Dynamic Graphs](https://people.eecs.berkeley.edu/~aydin/terrace.pdf)；SIGMOD 2021，**A**；[DOI](https://doi.org/10.1145/3448016.3457313) | 低度邻居内联数组，中度邻居共享PMA，高度顶点独立B树，适应偏斜度分布，兼顾更新成本与扫描局部性。 提供结构变化下如何组织邻接数据的强先例；我们不提出替代动态图容器，关注分析函数与版本之间的GPU准备状态和物理对象保留。 | 年份是2021，不能写SIGMOD2019；不能把其cache locality等同于跨请求cache-retention策略。 |
| 2.3 | **LiveGraph** — [LiveGraph: A Transactional Graph Storage System with Purely Sequential Adjacency List Scans](https://www.vldb.org/pvldb/vol13/p1020-zhu.pdf)；PVLDB 13(7):1020–1034 / VLDB 2020，**A**；[DOI](https://doi.org/10.14778/3384345.3384351) | Transactional Edge Log及其并发控制让事务更新下的邻接扫描保持顺序访问；按epoch可见性压缩/回收，旧读者继续访问旧块。 必须承认其已有版本可见性和旧读者保护。我们的lease/COW属于相关基础技术上的GPU视图维护，不是事务存储或MVCC的首创。 | 不可说旧系统不保护旧读者、不复用版本。 |
| 2.4 | **Aspen** — [Low-Latency Graph Streaming using Compressed Purely-Functional Trees](https://jshun.csail.mit.edu/aspen.pdf)；PLDI 2019，**A**；[DOI](https://doi.org/10.1145/3314221.3314598) | 压缩纯函数C-tree支持不可变快照、并发查询/更新；acquire/set/release管理版本，引用计数回收；flat snapshot支持全图算法，并明确可跨多次算法摊销构建成本。 与D的版本隔离、共享状态和多函数复用直接相关。我们的区别应具体落实到GPU布局/数值视图的合法维护和受预算限制的物理释放判断。 | 不能称首次跨算法复用快照、首次引用计数或首次旧版本安全。论文当时实现未支持weighted edges，勿混方法可扩展性与实际能力。 |
| 2.5 | **GraphOne** — [GraphOne: A Data Store for Real-time Analytics on Evolving Graphs](https://www.usenix.org/system/files/fast19-kumar.pdf)；FAST 2019，**A** | 结合edge-log与adjacency store，双粒度版本生成GraphView，支持批处理和流式访问并分离更新与读取；小量数据重叠维持视图组成，长分析可用私有未归档边副本。 已有多表示和稳定读取视图管理；我们把GPU设备/布局/权重依赖及共享物理释放作为运行时决策对象。 | 不能说前作只有单一表示或无法稳定读取变化图；其主张不是我们的GPU跨图留存算法。 |
| 2.6 | **GraphBolt** — [GraphBolt: Dependency-Driven Synchronous Processing of Streaming Graphs](https://www.cs.sfu.ca/~keval/contents/papers/graphbolt-eurosys19.pdf)；EuroSys 2019，**A**；[DOI](https://doi.org/10.1145/3302424.3303974) | 跟踪跨迭代顶点值依赖，以聚合值压缩依赖信息；图变化后逐迭代修正结果，并保留BSP语义。 这是计算结果的依赖驱动增量维护；我们的D主要维护输入权重/布局等准备状态，当前算法仍完整执行。两种依赖对象必须区分。 | 不能宣称我们首创dependency-aware graph maintenance，不能把D等同GraphBolt式增量求解。 |
| 2.7 | **KGraph** — [An Efficient Memoization Engine for Concurrent Graph Query Processing](https://gaosen.org/resources/papers/KGraph.pdf)；ICDE 2025，**A**；[DOI](https://doi.org/10.1109/ICDE65448.2025.00081) | 按并发查询和图分区选择并复用局部计算结果，权衡memoization准备、命中和存储。 是共同多源SSSP任务的直接外部复用对照；其CPU结果片段不是本文GPU输入执行视图。 | PPR随机游走不能代全局PR；不把论文排除磁盘和分区的时间直接填本轮T_first。 |
| 2.8 | **Shared Arrangements** — [Shared Arrangements: practical inter-query sharing for streaming dataflows](https://www.vldb.org/pvldb/vol13/p1793-mcsherry.pdf)；PVLDB 13(10):1793–1806 / VLDB 2020，**A**；[DOI](https://doi.org/10.14778/3401960.3401974) | 跨查询共享分片、多版本索引；引用计数和读者frontier协调维护、压缩及生命周期。 物理共享和多读者生命周期的直接先例；本文需证明GPU图版本、设备布局及逐卡释放闭包的具体增益。 | 不能称首次共享物理状态、首次旧读者保护、首次引用计数；其图实验不等于本文跨图有预算淘汰。 |
| 2.9 | **LRC** — [LRC: Dependency-Aware Cache Management for Data Analytics Clusters](https://home.cse.ust.hk/~weiwa/papers/lrc-infocom17.pdf)；INFOCOM 2017，**A**；[DOI](https://doi.org/10.1109/INFOCOM.2017.8057007) | 用尚未物化的依赖子块数量作为reference count，优先移除依赖需求少的块；LRC-Online随job到达更新应用DAG，承认未来顺序不能完全已知。 必须承认依赖感知缓存已有成熟研究。其reference count是计算消费者计数；我们的物理owner计数判断能否释放，需求分数来自近期请求，三者不可混写。 | 不是NSDI2018；不可说依赖缓存必须预知完整未来，LRC已有在线版本。 |
| 2.10 | **Blaze** — [Blaze: Holistic Caching for Iterative Data Processing](https://wonook.github.io/papers/eurosys24-song-blaze.pdf)；EuroSys 2024，**A**；[DOI](https://doi.org/10.1145/3627703.3629558) | 以CostLineage合并跨job重复RDD身份，动态估计重算/磁盘恢复时间；用ILP联合决定分区留内存、写盘或丢弃，并自动识别未来引用。 是C最需要正面承认的强先例。我们当前C是历史需求/实际可释放字节启发式，重点在GPU图版本和布局共享数组的释放闭包，不是更完整的恢复时间模型或通用最优缓存。 | 不能称首次联合缓存/回收/恢复，不能说前作忽略依赖或重复对象，也不能把Blaze的RDD身份合并与任意底层物理alias视作完全相同。 |

这一部分按“3 + 5 + 2”组织。KGraph 与 Shared Arrangements 比泛缓存材料更直接，因此进入十篇主池；原先核查过的 PACMan 可作为“命中率不等于任务完成时间”的补充，[Teseo](https://www.vldb.org/pvldb/vol14/p1053-leo.pdf) 可作为动态事务存储补充，均不挤占这二十篇。若使用 Teseo 的性能叙述，还必须同时核对[官方勘误](https://www.vldb.org/pvldb/vol14/p2271-leo.pdf)。

## 4. 软件背景与原 VLDB 稿的衔接（编者说明）

原稿确实引用的是 **nx-cugraph 官方文档**，不是一篇题为 cuGraph 的 CCF 论文。第一部分介绍通用算法接口时可以加一句：

> GPU 图分析库及其高层接口将加速算法提供给现有分析生态，例如 cuGraph 及其 NetworkX 后端 nx-cugraph，使应用能够在熟悉的调用方式下使用设备执行能力。

软件引用 S1 使用 [nx-cugraph 官方文档](https://docs.nvidia.com/cugraph/latest/nx_cugraph/)；具体版本和实际支持函数以本轮能力矩阵与运行环境为准。软件文档不计入十篇。原稿中的 NetworkX、EasyGraph、igraph 可以继续承担接口和功能背景，不为凑 CCF 数量更改出版身份。原稿 Gunrock 条目是 TOPC 2017 长文，当前表中明确选了 PPoPP 2016；可以二选一或同时引用，但不能把 PPoPP 等级贴在另一条期刊记录上。

原 VLDB 已有 CGgraph、SYgraph、EMOGI、HyTGraph、Grapin、Shared Arrangements 等条目，本稿可以继承其文献身份并按新主线重组。GUM、GRACE、C²graph 和 KGraph 是本轮补入的直接对照／定位材料；不要说它们都已出现在原稿。原稿引用审计的来源和关键映射保存在 [source_audit.json](references_20261105/source_audit.json)。

## 5. 引用等级与年份（编者说明）

本清单严格列出 **10 + 10 篇**：第一部分 9 篇 A、1 篇 B，第二部分 10 篇 A；等级指对应会议，不评价单篇论文质量。SYgraph 使用 ICPP 2025 正式论文，未纳入海报、Workshop 或 Demo；PVLDB 研究论文按相应 VLDB 会议周期归类，不把 PVLDB 偷换成 The VLDB Journal。

本次核对使用 CCF 2026 第七版原目录的[学者镜像 PDF](https://dengpingfan.github.io/papers/CCF-Recommend-2026.pdf)，实际检查物理第5、7、13、27、35页。CCF [官方目录入口](https://www.ccf.org.cn/Academic_Evaluation/By_category/)与[原始 PDF 地址](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=112CF3BF7E1140ACEB271ADAED12A67ADFABB8FF099E40C2759502A85C8A281F)本次返回405，未声称官网直接下载成功；镜像的字节、哈希和逐会议等级见 [ccf_audit.json](references_20261105/ccf_audit.json)。

EMOGI 的 PDF 推荐引用格式为 PVLDB14(2):114–127,2021，而 DOI 出版元数据为2020年10月；机器记录保留两者，最终 BibTeX 选择一种一致的出版口径，不同时计数。HyTGraph 的 arXiv 2022、ICDE 2023 与 TPDS 2025 扩展版也分别标识。

## 6. 收束句与实现范围（编者说明）

两部分可以用这一句收束：**已有研究提高了图执行、更新和复用的效率；EGGPU 将连续请求的内容依赖与物理持有关系结合起来，研究当前输入的合法维护和后续状态保留如何共同影响完整任务。**

正文统一使用“输入状态”“内容依赖”“物理持有关系”和“实际释放量”。D-Maintain 当前核心实现覆盖固定拓扑下的 FP64 权重及相关输入视图维护，分析算子仍完整执行；C-Retain 使用历史需求与实际可释放字节的启发式，预算作用于请求完成边界的受管驻留量。任意拓扑插删、增量答案求解、精确恢复时间预测和全局最优缓存规划不属于这里的实现主张。

Grapin 的更新与缓存、Aspen 的快照复用、Shared Arrangements 的共享索引和 Blaze 的整体缓存规划分别构成直接先例。论文的区分应由具体状态对象、合法决策范围和受控实验支撑。相关工作部分用于定位研究问题；E1–E6 的性能与适用边界留给正式实验结果回答，当前不填未经完成的领先比例或时间协同结论。

文献元数据和逐篇用途见 [papers.json](references_20261105/papers.json)；实验比较角色和图表方案见 [实验章节设计](PAPER_EXPERIMENT_DESIGN_20261104_CN.md)。
