# Related Work：两部分脉络、正文骨架与各十篇参考

更新：2026-11-05。下面先给“怎么讲”，再给可改写的中文正文和每部分十篇文献。**二十篇是供取舍的文献池，不要求正文逐篇点名。** cuGraph 等原 VLDB 软件背景另列，不占 CCF A/B 论文数量；这里没有新的性能结论。

## 1. 整体故事

**给定图任务如何高效执行 → 变化和重复请求中哪些状态仍可使用 → 有限显存下哪些状态应保留，以及撤销后到底释放多少。**

“按假设分组”应理解为归纳各类工作的主要优化前提，而不是断言所有论文都明写了同一个假设。第一部分回答“执行与数据供给”，第二部分回答“状态何时合法、保留什么”。两部分交汇到本文：不同函数、版本和设备布局所需的输入执行状态既有关联，也可能共享物理数组，维护与保留因而不能只按算法名或逻辑对象大小分别处理。

建议仍使用 **GPU-Accelerated Graph Processing** 和 **Graph State Maintenance and Reuse** 两个标题。每部分最终约两个自然段，总计先留半页至四分之三页；以下分组是写作提纲，不必在论文中全部变成小标题。先承认前作解决了什么，再用具体对象和边界引出本文，不写“现有系统都只执行一次”“以前没有共享状态”等过宽判断。

## 2. 第一部分：GPU-Accelerated Graph Processing

### 怎么把三层意思接起来

| 论证台阶 | 归纳的优化前提 | 这一层应讲什么 | 接到下一层的句意 |
|---|---|---|---|
| ① 给定输入后的执行 | 主要瓶颈在不规则并行、访存和设备间工作分配 | 用 frontier、负载平衡、异步通信、远程工作窃取概括；Gunrock/SYgraph 是接口与设备执行，Groute/GUM 是多卡协调 | 即使计算组织得好，容量和数据供给仍会使 GPU 等待 |
| ② 容量与数据供给 | 图数据可放在主机，但有限 GPU 容量和主机—设备带宽限制吞吐 | EMOGI 的按需访问、Subway 的活跃子图、HyTGraph 的混合传输、CGgraph 的驻留子图与异构协作，围绕“何时移动哪些数据”一起讲 | 数据运动也会在持续迭代、更新和多查询之间重复 |
| ③ 持续工作中的重复 | 多个查询或连续更新提供可利用的重用机会 | C²graph 连接压缩与多查询协作，Grapin 连接增量计算和热子图缓存；承认它们已跨越单次静态执行 | 本文进一步问：当函数、属性版本与图回访同时变化，如何维护合法输入状态并管理共享物理占用 |

### 大概正文

GPU 图处理研究首先关注给定图和分析任务后的执行效率。通过围绕活跃顶点或边组织计算、平衡不同规模的邻接工作并改进数据访问，图框架使不规则算法能够利用设备并行性；多 GPU 系统进一步通过异步通信和动态工作协调缓解分区之间的负载不均。通用软件库也将 GPU 算法暴露为可组合的分析接口，使已有应用能够使用这些执行能力。当图数据超过设备容量时，优化重点则转向数据供给：根据活跃集合压缩传输、按需访问主机存储、选择不同传输路径，或让 CPU 与 GPU 协作，减少带宽等待和重复搬移。

这些工作已经开始利用持续处理中的重复性。可驻留子图可以跨迭代复用，多查询可以共享压缩后的表示并协调异构资源，流式图系统也能结合增量计算和细粒度缓存减少更新后的访问。因此，“支持连续任务”本身不足以区分本文。我们关注的是连续分析中输入状态的另一组关联：一个属性更新可能只使部分表示失效，新的版本可能继续共享未变拓扑，而撤销某个函数或版本的视图也未必释放它所引用的全部存储。本文在已有高效执行能力之上，研究这些版本、函数依赖和物理持有关系如何共同影响合法维护、状态保留及完整任务的成本。

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
| 1.9 | **C²graph** — [C²graph: A Compression-Collaboration Algorithm for CPU-GPU Hybrid Weighted Graph Traversals](https://wzgcs.github.io/papers/32_ICDE_2026.pdf)；ICDE 2026，**A**；[DOI](https://doi.org/10.1109/ICDE65706.2026.00225) | 联合拓扑/边权压缩与消息补偿，再按多查询计算需求做CPU–GPU协作。 单查询压缩与同图不同源批任务分开，作为E1与E4-S不同角色。 | P与PM模式分开；论文runtime不含压缩，本轮完整任务含压缩，不混计时。 |
| 1.10 | **Grapin** — [Efficient Graph Data Access for Out-of-Memory GPU Streaming Graph Processing](https://www.vldb.org/pvldb/vol18/p3854-wang.pdf)；PVLDB 18(11):3854–3867 / VLDB 2025，**A**；[DOI](https://doi.org/10.14778/3749646.3749659) | GPU依赖驱动增量求解与细粒度热子图管理，减少动态长期计算的重复访问。 必须正面承认更新与GPU缓存已共同研究；区别落到多函数输入状态身份与物理释放合同。 | 它维护查询结果，我们D当前维护输入准备状态且算法完整重算；不能用“连续计算”本身声称新颖性。 |

上述文献可以按“4 + 4 + 2”选引，正文无需严格按表格顺序点名。特别补充：HyTGraph 的 [TPDS 2025 扩展版](https://neuzhangyf.github.io/assets/pdf/paper-2025-tpds-towards.pdf) 已加入 vertex-centric graph caching；当前十篇以用户指定的 ICDE 2023 版本计一篇，扩展版作为能力和新颖性补充，不能说 HyTGraph 从未考虑缓存。它也明确将多 GPU 扩展列为未来工作，不能因此给它补一条没有作者实现的多卡曲线。

## 3. 第二部分：Graph State Maintenance and Reuse

### 怎么把三层意思接起来

| 论证台阶 | 归纳的优化前提 | 这一层应讲什么 | 接到下一层的句意 |
|---|---|---|---|
| ① 局部变化与存储组织 | 可以通过布局和局部调整减少更新破坏的范围 | GRACE/Terrace/LiveGraph：动态邻接表示、重平衡、顺序扫描与并发更新的折中 | 低成本更新还必须保证读者看到合法版本 |
| ② 版本或依赖使复用合法 | 已有状态有明确身份、版本或依赖，就能安全服务后续工作 | Aspen/GraphOne 是图快照与视图，GraphBolt 是迭代值依赖，KGraph 是结果片段，Shared Arrangements 是共享索引；分清“复用什么” | 可复用不等于都能常驻，有限内存还要求判断保留价值 |
| ③ 有限内存与保留价值 | 对象价值由需求和恢复工作决定，最近访问时间并非充分信息 | LRC 的计算依赖消费者与 Blaze 的代价/lineage 已有成熟先例；需求预测与物理 owner 计数是不同概念 | GPU 视图共享数组时，名义大小不等于真实可释放量；保留版本还会改变下一次维护的可选路径 |

### 大概正文

图状态维护研究主要处理更新效率、读取局部性和一致性之间的折中。一类工作通过动态图容器和局部重建降低结构变化的代价；另一类工作使用持久快照、版本管理和稳定视图，让更新与分析能够并存。在这些基础上，增量图处理根据计算依赖修正已有结果，并发查询系统选择可复用的局部计算，而流式数据流系统进一步共享多个查询所需的索引状态。这里的复用对象从图存储、快照和视图延伸到中间结果与索引，已有工作也处理了共享状态的多读者生命周期。因此，版本隔离、依赖跟踪、引用计数和共享本身都不是本文的独立新颖性主张。

内存约束使合法复用进一步成为保留决策。依赖感知缓存利用尚未完成的消费者需求，整体缓存规划则综合复用身份、重新计算与恢复成本；这些研究已经表明，仅按最近访问或命中率优化并不总能缩短整个任务。本文关注这一问题在 GPU 图输入执行状态上的具体表现：不同函数和版本可能共同持有拓扑或属性数组，一个视图被撤销后，其共享数组仍可能被其他持有者占用；反过来，保留一个合法版本还可能减少下一次属性维护的工作。D-Maintain 根据内容、布局和函数依赖维护当前请求所需状态，C-Retain 根据全部物理持有关系计算真正可释放的空间，并结合历史需求决定请求结束后的保留。本文需要通过同一任务上的维护、回收、恢复和完整时间证据，说明二者结合在哪些条件下产生收益。

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

## 4. 原 VLDB 稿哪些内容继续保留

原稿确实引用的是 **nx-cugraph 官方文档**，不是一篇题为 cuGraph 的 CCF 论文。第一部分介绍通用算法接口时可以加一句：

> GPU 图分析库及其高层接口将加速算法提供给现有分析生态，例如 cuGraph 及其 NetworkX 后端 nx-cugraph，使应用能够在熟悉的调用方式下使用设备执行能力。

这句话使用 [nx-cugraph 官方文档](https://docs.nvidia.com/cugraph/latest/nx_cugraph/)；具体版本和实际支持函数以本轮能力矩阵与运行环境为准。软件文档不计入十篇。原稿中的 NetworkX、EasyGraph、igraph 可以继续承担接口和功能背景，不为凑 CCF 数量更改出版身份。原稿 Gunrock 条目是 TOPC 2017 长文，当前表中明确选了 PPoPP 2016；可以二选一或同时引用，但不能把 PPoPP 等级贴在另一条期刊记录上。

原 VLDB 已有 CGgraph、SYgraph、EMOGI、HyTGraph、Grapin、Shared Arrangements 等条目，本稿可以继承其文献身份并按新主线重组。GUM、GRACE、C²graph 和 KGraph 是本轮补入的直接对照／定位材料；不要说它们都已出现在原稿。原稿引用审计的来源和关键映射保存在 [source_audit.json](references_20261105/source_audit.json)。

## 5. 引用等级与年份的使用口径

本清单严格列出 **10 + 10 篇**：第一部分 9 篇 A、1 篇 B，第二部分 10 篇 A；等级指对应会议，不评价单篇论文质量。SYgraph 使用 ICPP 2025 正式论文，未纳入海报、Workshop 或 Demo；PVLDB 研究论文按相应 VLDB 会议周期归类，不把 PVLDB 偷换成 The VLDB Journal。

本次核对使用 CCF 2026 第七版原目录的[学者镜像 PDF](https://dengpingfan.github.io/papers/CCF-Recommend-2026.pdf)，实际检查物理第5、7、13、27、35页。CCF [官方目录入口](https://www.ccf.org.cn/Academic_Evaluation/By_category/)与[原始 PDF 地址](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=112CF3BF7E1140ACEB271ADAED12A67ADFABB8FF099E40C2759502A85C8A281F)本次返回405，未声称官网直接下载成功；镜像的字节、哈希和逐会议等级见 [ccf_audit.json](references_20261105/ccf_audit.json)。

EMOGI 的 PDF 推荐引用格式为 PVLDB14(2):114–127,2021，而 DOI 出版元数据为2020年10月；机器记录保留两者，最终 BibTeX 选择一种一致的出版口径，不同时计数。HyTGraph 的 arXiv 2022、ICDE 2023 与 TPDS 2025 扩展版也分别标识。

## 6. 收束句与不能越过的实现边界

可用以下句意收束两部分：**已有研究分别提高了执行、更新和复用效率；本文研究这些能力在连续 GPU 图分析中通过版本、函数依赖与物理共享联系起来时，如何减少正确完成任务的总代价。**

这句话需要 E1–E6 支撑。当前 D 的核心是固定拓扑 FP64 权重与相关输入执行视图的维护，不能写成任意拓扑插删或增量求解；C 采用历史需求与真实释放量的启发式，不能写成已学习精确恢复毫秒成本或具有全局最优保证的缓存规划器。Grapin 的更新/缓存、Aspen 的快照复用、Shared Arrangements 的物理共享、Blaze 的整体缓存规划都应正面承认。区别最终落在本文具体状态合同与受控证据上，不能靠省略强先例形成。

文献元数据和可机器核对的每篇用途见 [papers.json](references_20261105/papers.json)；实验比较角色和图表方案见 [实验章节设计](PAPER_EXPERIMENT_DESIGN_20261104_CN.md)。
