# 最新基线能力、适用范围与接入清单

后续接入更新：HyTGraph 四核心已完成 CPU 编译和 parser 地址/未定义行为检查；KGraph sparse/dense SSSP 已通过 CPU 微图完整输出资格。构建、补丁、限制与主代理 GPU 资格命令统一保存在 [新增基线接入说明](../experiments/campaign_20261104/baselines/README_CN.md)。七图正式比较仍待资格和运行，不能以编译成功代替正确性或性能。

日期：2026-11-04。对应冻结七图 US / EP / LJ / OR / IN / UK / TW 和 E1–E6。本文核对作者源码、既有适配器与源绑定历史台账；**没有启动 GPU，也没有把历史三图结果当作七图新测量**。机器可读版本为 [BASELINE_CAPABILITY_MATRIX_20261104.json](capabilities.json)。

## 1. 当前必须纠正的遗漏

1. **CGgraph 不能继续只列 BFS/SSSP。** 新 V1.5 的发布范围确实较窄，但本工程已找到旧完整作者版本并接入 WCC/PR，历史正式台账存在完整格。
2. **Grapin 不能只列动态 SSSP。** 作者有 BFS/SSSP/CC/PR，工程也已有四者的静态适配和历史测量；其 PR 是独立 FP32 质量面板，不能无条件并入严格 FP64 主表。
3. **SYgraph 不能只列 BFS/SSSP。** 冻结作者版本还有 CC、BC、TC；没有现成 PR 的结论仅限该源码版本。
4. **C²graph 不是只实现 SSSP。** 作者目录还有 PPR 和 Adsorption；当前四核心共同问题仅 SSSP 直接相交，PPR 不能替代全局 PR。
5. **HyTGraph、KGraph、GRACE 均有公开实现。** 前者四核心构建目标齐全；KGraph 有五类查询；GRACE 有三个存储系统的集成。尚未部署必须写为“待接入/待资格”，不能写“没有代码”。
6. **cuGraph SG 全局 TC 的“未测”旧注释已过时。** 历史 `formal_cells.json` 已有 `cugraph_sg_tc` 三图完整格。能力表生成器中的旧提示不得继续直接复制。
7. **Gunrock/igraph PR 的缺口是质量证据整合。** 历史严格 PR 时间已测；共同表尚缺完整 residual/mass/reference 绑定，不是 API 不支持，也不是已经观测到数值失败。

判定顺序为“论文确实评测过 → 找到作者函数 → 核查当前发布版/入口 → 对齐输入、数值与完整输出 → 当前机器资格 → 正式比较”。论文出现一个算法足以要求继续找实现，但不能推出任意公开版本都发布了该算法，也不能直接推出已经满足本文的输出和精度。

## 2. 系统能力与实验位置

以下“已适配”只表示代码存在；新七图、固定新权重和当前 release 仍须重测。SG 为单卡，MG 为真正多卡协同，CPU+SG 不等于 MG。

| 系统 | 作者能力 / 工程现有路径 | 适用实验 | 需要完成的资格工作 |
|---|---|---|---|
| cuGraph 26.2.0 | BFS/SSSP/WCC/PR/WPR，SG/MG；BC、TC、k-core 等有 MG；MST/SCC 为 SG。LCC 可由逐点三角计数与度合法派生 | E1 SG/MG；E2 同资源扩展参照；E3 更新后重建/重算；E4 持久图完整任务；E6 强外部组合 | 保留 SGGraph/MGGraph 和官方算法，实际同步与完整结果计入时间；新输入尺寸、边 offset、PR 质量、每种 MG 算法分别资格 |
| Gunrock Essentials + legacy CC | Essentials BFS/SSSP/PR、BC、k-core、TC、MST；legacy CC 提供 P1/2/4/8 | E1 SG；MG 的 CC 单独适用；E3 SSSP 持久图强对照；E6 可用函数的组合/补充 | 明示两份作者版本，不能称现代 Essentials 全算法 MG；严格 PR 补共同质量字段；现有 FP32 面板保留独立身份 |
| SYgraph | BFS/SSSP/CC/BC/TC，当前版本 SG | E1 SG 三核心；扩展覆盖补充 | 新七图输入、孤立点/不可达点、无向投影；PR 标当前源码无现成入口，不占耗时或胜率分母 |
| CGgraph | V1.5 BFS/SSSP；旧完整作者 WCC/PR，CPU+SG | E1 SG 强异构对照；需要相同任务时可进连续调用面板 | 记录 CPU 线程、原作者 CPU/GPU 分工；PR 合同适配公开；不能以新版裁剪目录为由漏掉旧版已有函数 |
| GUM | BFS/SSSP/WCC/PR，P1/2/4/8；工作窃取与作者非窃取 HostFragment SSSP 分开 | E1 同资源；E2 MG；E3–E5 可作完整任务外部对照 | 当前机器 P2P/RWS 实际可用性、源码路线、完整输出验证；不得把非窃取路径标成开启 RWS |
| Grapin | 原生 BFS/SSSP/CC/PR；动态图增量、热点子图缓存；当前 SG | E1 SG；E3 最相关的增量 SSSP 外部对照；E4 保留其原生缓存的完整任务参照 | 静态四核心与动态事件入口区分；权重替换用已审计的删除旧边+插入新边适配；PR 完整数值质量单列；不关闭原生增量或强制每次重建缓存 |
| HyTGraph | 官方 CMake 构建 BFS/SSSP/CC/PR；仓库另有 BC 源码但未列构建目标；SG 超显存传输管理 | E1 SG，尤其容量/传输边界；连续调用可作为外部持久图参照 | 现代 CUDA 构建、完整输出和计时适配；PR 源码阈值不一致，先最小正确性/数值合同修正再验收；未适配不等于不支持 |
| C²graph | SSSP/PPR/Adsorption；P 为压缩执行，PM 为同图多源 CPU+SG 流水线 | E1 的 SSSP；同图多源批任务复用补充面板 | 压缩准备、源补偿、CPU 尾部、所有源完整输出均收费；PPR/Adsorption 不放全局 PR 表；不把 PM 一批时间除以查询数冒充独立单次延迟 |
| KGraph | CPU 并发 SSSP/SSWP/SSR/PPR/DeepWalk；分区内 pivot 结果 memoization | 同图多源 SSSP 的状态复用对照，独立于 GPU 闭包预算消融 | 分区输入、pivot/query 文件、完整距离输出和 host 内存计费；先修容器越界 UB；保留作者 memoization。不是显存 C-Retain 策略的同接口替代 |
| GRACE | PPCSR/Terrace/VCSR 的 CPU 动态存储集成；论文测 BFS/SSSP/PR/TC | 图存储更新/重建方法参照；仅在建立相同更新合同后做独立对照 | 当前工作是固定拓扑权重版本，GRACE 主问题是拓扑插入/删除和 PMA 重建，不能直接并排称同一 D 算法；没有当前部署和相同语义证据时明示 NR/不同合同 |
| GraphBLAST | 作者 BFS/SSSP/PR；已有持久 BFS/SSSP 适配；SG | 适用单卡执行/持久表示补充，不进 MG 表 | 旧机器结果不能直接迁移；现适配 SSSP 用保守 FP32 精确整数距离准入，超出界限须有新证书或明确“不适用该适配”；PR 已有作者算法但共同语义未接入 |
| igraph | CPU 四核心/WPR、BC、closeness、constraint、k-core、LCC、SCC、MST；ES 可由无向简单图度/LCC公式派生 | E1 CPU 参考另组；E6 强 CPU/混合组合及独立检查 | 固定线程和 C 库调用，WF closeness 用正确可达集合归一；别把派生 ES 写成作者原生 API；严格 PR 补完整数值证据 |
| EasyGraph CPU / NetworkX | 结构洞等通用 API；适合作为语义参考和功能补充 | E6 输出参考/CPU 补充 | 不拿纯 Python 弱实现代替已存在的强 GPU 或 C 库组合；与 EGGPU 有代码来源关系应披露，不能当唯一外部 SOTA 参照 |

cuGraph 的[官方算法目录](https://docs.nvidia.com/cugraph/26.10/graph_support/algorithms/)用于核对作者能力；它的版本为 26.10，**本轮已冻结安装版仍为 26.2.0**，具体可调用性以本机 package/DSO 收据和资格结果为准。不能因网页更新就把新发布版功能当成本机已实现。[igraph 官方 API](https://igraph.org/python/versions/latest/api/igraph.GraphBase.html)同理；数值定义最终由冻结任务的独立参考判定。

## 3. 作者版本与公开复现状态

作者仓库与固定revision见 [capabilities.json](capabilities.json)。该文件仅列可核查的能力、适用域和缺口；内部适配器位置及旧格子时间不纳入公开能力表。当前发布尚未包含全部baseline runner，不能据“历史已适配”推断本次七图可直接运行。

| 系统 | 作者来源 | 固定版本 / revision |
|---|---|---|
| cugraph26.02 | https://github.com/rapidsai/cugraph | 26.2.0; exact deployed package/DSO identities require the current-run receipt |
| gunrock_essentials2.1 | https://github.com/gunrock/gunrock | 748f79e66cc26762cd3f99422b028e742a737bde |
| gunrock_legacy0.5.1_CC | https://github.com/gunrock/gunrock | 0c9a96dd64ddeecae9208644631983fc889b32b4 |
| igraph1.0 | https://igraph.org/python/ | 1.0.0 |
| CGgraph_V1.5 | https://github.com/PengBo410/CGgraph-V1.5 | 5945816f4ccf63fce8d13eac84c09bbbb4c7e58a |
| Grapin | https://github.com/Yongze-zzz/C-GpuStreamGraph | 7ffcb2949837ad9f05d5d77afe54c698e2887479 |
| GUM | https://github.com/alibaba/libgrape-lite/tree/gum | 267ffdd345cc65e7b5541fabd9e94c70e2fd8141 |
| SYgraph | https://github.com/unisa-hpc/SYgraph | 71dcb56aed0cd43fefa5752a552e0a10cb69a758 |
| C2graph_P_corrected | https://github.com/lihuaibei7951/C-2graph | 97af1ebd935a03f5adc180935644e786a624bdbe |
| HyTGraph | https://github.com/iDC-NEU/HyTGraph | 3e5b0e47035972cd8ba1e0a9c89d99c108041de9 |
| KGraph | https://github.com/Gawssin/KGraph | e09ca53695cb5a82d1bbb2418caac90b2932af7a |
| GRACE | https://github.com/DataSysResearch/GRACE | 9f81318ecb12cbb5531034a574f6136f65d6b8e2 |
| GraphBLAST | https://github.com/gunrock/graphblast | f56f9677276c574c6fe4f1cc104d51f13717e165 |

## 4. 两个核心机制应与谁比较

### E3：维护合法状态，而不是把所有动态图系统混成一个对照

核心内部对照保留 `Cold / Reuse+Full / D-Maintain`，其余执行内核、输入、更新、资源相同。外部优先 **Grapin 的增量 SSSP**，再加入 cuGraph/Gunrock 的合法持久图和必要更新路径，比较“更新到完整答案”的全过程。Grapin 的算法结果增量比 D 的布局/权重维护更强的一面必须保留，不能强制重置结果削弱它。

GRACE 处理拓扑变化的 PMA 重建，并不直接提供 GPU FP64 属性 patch。当前 E3 不宜把 GRACE 的插边吞吐和 EGGPU 的更新后答案时间放进同一速度图。如果为用户指定的参照补实验，应先建立双方都实现的更新事务及完整输出，再单列方法面板；若要比较原位拓扑更新，首先需要扩展我方当前能力并修订研究范围，不能靠改图注掩盖。

### E4：保留表示、缓存结果和查询共享是三种可相关但不同的对象

受管显存预算扫描以 **closure-LRU、同所有权判断的其他合法策略、C-Retain** 为直接控制，完整系统对照允许外部原生缓存和图对象长期存活。不能每次销毁 cuGraph 图或清空 Grapin 缓存，再把重建开销归为外部缺陷。

**KGraph 必须进入状态复用相关实验考虑范围。** 正确位置是同图不同源 SSSP 的批任务参照，与 C²graph-PM 及 EGGPU 的相同请求序列比较：所有实现处理相同输入、相同权重、相同源集合并返回全部结果；KGraph 的分区/pivot 构建和 memoization 计入首次任务，热任务允许复用。它使用 CPU 内存与查询结果，不能直接塞进 GPU 物理数组闭包的 β 曲线。该补充仍使用七图中的预登记图，不另换便宜数据集，不以结果挑选图。

KGraph 的 PPR 与当前全局 PR 不同，SSR 是可达性而非完整 BFS 距离，SSWP 是最宽路径而非最短路。若只跑 SSSP，就在功能矩阵登记其余原生能力并写明不同任务；不用为填满四核心表重写它的算法。

### E5 / E6：控制变量与完整外部组合分开

E5 四格 D×C 用内部强替代做机制因果判断；外部完整工作流只回答系统位置，不冒称其模块开关等价。E6 的 A–D 功能组消融是我方与外部函数的受控组合，必须另有从头到尾外部组合。当前已有 **CuPy + direct pylibcugraph** 强组合路径，不能退回只比较 CPU constraint 的旧版本。图转换、阶段交接、真实候选选择均计时；每次重新算答案。

E1 单卡主比较至少保留已适用的 cuGraph、Gunrock、SYgraph、CGgraph、Grapin、HyTGraph；GUM P1 和 C²graph SSSP 在同资源适用格保留。正文可用“最强适用外部系统”概览，但所有有效对照的绝对时间与缺失原因都进完整表。E1 多卡和 E2 只放真正多卡 GUM、cuGraph、legacy Gunrock CC 等对应函数；不把单卡系统复制成多个实例冒充协作强扩展。

## 5. 本次直接读源码发现的接入事项

| 项目 | 已观察到的具体事实 | 允许的最小处理 |
|---|---|---|
| HyTGraph PR | [固定作者文件](https://github.com/iDC-NEU/HyTGraph/blob/3e5b0e47035972cd8ba1e0a9c89d99c108041de9/samples/hybrid_pr/hybrid_pr.cu)中 `CombineValueBuffer` 用 `buffer > 0.01`，`IsActiveNode` 用 `m_error`。仅改变命令行 error 不能保证同一个停止条件 | 保存原版；明确统一数值条件的兼容修改，完整输出做独立残差/质量检查；不替换为新 PR 内核，不把当前未知质量写成已通过 |
| HyTGraph 构建 | 官方 CMake 使用旧 CUDA find 模块、sm_35 等目标和旧依赖 | 单独锁定现代编译兼容补丁，保留原调度与传输算法；未有实际构建失败前不填“不可构建” |
| KGraph sparse SSSP | [固定作者文件](https://github.com/Gawssin/KGraph/blob/e09ca53695cb5a82d1bbb2418caac90b2932af7a/apps/kgraph-sssp-sparse.C)对 `ShortestPathLen_l` 只 `reserve(k)` 后用 `[i]` 写入；其 size 仍为零，属于 C++ 越界未定义行为 | 以 `resize(k)` 或等价最小修复满足容器语义；先做 CPU 正确性和输出资格，不能利用 UB 的偶然表现作性能结论 |
| C²graph | README 主要展示 SSSP，但 `src/ppr`、`src/adsorption` 与对应压缩实现确实存在 | 能力登记完整；主表仅取同语义交集，其他算法不伪装 PR |
| Grapin | 原静态 CC 缺返回、PMA边界和显式权重入口已有本地修正链 | 使用最终修正 receipt/source manifest，不重新拼多个最快旧适配器；全部修改理由随 artifact 公开 |
| GraphBLAST | 当前 SSSP 适配的保守条件是 `n × max_weight ≤ 2^24`，对部分大图不成立；这不等于库永远不能处理该图 | 若使用精确整数参考，应先提供更精确的合法距离范围证书或新的数值适配并重资格，否则明确适配范围不足；不截断输入，不默许 FP32 距离误差 |

这些是接口、正确性和编译兼容工作，不是替外部系统设计新性能优化。相反，禁用其原有缓存、窃取、压缩、并行模式，或给它额外 Python/CPU 往返，都会改变比较对象。作者原生公开参数可按论文配置冻结；不按每图测速挑最快版本或用我方新算法补其实现。

## 6. 近期执行优先级与证据状态

1. **先统一已有四核心入口。** 将 CGgraph WCC/PR、Grapin 四核心、SYgraph CC、GUM 两种 SSSP 路线纳入当前能力注册；重新绑定最新源码与七图 manifest。大图上任何 OOM、超时、索引不支持均保留原因。
2. **把质量证据补实。** Gunrock/igraph 严格 PR 需要完整 residual/mass/reference 字段；FP32 PR 用单独共同质量面板，绝不将精度差异隐藏在一个速度分母里。来源为 `analysis/full_results_20261102/pr_coverage_correction_20261104/correction.json`。
3. **新增 HyTGraph 的真实接入。** 固定上列 commit，兼容构建后逐函数输出资格；BFS/SSSP/CC 可先资格，PR 待数值条件修复。未测格记 NR，不能直接从论文图读时间。
4. **新增 KGraph SSSP 批任务接入。** 合法分区/pivot准备、同源序列、完整输出和 CPU 预算；与 C²graph-PM 区分压缩、memoization、查询调度的不同贡献。
5. **GRACE 保持源码明确、合同明确。** 当前无需凭空把它加进不相交的 GPU 表；若做 CPU 动态重建对照，使用真实拓扑更新和其作者插入/删除接口，另立受控范围。

`formal_cells.json` 中历史状态已核对：Grapin 静态 BFS/SSSP/CC/FP32 PR 各有三张旧图的 COMPLETE 格；CGgraph WCC/PR 有完整格；C²graph-PM 有三图、两参数的六个 COMPLETE 格；cuGraph SG TC 已有三格；GUM 非窃取 SSSP 有 P1/2/4/8 的十二格。这里核对的是 **历史 collector 内容及来源绑定**，并未重新验证远端全部大向量，也没有声称新冻结输入已完成。

机器可读表同时保存：作者 commit、现有 runner、原生函数、适用资源、未接入原因和历史组摘要。新 campaign 生成的 qualification/formal receipt 才能把新状态从 NR 变为 PASS；基线选择、协议和输入先冻结，再查看速度。单卡领先率以完整合格矩阵计算，缺失质量、OOM 和不同资源不静默抹去，也不拿“函数多”代替性能证据。

## 7. 一手材料与审计留存

- [HyTGraph 作者仓库](https://github.com/iDC-NEU/HyTGraph)、[论文](https://arxiv.org/pdf/2208.14935)。
- [Grapin 作者仓库](https://github.com/Yongze-zzz/C-GpuStreamGraph)、[论文](https://www.vldb.org/pvldb/vol18/p3854-wang.pdf)。
- [GUM 作者分支](https://github.com/alibaba/libgrape-lite/tree/gum)、[论文](https://graphscope.io/publication/gum.pdf)。
- [C²graph 作者仓库](https://github.com/lihuaibei7951/C-2graph)、[论文](https://wzgcs.github.io/papers/32_ICDE_2026.pdf)。
- [KGraph 作者仓库](https://github.com/Gawssin/KGraph)、[作者论文](https://gaosen.org/resources/papers/KGraph.pdf)。
- [GRACE 当前作者仓库](https://github.com/DataSysResearch/GRACE)、[论文](https://me.sage.org.ai/contents/research_papers/2026/2026_grace_icde_2026.pdf)。
