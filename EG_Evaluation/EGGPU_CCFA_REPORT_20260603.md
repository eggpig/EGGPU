# EGGPU CCF-A Paper-Level Report, 2026-06-03

本文档是当前 EGGPU 论文第三章和第四章的工程化总报告。它只记录已经能由代码、预检、审计或局部测试支撑的结论；最终 SOTA 率仍必须以后续 idle GPU 上的 16 函数全量实验为准。

## 0. 当前结论边界

当前项目范围是 16 个 EasyGraph 函数：

```text
PageRank, MST, LCC, WCC, SCC, BFS, Dijkstra, BellmanFord, SSSP,
KCore, BC, Closeness, EffectiveSize, Efficiency, Constraint, Hierarchy
```

已经完成的关键改动：

- Closeness unweighted 路径从 weighted/Dijkstra-style CUDA 路径切到 exact source-parallel BFS CUDA 路径。
- Closeness 复用 device CSR 和 per-source workspace，并避免 unweighted 输入上传权重数组。
- Closeness CUDA event 在 D2H score copy 之前停止，`kernel` 不含返回拷贝；`e2e` 仍包含 CSR/transfer/sync/result wrapping。
- BC/Closeness 的 EGGPU Python backend 保留 C++ dense result 为 NumPy array 返回，避免百万级节点结果在 e2e 窗口内逐元素转成 Python float list。
- `run_full_baselines.py` 与 `library_baselines.py` 双层强制 EGGPU strict error，并禁用 SCC/KCore/SSSP host policy。
- full runner 在每个 EGGPU child process 启动前重新检查 GPU 是否空闲，避免长实验中途被外部进程污染。
- build script 固定 `CMAKE_CUDA_ARCHITECTURES=${EGGPU_CUDA_ARCHITECTURES:-80}`，原 A100 机器默认 `sm_80`，新机器按 GPU compute capability override。

当前还不能宣称完成最终目标：

- 2026-06-01 的 clean full run 是 15 函数历史结果，不包含当前 Closeness。
- 2026-06-03 正在跑的 full run 后半段被外部 `sglang` 进程占用 GPU 污染，不能作为论文最终 timing。
- 新 Closeness 路径只有 targeted log evidence：`ca-GrQc` 上 EGGPU Closeness e2e 约从旧目标结果的 `0.05696s` 降到 `0.00897s`，kernel 从 `0.03065s` 降到 `0.00744s`，并与 EasyGraph C++ correctness hash 对齐。该 targeted run 后续被中断，未生成完整 summary CSV，因此只能证明优化方向有效，不能替代 clean full rerun。
- 2026-06-04 targeted evidence 显示 BC dense return 优化有效：`web-NotreDame` BC E2E `0.1203s`、kernel `0.1170s`；`wiki-Talk` BC E2E `0.8741s`、kernel `0.8659s`。上一轮 full run 中 `wiki-Talk` BC E2E 为 `1.1306s`，该优化主要减少 Python result materialization。
- 2026-06-04 后续 BC warp-size sweep 显示大规模低平均度 directed graph 更适合较小 warp：`web-NotreDame` BC 默认 AUTO 从约 `0.1157s/0.1135s` 降到 `0.0915s/0.0892s`，`wiki-Talk` BC 从约 `0.8981s/0.8897s` 降到 `0.7323s/0.7244s`。当前代码只对 `directed && nodes >= 300000 && avg_degree <= 8` 自动选 `warp=2`，其余图保持原自动策略；`EASYGRAPH_GPU_BC_WARP_SIZE` 可显式覆盖并写入 metadata。
- 2026-06-04 KCore targeted evidence 显示 `EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS=1024` 与 int32 dense return 是正收益：`ca-HepTh` KCore E2E/kernel 降到约 `0.00102s/0.00069s`，`email-Enron` 降到 `0.00447s/0.00412s`，`ca-HepPh` 降到 `0.00218s/0.00192s`。后续收紧 single-block 默认策略，避免中等规模低平均度图只因高最大度进入 single-block；`ca-CondMat` default/no-single 对照后保留默认约 `0.00551s/0.00513s`。它仍慢于 easygraph-cpp 约 `0.00440s`，是保留的中等规模低 core peeling hard case。
- 2026-06-04 结果口径新增 `paper-core` 固定视图：在现有 small/low-work dataset filter 外，透明排除 `email-Enron/{WCC,SCC}`、`soc-Slashdot0811/SCC`、`web-NotreDame/SCC`、`wiki-Talk/SCC` 这类 component-set materialization dominated pair。将本轮 targeted EGGPU rows 合并上一轮 clean baseline 后，估算 `paper-core` E2E pair SOTA 为 `103/106 = 97.2%`，kernel pair SOTA 为 `104/106 = 98.1%`。最终仍必须通过 clean full rerun 确认。
- 2026-06-04 summary 口径明确了 `0.05%` relative timing-tie tolerance；超过该容差的 pair 不算 SOTA。final summary 同时输出 `2%` near-miss 表，用于定位需要重复测量或继续优化的 close loss，而不是事后放宽 SOTA 定义。
- 2026-06-04 安全审计收紧了 EasyGraph-mode timed path：EGGPU warmup 出错会使该 row fail，不再继续冷启动计时；PageRank 不再降低 `alpha` 重试；KCore 不再在异常后走 directed fallback retry。
- 2026-06-04 同轮排除了三个不应合入主路径的实验：unweighted BFS-Brandes BC kernel 在 `web-NotreDame` 上 kernel 变慢到约 `0.141s`；SCC `ACTIVE_TRIM_MAX_ITERS=64` 只对单图有小幅波动收益且历史 hard-case sweep 显示 trim16 几何均值更好；lazy component view 在 `web-NotreDame` SCC 上 e2e 回归到约 `0.519s`，因此全部不保留。

## 1. 第三章：系统设计与方法

### 1.1 设计目标

EGGPU 的核心问题不是只写 CUDA kernel，而是在 EasyGraph Python API 不变的前提下，把用户真正支付的端到端成本降下来。论文第三章应把贡献拆成三层：

- Public API dispatch：用户仍调用 EasyGraph 原函数；GPU enabled 时进入 EGGPU native backend。
- C++/middle layer：负责图对象转换、CSR 表示、GraphContext/device cache、workspace reuse、kernel time 回传和结果包装。
- CUDA layer：实现 16 个函数的 GPU kernel 或 GPU workflow，并把 kernel timing 与 e2e timing 分开。

论文里要强调 EGGPU 与 EasyGraph C++ baseline 的边界：

- EGGPU 可以在自身路径中使用 C++ binding 和 GraphContext，这是 GPU 集成路径的一部分，计入 EGGPU e2e。
- easygraph-cpp baseline 必须 GPU disabled，不能经过 EGGPU backend，也不能走 CUDA structural-hole binding。
- GPU strict error 开启时，GPU 模式失败必须 raise，不能 fallback 到 CPU。

### 1.2 数据表示与复用

第三章建议使用“GraphContext + device CSR + workspace”作为中间层主线：

- CSR 是 16 函数共享的主表示，降低 Python adjacency/set 操作开销。
- device CSR cache 避免同一图多次函数调用时重复 H2D transfer。
- workspace cache 避免每次 Closeness/BC/SSSP 重新分配大型 per-source buffer。
- result cache 对 timing 默认关闭，避免纸面时间被 cache hit 污染；workflow/return ablation 可以单独研究 cache 与 materialization 的收益。

这条线与 HyTGraph 的 transfer-management 结论一致：GPU graph analytics 的性能瓶颈经常不是单个 kernel，而是 CPU-GPU transfer、active data movement 和调度成本。EGGPU 不实现 HyTGraph 的 hybrid scheduler，只继承“数据移动必须作为一等公民优化”的系统洞察。

### 1.3 Closeness Centrality

Closeness 是本次新增和重点优化函数。语义对齐如下：

- EasyGraph/EGGPU 使用 directed outward shortest-path distance。
- NetworkX directed closeness 默认是 inward distance，因此 benchmark 对 directed graph 使用 `G.reverse(copy=False)`。
- igraph 使用 `mode="OUT"`，再乘以 reachable-fraction 的 Wasserman-Faust correction，使 disconnected graph 语义与 EasyGraph/NetworkX 对齐。
- nx-cugraph/cuGraph/Gunrock 当前不测 Closeness：官方 supported algorithms 未列出 Closeness，不能测 fallback。

实现策略：

- Unweighted Closeness：exact source-parallel BFS CUDA kernel，一块处理一个 source，按 frontier 层推进，累计 reachable count 和 distance sum。
- Weighted Closeness：保留 weighted Dijkstra-style CUDA path。
- Unweighted path 调用 `acquire_device_csr(..., include_weights=false, ...)`，不上传权重。
- `kernel` 仅包含 CUDA event 包围的 GPU kernel 区间；final score D2H copy 计入 `e2e`。

论文写法要保守：

- 可以说 EGGPU 将 Closeness 映射为 repeated/source-parallel BFS traversal，并用 CSR/cache/workspace 减少 e2e 成本。
- 不要说实现了 iBFS、HyTGraph 或 INFINEL。
- iBFS 可作为 CCF-A repeated/concurrent BFS traversal 背景；HyTGraph 作为 transfer-management 背景；INFINEL 作为 output/materialization 背景。

### 1.4 Connectivity/Core

当前弱项主要是 SCC/KCore：

- SCC 面对很多 singleton SCC、低直径碎片和大 Python set/dict 返回时，GPU kernel 优势容易被 result materialization 吃掉。
- `wiki-Talk` SCC targeted 对照显示 kernel 约 `0.20s`，但 e2e 约 `3.9s`，主要瓶颈是 EasyGraph API 需要 materialize 约 228 万个 component set；直接换成 lazy view 会改变/干扰当前 benchmark 路径且实测不降反升。
- KCore 在小图/低平均度图上容易输给 igraph/easygraph-cpp 的低常数 CPU 路径。
- EGGPU 当前策略是保持 strict GPU 路径，不用 host policy 伪装成 GPU 成绩。SCC/KCore/SSSP host policy 在 paper runner 中强制关闭。
- `paper-core` 过滤不是隐藏失败，而是把 EasyGraph public API 的 output-materialization dominated regime 单独列出；full/gpu-friendly/paper-core 三个视图都由 `summarize_final_result.py` 固定生成。

论文可引用：

- ECL-SCC, SC 2023：GPU-friendly SCC 应避免递归 DFS，采用并行传播/edge-centric 思路。EGGPU 只借鉴这一方向，不声称实现 ECL-SCC。
- Accelerating k-Core Decomposition by a GPU, ICDE 2023：高性能 GPU KCore 需要专门 peeling 优化。EGGPU 当前还留有进一步算法空间。
- INFINEL, PPoPP 2024：大输出不可预测图查询提示 GPU kernel 时间和用户可见输出 materialization 可能脱钩。EGGPU 用它支撑 return-path slimming 与 `paper-core` output-heavy 限制，不声称实现 INFINEL。

### 1.5 Path/Tree 与 Structural Holes

Path/Tree 类函数目前是强项：

- BFS/SSSP/Dijkstra/BellmanFord/MST 适合 CSR traversal 或 frontier-style GPU path。
- Gunrock/cuGraph 是合理 GPU baseline；Gunrock CLI 的 per-source/CLI timing 要在 notes 中说明。

Structural-hole 类函数是 EGGPU 覆盖优势：

- cuGraph/nx-cugraph 没有 EasyGraph-compatible Burt structural-hole API。
- EGGPU 把 Python ego-network/set loops 转成 CSR scan/intersection。
- 论文应把这类函数作为“拓展 GPU graph libraries 覆盖范围”的亮点，而不是只与标准 PageRank/BFS 重复。

## 2. 第四章：实验设计

### 2.1 baseline 支持矩阵

当前 baseline 支持策略：

- EGGPU：16 函数全测，warmup=2，strict error，GPU backend=`mine`。
- EasyGraph CPU：16 函数全测，GPU disabled，warmup=0。
- EasyGraph C++：跳过 BellmanFord 和 structural-hole 函数；structural-hole C++ binding 在 GPU-enabled build 中会 route CUDA，不能当 CPU C++ baseline。
- NetworkX：按官方 API 测，Closeness directed 使用 reverse view；Hierarchy unsupported。
- igraph：测 C-core 支持函数，Closeness 做 outward + WF correction；structural-hole 只测 Constraint。
- nx-cugraph/cuGraph：只测官方 supported algorithm 范围；Closeness 和 structural holes unsupported。
- Gunrock：只测本地有 executable 且语义可对齐的函数；Closeness unsupported。

### 2.2 时间定义

论文必须明确三种时间：

- `build`：baseline-native graph object construction after raw edge-list parse/import。原始文件 parse/import 不算 build。
- `e2e`：用户函数调用 wall time after graph construction，包括 per-call CSR/transfer/sync/result wrapping。
- `kernel`：CUDA event time。CPU backend 的 `kernel` 定义为 algorithm wall time；无 clean event 的 CLI/native baseline 用 best available algorithm timer，并在 notes 标注。

显存统计：

- 全卡 NVML memory：用于观测整卡压力，但容易受外部进程污染。
- process-tree GPU memory：NVML 可归因时使用，更适合 paper table。
- delta memory：相对 child process start baseline，减少常驻上下文影响。
- `EGGPU_GPU_VISIBILITY_MARKER` 默认关闭。需要共享服务器可见性时，可设置
  `EGGPU_GPU_VISIBILITY_MARKER_MB=<N>` 让长驻 runner 固定申请 `<N>` MiB；
  `EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB` 默认同为 `<N>`，只从全卡绝对
  memory 指标中扣除。process-tree memory 和 delta memory 不做扣减。

### 2.3 正确性与公平性门禁

正式结果必须同时通过：

- `benchmarking/preflight_full_eval_ready.py`
- `benchmarking/audit_full_result.py`
- `benchmarking/audit_backend_separation.py`
- `benchmarking/validate_correctness.py`
- final summary / pair-level SOTA summary
- `run_metadata.json` gate

关键安全原则：

- EGGPU GPU mode fail 不能 fallback CPU。
- EasyGraph CPU/C++ baseline 不能继承外部 `EASYGRAPH_GPU_BACKEND`。
- 结果缺失 metadata、缺失 validation、validation self-reference、backend log mismatch 都是 hard failure。
- GPU busy 时不能继续产出 paper timing。新 runner 在每个 EGGPU child 前都会检查。

### 2.4 当前历史结果解释

2026-06-01 15 函数 clean full run 的历史结果：

- full 17 datasets：E2E SOTA 222/255 = 87.1%，kernel SOTA 241/255 = 94.5%。
- nodes >= 10000 过滤后：E2E SOTA 161/180 = 89.4%，kernel SOTA 171/180 = 95.0%。
- Path 类强，Structural holes 强，Connectivity/Core 弱。

这些结果能支撑方法趋势，但不能作为当前 16 函数最终结果。加入 Closeness 后必须重跑全量。

### 2.5 当前目标达成状态

用户目标有两个可接受条件：

1. 95% 以上场景 E2E SOTA 和 kernel SOTA。
2. 去掉小图或不利于 GPU 的图后基本全面 SOTA。

当前证据状态：

- kernel 在历史 filtered result 已达到 95.0%，但 E2E filtered 仍为 89.4%。
- 新 Closeness 局部证据显示优化有显著收益，可能改善 16 函数后的总体表现。
- SCC/KCore 仍是主要风险；如果最终 E2E 达不到 95%，论文需要清楚定义 GPU-unfriendly graph regimes，并给出 filtered/large-graph 结论。
- 不能用被 `sglang` 干扰的 2026-06-03 当前 full run 证明目标。

### 2.6 2026-06-08 会话更新

本轮会话新增或沉淀的内容：

- 新增 `EGGPU_SESSION_PROGRESS_20260608.md`，集中记录最新工程进展、
  visibility marker、显存观测、GPU 选择和 clean rerun checklist。
- `EGGPU_GPU_VISIBILITY_MARKER_MB` 支持固定显存 marker，便于共享服务器上
  通过 `nvidia-smi`/`nvtop` 看到长驻 runner。该 marker 不启动 kernel，不应产生
  SM utilization。
- `EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB` 只校正全卡绝对 memory 指标；
  process-tree memory 和 delta memory 不调整。paper table 仍优先使用
  process-tree GPU memory。
- preflight 已把 marker 默认关闭、默认 0 MiB、脚本透传、`cudaMalloc` 支持和
  memory-adjust hook 纳入静态/结构合同检查。
- 这轮验证通过：Python bytecode compile、`bash -n`、GPU0 full preflight、
  16 MiB marker smoke、adjustment-rule smoke。

最近显存观测来自
`full_eval_gpu0_20260603_234802_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto`
成功 EGGPU memory rows。该 run audit 未通过，因此只能作为 sizing/diagnostic：

- EGGPU process-tree GPU peak：mean 456 MiB，median 420 MiB，P95 623 MiB，
  max 2002 MiB。
- gpu-friendly successful subset：mean 476 MiB，median 428 MiB，P95 675 MiB，
  max 2002 MiB。
- whole-device peak：mean 1232 MiB，median 1196 MiB，P95 1399 MiB，max
  2778 MiB。
- 最大点主要是 `wiki-Talk / BC` 2002 MiB 和 `web-NotreDame / Closeness`
  1240 MiB。
- 与 nx-cugraph 在共同 successful gpu-friendly rows 上相比，EGGPU GPU memory
  属同量级：process-tree peak mean 约 485 MiB vs 471 MiB，median ratio 约
  0.991x，EGGPU 在约 63% pair 上更低，但 BC outlier 让 mean ratio 略高于 1。

跨机器 GPU 对比建议：

- A100 80GB：保留为 datacenter reference。
- RTX 4090 24GB 或 RTX 5090 32GB：高端消费/工作站对比。
- 一个 16GB 消费级卡：用于说明当前 EGGPU 并不依赖 80GB 级显存。
- 8GB 卡可以作为补充 portability/stress 实验，但不建议作为主对比平台，
  因为 RAPIDS/nx-cugraph、CUDA context、碎片和未来更大图可能让失败更难解释。

## 3. 复现实验命令

A100 原机器或同等机器：

```bash
cd EGGPU
conda activate EGGPU
export EGGPU_CUDA_ROOT="$CONDA_PREFIX"
export CUDA_PATH="$EGGPU_CUDA_ROOT"
export CUDA_HOME="$EGGPU_CUDA_ROOT"
export CUDAToolkit_ROOT="$EGGPU_CUDA_ROOT"
EGGPU_CUDA_ARCHITECTURES=80 bash scripts/build_eggpu.sh
```

RTX 4090：

```bash
EGGPU_CUDA_ARCHITECTURES=89 bash scripts/build_eggpu.sh
```

正式 sequential full + ablation：

```bash
cd EG_Evaluation
RUN_LOG="benchmarking/results/main_then_ablation_$(date +%Y%m%d_%H%M%S).console.log" && \
MAIN_GPU=<IDLE_GPU> ABL_GPU=<IDLE_GPU> LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=300 \
bash run_main_and_ablation.sh |& tee "${RUN_LOG}"
```

不要在 paper run 中设置：

```bash
EGGPU_ALLOW_BUSY_GPU=1
RUN_PREFLIGHT=FALSE
EGGPU_USE_CONDA_RUN=TRUE
```

## 4. 下一次 Clean Full Rerun 检查表

运行前：

- `nvidia-smi` 确认目标 GPU 没有外部 compute process。
- `bash scripts/build_eggpu.sh` 已用正确 `EGGPU_CUDA_ARCHITECTURES` 重建。
- `benchmarking/preflight_full_eval_ready.py` pass；outer shell hygiene 允许默认 warning。
- `EGGPU_GPU_VISIBILITY_MARKER=FALSE`。
- `EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE`、`EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE`、`EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE`。

运行后：

- full audit gate pass。
- backend separation audit pass。
- EGGPU runtime bad rows = 0。
- EGGPU validation bad rows = 0。
- Closeness rows correctness pass，nx-cugraph/cuGraph/Gunrock Closeness unsupported rows没有 fallback timing。
- `run_metadata.json` 记录 dirty status、Python、CUDA root、loaded `.so` path/mtime/size。
- final summary 中报告 full、large/filtered、category-level SOTA rate。

若 clean rerun 后 E2E 仍未达到 95%：

- 使用 nodes >= 10000 或排除小图/明确 GPU-unfriendly regimes 的 filtered table。
- 同时报告 `paper-core` 固定视图；该视图排除小图/低 work 数据集和 component-set materialization dominated WCC/SCC pair。
- 单独讨论 SCC/KCore 的结构性难点。
- 对 Closeness 给出优化前后 targeted replacement evidence，但必须标注 clean GPU 条件。

## 5. 可引用来源

Primary/near-primary sources checked in current docs:

- NetworkX Closeness semantics:
  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
- python-igraph Closeness API:
  https://igraph.org/python/versions/0.10.0/api/igraph.GraphBase.html
- RAPIDS cuGraph supported algorithms:
  https://docs.rapids.ai/api/cugraph/stable/graph_support/algorithms/
- nx-cugraph supported algorithms:
  https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/supported-algorithms/
- Gunrock algorithms:
  https://gunrock.github.io/gunrock/gunrock.wiki/Graph-Algorithms.html
- HyTGraph, ICDE 2023:
  https://dblp.dagstuhl.de/rec/conf/icde/Wang0ZC023.html
- ECL-SCC, SC 2023:
  https://userweb.cs.txstate.edu/~burtscher/research/ECL-SCC/
- GPU k-Core, ICDE 2023:
  https://impact.ornl.gov/en/publications/accelerating-k-core-decomposition-by-a-gpu/
- INFINEL, PPoPP 2024:
  https://ppopp24.sigplan.org/details/PPoPP-2024-papers/17/INFINEL-An-efficient-GPU-based-processing-method-for-unpredictable-large-output-grap
- GraphCube, PPoPP 2024:
  https://ppopp24.sigplan.org/details/PPoPP-2024-papers/23/GraphCube-Interconnection-Hierarchy-aware-Graph-Processing

## 6. 不能写进论文的 claim

- 不能说 EGGPU 实现了 ECL-SCC/iBFS/HyTGraph/INFINEL；只能说这些工作启发或定位了相同系统问题。
- 不能把 nx-cugraph/cuGraph Closeness fallback 当作 baseline。
- 不能把 easygraph-cpp structural-hole CUDA binding 计为 CPU C++ baseline。
- 不能把 busy GPU 上的 timing 当作 paper-quality timing。
- 不能用 15 函数历史结果替代当前 16 函数结论。
