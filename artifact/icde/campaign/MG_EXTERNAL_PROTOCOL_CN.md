# 本轮 GUM / cuGraph MG 外部基线入口

当前状态：CPU 入口与契约已实现，尚未对新输入执行 GPU 资格或正式测量。仅主任务持有全局锁并执行 GPU。当前公开 artifact 第二增量不包含这些尚待 GPU 资格的新入口。

两个系统都使用 `sg_worker.load_input` 读取七图冻结的 `canonical/<缩写>/receipt.json`，验证全部输入 SHA、w16-v1、三个固定源点、int32 CSR 和完整 n。WCC 在任务内生成 A OR Aᵀ 的去重弱投影并保留孤立点，其他函数保留原始存储弧；SSSP 不重新生成权重。BFS/SSSP 每个 job 包含三个固定源点块，PR/WCC 一个。每块保留初次、warmup、ready 三份完整主机结果及精确 SHA，输出相同也保留原生调用和原始证据；没有按运行快慢删数据或自动选实现。

## GUM

入口 `gum_worker.py`，计划/串行控制器 `run_gum.py`。`gum_reused.py` 是之前 CSR-loader adapter 的明确函数抽取，`gum_rank_exec.py` 保持 MPI rank 的同 PID exec 取证；具体来源在 `gum_reuse_provenance.json`。作者算法二进制、CSR 构建计划、源清单及 runtime parser 全部由实际 build receipt 绑定并核验。不修改作者算法。

固定作者配置：标准 SSSP，不是 RWS；`--ws=false`、`partitioner=seg`、`lb=cm`；PR alpha=0.85、117 轮。SSSP 采用作者 uint32 域，明确要求全部 w16 值精确可表示且 `(n−1)*max_weight < UINT32_MAX`。超出该保守域记不支持，禁止隐式截断或改变权重。其完整原始距离向量保留 i64 文件；仅独立审核时将原生 unreachable sentinel 转为 inf。

当前 T_first 使用真实连续 CLOCK_MONOTONIC 区间：父进程在 weak/descriptor 适配前调用 `time.clock_gettime_ns(CLOCK_MONOTONIC)`；当前已构建作者输出 adapter 本来就在完整 dense host scatter 和 MPI completion barrier 后调用 `baseline_memory::monotonic_ns()` 并记录 `memory-markers-rank-N.jsonl`。首轮取全部 rank 终点的最大值减去**同一个父起点**，同时单独保存 rank0 完整主机输出的终点。没有相加阶段或 rank 的耗时。区间包含 weak/descriptor、MPI 启动、原生准备、首轮计算与完整主机输出；后续磁盘落盘和 Python 独立审核不在区间内。

`gum_clock.py` 核验已审阅三份 C++ adapter 与 CLOCK_MONOTONIC 头文件的精确 SHA；既有 build receipt 核验确保这些是编译源。父进程和每个 MPI rank 的 exec 之前分别记录 boot ID、time namespace、time-for-children namespace 和 offset；域不一致即拒绝。rank exec 与作者 C++ 保持相同 PID，marker 的 PID、首轮编号、成对 begin/end、起点/launch/终点/父返回的单调顺序都核验。跨进程时钟观测不改 C++、不重建二进制，也不改变原生算法或 T_ready。已有 ready-only 入口原始字节保存在 `gum_ready_only_v1/`；新的 worker epoch 必须重新 GPU 资格，不能把旧 ready-only 通过当作新 T_first 通过。

T_ready 仍是第三轮完整 query 的 rank 最大值；三轮原生数值结果和原始 timer 记录全部保留。首次结果的“主机就绪”是 rank0 内部拥有的完整 dense 数组，源码保证此前结果仍存活；此边界不包含将结果写盘再由 Python 加载。

GUM build receipt、mpiexec、core-python、LD_PRELOAD/LD_LIBRARY_PATH 都显式传入，不把机器绝对路径写死。使用 `baselines/baseline_binary_registry.json` 中的 `gum_law_build_v1` 标准 SSSP 项，不能误取历史 `gum_sssp_ws`。

示例（替换本机路径和 UUID；先运行一个 tiny case，由主任务执行）：

```sh
CORE_PYTHON -B gum_worker.py --execute-gpu --phase tiny --dataset TINY_D --function bfs --count 2 --gpu-uuids UUID0,UUID1 --cpu-set CPU0,...,CPU15 --input-root INPUT_ROOT --setup SETUP_JSON --lock-file GLOBAL_LOCK --reference-cache REF_CACHE --output NEW_JOB_DIR --build-receipt GUM_BUILD_RECEIPT --mpiexec MPIEXEC --core-python CORE_PYTHON --ld-preload PRELOAD_LIBRARY --ld-library-path AUTHOR_LIBRARY_PATH
```

## cuGraph MG

入口 `cugraph_mg_worker.py`，控制器 `run_cugraph_mg.py`。`cugraph_mg_adapter.py` / `cugraph_mg_platform.py` 来自已有 MGGraph+RAFT+Dask 适配，仅重命名 companion import；源码绑定在 `cugraph_mg_reuse_provenance.json`。使用 P=2/4/8；P1 外部基线已有 SG worker，不把 MG 模拟为单卡。

T_first 是一个连续计时：已核验主机 CSR → 必要弱投影 → imports、Dask cluster/RAFT communicator → 各 rank CSR→COO 转换 → MGGraph → 实际查询 → 所有 rank 主机结果收集 → 完整唯一顶点域校验和 owned output。T_ready 在一次明确 warmup 后执行同一完整查询，无初始解复用。清理后再审核仍持有的三份输出，证明没有悬空返回。CPU参考与 SHA/readback 不计入任务时间。

保留旧配置：16 个固定 CPU，单 rank 线程 1；每 rank host32GiB、device20GiB、RMM256MiB/上限16GiB；RAFT `p2p=False`，`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、loopback socket；dashboard 禁用且实际 socket/domain 被检查。每 rank GPU UUID、PID、CPU affinity、CUDA context 和载入库都审核；parent 观测整个 owned process tree RSS，超时只终止自己的进程组。

PR 使用未加权 FP64、alpha=.85、global-L1 epsilon1e−9、max_iterations10000、每次无 initial guess。SSSP 保留当前 pylibcugraph 必需的 compute_predecessors=True，额外工作不会通过修改作者库消掉。BFS 保留原已用 adapter 的 depth_limit=0，不根据推测改公开 API；tiny 正确性门将检验实际库语义。

```sh
RAPIDS_PYTHON -B cugraph_mg_worker.py --execute-gpu --phase tiny --dataset TINY_D --algorithm bfs --count 2 --gpu-uuids UUID0,UUID1 --cpu-set CPU0,...,CPU15 --input-root INPUT_ROOT --setup SETUP_JSON --lock-file GLOBAL_LOCK --reference-cache REF_CACHE --output NEW_JOB_DIR
```

## 资格与记录边界

可用 `native_worker.py --make-tiny-native NEW_INPUT_ROOT` 创建两份 16 点含孤立点 fixture；不要覆盖已冻结正式输入。两套控制器不带 `--execute-gpu` 时只给计划。`--phase target` 必须显式指定匹配函数、P、代码/环境 epoch 和有向性的 tiny proof；formal 再要求同完整输入 SHA 的 target proof。控制器 `--proofs` 是 `{ "EP:bfs:p8": {"tiny": ".../result.json", "target": ".../result.json"} }`，不会自动挑最新或最快。正式 repeat 仅 0..9，控制器当前每次执行显式一次 repeat；十次必须保留并完整编排。

独立质量门共用 SG：BFS/SSSP exact distances/unreachable；WCC partition equivalence；FP64 PR residual/mass≤1e−6、referenceL1≤1e−5。GPU tiny、target 或正式失败均保留失败记录，不能改参数后继续视为原 source epoch。

当前尚未覆盖：GUM 新连续 T_first 观察器的实际 GPU 资格；新图新 epoch 的完整正式运行；controller 与系统级调度整合；全局编排缺失条件/OOM/不支持状态的统一汇总。以上不被 CPU 通过替代。
