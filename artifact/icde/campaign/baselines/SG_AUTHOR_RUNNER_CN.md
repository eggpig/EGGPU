新的入口是 `sg_author_worker.py`，批量串行入口是 `run_sg_authors.py`。它们读取本轮 `canonical/<七图缩写>`，复用同目录上一级 `sg_worker.py` 的输入、三源、独立参考、完整输出与 PR 质量合同。CPU 测试通过不代表 GPU 资格通过；GPU 只由主代理执行。

| 系统家族 | 固定 variant | 算法 | 来源与接口 |
|---|---|---|---|
| CGgraph | `cggraph_legacy_full` | BFS/SSSP/WCC/PR | 同一 `3b23a7d` 作者版本；按函数固定整数 BFS/SSSP/WCC 和 FP64 PR 构建，遵守作者类型门槛；接口仅补不可达值到 `+inf` 的转换 |
| CGgraph | `cggraph_v15` | BFS/SSSP | 独立 `5945816f` 新版遍历组；不能把缺失函数以另一版本填入后冒称同一构建 |
| Gunrock | `gunrock_essentials` | BFS/SSSP/PR | `748f79e6`，现有 FP64 SSSP/PR adapter；BFS 整数 |
| Gunrock | `gunrock_legacy_cc` | WCC | `0c9a96dd`；作者 CC，保留实际 owning 三次输出；新增首请求计时观察，无算法改动 |
| SYgraph | `sygraph_author` | BFS/SSSP/WCC | `71dcb56a`；已有真实 CC 正确性适配，当前源没有可用 PR |
| Grapin | `grapin_author` | BFS/SSSP/WCC/PR | `7ffcb294`；沿用 PMA 边界修正和静态完整输出入口；PR 原生 FP32 |
| HyTGraph | `hytgraph_author` | BFS/SSSP/WCC/PR | `3e5b0e47`；v4 修复 CC `bool Single()` 缺 return，首请求覆盖 CUDA property probe，PR 原生 FP32 |

表格可以以家族名显示，但每个函数的 variant/commit 必须保留。`cggraph_legacy_full` 是完整 baseline；新版遍历是明确附加 variant。不得按图选最快版本后删除身份。

CSR 接口直接读取 canonical 的非负 int32 offsets/indices；需要 uint32 权重的作者函数进行精确转换。Legacy CGgraph 的非零出度前缀重排同时保存 `order/degrees`，还原完整顶点顺序；SSSP 权重和源点一起重排。WCC 使用显式去重弱图，所有孤立点保留。Gunrock legacy CC 读取带完整 n 的 MatrixMarket 上三角；Grapin/HyTGraph 读取作者文本格式。HyTGraph 必须用 `weight_num=0`，否则作者 parser 会重新生成权重。

每个 BFS/SSSP 点有三个预先冻结的源，每源启动独立原生进程，连续 first/warmup/ready 三次完整调用。其他函数一个进程、三次调用。`T_first` 包含 CPU 格式/弱图/重排和 native 导入、准备、首完整调用；人工跨进程输入序列化单独记录。`T_ready` 使用第三次完整调用，不能拿作者 kernel-only 数值代替。完整输出使用统一 oracle；PR 的 residual/mass ≤ 1e-6、reference L1 ≤ 1e-5。各 baseline 的原生返回数组是否跨调用保留，以对应原始适配器为准；该入口不声称额外的“所有原生输出同时常驻”内存实验。

使用 `--cpus` 固定 16 个实际可用 CPU ID。CGgraph 的 OMP/TBB 上限为 16，其他系统保留其已有单线程宿主配置。每个任务持全局 GPU 锁，核查八卡占用、指定 UUID 和真实 host/cgroup 内存余量。CPU bridge 与输出有独立磁盘门槛；空间不足必须记录资源失败，不能删除图边、改图或声称作者算法 OOM。

```sh
# R 为已构建 artifact 根，B 为本轮 campaign 目录；由主代理执行。
$PY -B "$B/baselines/run_sg_authors.py" \
  --variants cggraph_legacy_full,cggraph_v15,gunrock_essentials,gunrock_legacy_cc,sygraph_author,grapin_author,hytgraph_author \
  --datasets TINY_D,TINY_U --phase tiny \
  --root "$R" --input-root "$INPUT_ROOT" --setup "$B/setup.json" \
  --reference-cache "$REFERENCE_CACHE" --lock-file "$R/gpu_experiment.lock" \
  --gpu-uuid "$GPU_UUID" --cpus "$FIXED_16_CPUS" --python "$PY" \
  --output "$B/author_tiny_v1"
```

默认只打印 CPU 计划，加 `--execute-gpu` 才执行。`target` 要提供 `--proofs`，其 JSON 是显式 tiny `result.json` 路径数组；`formal` 同时要求 tiny 和该真实图 target 证明，精确绑定 variant、source/binary、环境、输入和源点。正式默认 10 个独立 repetition，不筛除慢值。派生大 staging 在每个源完成后清除，完整数值按内容哈希只存一次；所有重复的时间、原始 native 日志、正确性指标、失败记录保留。

`baseline_binary_registry.json` 保存读到的历史与新构建入口。`evidence/hytgraph_tiny_gpu_v1_receipt.json` 是主代理实跑的失败记录：BFS/SSSP/PR 完整输出通过，两个 CC 进程在返回路径崩溃；不能把该记录记为全系统通过。主代理已执行 v4 的 `hytgraph_tiny_v2/plan.json`，12 项全部通过，回执为 `evidence/hytgraph_tiny_gpu_v2_receipt.json`。它证明四函数作者 binary 的 tiny 正确性，尚不等于新 generic E1 worker 的 tiny/target 资格，也不是正式性能结果。KGraph 的 sparse/dense 作者入口、CPU tiny 与批处理范围仍见本目录 `README_CN.md`；它不被虚构成 GPU 单卡四核心 baseline。

2026-11-04 v1实际GPU资格为26/42通过；其余16项是CLI/运行库/类型构建接入错误，未归因算法慢或质量失败。v2修复SYgraph的libstdc++ ABI、Grapin非源函数的无效source_node参数、CGgraph整数函数的合法类型构建。v2完整42格重新资格，不能继承旧worker的26格为新worker通过。原worker保存在 `history/sg_author_v1/`。
