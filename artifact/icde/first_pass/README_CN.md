# 首轮覆盖优先：已有完整测量直接呈现

当前入口是 首轮比较表（内部资产未随本增量分发：`current/FIRST_PASS_CN.md`）、全系统表（内部资产未随本增量分发：`current/first_pass.html`） 、E1 单卡图（内部资产未随本增量分发：`current/e1_single_gpu.pdf`） 和 E2(a) 八卡图（内部资产未随本增量分发：`current/e2_eight_gpu_comparison.pdf`）。所有适用系统保持原 47 个函数/资源义务 × 7 图，共 329 格；缺项和失败不隐藏。静态主图只画准备完成后的完整执行时间；首次任务时间留补充表。表中最快已测对手仍按各指标分别计算，其身份与覆盖数逐格列在表中，不宣称完整矩阵 SOTA。

本轮直接复用完整 target 的既有 T_first/T_ready，明确保留 target 身份，不改成 formal，不重跑资格，不重读大数组。只有 source/input、既有全输出质量和完整计时满足原合同才呈现。EGGPU 固定为统一源码 71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce；旧 d84 结果只保留在 previous，不能填充 current 的自系统缺项；GUM PR 固定为作者 FP32/117，旧 P8 FP64 结果保留在 excluded 中，不进入当前对比。原先未命名的 P8 WCC 义务现在明确对应 Gunrock legacy CC，实现身份与已有 e1_missing_baselines_v1 适配器一致。

当前优先补齐首轮全部适用格，再做第二次独立观测。计数单位是一次完整观测。BFS/SSSP 三个固定源仍按原任务等权汇总，不能把它们当三次独立重复。首轮不输出重复均值、置信区间或显著性。完整源表保留首次与执行两种时间，静态主图采用执行时间；队列整作业 wall time 不进入性能表。

`collect_first_pass.py` 读取显式计划、完成队列与它们引用的 JSON；失败额外保存绑定哈希的原日志尾部，不读取结果向量。现已包括 `first_pass_20261106/core/queue/state.json`、GUM 的 TW 容量队列以及 cuGraph TW narrow/wide 补充队列。`build_first_pass.py` 从快照输出 JSON/CSV/Markdown/HTML；`render_icde.py` 生成 175 mm 左右双栏图，保留全部七图位置、简短坐标和固定配色。`current/receipts.json.gz` 是原采集 JSON 的无损压缩副本，其中包括输入 metadata 的来源绑定。源结果向量及失败原文件仍在其原服务器路径。

作者时钟的只读核对以各原始回执和当前 `first_pass.json` 的来源绑定为准：ready 取 complete_call_seconds、complete_query_seconds 或 query_complete_host_seconds，不取 CUDA kernel/enact 毫秒值；first 取原 CLOCK_MONOTONIC 连续区间。这是复用先前生产者证据，不是新增资格轮次。

```bash
python3 -B build_first_pass.py --bundle /your/output/receipts.json.gz --authority e1_scope_authority.json --output /your/output/current
python3 -B render_icde.py --report /your/output/current/first_pass.json --output /your/output/current
```

当前快照的采集时间和覆盖以 `current/export_receipt.json` 为准。E2/E4-S/E6 的逐格状态与图在 remaining（内部资产未随本增量分发：`current/remaining/remaining_results.html`），E3/E4/E5 由状态实验汇总器导出。线上watcher的启动状态由执行者单独记录；公开包本身不启动watcher；本说明中的命令仅重建已有证据，不启动 GPU。后续更新只滚动 current/previous 两套派生展示。

刷新时由主代理在实验机执行 `python3 -B collect_first_pass.py --campaign /your/campaign --output receipts_latest.json`，取回后无损压缩替换 `current/receipts.json.gz`，再执行上述导出和绘图命令。先将当前派生展示滚动为 previous，始终保留原始结果路径与输入/epoch 绑定；本说明不启动采集或实验。

论文中的 E1 只展示单卡四算法。原始 E1-P8 测量在论文中映射到 E2(a)，E2(b) 仍使用固定后端的 1/2/4/8 卡扩展；这仅改变图表组织，不修改 raw ID、执行后端或测量口径。

## 公开工具的增量与使用范围

生成物放在checkout之外。可移植collector保留显式`--plan`、`--queue-state`、`--source-freeze`、`--native-protocol`参数。旧源码先归历史，当前协议与质量错误继续复核。新增watcher命令：

```sh
python -B artifact/icde/first_pass/export_on_progress.py --campaign /your/campaign --output /your/output/first-pass --authority artifact/icde/first_pass/e1_scope_authority.json --once
```

移除`--once`后每60秒轮询，同一generation输出E1、E2、E4S、E6；保留current/previous两代，失败不切current。需要执行者自己的`first_pass_20261106/unified_main_continuation/plan.json`、`unified_scaling_continuation/plan.json`、`unified_mechanism_application/{plan,receipt,selection}.json`和原始回执，这些私有运行计划未随本包分发。E2固定80格、E4S36格、E645格；20个P8只按`job.continuation.original_receipt`唯一映射复用，不按最快结果选路。E3/E4/E5仍由原state_main汇总器负责。CPU placement默认继承，可显式给`--cpu-set`；完整明细在JSON/CSV/HTML，主图只显示已验证时间，统一Time(s)。

匹配调度证据的作者二进制前取消仍为DEFERRED_PREPARATION。缓存输入ready保持原native call2，cached first独立补充，不进canonical-CSR cold-first排名/分母。两格TW WCC仅在实际原CPU graph_layout触发int32限制且输入/原worker/准备回执全匹配时记为INPUT_LAYOUT_UNSUPPORTED_CPU；没有GPU时间，不表示函数在其他图未实现。源码包不包含滚动结果、数据集、机器二进制或原日志。

`check_remaining.py`保留原15项元数据回归检查源码，使用显式`--runtime --snapshot --current-bundle --compatibility --scratch`参数。该检查绑定2026-11-06已捕获的特定回执集合；这些私有原始JSON未随包分发，不应对未来滚动快照套用固定历史数量。公开适配验证另见`../provenance/followup_public_validation.json`。
