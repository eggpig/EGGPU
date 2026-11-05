# 正式矩阵逐格处置：独立 CPU 检查候选

本候选解决的是“真实容量/索引/平台终态与未完成错误混在一起，无法清楚审查完整矩阵”的记录问题。它保留原 `formal_admission_d84_v2`、所有冻结计划和运行队列，只创建独立检查结果。**当前没有正式准入、派生执行计划或 GPU 启动能力。** 即使所有格都已经解析，结果仍是 `INSPECTED_NOT_ADMITTED`，`formal_admitted=false`。

`inspect_ledger.py` 固定读取 SHA-256 为 `467c9b4cddd1c21d43c848b74364c6f817b51da14fd0bc6aba26531030ae54cc` 的原 v2 helper，复用其中纯 CPU 结构与深层证据验证函数；不调用其 `inspect(..., mode='admit')`、worker 入口或配置 CUDA 的函数。若父 helper 字节改变，候选直接拒绝，应审查新版本。

## 已实现

- 从绑定的原 manifest、qualification plans、formal pending 推导完整 target 矩阵；每个原 target 恰出现一次。formal 必须仍完整覆盖原矩阵，每格保留重复 `0..9`，命令、环境、解释器、worker、依赖和资源参数不得改变。
- 保存每个原 target/tiny job 和十个完整 formal job，包含原源码/输入/runner/资源/超时/磁盘门/维护绑定。`cell_identity_sha256` 对整个原 cell 和十个 formal job 摘要，处置记录不能跨版本、跨卡数、跨参数套用。
- `QUALIFIED` 必须通过原 v2 的真实 proof reader、当前源码全树和 DSO 哈希、epoch、完整 canonical 输入文件、完整输出与共同质量检查。额外的 native 数值域资格不能被逐格处置绕过。
- E4-S 继续使用原恢复归档合同，要求每份资格的 `recoverable_outputs.json`，执行既有 `validate_sidecar` 并重新核对对象及 reference metadata。只有 SHA 没有可恢复输出时不能成为 `QUALIFIED`。
- 终态仅接受主代理显式提供的逐格 review 文件，不能从 `FAILED`、日志关键词、missing 或当前 PASS 子集自动推断。逐格绑定原失败/小图收据、epoch、输入、setup、资源观察及原始日志；日志内容变化或所声明错误标记不存在时拒绝。
- 分类为 `QUALIFIED`、`OOM`、`TO`、`INDEX`、`PLATFORM`、`PENDING`、`CORRECTNESS`、`INFRA`。worker 异常时默认的 `quality_pass=false` 不等于实际数值错误；明确 `FAIL_QUALITY` 或完整质量检查失败才记为数值错误，也不能通过能力处置覆盖。
- 所有展示槽位 `time_seconds=null`、`counts_as_win=false`、`execution_scheduled=false`。资格通过本身不产生正式测量。终态仍占原矩阵格，不生成虚假十次成功或零耗时。

已用原固定四组计划作 CPU 结构核对：SG 56 格/560 槽，native 88/880，sequence 6/60，E4-S 36/360。这里的 1860 是这四组正式槽位总数，不是整个 E1–E6 campaign 的总任务数。

## 输入和使用

运行目录须能访问原 manifest 内的真实绝对路径。`binding` 统一是 `{ "path": "绝对路径", "bytes": 整数, "sha256": "完整SHA256" }`；程序会读取全部文件字节核验，不把 mtime 当内容证明。

请求文件形状：

```json
{
  "schema": "eggpu.formal.disposition.inspect.v1",
  "group": "sg",
  "manifest": {"path": "原manifest绝对路径", "bytes": 0, "sha256": "需要实算"},
  "expected_cell_ids": ["必须列完整原target IDs，或省略此字段由原矩阵自动推导"],
  "dispositions": []
}
```

这是字段说明，示例占位值不是可用证据。`group` 可为 `sg/native/sequence/e4s`。`dispositions` 是由主代理审查并明确提供的处置 JSON 文件 binding 数组；空数组表示不豁免任何失败。程序不会自己写 review，也不会自己把记录标成已批准。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -B inspect_ledger.py --request REQUEST.json --output NEW_LEDGER.json
```

输出必须是不存在的新文件。这里只产生一个完整 ledger 报告，不能当队列计划交给 runner。它保留所有 proof/输入/日志 binding，完整数组哈希可能有显著 CPU/I/O 开销；应选适当 CPU 与时段，不能和计时实验争用资源。

## 逐格主代理 review 合同

处置文件必须含以下字段，且必须已经过主代理对原始证据的实际审查：

| 字段 | 要求 |
|---|---|
| `schema` | `eggpu.formal.cell.disposition.v1` |
| `cell_id` | 原 target job ID，只能是原矩阵的一格 |
| `classification` | 仅 `OOM/TO/INDEX/PLATFORM`，不能填 PASS 或 QUALIFIED |
| `cell_identity_sha256` | 首次检查 ledger 中该格的摘要，涵盖十次原任务及资源命令 |
| `source_sha256 / epoch_sha256 / input_sha256` | 与精确失败目标及当前冻结源一致 |
| `failure_stage` | 实际失败阶段，不以“失败”笼统替代 |
| `review` | `authority=primary_agent_explicit`，`decision=accept_capability_terminal`，非空 `rationale` |
| `evidence.target_receipt / tiny_receipt` | 原精确 target/tiny 收据 binding，必须存在；tiny 必须深层通过且与 target 同 epoch |
| `evidence.setup` | 与冻结命令 `--setup` 相同的 binding |
| `evidence.raw_logs` | 非空且无重复的原始日志 binding 列表 |
| `error_marker` | `{path,text}`，必须在其中一份原始日志中逐字存在 |
| `evidence.resource_observation` | 下述主代理审查后的资源观察文件 binding |
| `facts` | 下述分类专有字段 |

资源观察文件 schema 为 `eggpu.reviewed.failure.resources.v1`，包含同一 `cell_id/source_sha256/epoch_sha256/input_sha256/failure_stage`，并明确：`execution_attempted=true`、`unrelated_gpu_contention=false`、`disk_preflight_passed=true`、`host_preflight_passed=true`、`gpu_uuids` 与原命令完全一致。其非空 `evidence` 是原始资源记录的 binding 集合，递归核验全部字节。这里是主代理显式审查记录，不假称程序能够仅靠文本自动证明故障归因。

分类约束：

- **OOM**：`memory_domain=device`；`error_class` 为 `cudaErrorMemoryAllocation/CUDA_ERROR_OUT_OF_MEMORY/rmm::bad_alloc/std::bad_alloc_device` 之一，并实际出现在原日志标记里。主机内存、磁盘不足、他人 GPU 占用不能归为此类。
- **TO**：原日志必须有 timeout/timed out 标记；`frozen_timeout_seconds` 必须精确等于 target job 原 `timeout`；`observed_elapsed_seconds` 已达到该期限；`owned_processes_terminated=true`。本版只覆盖冻结控制器超时。作者预处理内部另一个 1800 秒时限若不同，仍保持未解决，不能拿它冒充控制器时限。
- **INDEX**：原日志含 `INDEX_UNSUPPORTED`；`required_entries > supported_max_entries >= 0`；`index_bits` 为 32/64，`index_domain` 为 `global_offsets/local_offsets/local_vertices`；另需 `evidence.index_geometry` binding，含相同输入身份、完整弧保留标记及实际/上限规模。
- **PLATFORM**：本版仅覆盖作者功能实际启用的 peer-access 错误 217，要求 `required_feature=peer_access`、`native_error_code=217`、`author_feature_enabled=true`，且 217 出现在原日志标记。其他平台限制仍需新审查实现，不能以泛化 unsupported 放行。

由于本版要求有精确 target epoch 和已通过 tiny，**在构建阶段或 tiny 阶段就失败的平台案例仍不能由本版终态合同豁免**。它们保持 PENDING/INFRA，不能伪造 target。此限制是有意的保守范围，后续扩展应增加独立、来源绑定的平台能力门。

## 暂未实现与后续准入要求

本版无任何执行计划生成能力。`all_cells_resolved_and_proofs_checked=true` 只表示本次只读检查中，所有原格均为深层合格或主代理已审查能力终态，额外资格及原 CPU 门也通过；不表示获得正式准入。

仍完整保留原 v2 的全局 CPU reference gate，尚未按实验依赖拆分，因此无关 reference 可能继续阻塞本次“全解析”状态。这一版没有降低任何原 oracle 门。

正式派生的下一版本仍需：在全局 controller 锁下重验源/证据和处置快照；仅为合格格保留全部十次独立测量；在完整展示矩阵保留能力终态；复用 E4-S 必需 postprocessing controller；逐一检查派生 command/environment/维护/资源门未改变；原子写新的 admission receipt/plan；不能把本版只读 ledger 当作这些步骤的替代。

## CPU 验证

`test_inspect_ledger.py` 使用原 v2 真实 proof reader 和独立合成字节证据，隐藏 CUDA、固定两个 CPU、临时目录在 EGGPU_TMP。36 项通过；含遗漏/重复格、缺重复/越界重复、旧 epoch、篡改日志/完整输出、假成功、主机 OOM 误分类、数值失败被能力标签覆盖、E4-S 缺可恢复输出、CPU 参考未完成等拒绝路径，也验证四种显式能力终态保留十个展示槽位。独立审核指出的 numeric 协议早返回、能力终态方向语义、正式文件绑定及 E4-S 源序列合同也已加入拒绝回归。

这些是 CPU 工具资格，不是实际七图的处置审批、正式准入或 GPU 性能结果。见 `cpu_test.log`、`cpu_receipt.json` 和 `manifest.json`。
