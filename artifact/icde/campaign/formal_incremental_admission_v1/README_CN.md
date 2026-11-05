# 固定矩阵逐格正式准入

本候选解决“一个目标格尚未通过，阻断整个组全部正式测量”的调度问题。保留原完整矩阵与每格 10 次独立重复；只有当前源、完整 tiny／target 深层证明合格且从未尝试或预约的格，才进入独立批次。未完成、质量失败、基础设施失败、经主代理逐格审核的容量／平台限制仍留在完整账本，没有缩小分母或把资格当正式结果。

**当前支持 SG、连续 16 请求序列和 native。E4-S 明确 HOLD。** E4-S 仍执行原 recoverable sidecar 检查，但其冻结专用控制器不接受 CPU gate 前置项；本实现不会改用普通控制器或取消归档要求。后续须有独立审查的 E4-S 增量控制器。native 的四个附加 numeric 资格是共同门，缺一项不能逐格绕过。

## 保留什么，改变什么

- 复用 SHA 冻结的 `formal_admission_disposition_v1/inspect_ledger.py` 和 `formal_admission_d84_v2/admit.py` 深层证明读取器：完整输入 SHA、源树、DSO、实际数值质量、独立完整输出、tiny／target 的相同 epoch／方向／准确绑定、CPU 准备与 E4-S 参考／输出合同均保留。
- 每次观察保存 `complete_ledger.json`，仍包含所有原始目标格和全部正式槽。`QUALIFIED` 仅说明有资格进入正式测量，正式时间仍为 null，不能算胜出。
- 原正式 job 的整个 JSON 对象不改：命令、Python 入口、环境、配置、bindings、10 个 repeat、输出路径、depends_on、timeout、资源 guard 和维护均相同；选中后的相对执行顺序也与完整原计划相同。
- 新批次在前面放一个每格 CPU gate，ID 使用该格原 target ID，满足原有 depends_on。该 gate 再验证预约、当前源树与精确 proof 身份；实际 worker 仍执行原有 proof 校验。CPU gate 的 phase 是 cpu，不计入正式次数。
- 每次读取都使用冻结父读者完整 SHA，**不使用 stat／时间戳缓存**。独立审查实测同长度改写可能保留相同时间戳，因此即便 inode、size、mtime、ctime 全部相同，也不能跳过完整字节验证。

默认 `--output` 只生成观察报告。显式 `--admit` 也只生成计划和独占预约，**不启动 GPU 或任何 worker**。`run_incremental_queue.py` 是唯一后续执行入口，仅主代理在资源检查后调用。

## 防重与失败边界

所有检查、预约和正式准入提交都在原 `root_controller.lock` 下。首次提交会在原整组 admitted 目录与原 queue 目录建立带源／完整矩阵身份的 fence；旧准入器的“目录不得存在”与旧控制器的独占创建会阻止它们重复启动。若旧目录已有真实 plan／queue／state，或原 controller log 已存在，本候选拒绝迁移，不能自行覆盖。

原输出目录或维护目录只要存在，或显式提供的其他旧 queue state 已认领一个格中的任意重复，整格都不再被自动发放。所有批次在新的 group namespace 中保留不可变 reservation。预约先落盘，计划随后提交；崩溃后的 RESERVED、孤立 batch 或索引不一致均失败关闭，不能自动重试而产生第二组重复。改变 source、pending 矩阵或 namespace 所有者必须重新审查，不能继承旧预约。

执行入口再次在原锁下确认 spec／plan **路径和字节**与 registry 中唯一 COMMITTED 批次相同，输出必须是其唯一 `batch/queue`。复制合法计划后改命令、删 CPU gate、换路径或换输出目录均不允许。没有隐式 resume／retry；既有失败需保留并另行评审。

## 原维护如何看到新批次

原维护的 registry 已绑定原整组 queue 路径。为不修改正式 job 的维护命令及 bindings，新 wrapper 调用原 v8 控制器，仅在其 `run_process` 前插入已完成记录投影。此时原 controller lock 仍持有；原 worker、监督、GPU lock、时间边界与维护代码不变。

投影将已注册批次真实 v8 state 中的原始 completed formal 记录，按原 ID／receipt 原样汇集到已封锁旧 queue 的 `state.json`，供既有无损输出去重发现它们。仅接受绑定计划的实际完成前缀，拒绝伪造 receipt、跨批重复 ID 或计划变化。当前 active worker 不作为完成项。投影有独立 schema，`projection_only=true`、`physical_worker_runs=0`、`performance_eligible=false`，**没有另算 formal／attempt 次数**。收集器及导出器必须排除该 schema，按原始 receipt 身份统计一次。

下一次 worker／维护前会刷新此前维护状态；原 v8 最终退出（包括最后一项维护失败）后，wrapper 再取相同锁刷新最终字段。投影中的 worker PASS 不代表维护或正式可用性通过；失败字段不会被清除，最终性能导出仍应直接审计真实批次队列和完整测量。

实际冻结维护链为 `completed_job_audit_maintenance_v2.py` → `completed_output_maintenance_incremental_v1.py` → `incremental_completed_output_v1.py`。最后者的 `completed_arrays()` 仅读取 `state['completed']` 及每条真实回执，不要求原队列 schema 或 plan 字段。它完整核对顶层和嵌套 JSON 回执，只对 `.npy/.npz/.i64/.f64` 数组做原路径硬链接去重；所有运行路径保留，JSON 回执不会移动、删除或改写。投影因此可在维护后继续严格验证原 receipt 路径／大小／SHA。数组内容变化或回执缺失、替换仍是错误，不能由投影修补。

## 主代理创建请求

请求为 `eggpu.formal.incremental.request.v1`，字段如下；所有 binding 都必须是实际文件的 `{path,bytes,sha256}`，路径采用解析后的真实文件路径。Python 调用路径单独保存，例如 `envs/core/bin/python`，不能把其 symlink 指向的共享可执行文件路径用作运行入口。

```json
{
  "schema": "eggpu.formal.incremental.request.v1",
  "group": "sequence",
  "parent_manifest": {"path": ".../formal_admission_d84_v2/manifest.json", "bytes": 0, "sha256": "实际值"},
  "namespace_root": ".../formal_incremental_admission_v1/runtime",
  "controller_lock": ".../root_controller.lock",
  "controller": {"path": ".../run_campaign_queue_v8.py", "bytes": 0, "sha256": "实际值"},
  "supervisor": {"path": ".../owned_process_supervisor_v2.py", "bytes": 0, "sha256": "实际值"},
  "python": {"path": "Python真实可执行文件路径", "bytes": 0, "sha256": "实际值"},
  "python_invocation": "EXPERIMENT_WORKSPACE/envs/core/bin/python",
  "additional_prior_states": [],
  "additional_legacy_directories": [],
  "dispositions": []
}
```

`additional_prior_states` 和 `additional_legacy_directories` 应列出主代理检查到的其他旧正式控制器／派生计划命名空间；不能故意漏报已有任务。原 manifest 的 queue、admitted 和 controller log 自动检查。`dispositions` 只接受既有 inspector 的主代理逐格审核回执，不能自动生成 OOM／平台终态，更不能豁免质量失败。可选 `max_new_cells` 只按冻结矩阵顺序限制批量大小，不读取性能来挑选。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  EXPERIMENT_WORKSPACE/envs/core/bin/python -B admit_incremental.py \
  --request /明确的新请求.json --output /明确的新观察目录

# 完成审查后生成独占批次，仍不执行 GPU：
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  EXPERIMENT_WORKSPACE/envs/core/bin/python -B admit_incremental.py \
  --request /明确的新请求.json --output /另一个新观察目录 --admit
```

主代理使用返回 `admission.startup` 的参数启动该批次，先完成实际 UUID、GPU／主机内存、磁盘与进程占用检查。不要直接启动原 v8 跳过 wrapper，也不要另启动被 fence 的旧整组队列。

本目录 CPU 测试使用真实冻结 proof reader 和明确的合成文件，覆盖两格之一失败仍保留完整矩阵、逐批十次、并发预约、防重、原 source／输出篡改、旧队列、崩溃预约、runtime 入口替换、投影锁和最后维护失败。它们不产生真实 GPU 性能证据。实际运行状态以主代理部署后的源绑定回执为准。

两项集成测试实际运行原 v8、原进程 supervisor 和上述三层维护子进程链，worker 使用明确的 CPU 合成数组入口，CUDA 不可见：成功场景执行 1 个 CPU gate、10 个 CPU 合成 formal worker 和 10 次真实维护，20 个数组路径去重后共享一个 inode，全部原 JSON 回执仍逐个通过 SHA；失败场景故意破坏最后一个合成数组，真实维护在完整 SHA 检查中拒绝，最后 worker PASS／maintenance FAIL 与 finished 时间原样刷新到投影。测试没有模拟 v8 或维护的执行逻辑，所写 formal 字段仅为隔离的 CPU 调度测试数据。
