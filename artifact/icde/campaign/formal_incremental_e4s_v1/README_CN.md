# E4-S 固定矩阵逐格准入

本目录接续已冻结的 `formal_incremental_admission_v1`，仅处理 E4-S。通用入口仍明确 HOLD E4-S；本入口使用专用恢复控制器。完整原矩阵、每格十次独立正式重复、未完成／失败／经逐格审查的容量状态均保留，不删除分母，也不把资格或 CPU 测试算作正式测量。

## 准入和归档

每次在原 `root_controller.lock` 内先扫描冻结资格计划，只有原始状态为 `PASS_E4S_QUALIFICATION` 的完整输出才能调用既有 `archive_outputs.py`。缺失 sidecar 时补建，已存在时重新完整校验；原 worker 回执不改写，不重跑 worker。归档失败保留错误，随后原 `inspect_ledger.py`／`formal_admission_d84_v2/admit.py` 的完整字节、source／DSO、epoch、数值、参考与 sidecar 深审仍会拒绝该格。

只有当前完整合格且没有任何旧预约／输出／维护尝试的格才发放十次重复。使用与通用候选相同的独占 reservation、完整原矩阵 ledger、旧整组 fence、原 controller lock、失败关闭及跨批防重。执行入口仍把 spec、plan 的路径和字节以及唯一 batch/queue 输出绑定到 COMMITTED registry。没有自动 retry 或 resume。

每个格的 CPU gate 再核验当前源树、DSO、原 tiny／target／epoch 身份、准入时精确 sidecar 字节，以及所有可恢复对象的完整内容。随后原正式 worker 继续原有证明检查，原命令、环境、timeout、资源守卫、repeat 和输出路径不变。

## 专用控制器和维护

原 `decorate_formal_plan.py` 只将正式 job 的原维护包装为强制归档后处理，并增加原协议所需字段；原维护完整嵌入绑定 spec。原 worker 的其余字段逐项比较相同。本目录专用控制器相对 `e4s_recoverable_outputs_v1/run_e4s_formal_queue_v1.py` 仅允许带明确标记和固定入口、CUDA 不可见的 CPU cell gate，并将其排除在 worker_passed/worker_failed 计数之外。

正式路径仍采用原专用规则：worker PASS、可恢复输出 PASS、原维护 PASS 必须全部成立，才计一个 formal measurement。后处理失败保留 `worker_outcome=PASS` 和 `outcome=FAIL_REQUIRED_POSTPROCESSING`，记录 archive 或 original_maintenance 失败阶段，停止当前批次；原原始回执不改成失败，也不把它作为可用正式样本。出口仍应调用原 `export_eligible.py` 完整核验。

维护投影只汇集真实专用队列已完成的原 ID／receipt，带 `projection_only=true`、`physical_worker_runs=0`、`performance_eligible=false`，不产生第二份计数。当前项在后处理完成前位于 `pending_postprocessing`，不会提前投影成完成；因此原数组去重至多晚一项看到它，最后一项由 finally 刷新投影但不自动追加第二次维护。E4-S sidecar 的内容寻址对象独立完整，不依赖这项去重延迟。投影绝不作为维护成功或正式性能的证据。

## 使用

请求沿用 `eggpu.formal.incremental.request.v1`：`group="e4s"`；parent_manifest 为当前原冻结 manifest；新的 namespace_root；原 controller_lock；controller 必须绑定本目录 `run_e4s_formal_queue_v1.py`；supervisor 必须绑定本目录原字节拷贝；Python 的真实文件 binding 与实际环境入口分别记录。完整字段和旧队列清单要求参照通用候选 README。

`archive_qualified_outputs` 默认 true，在原锁内补归档。显式 false 只观察已有资格恢复状态，缺 sidecar 的格仍不会准入。它不是跳过归档检查的开关。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  EXPERIMENT_WORKSPACE/envs/core/bin/python -B admit_e4s_incremental.py \
  --request /新的明确请求.json --output /新的完整观察目录 --admit
```

此命令仅做 CPU 归档／审计／预约和计划生成，不启动 GPU。主代理完成实际资源检查后才能执行返回的 `admission.startup`。不要直接用普通 v8 执行本目录的正式计划。

CPU 测试使用隔离的小型合成测量文件，实际调用原 sidecar 归档、原专用进程监督、原三层维护链和原出口，验证成功、最后归档失败、最后原维护失败、最后 worker 失败四种路径；测试不提供论文性能结果。独立审查记录和最终测试状态见 `cpu_receipt.json`。
