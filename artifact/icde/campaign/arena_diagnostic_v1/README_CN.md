# 单卡现有分段指标独立诊断

当前 d84 资格证据发现 US 序列 PR、IN 热 WCC 和 TW 首次 PR 值得定位。现阶段没有内核根因证据，本候选只读取已有指标，未做算法或源码优化。

`worker.py` 在独立进程导入 SHA 固定的 `pr_global_l1_i64_v6` worker/adapter，并通过其原执行器跑相同 `[BFS,SSSP,WCC,PR]×4`、源序列 0/1/2/0、相同 FP64/global L1/完整主机输出与原 independent reference。新协议为 `e1-arena-existing-metrics-diagnostic-v1`，拒绝 formal、repeat!=0、cuGraph；所有结果强制 diagnostic_only=true / performance_eligible=false。它不会改正式 worker 文件或正式 epoch；诊断 epoch 额外绑定自身与 capture 源文件，新 tiny 才能资格放行新 target。

`capture.py` 只追加已计算的 QueryResult.backend_metrics、QueryResult.seconds，以及现有对象字典中的 host_view_seconds / graph_host_prepare_seconds / counters。读取用 vars(state)，不会触发 DemandState.__getattr__，不会预构建 incoming/degree，也不调用额外 query、transfer、GPU sync 或 kernel。返回原完整 owned readonly `.values`，不替换结果。观测开销包含在诊断连续时间且单列，不能与正式时长混合。

## 现有指标能回答什么

- prepare_seconds、execute_and_download_seconds、packaging_seconds、housekeeping_seconds：主机准备与执行/下载的粗分段。
- host_view_seconds_delta.incoming：首个 PR 时 DemandState 用完整 CSR `.tocsc()` 构建 incoming 的时间；degree/sums/in_weight 也有独立已有计时。此前 BFS/SSSP/WCC 使用 outgoing，因而首次 PR 可能才支付 incoming 代价，但尚未实测证实 99.78 秒的主因。
- factorized_kernel_load_seconds、eligibility_seconds、low_degree_fused、row_cooperation：模块加载与版本化低入度判断、实际 PR 路线。
- iterations、queued_iterations、convergence_checkpoints、checkpoint_interval、final_global_l1、converged、device_loop_seconds：实际收敛与 device 区间，检查是否为迭代数或设备执行主导。
- download_seconds、d2h_seconds、result_d2h_bytes 与 pr_public_output_ownership：完整输出与 bridge adoption/copy。
- connectivity_module_load_seconds、device_loop_seconds、download/d2h、wcc_public_output.canonicalization_seconds、ownership：将 IN 热 WCC 的 union、主机证书和完整输出区分。
- counters_delta.h2d_bytes 与 host_prepare 时间结合定位上传规模；现有代码没有独立覆盖所有 H2D 的持续时间。不得把 prepare-host_view 简单命名为精确 H2D 时长。

这些 timer 有嵌套和不同 clock 范围，不能相加重造总时长。诊断保留全部 16 调用，不能只挑快调用/源点，也不把失败或超时转换成测量值。

## CPU 验证和运行前条件

6 项 CPU 测试通过：读取不触发 lazy materialization；每个请求只执行一次且返回同一完整 owner；复用真实冻结 sequence 执行器保持全部 16 请求/源序列和关闭后输出；新 epoch/child 路径绑定并拒绝 formal。

`diagnostic_draft.json` 含 2 tiny + US/IN/TW 3 target 的独立 job_specs；jobs=[]，尚未部署或运行。`prepare.py` 已提供 CPU 部署：检查实际 d84 sequence plan SHA、全部冻结源码与 DSO、原 runner/job bindings，并保持除诊断入口/output/tiny-proof 外的全部 CLI、环境、资源和 timeout；自动生成独立 queue registry、增量维护和完整 output+oracle+最大 temporary 容量 guard。仍不启动 GPU。原正式队列与原始结果不得复用本诊断作为资格。

CPU入口：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider test_capture_cpu.py`。真实 GPU 仅由主控显式 --execute-gpu；本 agent 未执行。

部署入口：`corePython -B P/arena_diagnostic_v1/prepare.py --campaign P --base-plan P/unified_release_d84_v2/sequence/plan.json --base-sha256 <exact plan SHA> --output P/arena_diagnostic_v1/deployed_v1`。产物 receipt 含主控可审核的 startup。新 tiny/target 的 output 不覆盖任何正式或旧资格结果。
