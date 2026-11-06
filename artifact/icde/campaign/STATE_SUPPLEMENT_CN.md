# Source71 后续实验准备补充

本目录是提交 `8953673f212eeaa7c7bd5bc4775b00dd1af9b45d` 之后的新增源码与 CPU 证据，不替换现有公开实现、旧计划或结果。运行实现仍为 source71；这里没有新 GPU 性能结果，也没有输入图、完整输出向量或二进制。SSSP 观察版与文本转换器已在目标机器完成 CPU 构建，39 项计划已准备。主任务已经接线，当前新测量仍待前置阶段与实际资格检查；实验尚未完成，SG/MG 性能目标尚未实现。

| 范围 | 配置与资格检查 | 主时间 |
|---|---|---|
| 主 state | 504 个目标；新的 focused 与六个物理字节 pilot | 直接记录的 `workflow_complete_host_seconds` |
| 独立追加 | 32 个热点 + 6 个 unweighted PR；两个新 PR tiny | 对应直接记录的 ready 连续工作流 |
| SSSP 子实验 | IN/OR × 六个密度 × D0C1/D1C1/Grapin = 36；三个 tiny | 直接记录的 `post_base_update_workflow_seconds` |

三组分母分别保留。SSSP 固定第一个 primary source，执行一个基础查询、20 次相同权重提交和20 次完整 SSSP 查询；21 个完整输出均按原精确数值契约检查。`prepared_service_seconds` 是补充列，不能当作 cold-first，也不通过阶段相减生成主时间。作者的删除、插入、显式权重和父节点修复路径不变；观察版只增加基础查询结束时间戳与 JSON elapsed 字段。实际 OR 总计数4,253,968,360仍小于 uint32 上限；资源不足明确记录，不能缩短轨迹。

新 PR 策略显式绑定 auto/implicit/pull_only/cpu/transpose，并在计时外检查实际路线。它是新的测量配置，不复用旧 general/materialized/full 结果。残差证书只改变计时后的独立输出审计：保留原数值阈值，报告误差上界，未满足证明条件时回退原 solver。tiny 使用原 oracle。

这些源码使用 `artifact/icde/campaign/<模块>` 公共布局；运行时仍需要外部保留的完整冻结 campaign、source/DSO、输入和原作者 build receipt。不要把经过路径映射的 JSON provenance 当作可执行绑定。CPU 回执的 `public_projection.original_receipt` 保留原字节 SHA；它们证明原本地/远端 CPU 检查，不声称公共目录中的所有历史测试已重新执行。测试的完整 frozen runner 依赖未复制进此小型补充。

`state_activation` 的三个脚本改为显式 campaign/Python 参数；替换旧 idle watcher 的脚本还需要明确旧 PID，SSSP scheduler 需要明确计划 SHA。它们属于主任务的调度入口，包含提交动作。`state_scope_additions/watch_fresh_additions.py` 的 bootstrap 只准备 WAIT_FOCUSED；watch 等新的实际 focused PASS，再准备追加计划，等核心队列结束后才按显式启动参数提交。它不等待 tail。ready 与 SSSP watcher 仅在回执变化时更新图表，pending/failed 不填时间。

主任务核实的接线状态见 `state_activation/wiring_status.json`：新的 after-main 与 SSSP 等待主表和缺失基线，尾部等待八个主要阶段（包括新38与SSSP），新38等待 focused。ready 导出为 WATCHING，当前新主 state 为0/504。SSSP持续导出已启动。`author_priority/pause_then_resume_author.py` 只在所绑定作者 controller 的完整作业边界保存与验证 checkpoint，再等待主表前置条件恢复原队列；它不终止正在测量的 worker、不重跑已完成前缀。打包时状态为等待该边界，不能据此声称后续任务已经完成。

历史胜率审计在 `state_audit_scheduling/history_metric_audit.json` 与配套 CSV。旧164/167、100/105等分母与当前七图矩阵的输入、函数、比较系统及估计方式不同，不能转写为当前“90%已完成”。历史四函数50/50 E2E与48/50 device子集保留，V15 minimum/mean口径差异明确记录。

本次唯一覆盖895既有代码的改动是 `../first_pass/export_on_progress.py` 的元数据触发条件：已有队列新增 target/formal 完成、失败或阻塞记录即更新；tiny/pilot不因数量累积重复绘图，继承的 completed 前缀去重。原采集器、数值检查、绘图逻辑、current/previous两代保留与公开 `--cpu-set` 参数保持不变。测试与原文件快照在 `../first_pass/export_trigger/`。

该触发修复已在目标机器完成一次完整导出，plots 为 PASS；公开布局的同一组五项 CPU fixture 也通过。该次 E1 元数据为329个格中218可用、89待完成、2个输入布局不支持、20失败；其中5个准备延后属于待完成项的子集，不另加分母。E4S为2个可用。这里只刷新已有证据，未产生新的GPU测量，也不能由可用格数量推断胜率。
