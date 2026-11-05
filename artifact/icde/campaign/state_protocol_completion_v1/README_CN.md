# E3/E4 缺失对照补齐：独立执行协议

本目录补齐三类缺口：E3 的完整重建臂、独立只读函数顺序对照，以及 E4 的三种保留对照。**CPU 实现和验证不代表 GPU 资格或正式结果。** d84 源码、6dfc DSO、旧 worker、旧计划和旧回执均未修改。新协议为 `state-complete-workflow-completion-v1`；源代码/DSO 之外还绑定本目录全部执行文件，因此旧资格不能自动转入。

## 1. 每个对照实际执行什么

| 实验/臂 | 实际执行 | 解释边界 |
|---|---|---|
| E3 Rebuild | 每次权重提交从规范 CSR 重新物化本版本完整权重、range 分区、局部编号/owner、BFS 和 weak-WCC 表示；创建新 manifest 并重新导入完整 host 分区。释放所有无 lease 的旧设备 owner，禁止 shared catalog 保留；随后四个原生函数重新构建设备状态 | 使用已有 **文件 manifest 接口**，写入/读取/校验与 host 构造均计入完整轨迹，回执单列。不是专门优化的内存重建器，也不把它当唯一强对照 |
| E3 D0C1 | 原有 `persistent_full + closure`：复用有效拓扑，未命中权重全量维护 | 主要强对照，不改名为 Rebuild |
| E3 D1C1 | 原有 `lease_sparse + closure` | FP64 固定拓扑权重维护，完整求解各函数 |
| E3R | 每函数四次，共 16 个只读请求；版本和请求多集合相同，只改变 grouped/interleaved 顺序 | 源点为预定 0/1/2/0 索引，不跨版本重排；两种顺序×两D路线按同repeat组成四格配对块并轮换执行次序，不冒充真实生产日志 |
| E4 No-retention | 完整结果已独立持有之后，真实释放所有可合法撤销的受管 owner 和 idle catalog，要求受管物理占用为零 | 不表示整个 CUDA 进程/上下文总显存为零；保留与其他臂相同的预算账本 |
| E4 view-LRU | 对单个 view 按最近使用时间排序；workspace 作为后备。每次实际移除 holder 并核验原生收据 | 不是旧 `lru` 的“当前调用优先级”策略。共享分配可能先释放 0 字节；此时只算删除 holder，不能计为容量已满足 |
| E4 closure-LRU / C-Retain | 原生 `closure_lru` / `closure`，相同合法闭包菜单 | 原有强对照和完整方法不变 |
| E4 logical-size demand | 原有 `gdsf` 路径：历史需求除以 owner 引用字节数；共享大小按逻辑 owner 计分 | 不是经典 GreedyDual/GDSF 实现；容量判断仍用真实物理占用，未使用未来请求或实测时间 oracle |

Rebuild 不能创建并存的两个 Session：当前原生 runtime 明确只有一个 owner。实现因此在同一独占 runtime 注册新的完整 host snapshot，并让真实 leased 旧 view 保持存活直到该批四查询结束。不会通过关闭 Session、全局清空 runtime 或伪造拓扑摘要绕过旧读者保护。新的 topology/weight/view 摘要都由实际新 payload 得到。

完整重建沿用原始规范输入的 `np.linspace` range cuts。第 r 版本只在冻结 positions 上写入 `1 + ((w_initial - 1 + r) mod 16)`；其余位置保持原值。WCC 重新构造完整去重弱投影；超过 int32 域时明确失败，不裁边。

## 2. 规模、隔离与计时

| 范围 | 配置数 | 资格任务 | 正式重复 |
|---|---:|---:|---:|
| E3 三臂完整密度/局部性/reader 网格 | 300 | 与下列合计 | 3000 |
| E4 五臂预算/顺序/无回访网格 | 160 | 与下列合计 | 1600 |
| E3R 只读顺序，两 D 路线 | 20 | 与下列合计 | 200 |
| 合计 | 480 | 180 tiny + 480 target + 4 byte pilot = 664 | 4800 |

这是完整补充协议的计划数，不是完成数。为保证同一 runner/计时边界，原 D0/D1、closure-LRU/C-Retain 也在新协议中重新测量，不与旧执行直接拼表。保留当前 E5、热点和无权 PR 的独立协议；本目录没有重启或替换它们。固定图/方向/来源、源点、WPR 语义、质量阈值、独立输出保存与十次重复规则均保持。

共享 catalog 对所有非 Rebuild 臂显式固定为原默认 512 MiB；Rebuild 固定为 0，这是重建干预的一部分，并进入配置和环境身份。E4 仍按相同新 byte pilot 得到逐设备预算。No-retention 的目标为零，不通过修改 budget 值伪造更低容量。

`first_complete_workflow_seconds` 包括初始化和整段真实请求；E3 同时单列 base 初始化以及更新轨迹。完整输出 hash、独立 CPU 参考、归档和最终 Session 关闭仍在主时间之外。Rebuild 每次的分区构造/文件写读/注册进入主时间；临时分区文件只在 Session 和 lease 结束后移除，参数、逐文件 SHA/字节、manifest SHA 和重建分项留在回执。容量检查额外预留全部重建 host snapshot 和临时磁盘，不能按仅权重增量预估其资源。

## 3. CPU 检查与尚待执行

`cpu_validation/current/cpu_receipt.json` 和 `cpu_validation/current/cpu_test.log` 记录实际结果；测试使用独立 CPU owner/allocation 图模拟原生 ABI，但运行真实生产 controller 的排序、预算、receipt 合并和释放逻辑。它检查零释放、共享 union、idle catalog-only 数组、leased child、错误 receipt、多设备、完整重建 CSR/权重/弱投影以及实际 planner 生成。原 worker 的完整输出/旧读者/proof 合同回归也保留。没有 CUDA 执行，不替代真实 GPU tiny/target。

当前 `draft/qualification_draft.json` 和 `draft/formal_pending.json` 均有 `jobs: []`。具体目标和全部 4800 repeat 仅在 `job_specs` 中列明，不能作为 GPU 队列提交。

部署前仍需主代理检查资源和候选。部署只执行 CPU 构建，在选定机器的**原 lexical venv Python** 下运行：

```bash
<existing-core-python> -B <candidate>/prepare.py deploy \
  --base-deployment <campaign>/state_unified_full_v1/main/deployment.json \
  --base-coverage <campaign>/state_unified_full_v1/main/coverage.json \
  --output <candidate>/deployed_v1 \
  --results-root <campaign>/state_completion_results_v1
```

`deploy` 核查现有源码、DSO、source signer、输入、环境和 runtime 身份，再生成实际 664 项资格队列；主代理审查后才能交给现有锁/资源监督器。正式输出仍为 `jobs: []`，只有 `state_plan.py admit-formal` 在每格精确 tiny/target 和完整输出证明通过之后才生成正式队列。任何失败保留，不能删配置或缩短十次重复。所有资格和正式执行都尚待实际回执验收。

CPU复现入口：在已有 numpy/scipy/pytest 环境执行 `python -B run_cpu_tests.py --source <Easy-Graph> --scratch <task-scratch-new> --output <receipt-new>`。该入口显式隐藏GPU、限用最多两CPU、检查测试期间runner文件未变。测试的所有权oracle已自包含；提取来源见 `cpu_oracle_provenance.json`，不依赖旧工作目录的测试模块。
