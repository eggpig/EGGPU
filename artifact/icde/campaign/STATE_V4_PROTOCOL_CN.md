# E3–E5 独立 v4 入口与证据合同

此目录从 `state_cache_candidate` 派生，不覆盖活动 `campaign_20261104/state_worker.py`。GPU 运行仍只由主代理提交。当前 CPU 检查通过；本入口的 GPU tiny、真实目标资格及正式十次重复尚待执行，不能沿用旧 v3 的资格标记。

## 保持的工作量与干预

- E3：单图 US / IN / OR 分别运行，每条正式 trace 为 20 次权重提交、80 次查询。
- E4：US / OR / IN / UK 的 32 次访问、128 次查询；`once` 仍是单独的无重访控制，不与 128 查询主项混合。
- E5：IN / UK / OR 的 24 次权重提交、96 次查询。
- 查询顺序固定 BFS → SSSP → WCC → weighted PR，输入与三投影、确定性权重更新、源点轮换沿用 v2 输入协议；质量阈值不变。
- D0 为 persistent_full，D1 为 lease_sparse；C0 为 closure_lru，C1 为 closure。E4/E5 物理容量仍为 `floor(B_min + beta*(B_all-B_min))`，由当前源、实际 GPU 配置及输入池的独立 physical-byte pilot 冻结。pilot 不按速度选择预算。

E3 新增 `--old-readers none|per_update`。两种情况均先完成相同的四次 base 查询。per_update 在每次 commit 之前调用真实 `acquire_snapshot_lease(session, base)`，持有该 base 的 FP64 SSSP 与 weighted-PR 原生 resident views，在新版本四个完整查询结束后显式 close。它表示已完成查询的输入视图仍被读者持有，不表示 CUDA kernel 仍在运行。D0/D1 使用同样的 acquisition、close 与记录边界；per_update 的三次 owner inventory 审计也在该边界内；none 不制造虚构读者。旧输出数组的存活与此原生输入租约是两种独立所有权。

每次租约记录取得前、持有时、释放后的 owner inventory，逐 owner 检查 active_leases 增一和恢复，每设备应有两个 weighted views。query 原生 `per_device_delta` 记录保留，额外汇总 COW、patch 与 lease 计数。tiny 会在持有前后对旧视图的完整 FP64 weights 读回并逐 view 比较 SHA；target/formal 完全不调用 `read_weights()`。tiny 因含这些额外资格工作，不能作为论文性能结果。

## 连续计时、完整输出与失败保留

`first_complete_workflow_seconds` 是 Session 构造至 trace 完成的连续真实时间；`workflow_complete_host_seconds` 从 E3 固定 base 初始化后、或 E4/E5 创建空执行历史后开始。每次 commit、四函数请求、真实读者取得/释放及统一 event 记录均在区间内。两列不得用阶段和替代。最终 inventory、输出哈希、CPU oracle、落盘和 Session close 位于区间外，并明确记录。

每个请求必须返回 canonical n 个元素的 owning、readonly、FP64 完整向量。所有请求输出独立且保留至 Session 关闭，关闭前后核哈希。跨 run 的 `--output-store` 按 dtype/shape/完整字节 SHA 寻址，用无损 NPZ 保存实际向量；相同字节只存一个对象，每个请求保留独立指针与质量结果。现有对象不会覆盖。读回验证压缩文件 SHA 和未压缩向量 SHA；`output_index.jsonl` 在每个完整输出入库后刷新，后续质量失败不删除已有证据。没有按速度或优劣筛选输出。

CPU 参考缓存只缓存由 canonical 输入独立算出的答案，绑定原始字节、修改位置、版本、源点、CPU 实现、NumPy/SciPy 版本与数值门限。每个请求仍核完整向量；每个 PR 请求仍独立重算 fixed-point residual。参考缓存不接收 GPU 答案，位于计时外，默认 4 GiB，可淘汰重算。原始输出对象库不采用该淘汰策略。

## tiny → target → formal

`--phase tiny` 只接受 `qualification_only=true` 的小图输入；`target` 和 `formal` 只接受冻结真实输入。`--repeat` 正式为 0–9，tiny/target 为 0。target 必须提供匹配 tiny proof；formal 必须同时提供 tiny 和 exact target proof。

匹配项包括实验、D/C arm、old_readers、beta、order、GPU UUID/数量、CPU affinity、控制环境、质量协议、Python/runtime 身份、实际 DSO、全部 Python/C++/CUDA/header 源码，以及完整 worker/protocol/reference/input/resource-gate 文件。目标资格还精确匹配输入回执、图池顺序、已冻结物理预算和 pilot 回执。proof 的每个完整输出对象都会重新读回校验，旧 hash-only 收据或旧源资格不能通过。源码变化后必须重新 tiny/target/pilot，不能编辑旧收据绕过。

主代理先指定 CUDA DSO 所在目录和当前源路径，使用同一物理 UUID 与 16 CPU 核绑定。示意命令如下，变量均由运行机器显式传入，不含内部 SSH 信息：

```sh
python state_worker.py run --execute-gpu --phase tiny --repeat 0 \
  --experiment E3 --arm D1C1 --old-readers per_update --datasets TINY_A \
  --inputs "$TINY_INPUTS" --source "$SOURCE" --native-dso "$DSO" \
  --gpu-uuids "$GPU_UUIDS" --cpu-set "$CPU_SET" --lock "$GPU_LOCK" \
  --output-store "$RESULTS/objects" --reference-cache "$RESULTS/references" \
  --output "$RESULTS/e3_tiny_D1C1_old"
```

E4/E5 先以 `action=pilot` 对对应图池运行，再给 `run --pilot <pilot/receipt.json>`。target 增加 `--tiny-proof <tiny/receipt.json>`；formal 再增加 `--target-proof <target/receipt.json>`。tiny 的实验/arm/读者/beta/order 要与 target/formal 一致。每次正式 repeat 运行独立进程和独立 output 目录，共享 immutable output-store；沿用主代理全局串行 GPU 锁、进程组超时与资源检查。

本 worker 没有在进程内隐藏启动全部 sweep，也没有声称该目录能一键完成七图及所有 baseline。输入下载/生成、全部配置计划、进程组 deadline 由 campaign controller 提供；入口在 GPU 前检查实际 GPU 空闲与主机余量，并按所有完整向量的未压缩上界预留磁盘，不预支去重收益。

## CPU 验证与未完成项

34 项 v4 契约测试通过：四臂×两种读者生命周期、目标阶段禁止旧权重 readback、错误视图清理、全向量内容寻址、旧 proof/配置/目标输入/预算拒绝、80/128/96 请求数等。28 项独立参考缓存检查与 25 项实际生产输入/SSSP policy 门检查通过，共 87 项。CuPy/CUDA 算法没有由本代理执行。GPU COW/释放计数、完整结果及性能必须由当前入口与当前 source/DSO 的实测收据给出。
