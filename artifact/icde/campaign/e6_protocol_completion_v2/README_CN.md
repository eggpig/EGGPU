# E6 两扩展组完整 2×2 候选

本目录是独立的执行协议候选。保持 `native_memory_input_v1` 的 d84 源码、原 DSO 和既有 GPU 队列不变；本目录尚未执行 GPU，也未准入正式测量。CPU 结果见 `cpu_receipt.json`，计划草稿见 `draft/receipt.json`。旧 1800 秒协议的结果保留原身份，不并入本候选。

## 补齐的实际执行路径

| 配置组 | 配置数／图 | 实际工作 |
|---|---:|---|
| 完整 EGGPU | 4 | P1、P2、P4、P8；多卡结构指标按 ego 顶点拆分、closeness 按已选源拆分，尾部 BFS/WCC 仍在逻辑 GPU 0 |
| CPU 外部扩展组 | 3 | structural、closeness、structural+closeness 各自委托现有 igraph 路径；BFS/WCC 留在 EGGPU |
| GPU 外部扩展组 | 3 | 相同三个切换，委托现有 cuGraph/CuPy composite；BFS/WCC 留在 EGGPU |
| 旧基础函数补充控制 | 2 | 单独 BFS 或 WCC 委托 igraph，其他函数 EGGPU |
| 完整外部系统 | 3 | igraph、cuGraph+igraph、cuGraph/CuPy composite |

两个扩展组 2×2 各由“都在 EGGPU、仅结构组外部、仅 closeness 外部、两组均外部”组成，共用同一份 P1 完整 EGGPU 控制，避免重复记录。每图共 15 个唯一配置；EP-U、LJ-U 是主应用输入，OR 是固定压力输入。资格为 15 个 tiny + 45 个 target = 60 项；正式候选为 3 × 15 × 10 = 450 个独立重复。这个数量是**计划数量**。

GPU 替代路线复用 `e6_reused.py` 的既有 `GPUCompositeBackend`：cuGraph triangle count 和 GPU 归一化组成 effective size；CuPy CSR 乘法、邻接掩码与按行归约组成 constraint；pylibcugraph BFS 加 GPU 归约组成精确 WF closeness。这些是明确标注的外部组合，不冒称 cuGraph 自带同名高级 API。没有新增或优化外部内核，`e6_reused.py` 与上一候选字节完全相同。GPU 失败会保留失败，不能回退 CPU 后仍记录 GPU 臂。

## 环境、完整任务和数值合同

所有 EGGPU 配置统一使用已有 `envs/rapids/bin/python` 进程入口，包括两个 2×2 共用控制与 P1/P2/P4/P8，避免在消融中同时改变 Python 环境。IGraph 包通过原先明确绑定的 core site 路径供实际 CPU fallback 使用。RAPIDS 与 EGGPU DSO 在同一进程能否共存仍须当前 tiny 资格验证，CPU 测试不代替它。完整外部系统使用各自原有适用环境。

每次完整调用限时 3600 秒；first、warmup、ready 是三次独立完整调用。外层队列的 86400 秒是包括三次调用、输入检查和审计的监督余量，不能把它当任务时限。`T_first` 从公共 host CSR 就绪开始，包含弱投影、导入、系统专有表示、上传、五个实际函数、两次依赖选择与完整 host 输出。预先缓存的 CPU 参考不进入实际选择或执行；准备参考、输出归档、审计与 close 单独记录。

沿用现行更严格的 `atol=1e-10`、`rtol=1e-8`；候选列表、winner、BFS 与 WCC 分区精确校验。所有顶点（含孤点）的 ES/Constraint、32 个真实选中源的 closeness、winner、完整 BFS/WCC 输出都必须恢复，且每次调用结果独立持有、不可变、存活于后续调用及 close。此阈值比最初选图文档更严，已显式版本化，不声称它就是最初的宽松阈值。

## 旧独立参考的窄桥接

`e6_cpu_reference.py` 只接受与 `legacy_contract.json` 完全一致的旧生产者合同，并检查新旧唯一差异为执行配置菜单与 1800→3600 秒。输入、算法公式、选择规则、输出域、精度、计时边界与重复次数不变。旧 producer 源码、原 runtime 版本、每个 oracle 的 receipt 和 NPZ 字节均校验；随后由未经修改的 `legacy_cpu_reference.py` 做全字段读取。这里复用的是 CPU 数值参考，不继承任何旧 GPU 资格或耗时。桥接 manifest 标记 `performance_eligible=false`。

`e6_reuse_provenance.json` 保留上游原记录：其 `output_sha256` 是最初机械抽取模块的哈希，后续 complete-output 适配已在同文件说明。当前模块的精确哈希以 `upstream_bindings.json`、`runner_freeze.json` 及新计划 binding 为准，不能拿最初抽取哈希冒称当前模块。

## 主代理部署与资格生成

1. 将本目录代码、测试及回执完整复制到远程同名目录，核对 `runner_freeze.json` 所列字节。不要上传本地 `draft/` 为可执行计划。
2. 在现有环境以 `CUDA_VISIBLE_DEVICES=''` 运行 CPU 预检。生成器会核验原计划所有输入／源码／DSO／参考／维护 binding 与全量 oracle 字节，失败时不生成可执行计划。
3. 由主代理在部署路径运行下列命令。新 `deployed/` 必须不存在；生成器拒绝覆盖。此命令仅生成资格任务，**不启动 GPU**。

```bash
P=EXPERIMENT_WORKSPACE/experimental/campaign_20261104
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
taskset -c 94,95 EXPERIMENT_WORKSPACE/envs/core/bin/python -B \
  "$P/e6_protocol_completion_v2/build_plans.py" \
  --campaign "$P" \
  --original "$P/e6_memory_input_v1/full_profile/qualification_plan.json" \
  --output "$P/e6_protocol_completion_v2/deployed"
```

4. 主代理复核部署回执、实际 GPU UUID／空闲显存、主机内存、磁盘与全局锁后，才能将 `deployed/qualification_plan.json` 的新 60 项交给现有串行控制器。每个 target 只接受本候选同配置的 tiny；每个正式重复还要同输入 target，`coverage_backend` 属于 epoch 与 proof 身份。
5. 即使部署通过，`formal_pending.json` 仍只有 `job_specs`，没有可直接提交的 `jobs`。必须完成新资格、逐格处理真实适用限制并由主代理另行准入；不存在“CPU PASS 后自动把 450 项算作正式完成”。旧队列、旧失败与原始输出均保留。

CPU 测试命令（本地 scratch 路径可调整到任务临时目录）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -B -m pytest -q -p no:cacheprovider test_protocol_cpu.py \
  --basetemp INTERNAL_EVIDENCE_ROOT/e6_protocol_completion_v2/pytest
```

目前完成的是代码、CPU 合同验证与完整计划草稿。GPU 相容性、60 项资格与正式 450 次结果仍待主代理执行；不能据此更新 E6 性能表。

## v2 部署修复

v1真实远端预检发现虚拟环境解释器路径与其symlink目标在元数据中不同，文件SHA及长度相同。v2仅修正模板binding的路径比较：保留原envs/core/bin/python调用路径，仍严格检查SHA及长度；独立oracle consumer和桥的检查完全未改。原失败保留在v1部署日志。新增两个真实symlink/长度反例，合计59项CPU检查通过。
