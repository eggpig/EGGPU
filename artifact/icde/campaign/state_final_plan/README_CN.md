当前公开版本以 artifact/icde/provenance/current_source.json 的 d84 源码为准。以下旧部署叙述仅说明历史计划格式；原 GPU 证明不能用于当前源码。可运行生成器为本目录 state_plan.py（v7），当前完整主网格288格、hotspot另32格。

# E3–E5 最终源码计划与准入 v3

本工具只生成计划、验证证据；不会自行启动 GPU。v3修复旧v2计划对Python符号链接调用路径的错误解析：调用保留`core/bin/python`词法路径，同时另绑实际ELF字节、`sys.prefix/base_prefix`以及NumPy/SciPy版本和实际文件。相同解释器ELF不代表相同虚拟环境。旧v2两次prelaunch的SciPy缺失失败保留，不算算法失败或性能结果。冻结 `state_v4`、既有输入和既有运行记录均未修改。`guard --execute-gpu` 是主代理统一队列调用的显式执行入口，最终仍由原 worker 获取全局 GPU 锁并检查 GPU 空闲、主机/cgroup 余量和磁盘。

当前部署绑定 `arena_output_ownership_v2` 源码 `cb14360e2dfb934b8a28f9875fb80639ce4503bc6ed27956201b18e9edec6869`，原生 DSO `bc1b7cf247222beec1f33afe4d184bdc715eea4ac65a07e8e2a567ec196aec46`。计划生成器仍要求显式传入 source/freeze/SHA，不能根据目录时间自动选版本。旧输入由 host_v1 准备的事实保留；最终源码的 manifest signer 与旧输入绑定均为 `98fcb74a7a3b97bfa6aa9e325894f634e09084a8aff58a2b62ab773cc94526ee`，允许复用其字节，不复用旧源码 GPU 资格。

## 主协议及完整性

- E3：US/IN/OR 的 P1、IN/OR 的 P8；D0C1 与 D1C1，分别 none/per_update。每条20次权重提交、80查询，另有相同四次初始化查询。D0是复用拓扑并对新权重身份全量维护，不是每次重建整个图。C固定且无容量预算。
- E4：US/OR/IN/UK、P1/P8；D1C0 与 D1C1。β为0/.25/.5/.75/1，三种顺序分别 grouped/interleaved/permuted，各32访问、128查询；once在β=.25单列为4访问、16查询。
- E5：IN/UK/OR、P1/P8，四臂完整2×2；24提交、96查询。主点ρ=1e-4、β=.25；另外ρ=0、β=.25和ρ=1e-4、β=1两个负控制。
- 完整网格共288配置、每配置10独立进程重复。E3密度为0/1e-6/1e-4/1e-2/.1/1，0和1不重复等价局部性。正式顺序事前固定：同一配置条件的同一repeat构成匹配块，臂顺序按repeat循环轮换，不看测得时间。E5交互用这些预注册块；缺任意臂时该块不能估计交互。
- 无权PR权重不变性负控制、显式分区集中轴仍待独立协议/入口，列在coverage中。它们不能被现有四函数或行聚集替代。冷启动整图Rebuild臂不在冻结state_v4中，不宣称已实现。

初始五个P1输入只支持选定的18个主点配置：E3十二个、E4两个、E5四个。`qualification_plan.json` 共42任务：两种卡数共22个重新tiny、两个P1真实图池字节pilot和18个目标资格。`formal_pending.json` 只有180个待准入请求；它不是可执行正式队列。完整coverage仍列出其余输入缺口和P8容量限制，不能把18主点写成288配置全部完成。

## 使用

`build` 接收旧tiny计划作为显式部署模板，只提取同一Python/DSO、UUID、16核、锁、tiny输入和环境；从模板读取的旧结果、旧source与旧输出路径不被继承。`--inputs` 必须逐项提供真实输入JSON；不扫描“最新”目录。参数示意：

```sh
python state_plan.py build \
  --source "$SOURCE" --freeze "$FREEZE" --expected-source-sha256 "$SOURCE_SHA" \
  --runner "$FROZEN_STATE_V4" --registry "$INPUT_REGISTRY" \
  --deployment-template "$EXPLICIT_TINY_DEPLOYMENT" --native-build-receipt "$BUILD_RECEIPT" \
  --inputs "$E3_US_INPUTS" "$E3_IN_INPUTS" "$E3_OR_INPUTS" "$E4_INPUTS" "$E5_INPUTS" \
  --output "$NEW_PLAN_DIR" --results-root "$NEW_RESULTS_DIR"
```

默认`--scope initial`只选择上述18主点。`--scope available`可选择完整网格中已有明确输入的其他配置；P8真实目标还必须明确`--enable-p8-targets`，不会由小图成功推断大图容量足够。

`guard --spec <jobs/...json>` 是CPU准入检查：核完整源码freeze与额外可执行源码、冻结runner、Python/DSO/build receipt、registry、原始规范输入/投影/修改位置全部SHA、manifest签名、精确资源和proof。不会导入GPU后端。目标资格要求同源码/资源/臂/顺序/β/读者的tiny；正式资格再要求精确目标输入和预算。tiny/target证据逐个核完整NPZ、原向量hash、质量和close后存活，不接受只有哈希标签的旧收据。失败与已存在输出目录拒绝继续，不覆盖原运行。

E4/E5预算必须来自实际pilot，各设备`B_d=floor(max_g S_gd + β(sum_g S_gd−max_g S_gd))`；pilot绑定源码、输入、GPU UUID、CPU affinity、环境，不能只比设备数。它不使用计时选择预算，同图池同源同卡数可以跨臂/β/访问顺序共享一份pilot。tiny池和目标池分别pilot。

全部对应资格完成后执行CPU命令：

```sh
python state_plan.py admit-formal --bundle "$NEW_PLAN_DIR" --output "$NEW_FORMAL_QUEUE"
```

仅此命令在验证proof后导出正式队列；队列再绑定proof字节，每个任务执行前复核。v3每次真实发起worker之前另写不可覆盖的launcher admission sidecar，记录精确spec/runtime；后续pilot/tiny/target proof都必须同时有该sidecar，旧v2无sidecar证明不能继承。准入成功仍是零测量，不产生论文结果。主代理通过统一串行controller运行，worker自身资源锁和60分钟进程组超时继续有效。额外的输入/资格哈希与NPZ读回是测量区间外I/O，必须在同一串行维护窗口完成，不能与其他正式测量重叠。

## 资源和保存合同

原worker的每次磁盘门限是`sum_query 8*n(query)+2GiB`，不提前计入压缩或去重收益；所有完整输出在Session内保持独立并存活，计时后写无损内容寻址NPZ。目标/正式间的相同完整向量可共用对象，每个请求仍有独立receipt。该门限不包含将来JSON日志的精确字节（固定2GiB为保留量），不是整个campaign的总存储上界。

| P1主点 | 完整向量上界（bytes） | 向量+2GiB门限（GiB） | worker主机余量门限（GiB） |
|---|---:|---:|---:|
| E3 US | 15,326,302,080 | 16.274 | 28.053 |
| E3 IN | 4,745,514,240 | 6.420 | 39.278 |
| E3 OR | 1,918,186,240 | 3.786 | 39.925 |
| E4 | 13,537,245,440 | 14.608 | 26.219 |
| E5 | 7,406,724,608 | 8.898 | 60.847 |

主机门限按现有worker公式`4GiB + Σ_g(32*n_g*r + 8*m_g*(E4?1:r+1) + 4*m_g + 24*n_g)`，E3的r=20，E4/E5的r=8；不是测得峰值。运行前取主机可用与cgroup余量的较小值。磁盘余量不足会诚实停止，即使历史去重率很高也不跳过门禁。tiny/target proof及其对象在后续准入完成前须保留于其绑定路径；正式raw可经verified CAS归档，但需要明确映射和原hash，不能删证据或自动换路径。E1呈现工具已支持冻结selection的显式`path_remap`，state_v4的在线proof暂不支持离线映射。

文档中的实验图编号、颜色、轴标题、图注仍以 `docs/PAPER_EXPERIMENT_DESIGN_20261104_CN.md` 为准。当前工具没有生成E3–E5正式图，也没有把资格时间写进论文。

## 解释器回归

27项CPU测试包括真实symlink venv复现：同一ELF SHA、不同prefix，错误base调用被拒绝。实际机器对P1/P8使用生成计划中的原解释器/环境/guard参数，仅去掉GPU执行flag，完整tiny CPU准入通过；同时对PR负控制按相同解释器和原参数解析及实际TINY_A独立oracle通过。CPU准入未发起任何GPU任务。最终源码仍由显式source/freeze参数决定，不能将此CPU通过当作新源码GPU资格。
