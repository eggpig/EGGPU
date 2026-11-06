> 历史协议更新：此文保留当时源码与验证状态。当前统一源码、构建及首轮状态以 [FIRST_PASS_UPDATE_CN.md](FIRST_PASS_UPDATE_CN.md) 为准。

# 本轮协议与论文资产增量

这是基于 `4cb83e75a98210e0106ee18cc581a714fc584a8a` 的开发检查点。`Easy-Graph/` 主源码仍逐字节保持 d84；6cce 仅作为独立实验 overlay。正式队列已启动，但完整矩阵与单卡 90% 领先目标尚未被完整正式证据确立。机器进度仅见 `campaign_status.json` 的带时间快照，不把 CPU 测试或资格结果当性能样本。

本轮公开目录重新运行 190 项 CPU 测试：通用增量准入 38、E4-S 增量 24、state 队列保护 5、arena 诊断 6、严格结果导出 59、WCC runner v3 50、可移植 80 格 recipe 拒绝测试 8。原始作者测试与独立审查的哈希保留在各候选 `original_cpu_summary.json`；它们是来源摘要，不是迁移后的 GPU 证明。公开目录的重新验证见 `provenance/protocols_cpu_validation.json`。

## 复验

需要 Python、NumPy、SciPy、Matplotlib、pytest，WCC 的实际 C++ 参数解析测试另需 g++、OpenMP 与 taskset。WCC 历史 profile 采用公开 setup 中的固定 16 核配置；该项测试应在能提供相应 CPU affinity 的主机执行，不能静默弱化 placement 检查。测试不会初始化 GPU。

```bash
python3 artifact/icde/tools/test_protocols_cpu.py --work /task/new_cpu_check
```

缺少依赖时可在任务目录安装并通过 `--dependency-path /task/dependencies` 指定；测试临时目录必须位于 checkout 之外。各组在独立进程运行，保留 JUnit、日志和哈希。`--groups paper,incremental` 可显式选择部分测试，不能据此宣称未运行组已通过。合成图只出现在测试 scratch，带水印，未作为 artifact 的正式性能图发布。

## 新入口及准入边界

| 入口 | 用途与边界 |
|---|---|
| `campaign/formal_incremental_admission_v1/admit_incremental.py` | 对完整预声明矩阵逐 cell 核验、登记已完成的资格和显式终止；只派生未执行的合法 cell，不缩分母、不改原正式 job 的计算参数 |
| `campaign/formal_incremental_e4s_v1/admit_e4s_incremental.py` | 同样的增量语义，另要求 E4-S 完整可恢复输出及原维护通过；不存在的资格向量不能由哈希声明替代 |
| `campaign/state_completion_queue_guard_v1/derive.py` | 对 caller 给定的完整新 state 资格计划加 phase、完整输出/参考/Rebuild 临时空间和维护保护，保留原命令、顺序与依赖 |
| `campaign/arena_diagnostic_v1/prepare.py` | 五格独立资格诊断，观察既有 arena 指标；始终 `diagnostic_only`、`performance_eligible=false`，不改正式算法路径 |
| `paper_results_current/prepare_selection.py` | 产生 561 格显式待绑定清单：E3 300、E4 160、E3R 20、E4-S 36、E6 45；待绑定清单不能直接出图 |
| `paper_results_current/render.py` | 拒绝缺十次重复、混 epoch、错误质量及缺失完整归档；只允许完整正式覆盖进入受管理的 current/previous |

每个入口的 `--help` 和候选 README 给出参数。`campaign/sg_current` 是诊断所需的四个逐字节冻结 runner 的公开别名；原始机器计划不随包提供。arena 的 fixtures 是明确标记的 CPU 模板，不可当作部署计划。

新导出工具支持 E6 的 igraph CPU 和 cuGraph composite GPU 两套完整 2×2，保留外部双组臂与共享 native P1；复制图的多卡执行不解释为聚合显存容量。E4-S 必须实际解压 pinned NPZ，核对全部 FP64 输出字节与测量输出 SHA，并核验 archive/maintenance 双门。旧 `paper_results` 仅服务其历史协议；新论文结果使用 `paper_results_current`。

## 独立 WCC v3 候选

`experimental/native_wcc_wide_v3` 只改变 qualification runner 对 label 分支的配置及实际 profile 审计。生产源仍是上次公开的 `native_wcc_wide_v2/source_overlay` 对应 6cce，不重复打包整份源码，也不替换 d84 主目录。先用 v2 的 `materialize.py --output /task/new_wcc_v2` 在 checkout 外重建完整候选；其结果逐文件验证后应为 6cce。

`test_profile_compatibility_cpu.py` 用 `EGGPU_WCC_V2_SOURCE` 指向该重建目录，编译原始 C++ 参数验证代码；没有加载 GPU DSO。公开测试 runner 会自动完成这一步。50 项通过只证明 CPU 选项合同与回归检查，不证明 80 项 GPU 资格已完成。

`prepare_qualification.py` 是原 80 格生成器的可移植投影，显式接收 campaign、v2/v3 候选目录、完整父计划/registry、构建回执、controller Python、CPU affinity 和全新输出目录。它验证 6cce 的全部源码、调用者新构建的 DSO 字节与回执、公开 v3 runner CPU 回执，逐格核对 `qualification_recipe.json`，然后重建 4 项数值 + 48 项 WCC 分支/宽度 + 24 项其他算法 + 4 项 TW 资格。所有 TW 项依赖本轮完整 76 项前置资格；不继承旧 GPU PASS。新的 build/plan 必须来自复现者机器，不能将原机器回执改路径后继续使用。

80 格 recipe 记录准确的算法选项、超时和依赖，是可审阅的语义输入；它不是可执行计划或结果。生成器依赖已有、重新绑定的完整 v2 父计划及对应维护链，拒绝少格、重复、改算法/分支、缩短超时、缺维护或缺前置项。包中不提供本机私有部署计划。生成器只准备，不启动队列；`startup` 是生成回执中的显式命令。

## 论文导航

当前入口为 [paper/README_CN.md](paper/README_CN.md)。Intro 与首页图是明确构造的共享分配反例；Related Work 两部分各十篇主源，软件另列；机制图、框架图继续保留位置。旧论文性能图不作为本轮最新结果。当前没有新协议完整正式性能图。

尚未冻结的 E5/E6 formal admission 新代码未纳入本轮检查点。运行实际实验仍须满足 caller 当前源码、输入、构建、资源和资格一致性；CPU 源码包无法转移原机器 GPU 资格。
