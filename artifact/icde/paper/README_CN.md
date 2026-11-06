# 当前论文与图表入口

当前研究进展以[本次公开范围](../TOOLING_FOLLOWUP_CN.md)为准。以下是本轮当前正文与图表入口。已生成明确标注 n=1、覆盖未齐的实测首轮表与图；完整评测仍在补齐。

| 资产 | 当前入口 | 证据类别 |
|---|---|---|
| Intro与首页问题图 | [中文初稿](intro/INTRO_DRAFT_CN.md)、[图注与源代码说明](intro/SOURCE_AND_FIGURE_NOTES_CN.md)、[双panel PDF](intro/first_page_problem_85mm.svg) | IN/UK 八卡真实受控轨迹，初始 G⁰ 就绪后计时：Full upload + LRU 8.94 s，EGGPU 7.90 s（n=1，源码71ea）；包含初始准备的四请求总时间为21.79/21.72 s，不能混用分母；旧共享分配构造例仅留 previous |
| 数据集与机器表 | 来源回执（内部资产未随本增量分发：`setup_tables/current/provenance.json`）、数据集PDF（内部资产未随本增量分发：`setup_tables/current/datasets.pdf`）、设置PDF（内部资产未随本增量分发：`setup_tables/current/setup.pdf`） | 实际输入与硬件事实，不是性能结果；Matplotlib预览不是LaTeX编译通过证明 |
| Related Work | [两部分各十篇](RELATED_WORK_NARRATIVE_20261104_CN.md) | 主源、CCF版本与适用边界明确；软件另列 |
| 首轮实测 | 对照表（内部资产未随本增量分发：`../experiments/campaign_20261104/first_pass_20261106/current/FIRST_PASS_CN.md`）、全系统矩阵（内部资产未随本增量分发：`../experiments/campaign_20261104/first_pass_20261106/current/first_pass.html`）、E1单卡图（内部资产未随本增量分发：`../experiments/campaign_20261104/first_pass_20261106/current/e1_single_gpu.pdf`）、E2(a)八卡图（内部资产未随本增量分发：`../experiments/campaign_20261104/first_pass_20261106/current/e2_eight_gpu_comparison.pdf`） | 当前范围为源码71ea；数值及采集时间见交付入口和表内回执。n=1，target 身份保留；旧d84仅留previous，先补第一轮，再补第二轮 |
| 实验正文 | 实验章节中文稿（内部资产未随本增量分发：`EXPERIMENT_DRAFT_CN.md`） | Setup、共同比较与计时合同、E1–E6正文及图表位置；E1 含部分首轮观察，其余结果待填 |
| 实验设计 | [E1–E6设计](PAPER_EXPERIMENT_DESIGN_20261104_CN.md)、[协议替换和待办](EXPERIMENT_SCOPE_RECONCILIATION_20261105_CN.md) | 设计、CPU检查、GPU资格、正式重复分别标识 |

首页问题图为Fig. 1；机制图M1与框架图M2由协作者绘制，仍仅留位置，最终编号排版时确定。不要沿用旧README中“Fig. 1=机制图”的映射。

`intro_20261105/previous`只保留上一版正文/同源图；`setup_tables/current`当前只有一版。以后派生展示最多current/previous两代，重复测量、失败和来源记录不按这个规则删除。

其余paper目录（包括icde_v3、icde_method_v4/v5/v6、integration_revision_20261028、figure_atlas_v3等）及顶层theory_method_v2均为历史来源材料，保留原路径和哈希。它们内部的“当前”“最终交付”和旧性能数字只属于当时版本，不能作为新七图campaign结果。完整分类见[NAVIGATION_STATUS.json](NAVIGATION_STATUS.json)。旧渲染物归档入口见figures说明（内部资产未随本增量分发：`../figures/README.md`）。

早期 `experiments/campaign_20261104/paper_results_v1` 与 `presentation` 是相应冻结协议的工具源，不能用旧200/64/33配置验收新300/160/45配置。尤其旧E4-S `accept_execution_hash_attestation=true` 不符合当前完整输出恢复要求，不能用于新正式图。新协议导出/绘图适配已在独立候选（内部资产未随本增量分发：`../experiments/campaign_20261104/paper_results_protocol_completion_v1/README_CN.md`）实现并通过59项CPU测试及480格state合成渲染检查；561格待绑定清单（内部资产未随本增量分发：`../experiments/campaign_20261104/paper_results_protocol_completion_v1/pending_grid.json`）明确不是结果。这些是旧协议工具的历史要求，不构成当前首轮展示的十次重复门槛。当前使用轻量首轮导出器（内部资产未随本增量分发：`../experiments/campaign_20261104/first_pass_20261106/README_CN.md`），复用源绑定完整目标测量并保留缺项、失败与 n=1 身份；第二轮随后补充。合成图仍不能发布为实测结果。

上次核验的公开 artifact 为 [62883ed 统一源码检查点](https://github.com/eggpig/EGGPU/commit/62883ed29701e09d4e655c7a92575d5566613406)，对应发布指针（内部资产未随本增量分发：`../experiments/campaign_20261104/publication_current.json`）。此后的文字更正、输入缓存和导出器更新尚待后续推送，以发布指针为准。

本轮正文统一主线：当前请求按内容依赖维护合法输入；请求完成后按物理持有关系选择保留与实际回收。Intro 用具体例子建立问题，Related Work 解释已有工作的前提与决策范围，实验正文逐节给出验证问题。运行器、资格进度和调试记录不进入论文正文；首轮实测明确披露样本数与覆盖，缺失结果集中保留占位。

当前论文呈现：E1 是单卡静态执行；E2(a) 是八卡同资源比较，E2(b) 是固定后端的 1/2/4/8 卡扩展。原 E1-P8 证据 ID 保留。静态主图只画执行时间，首次任务时间留补充表；方法段统一解释完整结果、同步和必要传输。E3–E6 优先实际支持的八卡面板，资源不同的 E4-S/E6 对照仍明确区分。
