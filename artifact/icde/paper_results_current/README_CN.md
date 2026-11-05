# 最新协议结果导出与绘图候选

这是独立的 CPU 导出/绘图适配器，不修改旧 `paper_results_v1`、已冻结实验 runner 或之前已公开 artifact stage。它不启动 GPU，不补写实测结果，不选择“最快一次”。目前尚无经本工具验收的新协议完整正式结果；没有创建论文性能 `current`。CPU 验证入口为 `run_cpu_tests.py`；本次公开目录验证见 `../provenance/protocols_cpu_validation.json`。CPU 测试通过不替代实验 GPU 资格或十次正式重复。

当前入口为 [paper/README_CN.md](../paper/README_CN.md)。文稿中的 Fig. 1 是构造的首页问题反例；机制图、框架图仍由协作者提供。

| 协议 | 固定配置数 | 新适配内容 |
|---|---:|---|
| E3 | 300 | Rebuild、D0+C、D1+C；实际 20 次 commit / 80 次 query；Rebuild 的 commit 释放量单列，不混入 C 的 query 后释放 |
| E4 | 160 | 不保留、view-LRU、closure-LRU、逻辑评分、C-Retain；32 visits / 128 queries，once 对照单独 16 queries |
| E3R | 20 | grouped/interleaved 含相同 16 次只读请求；不并入更新密度图 |
| E4-S | 36 | 四图、三批长、三种适用系统；完整 pinned NPZ 可恢复输出、正式 archive 与原维护双门 |
| E6 | 45 | 每图 15 配置：native P1/P2/P4/P8、igraph 和 cuGraph composite 各一套真实双因素 2×2（共享 native P1）、两个遍历补充、三个完整外部组成 |

E2 与 E5 沿用其原冻结合同，不把这里的 480 个 state 配置误写成包含 E5。固定七图注册表仍为 US/EP/LJ/OR/TW/IN/UK，具体面板使用上游预声明子集。`upstream_provenance.json` 记录复用原始工具与当前协议文件的 SHA；`protocol.py` 只执行 CPU 元数据/纯网格函数，不导入 E6 GPU worker。

## 准入与输出

`normalize.py` 要求显式 `FROZEN_EXPLICIT_SELECTION` 和 `export_protocol=paper-results-protocol-completion-v1`。一次选择可以覆盖一个或多个实验；每个选入实验必须具备完整固定网格，不能根据胜负缩小格子。每个成功 cell 必须有独立 repeat 0–9；失败必须绑定该 cell 的原始失败记录，失败没有零秒值。资格只能生成诊断，不能进入正式均值。执行源码、runner、资源、输入、质量、完整输出及 tiny/target 证据必须对应同一选择。

新 state 使用 `state-complete-workflow-completion-v1`，校验真实 trace、语义版本、独立参考审计、完整 FP64 NPZ、输出存活、预算、零释放不能算容量进展，以及 Rebuild/no-retention 特有行为。E6 使用 `coverage2x2-v2` schema 和完整 v2 contract，逐调用检查 first/warmup/ready、完整输出、owning/immutable 审计、实际阶段 backend/device 与依赖选择；closeness/candidates 是 32 项，ES/Constraint/BFS/WCC 是完整 n 项。两个外部 backend 的同名 ablation 不会互相覆盖。

E4-S cell 需要 `archive_helper` 的绑定，以及 `recoverable_outputs` 列表。列表每项包含 `result`、`archive`，正式项另含 `maintenance`，均为 `{path,bytes,sha256}`。每个正式重复以及 tiny/target 资格都必须列出唯一对应归档。工具逐个解压 `NPZ['reference']`，验证 shape/dtype/原始字节 SHA 等于测量时 output SHA；核对独立参考 metadata 和生产者；正式结果额外核对相同原始结果的 archive、原维护与 wrapper 冻结链。`accept_execution_hash_attestation=true` 一律拒绝。源路径可通过显式 `path_remap` 迁移，原始逻辑绑定及哈希不重写。

准备尚未绑定结果的完整格子（该文件刻意不是可用于出图的 selection）：

```bash
python3 -B prepare_selection.py --experiments E3 E4 E3R E4S E6 --output /task/pending_grid.json
```

已有显式 receipt index 后可使用 `--receipt-index /task/index.json`。其 `cells` 按 scaffold 的 `id` 给出 `runs` 与 configuration/epoch/input/archive 等绑定，顶层提供 `final_eggpu_release`、`input_registry` 和必要 `path_remap`，状态为 `FROZEN_EXPLICIT_RECEIPT_INDEX`。程序拒绝缺少、多出、重复 cell 或改写预声明维度，不扫描目录寻找最新/最佳结果。

```bash
python3 -B render.py --selection /task/selected.json --mode formal --output /task/paper_outputs --scratch /task/staging
```

未完成时只允许在普通诊断目录写 CSV/JSON/HTML；正式图不会生成。完整正式覆盖（含显式失败状态）才可发布受管理的 `current/previous` 两代；删除 previous 前校验整个文件清单与 SHA，不删除 raw、repeat、failure 或 provenance。合成测试不能发布 current/previous，测试 SVG/PDF 和图注均带 `SYNTHETIC TEST DATA — NOT A RESULT` 标记。

## 图表与统计

沿用已定稿的蓝/绿/橙/紫配色、简洁轴标题、矢量 PDF/SVG、十次均值和 95% Student-t 区间。E3/E4 分别按请求维护与物理驻留问题分面；E3R 单独画顺序对照。E6 把完整系统阶段堆叠、两套 2×2、遍历补充和资源敏感性分开，外部 CPU 组成不纳入单卡同资源胜率。无配对调度回执时，2×2 交互使用独立组 Welch–Satterthwaite 线性对照，不按 repeat 编号伪配对，不预设有利符号；复制数据的多卡执行不宣称聚合显存容量。

`prepare_selection.py`、`normalize.py`、`analysis.py`、`render.py` 是可组合的入口；正常结果包含每次重复 CSV、覆盖 CSV、源码/输入链、比较 JSON、中文报告与图注。没有正式数据时不借合成值预填论文结果。

## CPU 验证

需要 Python、NumPy、SciPy、Matplotlib、pytest。可将缺失依赖安装在任务临时目录，再重复传入 `--dependency-path`；不会修改全局环境。

```bash
python3 -B run_cpu_tests.py --scratch /task/new_test_scratch --output /task/new_cpu_receipt
```

入口隐藏所有 GPU 并限制 CPU 线程为 2；只生成明确 synthetic 测试数据。覆盖完整网格、450 条 E6 合成正式形状回执、10-repeat 门、源码/epoch/资格混用、完整输出与质量拒绝、Rebuild/no-retention/E3R、E4-S 实际解压及 archive/maintenance 拒绝、双 2×2 与水印/禁止发布。实际测试数量、依赖版本和源码 SHA 以回执为准，不将这些测试称为新的实验实测。
