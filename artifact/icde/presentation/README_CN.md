# E1 结果汇总入口

本目录实现 SG/P8 的显式版本选择、完整证据检查、统计与 LaTeX 表格导出。当前没有正式输入清单，也没有生成论文结果。合成测试只在临时目录运行，不能引用为实验数据。实验问题和展示原则仍以 [论文设计](../paper/PAPER_EXPERIMENT_DESIGN_20261104_CN.md) 为准。

依赖 Python 3.10+ 和 SciPy；不导入 GPU 库。校验会顺序读取输入和输出的完整 SHA256，需安排 CPU/I/O 维护窗口，不能与正式 GPU 计时交错。相同不可变 inode 的内容只哈希一次。

## 选择版本与运行

`selection.example.json` 是无法直接执行的路径模板。填写所有预先声明的适用系统/作者版本、每算法 epoch、最终 EGGPU freeze、七图 input registry，以及每次运行的确切 result.json 路径。不得用通配符、“最新目录”或单格最快尝试替代明确选择。

```sh
python presentation/bind_selection.py --draft selection.draft.json --output selection.lock.json
python presentation/e1_summary.py --selection selection.lock.json --mode qualification --output review/e1_qualification
```

正式清单必须另行填写 `mode: formal`，明确列出每格重复 0–9 的全部十条记录，再运行：

```sh
python presentation/e1_summary.py --selection formal.selection.lock.json --mode formal --output review/e1_formal
```

绑定工具只给显式路径补充哈希；遇到已有哈希不同就拒绝，不替换旧版本。若搬移 artifact，通过清单 `path_remap: [{"from": "/original/root", "to": "/relocated/root"}]` 显式映射；原始 JSON 字节不改。一个算法在七图采用同一预选实现和 epoch，不能按图选择版本。CGgraph legacy/modern 等真实不同实现须使用不同 system ID 和 variant，不能把逐格最快值伪装为同一作者版本。

适用性是算法级声明：`algorithms` 列出实际共同任务，`not_applicable` 为其余函数写出语义/作者能力依据；二者必须覆盖四函数且不重叠。C²graph-P 只列共同单源 SSSP，PM/KGraph 批任务不进入 E1。单卡实现不进入 P8。内部 FP32 并不自动排除系统：以冻结的共同完整输出质量为准，同时在 label/epoch 中保留作者精度、准则和资源说明。

## 统计和拒绝条件

- 每个系统×图×函数是一格，每个 repeat 内先对 BFS/SSSP 三个冻结源点等权算术平均；PR/WCC 只有一份。十个 repeat 均值才是十个统计样本，不能把三个源扩为三十次独立重复。
- T_first 与 T_ready 分开。作者系统第一项必须有连续时钟证明，旧 layout+native 的分解和式不能进入 T_first。GUM 使用实际连续首次区间和完整 native ready；warmup 只作质量检查。
- 每种资源和计时边界固定 28 个计划格。对照对象只叫 **fastest qualified measured external baseline**，按十次均值选择并保留作者/版本身份；不取某一次最小值。报告 W/Nvalid、W/28、valid coverage。
- FAIL、OOM、TIMEOUT、INDEX_UNSUPPORTED、质量失败均保留；没有合格外部结果的格不能自动获胜。至少存在一个有十次合格结果的外部系统才形成实际配对格。另列 `strict_full_baseline_coverage`：所有预先适用外部系统均完成十次测量，或有明确绑定的终止状态；`all_external_qualified_coverage` 则要求它们都合格。缺测不悄悄删除，也不能据部分对照泛称 SOTA。
- 提前终止某个系统/格时，在 `terminal_cells` 写入已选失败 run 的 evidence 和人工解释；该失败仍不产生任何时间均值。如果失败发生在 epoch/input 绑定之前，它是未绑定管线错误，不能作为最终版本的已解析比较格。
- 每格时间均值的95%区间采用 n=10 的 Student-t。差值为外部减 EGGPU，使用独立组 Welch 区间，不假称同步配对。选出最小外部均值后的区间属于探索性、条件于该选择的区间，未作多重选择校正。均值领先和区间是否跨零分别记录。
- 混合源码/runner/协议/输入、不同 source 顺序、重复 repeat/多次尝试、非正或非有限时间、完整输入、正式输出、tiny/target资格输出及子收据哈希变化、质量门槛不符都会拒绝。固定输入 registry 的七图必须全部 ready；最终 EGGPU runtime source 必须与选定 freeze 相符。

十次未齐的格没有正式均值。只有所有计划系统格都已完成或明确终止、且确有完整正式格时才生成 `.tex`；其余仅输出检查表。资格数据永远没有论文 LaTeX，HTML 顶部显示大字 **QUALIFICATION — NOT PAPER RESULTS**，并拒绝写到 `paper/current/formal` 目录。这里不会更新论文 `current` 指针。

输出 `summary.json` 包含选择哈希、全部来源、均值/区间、覆盖和未完成状态；`sources.csv` 是逐源逐重复时间；`cells.csv`、`comparisons.csv`、`coverage.csv`、逐图/逐函数的 `breakdowns.csv` 可复查统计；`checks.html` 是检查表。相同锁定清单可以原位刷新，不同清单不能覆盖同一目录。正式十次原始收据始终保留，不被汇总或去重成一次。

## 稳定图表位置、颜色和图注

沿用现有设计的稳定文件名。以下作为当前投稿排版映射；若最终模板调整编号，只改映射和正文引用，不改结果身份。机制图/框架图由用户提供，保留位置，不在此生成。

| 排版位置 | 稳定名/当前产物 | 内容、轴和必要图注 |
|---|---|---|
| Fig. 1（M1） | mechanism | 双栏机制例子占位 |
| Fig. 2（M2） | framework | 双栏系统框架占位 |
| Table I / II | setup / datasets | 实测机器与版本；七图方向、V、存储弧A、A/V |
| Table III | e1_overall_first.tex | SG/P8 分块，行=函数×系统，列=`US EP LJ OR TW IN UK`；统一 Time (s)，三位有效数字，只加粗最低合格已测均值 |
| Table S1 | e1_overall_ready.tex | 相同布局的 T_ready 完整时间，保留首次与驻留边界区别 |
| Fig. 3 | e1_sequence | 预先固定 US/IN/TW；横轴 Requests、纵轴 Cumulative time (s)，真实16请求累计前缀，禁止用独立单函数时间拼接。此工具尚不导出该图 |
| Fig. 4 | e2_scaling | 四函数分面；横轴 GPUs、纵轴 Speedup，同一原生后端 P1 为分母；另附绝对 Time (s)，不以优化 SG 代替 native P1 |
| Fig. 5 | e3_maintenance | Updated edges (%) 对 Time (s)/Transferred data (GiB)，零更新单独标；真实旧读者语义 |
| Fig. 6 / 7 | e4_retention / e4_state_reuse | Memory budget (GiB) 对完整时间/重载；共同批 SSSP 另面板，不与物理预算图混比 |
| Fig. 8 | e5_joint | 四组合完整时间与配对交互区间，保留相同 workload/repeat 配对关系 |
| Fig. 9 | e6_workflow | 完整依赖任务阶段堆叠；计入数据交接，候选与最终结果每次重算 |

Table III/S1 注：完整任务时间；十次独立 repeat 的均值，BFS/SSSP 每次先等权平均三个冻结源。首次从共同 host CSR ready 到完整 host 输出；驻留仍重新计算。最低合格均值加粗；区间及各源数据见 artifact。分别解释 FAIL、OOM、TO、IX、QF、NS，不用一个破折号掩盖原因。不能从设计模板复制性能结论。

图表固定配色：EGGPU 深蓝 `#356A9A`/浅蓝 `#B8D3E7`；cuGraph 橙 `#C88646`；GUM 绿 `#6B8E67`；其他适用外部系统用紫 `#8C79A6`、灰 `#8A8A8A` 并辅以不同标记/纹理，系统数量多时分面。D 浅蓝、C 浅绿 `#BED4AA`、传输浅橙 `#E9BE96`、其他浅灰 `#D3D3D3`。系统颜色和阶段颜色不混用。白底、细轴、缩放后8–9pt，导出矢量 PDF/SVG；条形零起点，无双纵轴、渐变或图内长标题。图注按“测什么→固定条件→统计/分母”组织。

当前交付限于 E1 SG/P8 汇总、完整证据校验和主表入口。E2–E6 专门数据适配和矢量绘图器位于 ../paper_results；其正式数据必须独立通过完整验证，不能把本目录的 E1 结果冒充其他实验。
