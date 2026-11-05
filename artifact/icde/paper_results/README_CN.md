# E2–E6 正式证据汇总与矢量图表入口

这是 CPU 读取/统计/排版工具，不提交 GPU，也不产生实验数据。E1 同资源比较继续使用相邻 `presentation/e1_summary.py` 的固定 28-cell 规则；本目录复用它的内容校验与统计函数。工具参数化绑定最终源码，不默认选取当前候选。**当前没有由本工具生成的真实正式论文图。**

## 输入与执行

先按 `selection.example.json` 写显式选择文件：绑定源码 freeze、七图 registry、每个配置的 exact epoch 或 configuration、每个 repeat 的 receipt。所有 binding 均包含原逻辑 `path`、`sha256`、`bytes`；epoch 的内容摘要使用规范 JSON 的 SHA，不能用 epoch 文件字节 SHA 替代。只读明确列出的文件，禁止 glob/latest、跨图选择后端或按历史最快结果拼接。

```sh
python artifact/icde/paper_results/render.py --selection selected_qualification.json \
  --mode qualification --output checks/selected_qualification
python artifact/icde/paper_results/render.py --selection selected_formal.json \
  --mode formal --output paper/results --scratch "$EGGPU_SCRATCH"
```

`EGGPU_SCRATCH` 必须指向 artifact 源码目录之外的显式临时目录。执行环境需要 NumPy、SciPy、Matplotlib，精确 CPU 测试版本见 `requirements.txt` 和 `cpu_receipt.json`。不安装或改变实验环境；建议独立的 CPU 分析环境。输出清单记录实际分析源码及包版本。

正式成功配置必须具有 repeat 0–9 的十个独立完整任务；warmup 不算重复。E2 BFS/SSSP 每个 repeat 先对三个固定源等权平均，再跨 repeat 统计。完整 host 向量、tiny/target 证明、实际 source/DSO/runner/input 及数值检查逐链验证。任何资格结果、缺测、混合 epoch、缺失 payload、重复 receipt、重复 repeat 都不能成为正式均值。

`runs` 只列预先规定的当前配置独立重复。失败值保留为空，不填零。`terminal` 必须绑定本配置已观测且 source-bound 的失败收据，附明确终止原因；它能解释覆盖，却不产生均值或胜利。旧失败、重试和发生在 epoch 建立前的 prelaunch 失败放入显式 `attempt_history`，包含 `cell/evidence/reason`：写入 repetitions.csv，标记 `selected=false`，不补足正式重复或配置覆盖。工具不自动挑选重试中最快的一次；重试选择必须来自冻结执行计划。

完整正式选择要求：E2 80 配置，E3/E4/E5 分别 200/64/24 配置，E4-S 三系统四图三个 Q 共 36 配置，E6 三图十一配置共 33，独立 PR 控制至少三图两臂六配置。可以分实验发布，但不能把小范围 target 或初始输入网格叫作整个正式实验。root 的 `state_grid_prepare_v3` 已提供覆盖主网格的 56 个唯一输入准备方案；实际准备/资格完成以输入回执为准。本工具不把准备代码存在视为输入就绪。

允许为 E2 不同函数使用不同的事前冻结通信配置，例如 PR requested NCCL、BFS/SSSP/WCC host staging；同一函数与卡数在五图必须绑定同一个 epoch，所有 P 绑定相同 runner/native DSO。P1 无跨卡通信时 observed local/none 的含义由绑定协议解释。强扩展始终使用固定 native P1，不能换成优化单卡 Arena。

## 合法归档和输出保留

选择文件可用显式 `path_remap`：`{"from":"/original/answer.npy","to":"/archive/sha256/object"}`，也可目录前缀映射。收据的原路径和摘要不改；实际读取归档字节，仍核 SHA/bytes。没有自动搜盘、自动找同名文件或静默忽略缺失输出。tiny/target pin 在远端是调度策略；本工具同样支持其经验证的显式只读映射。

正式产物包括每幅 PDF/SVG、`repetitions.csv`、`cells.csv`、`coverage.csv`、`normalized.json`、`comparisons.json`、`captions.md`、`report_CN.md` 和内容清单。只在全部配置完整或有明确终止状态且至少一个配置完整时发布。qualification/不完整正式选择只产生带大字 `QUALIFICATION / INCOMPLETE — NOT PAPER RESULTS` 的 HTML/CSV，拒绝进入 paper/current/previous。

正式目录只维护 `current` 与 `previous` 两个展示版本；原始实验文件不删除。新包先完整生成并复制到同文件系统暂存；旧目录必须符合 renderer 清单且未有人为改动，才允许轮换/清理。CPU 合成绘图强制水印，禁止 publish 到 current/previous。

## 图号、视觉与结论边界

沿用 [论文实验设计](../paper/PAPER_EXPERIMENT_DESIGN_20261104_CN.md)，不另建冲突的实验协议。图注通过 renderer 与所选证据共同生成，最终文章可压缩正文图，将全部敏感性保留在 artifact。

| 编号 | 主展示与横轴 | 纵轴/配套材料 | 解释边界 |
|---|---|---|---|
| Fig. 1 / 2 | 机制图 / 框架图占位 | 由用户的绘图者提供 | 本工具不绘制 |
| Tables I / II | setup / 七图 | 独立记录真实软硬件及存储 arcs | 不把输入选择冻结当实验完成 |
| Table III / S1 | E1 SG/P8 first / ready | 相邻 presentation 严格同资源表 | 固定 28 分母、适用外部基线覆盖 |
| Fig. 3 | 独立连续 16 请求 | sequence16 的真实 continuous total | 不与 E6 或独立单调用均值混合 |
| Fig. 4 | E2，GPUs=1/2/4/8 | Time (s)，另图 Speedup | first/ready 各图；实际非单调保留；native P1 基准 |
| Fig. 5 | E3，Updated edges (%) | Workflow time (s)、Weight H2D (GiB)、COW D2D (GiB) | 正密度 log x；ρ=0 单独面板；两 D 臂、C 固定；20 commits/80 queries |
| Fig. 5S | 独立无权 PR 控制 | Workflow time (s) | 21 次真实 PR；三个图、两臂、old reader none；不改主 trace |
| Fig. 6 | E4，Budget fraction | Workflow time (s)、New shared arrays (GiB)、Freed memory (GiB) | D 固定，C 两臂；32 visits/128 queries；once 16-query 单独 |
| Fig. 7 | E4-S，Queries=32/128/512 | Time (s)，Q=128 为主点 | KGraph CPU 资源另列；完整批任务，不除 Q 伪装 E1 单源 |
| Fig. 8 | E5，四个 D/C 组合 | Workflow time (s)、Weight H2D (GiB)；独立交互点图 | joint 更快与 I<0 是不同命题；保持负控制 |
| Fig. 9 | E6，完整配置、四个单组回退及三种外部组成 | Application time (s)，真实阶段堆叠 | EP/LJ 主图，OR 压力；first/ready 分开；扩展函数是主阶段 |

统一 6.89-inch（约175mm）双栏宽，8pt 字体、细框、浅水平网格，PDF 矢量字体嵌入和 SVG 可编辑文字。调色为深蓝 `#356A9A`、绿 `#6B8E67`、橙 `#C88646`、紫 `#8C79A6`、灰 `#8A8A8A`，七图曲线补充红 `#B76A64` 与青 `#76B7B2`；D/C分别用浅蓝 `#B8D3E7` / 浅绿 `#BED4AA`，完整组合保持深蓝；同时用不同点形保证灰度区分。图内不写冗长协议、硬件或解释段；坐标只写指标与单位，图注写必要任务/时间/CI/coverage。参考论文的具体图表与出处在既有实验设计中；颜色是本论文统一设计，不声称逐像素复制其配色。

所有正式均值的误差线为十次完整重复的 Student-t 95% CI。独立配置差用 Welch，速度比用独立组 log-delta 近似；不因 repeat 编号相同就伪配对。E5 只有同时绑定事前四臂 block schedule 和实际完成 execution，才计算十个 block 的 `I=T11−T10−T01+T00` 与 CI；CI 上界低于零才能支持超加和时间收益。各配置 CI 未作多重比较校正，不能扩大成全图全函数保证。

权重维护 H2D 为 full-weight+patch，dense-lease 是其中子集而不重复加；COW D2D 单列。共享数组新分配字节不等于全部重载流量，更不等于所有 GPU 复制。E4-S 冻结 runner 在内存完整核验 Q 个答案，归档的是全量答案 SHA 执行证明；必须显式设置 `accept_execution_hash_attestation=true`，不冒称仍有 Q×n 向量文件可重读。E6 阶段堆叠来自连续阶段实测，剩余项是准备、交接与控制开销，误差线属于实际 continuous total。GPU 扩展组的数据复制执行不证明聚合显存容量。

## CPU 验证

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q artifact/icde/paper_results/test_paper_results.py \
  --basetemp "$EGGPU_SCRATCH/paper_results_tests"
```

测试使用明确合成数组、伪造但自洽的 CPU 收据，验证拒绝篡改、混版本、缺重复、假资格、错误设备/输出维度、错误配对、CAS 映射，以及非单调/正交互的如实报告和矢量排版。它不是 GPU 正确性或正式性能证明；实际 runner 首批正式收据到齐后仍须做完整端到端 schema/证据校验。

公开版将主网格绑定到当前 `campaign/state_final_plan/state_plan.py` 的288格，并支持单独发布32格 `partition-hotspot` 域（E3、IN/OR、P8、K=1/2/4/8、两D臂、两reader策略）。每个cell必须显式声明 `grid_domain: partition-hotspot` 与一致的 `hotspot_k`；不得把这一固定密度覆盖干预混作密度扫描。额外图以 Touched partitions 为横轴。

Fig.3现由 `sequence.py` 直接读取 continuous16 worker 的 monotonic timestamps、16份完整输出与同epoch tiny/target证明。selection的experiment为`E1_SEQUENCE`，adapter为`sequence`，固定US/IN/TW × EGGPU/cuGraph共六格、每格十重复、P1；同一系统不能按图换epoch。每个prefix来自共同CSR ready到该请求返回的连续时钟；独立workflow total仍保留，不把单函数平均耗时相加。所有正式图表均需真实formal receipts；本目录没有预制的论文结果图。
