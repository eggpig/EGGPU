# E6 完整依赖应用：当前可执行范围

2026-11-04。此入口只准备了源码、CPU 契约检查和受控执行器；未在本代理运行 GPU，不能写作新 E6 结果。它不修改已经冻结的 `artifact/icde_release`。本轮不新造结构洞或中心性算法，复用既有 workflow v2、dense_v1 和 addon_direct_v1，来源和提取范围见 `e6_reuse_provenance.json`。

## 固定任务

主工作负载为 **EP-U、LJ-U**；OR 是单独标识的压力工作负载。这里的 U 表示对指定 canonical 输入构建简单、无权、无向弱投影，不是换成另一个数据集。保留原 canonical 的全部顶点，包括孤立点。原始 w16-v1 权重和 sources.json 继续绑定输入身份，但这些 E6 算子明确采用无权语义，BFS 源点由本次应用结果决定，不使用预设主源点替代。

执行顺序固定为：

1. 对**全部顶点**计算 Effective Size 与 Constraint，返回两个完整主机向量。
2. 对有限分数顶点，先按 Effective Size 降序，再按 Constraint 升序，最后按 canonical ID 升序排序；分数使用既有 `1e-6`、round-to-even 量化。固定选 32 个候选。未根据性能或某次胜负更换排序、候选数或函数。
3. 对这 32 个实际候选计算精确 Wasserman–Faust Closeness；`r² / ((n−1)·Σdistance)`，没有可达其他顶点时为 0。它是精确选定源计算，不是全图近似排名。
4. 以相同量化规则取最高 Closeness，canonical ID 决胜，得到实际 winner。
5. 以 winner 运行 BFS，并运行全图 WCC；返回完整距离与分区，核对 BFS 可达集合等于 winner 的弱连通分量。

这是受控的结构筛选与可达性核验应用，没有声称是生产日志。`E6_TINY_U` 专用资格图有 40 顶点、两个非平凡分量、4 孤立点与 32 个实际候选，足以让 8 卡 source-shard 都有工作；不能用此前只有 7 顶点的 SG tiny 图冒认 8 卡资格。

## 真实实现与比较边界

| 配置 | 实际实现 | 使用范围 |
|---|---|---|
| EGGPU SG | 既有 `cpp_egmgpu_single_execute`，复用 bulk CSR 与结构投影 | 完整任务与四组覆盖消融的控制组 |
| EGGPU 2/4/8 卡 | 结构洞走既有 dense replicated ABI，按 ego 顶点分片；Closeness 按所选源分片；BFS/WCC 在 logical GPU0 | 全图复制模式的扩展；不是图分区容量扩展 |
| igraph | 原生 C Constraint、distances、components；Effective Size 由原生 transitivity/degree 代数归一化 | 完整 CPU 组合，GPU 数明示为 0，不冒充同卡 GPU 系统 |
| cuGraph + igraph | cuGraph triangle_count/degree 导出 Effective Size，cuGraph BFS/WCC；igraph 实际执行 Constraint 与 Closeness | 既有完整 SG 混合组合，CPU 回退和格式转换都计时 |
| cuGraph + CuPy 组合 | 沿用 addon_direct_v1：pylibcugraph BFS 和 GPU 归约导出 WF Closeness、cuGraph triangle_count/WCC、CuPy CSR 乘法/掩码/归约导出 Constraint | 所有五个数值阶段在一张 GPU，仍需新的 tiny/target 资格；应标为组合实现，不称 cuGraph 原生提供 Constraint API |

这里**不使用 Session.create/arena 调度扩展函数**：当前该入口不支持 Effective Size、Constraint 或 selected-source Closeness。SG/native 与 MG/replicated 的资源身份逐阶段保留，不能拿本实验表示分区后端的强扩展结果。每卡都必须能容纳完整复制图。

覆盖消融固定四组：结构洞（Effective Size + Constraint）、Closeness、BFS、WCC。每次只把一组替换为实际 igraph C/代数实现，其余阶段走原 EGGPU ABI；回退图构建、数据转换、结果返回与应用逻辑全部执行。不是跳过函数、用预计算值或拼接独立计时。覆盖消融仅在 SG 做，同一 32 候选规则与输出合同保持不变。对照并不声称只能使用 Python NetworkX；CPU 回退沿用已有较强的 igraph 实现。

## 计时、验证和资源

共同起点是已核验并触页的原 canonical 主机 CSR。**T_first** 从构建 U 投影之前开始，覆盖弱投影、后端导入与初始化、bulk CSR/结构投影文件与格式构建、图上传、全部计算、两次结果依赖选择以及全部主机结果物化。后端需要的临时文件写入也计费，不能把 native 专用预处理免费移出。

一个新进程连续执行 first、一次完整 warmup 和 ready；**T_ready** 使用该进程已建立的图状态。每次工作流计时均是一个连续墙钟区间，不是五个 stage 耗时之和；stage 仅作归因，候选与 winner 选择分别记录。close 单列，校验/hash/写结果文件在计时之后。每个正式重复仍是全新的进程，默认 10 个独立重复；奇偶重复仅反转配置顺序，不查看实测时间。

所有 ES/Constraint 顶点、32 个 Closeness、完整 BFS/WCC、候选及 winner 都保存并校验，主机数组必须独立拥有、只读、跨后续调用和 close 继续有效。ES/Constraint/Closeness 的精度阈值沿用 `rtol=1e-8, atol=1e-10`；孤立点 NaN 掩码、候选和 winner 必须一致；BFS 精确相等，WCC 比较分区等价。中间顶点 ID 返回必须覆盖全部原始顶点，不能静默丢孤立点。

独立参考沿用 SciPy 精确结构公式、每源 BFS 和连通分量。为了适配大图，只对矩阵乘法的行块划分作内存规划；数学公式和全部顶点范围不变，规划不依赖性能。32 次 BFS 逐个计算，避免同时保存 32×n 距离矩阵。tiny 结构分数及 WF Closeness 已与 NetworkX 独立核对。正式目标图的 CPU 参考必须先显式准备成功；`target/formal` 在没有绑定参考时拒绝启动 GPU。

CPU 参考在 LJ/OR 上可能昂贵；不能因昂贵改为抽样后声称全向量正确，也不能把未资格配置补为数值。若单行稀疏乘积估计超过参考规划上限，准备会明确失败，须由后续 CPU 资源实现改进解决，不减少输入。

每项启动检查真实 GPU UUID 与 setup、一台机器的 GPU 空闲状态、现有全局串行锁、磁盘和 host/cgroup 可用内存。进程内 1800 秒完整任务超时配合父进程 watchdog；仅终止自己创建的进程组。所有失败、超时和原始输出保留。source epoch 绑定 wrapper、当前源、GPU 源、DSO、原编译源 SHA、实际依赖包、配置、设备与环境；tiny→target→formal 必须完全匹配，不自动找最新或最快 proof。

## 执行顺序

下面都在本目录执行。`${...}` 必须由主代理替换成现有真实路径/UUID，普通计划不提交 GPU。

```sh
# CPU-only fixture and independent reference; common canonical inputs remain immutable.
python -B e6_worker.py --make-tiny "${INPUT_ROOT}"
python -B e6_worker.py --prepare-reference --dataset E6_TINY_U --input-root "${INPUT_ROOT}" --reference-cache "${REFERENCE_CACHE}"
python -B e6_worker.py --prepare-reference --dataset EP --input-root "${INPUT_ROOT}" --reference-cache "${REFERENCE_CACHE}"
```

LJ 下载与 canonical 资格完成后同样先准备 LJ 参考；OR 单独加压力组。先做每个实际 system/ablation/GPU-count 的 tiny 资格，再用显式 `--tiny-proof` 做相同配置的 target 资格。正式运行再同时给 `--tiny-proof` 和 `--target-proof`。native 配置还要求 `--build-receipt` 指向成功的 `Easy-Graph/scripts/gpu/build.py` receipt，逐个 compiled source SHA 与 native DSO 必须吻合；不接收没有 source/output 绑定的“编译成功”状态文字。

`run_e6.py` 默认只生成命令计划，`--execute-gpu` 仅供主代理。`--proofs` 是人工明确选定的 E6 worker `result.json` 路径列表。主图用 `--datasets EP,LJ --native-gpu-counts 1 --coverage-ablation`；复制式扩展另用 `--systems eggpu --native-gpu-counts 1,2,4,8`。OR 必须保持 pressure 标识。不能在多卡比较中混入仅支持 SG 的外部组合。

入口依赖同目录 `sg_worker.py`、`cugraph_csr_adapter.py`、`e6_reused.py` 和 `e6_reuse_provenance.json`。CPU-only 测试用 `python -B -m unittest test_e6_campaign -v`；它只在合成图上验证合同，不能替代新目标图 GPU 资格。CPU 测试回执单列 `e6_cpu_receipt.json`；图表和结论要等实际测量收据。
