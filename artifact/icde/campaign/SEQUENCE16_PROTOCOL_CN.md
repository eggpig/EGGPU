# E1 独立连续 16 请求工作流候选

状态：实现及 CPU 契约检查完成，GPU 尚未执行；不是现有单查询 E1 的替代结果，也不继承其 GPU 资格。不会修改活动 `sg_worker.py/run_sg.py` 或 public_csr_v1 源码。

## 固定任务

同一冻结 canonical 图，一次全新进程、一个持续会话，严格执行 `[BFS, SSSP, WCC, PR] × 4`。四轮 BFS/SSSP 源点依次为 `primary_sources[0,1,2,0]`；WCC/PR 无源点。所有正式重复都执行完全相同 trace，候选和图的选择不使用性能结果。权重保留同一 w16-v1，PR 明确未加权。共有 16 个完整 n 维 host 输出，不是 3 个源点块各 16 请求，也不是将已有单查询结果相加。

主指标 `T_workflow_seconds` 是一个连续 `perf_counter_ns` 区间：共同、已逐字节校验并访问过的 host CSR → 正常包导入 → 公共输入检查/拥有副本 → 初始化、布局、上传 → 全部 16 请求及各自的完整独立只读 host 输出。首次遇到某函数的额外结构准备发生在该请求计时内；16 次之间没有插入 oracle、文件保存或 SHA。每次原始 begin/end、单次时间和必要 ownership copy 都单独记录，但总时间不由它们求和。没有额外 warmup，也没有把预热的单查询组合成流程。

另外记录首个完整结果的端到端时间、构造时间、清理时间和从同一起点到关闭的连续时间。主区间结束于第16个结果就绪；关闭在后，数值审核与落盘再在关闭后进行，证明所有结果不依赖活着的会话。失败时已完成请求、输出、错误和 attempted interval 保留，完整流程时间为空。

## 两个系统的真实入口

EGGPU：显式 `--source` 指向带公共 CSR 入口的源码，例如正在资格化的 public_csr_v1 或经审阅整合的后继。仅正常 `from easygraph.gpu import Session`，调用 `Session.create(input_format='csr')`，不构造伪包、不私自绕过输入 seal，不按图/速度选后端。payload 含全部 n 个节点，保证孤立点不丢失。CLI 不硬编码旧源码路径；不支持公共 CSR 的源码直接失败，不偷偷回退。

cuGraph：沿用本轮 `cugraph_csr_adapter.py` 的四函数作者 API。BFS 与 SSSP 共用一个保留的 weighted outgoing SGGraph；BFS 接受带权 Graph 的合法性由 [官方 pylibcugraph.bfs 示例](https://docs.nvidia.com/cugraph/latest/api_docs/api/plc/pylibcugraph.bfs/) 支持，示例将 `weight_array` 传入 SGGraph 后调用 BFS。这里仅复用图对象，BFS 仍计算 hop distance，SSSP 仍读取原始权重。弱连通使用单独的弱投影图，PR 使用单独的单位权重转置图；三种表示在首次需要时创建，之后保持。无额外结果缓存，没有重写作者算法；现有公开 API 自己的正常缓存行为保留。与单查询适配相同，PR 每次无初值复用、统一数值门；SSSP 作者 API 必需的 predecessor 工作保留。

每个 query 必须返回独立的完整 host 所有者。公共 EGGPU 的不可变 owning 输出直接保留；cuGraph 现有 adapter 每次 `cp.asnumpy` 生成独立 host 数组后冻结。若 API 返回借用、非连续或可写数组，worker 立即在当前 query 计时内复制；检测到与历史输出 alias 也复制。CPU 测试包含故意重复写入同一个返回 buffer，验证先前结果仍独立。

## 资格、资源与 provenance

仍采用单卡 UUID 与全机 occupancy、host/cgroup headroom 检查；显式固定16 CPU；父进程持有全局 flock，子进程继承相同 inode/FD，并验证父锁仍占有。一整个流程的 timeout/watchdog 连续覆盖，不在每个请求重置计时。只终止自己的子进程组，不清理他人任务。

epoch 绑定选择的完整源码、native DSO、arena libraries、相关已安装 Python/DSO 文件、显式 preload、环境和源码参数；实际映射的 `.so` 在计时后另记 SHA。输入绑定 receipt/全部 CSR/weights/sources/optional mapping；每个请求绑定完整输出 SHA 和独立 CPU oracle receipt。数值门复用当前 SG：距离 exact/unreachable 一致、WCC partition equivalence、FP64 PR residual/mass≤1e−6 和 referenceL1≤1e−5。所有16个结果都在关闭后审计；相同输出仅在本次 job 内共享一个落盘文件，每个请求仍有独立执行记录和内存所有者。

需要独立的 tiny→同图target→formal 资格。单查询 proof 被明确拒绝。formal 默认十次重复，不按结果挑重复；系统运行顺序按重复号交替，工作流内部顺序始终不变。

## 入口

- `sequence16_worker.py`：一个系统/图/重复；`--source` 为公共 CSR 源码目录。
- `run_sequence16.py`：默认仅输出计划；`--phase formal` 未指定 `--repeat` 时计划完整 10 次。
- `sequence16_adapter.py`：固定 API 映射与合法图复用。
- `test_sequence16_campaign.py`：CPU-only 契约。

```sh
CORE_PYTHON -B sequence16_worker.py --execute-gpu --system eggpu --phase tiny --dataset TINY_D --gpu-uuid GPU_UUID --cpu-set CPU0,...,CPU15 --input-root INPUT_ROOT --source CANDIDATE_SOURCE --native-dso CPP_DSO --arena-library-dir ARENA_LIBS --setup SETUP_JSON --lock-file GLOBAL_LOCK --reference-cache REF_CACHE --output NEW_JOB_DIR
```

cuGraph 用对应 RAPIDS Python 和 `--system cugraph`，其他资源、输入、协议一致。控制器额外传 `--python-eggpu CORE_PYTHON --python-cugraph RAPIDS_PYTHON --systems eggpu,cugraph --datasets ...`。`--proofs` 是显式通过的 workflow result.json 路径列表，不能填写旧单查询 result.json。GPU 输出资格和完整正式测量仍需实验所有者执行。
