E4S 可恢复输出与正式计数候选（无 GPU 提交）

原 worker、epoch、计算时间和超时保持原值。逐完整向量的实测 output_sha256 必须与已存独立 CPU NPZ 解压后的 float64 原始字节 SHA 一致；数值相等不足以通过（+0/-0 测试必拒绝）。同时核对图 CSR 内容标识、源点、向量形状、独立参考代码和 NPZ 完整 SHA。

archive_outputs.py 打开普通文件 fd（拒绝符号链接），读完整归档并核 SHA、解压单个向量核原始字节，重读同一 fd 验证未改变。同文件系统用硬链接把原 NPZ 固定到 object_store/objects/<SHA前两位>/<SHA>.npz，参考缓存删除原路径后仍可恢复。只有 EXDEV 才逐块复制，并保留 2 GiB 磁盘余量；权限、空间、数据、fsync 等错误全部失败，不自动改路。目标独占原子发布，文件和目录 fsync；不会覆盖已有对象或旧sidecar。已有对象也重新核完整 SHA。每次最多解压一个向量，不保存整个 batch 副本。

backfill_qualification.py 只处理明确 SHA 绑定的 E4S 资格 plan。默认仅列缺项；--archive 在 root_controller.lock 内给已经完成的合格作业补 recoverable_outputs.json，不重跑 worker，不将旧源或资格结果转成新源正式结果。未来每个实际新 E4S tiny/target 均需补此恢复证据。原计划中目前没有可读取的旧 E4S tiny 实测 receipt；真实 E4S 端到端归档仍待第一条新资格完成。CPU fixture 流水线与实际 d84 SG/sequence/native tiny 校验是分别标注的证据。

decorate_formal_plan.py 只替换正式计划的 maintenance，保留 worker argv、输出身份、重复0..9、3600秒计算合同与原 worker 外层时限。新 maintenance 先 archive_outputs，再原 maintenance，最后写 recoverable_maintenance.json。CPU归档/审计的独立超时最多86400秒，不改写基线计算 timeout/OOM 类别。归档失败不执行后续维护；原维护失败保留已完成的归档和所有原 receipt。

run_e4s_formal_queue_v1.py 是独立控制器；原 v8 与已运行的资格队列完全保留。它分别记录 worker_passed/failed、archive_passed/failed、original_maintenance_passed/failed。只有 worker、归档、原维护均通过才增加 formal_passed/formal_measurements。失败样本保留原 worker 成功/失败与原计算时间，正式可用数保持0。已有 worker 输出目录禁止重跑，自动 resume 禁用。

显式恢复仅限 CPU后处理：新 spec 的 recovery_of 必须绑定旧 FAILED_E4S_OUTPUT_MAINTENANCE receipt；使用新 attempt 输出文件及必要的新原维护输出路径。完整重新核验已固定的归档；不得执行 worker。旧失败 queue 和 receipt不改成成功，后续 continuation 与恢复证据必须由根审阅，不自动重放 GPU。

export_eligible.py 必须同时验证原正式 worker、成功 maintenance、原维护成功回执、精确 sidecar 与所有固定归档。仅 raw worker PASS、仅质量+SHA、仅 archive PASS 均不足以导出。导出时间直接来自原 measurements.first_complete_seconds；恢复与维护时间不进入该时间。实际调用方应以此门替代旧的 hash-only acknowledgement。

formal_admission_d84_v2/admit.py 要求全体冻结资格目标（SG56/native88/sequence6/E4S36）精确覆盖，所有格保留10次；深度完整输出校验持有 root_controller.lock。E4S还必须具备资格恢复sidecar，且正式计划绑定本候选控制器和maintenance。没有实际资格/参考/恢复证明便不准入。

尚未启用的能力终态接口：由根单独批准全矩阵 disposition，每个冻结target ID都必须出现；绑定源、DSO、runner、原receipt与日志、失败阶段。成功格执行10次；经确认的OOM/CPU预处理TO/索引限制/平台限制保留原展示槽位及状态，时间为null，不计速度胜利。此接口目前仅设计，现有v2仍采用保守全PASS门，不自动按速度或成功子集筛选。
