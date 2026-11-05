# Queue v4：正式计数与存储等待

独立文件 `run_campaign_queue_v4.py`，不修改 v3。CLI 仍为 `--plan --output --controller-lock`。

- phase 来自 command 的显式 `--phase value` / `--phase=value` 或 `job.phase`；两者必须一致，重复 flag 拒绝。没有声明为 unspecified，不能凭未声明的 formal receipt 计入正式测量。
- attempted 只在真实子进程成功启动时增加；PASS 同时要求退出码0、预期 receipt 状态及 phase 一致。预启动绑定失败、依赖跳过、formal 失败分别记录。`formal_measurements` 是成功 formal worker job 数，不是输出向量数量或内部源点样本数。
- 可选 `job.disk_guard={"path":"/your/output/volume","min_free_bytes":17179869184}`。不足时保持 `WAITING_STORAGE`，释放 controller lock，不启动 worker、不计失败/attempt。默认每5秒重查，可用 plan.storage_poll_seconds 调整。固定顺序等待，不自动挑后续更小任务，不重跑失败 worker。
- 可选 `job.maintenance`：`command` 显式argv，`command_sha256` 为紧凑 JSON argv 的 SHA256，`bindings` 绑定实际解释器/可执行文件及直接 `.py/.sh` 脚本；还需 `gpu_lock` 指向现存 worker GPU锁。可加timeout、environment、receipt、expected_statuses。
- worker结束后，maintenance 在 root controller lock 内运行；先实际尝试 GPU锁证明 worker已释放，再解锁，允许现有dedup工具自行获取同一 GPU锁。因此不会和会自行加锁的既有工具死锁。maintenance失败停止本队列并单列失败，不把已经PASS的formal改写成失败，也不静默继续填满磁盘。

维护命令的绑定可用同模块的 `bind(path)` 与 `command_digest(argv)` 生成；不得省略 Python解释器绑定。小型CPU测试使用真实子进程验证PASS/FAIL/依赖/真实formal计数、存储等待期间controller锁释放且只启动一次、绑定拒绝、维护子进程能持GPU锁且controller仍被父控制器持有。测试不启动GPU，也不生成论文结果。
