# 作者输入缓存与计时边界

此适配器用于已明确延期的TW Grapin/HyTGraph首轮任务：CPU转换器生成与原NumPy writer逐字节相同的CSR顺序文本，多个消费者以只读硬链接复用。原作者二进制、参数、数值检查和first/warmup/ready调用不变；缓存构建在独立CPU准备回执中记录。实际原CPU弱投影若触发int32限制，则记录INPUT_LAYOUT_UNSUPPORTED_CPU，不启动该GPU任务、不伪造耗时。

`result_cached_input.json`是原`result.json`的加字段副本。协议为`author-ready-native-call2-cached-input-first-supplement-v1`：ready仍为作者原native call2；first已有精确文本输入，另列cached-input补充，不能与canonical-CSR冷启动首次时间混比。原结果、每次call、缓存使用和准备回执都保留绑定。

`disk_admission.py`严格按AST匹配，仅替换原adapter中重复预算文本staging的一个磁盘准入比较。真实磁盘free/used/total没有修改；其他函数AST、host/GPU/numerical guards均保留。按源预算三份raw FP64输出、三份独立NPY归档、一份新参考以及2GiB元数据余量，不依赖之后的删除/去重节省。TW BFS/SSSP需要9,145,061,360B，PR需要4,480,009,552B；旧75,627,958,640B门槛包含已存在文本的重复预算。该修订不改变算法时间；source audit明确检查了实际冻结二进制对应的两次只读text解析和无边规模磁盘缓存写入。

公开包中的原author worker与既有冻结版本逐字节一致。binding/placement helpers在`../timing/`。CPU验证可在已有NumPy/SciPy环境中运行；临时输出指定到checkout之外：

```sh
python -B artifact/icde/campaign/author_text_cache/check_cpu.py --scratch /your/scratch --receipt /your/scratch/text-cache-cpu.json
python -B artifact/icde/campaign/author_text_cache/check_disk_cpu.py --scratch /your/scratch --receipt /your/scratch/disk-admission-cpu.json
```

`cpu_receipt_v7.json`是原序列化/生命周期验证，早于storage-only修订；`disk_cpu_receipt.json`单独验证该修订，并明确使用模拟文件系统读数。当前公开路径投影另经CPU验证，见`public_cpu_validation.json`。这些验证不代表TW GPU作业通过或有性能提升。

准备工作使用显式campaign、缓存/build/output目录、原worker SHA和当地可用的16个CPU；`prepare_inputs.py --worker`可覆盖默认的公开原worker。`prepare_plan.py`读取既有八格deferred计划与实际CPU preparation receipt，生成六个可缓存调用与两个有证据的索引限制条目；本包没有私有计划、任务PID、数据集或二进制。
