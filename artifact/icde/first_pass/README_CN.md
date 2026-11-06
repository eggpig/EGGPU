# 首轮证据采集与导出

这三个工具读取已经存在的 JSON 收据，生成每格一次完整观测的 CSV、HTML、中文表格和静态图；不提交 GPU 作业。固定七图 US/EP/LJ/OR/IN/UK/TW 与适用 baseline 来自同目录 `e1_scope_authority.json`。当前源码身份由调用者显式选择的 freeze 决定，旧源码 EGGPU 行不能进入新源码表。采集器默认选择 `native-host-csr-static-all4-public-return-v2`，`--native-protocol` 可显式指定另一个历史协议；native 计时终点不一致的观察被拒绝，不能在同表混合。每行保留原 `measurement_protocol`。

在自己的实验 workspace 中运行，所有输出放在 checkout 之外。`--plan` 可重复传入相对 campaign 的计划，`--queue-state` 可重复传入队列状态；默认历史搜索列表只负责发现证据。新执行布局必须显式传入对应计划与队列。

```sh
python -B artifact/icde/first_pass/collect_first_pass.py \
  --campaign /your/campaign \
  --source-freeze candidates/unified_release_20261106/freeze.json \
  --plan first_pass_20261106/unified_main/plan.json \
  --queue-state first_pass_20261106/unified_main/queue/state.json \
  --output /your/output/receipts.json
python -B artifact/icde/first_pass/build_first_pass.py \
  --bundle /your/output/receipts.json \
  --output /your/output/first_pass
python -B artifact/icde/first_pass/render_icde.py \
  --report /your/output/first_pass/first_pass.json \
  --output /your/output/first_pass
```

采集器需要可读的完整 canonical 输入 metadata、原始测量与其引用的证明 JSON；不重读结果数组或生成新的正确性证明。`build_first_pass.py` 核对已有完整输出、源码、输入、时钟和质量合同；首次/执行时间只读取原结果。BFS/SSSP 的三个固定源按等权汇总，仍算一次观察。相同源码的完整 target 可使用，但不改名为 formal。

导出保留全部适用系统、缺项、失败与重复尝试。图中的 Best observed 仅为该格已经测到的最快适用对手，具体系统和覆盖见 CSV；覆盖未齐时不表示完整对手集合上的领先。失败没有替代时间，也不自动计为 EGGPU 胜利。n=1 不生成误差条或最终 SOTA 比例。

收据包可能含本地部署路径和失败日志；本地导出不等于可直接公开。公开前须审查并去除凭据、私有绝对路径与机器标识，同时保存内部原件及内容哈希。当前包只附带首屏轨迹的审阅摘要，不附带未完成的 SG/MG 结果图。执行状态见 `../FIRST_PASS_UPDATE_CN.md`。
