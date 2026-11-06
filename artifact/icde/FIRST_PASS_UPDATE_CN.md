# 统一源码与首轮执行更新（2026-11-06）

当前唯一实现源码为 `71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce`，1162 个文件。相对上次公开 d84 版本有 8 个实现文件变化，整合不可变 SSSP 主机载荷共享、resident owner 保留、Arena WCC path halving 与注册主机输出。原有 CPU 验证保留原源码身份，不能直接计为新源码的完整验证。

实际绑定的 native DSO 已构建/暂存通过，SHA-256 为 `e50bcad821a4972c6a8039edf58837229849c4f75e38aa56a051268afb5907ac`，8,551,264 字节。native 源码为 `959cb6bac81b36c9c56d7ec02427bf9c7c1dae6d6812e7fc381ae42be59b0d65`；统一版本只额外修改两个 Python runtime 文件，所有 native/build 源码逐字节一致。公开包不含机器二进制，重新构建后仍须匹配当地部署验证。

首屏固定连续轨迹已实际通过：IN 的 G0 → 权重更新后的 G1 → UK 的 H → G1 重访，8 卡、SSSP、每个策略一次完整观察。主时钟从初始 G0 ready 后的权重提交开始，到 G1 重访完整结果结束；保留更新、查询、同步、完整主机输出、实际 retention 与区间内控制开销。

| 策略 | ready 后更新至重访 (s) | 原四请求总时长 (s) | 次数 |
|---|---:|---:|---:|
| Full upload + LRU | 8.938643417 | 21.794597326 | 1 |
| EGGPU | 7.899269512 | 21.716015897 | 1 |

两个策略均保留四个完整输出与独立正确性检查，结果在 Session 关闭后仍有效；实际压力与释放字节有记录。可移植摘要见 `provenance/first_page_unified_summary.json`，原始区间、receipt SHA 与计时定义可核对。此结果仅说明该固定内部控制轨迹，不能代替完整静态矩阵、外部 baseline 优势或稳定性结论。

本轮先完成每格一次的首轮矩阵，再做第二次。已经完成且源码、输入与合同一致的 target 测量可按 n=1 使用，保留 target 身份。历史快照有 329 格中的 273 格可用、17 格失败、39 格待完成；它们属于各自原始源码与 baseline 证据，不迁移为 71ea 结果。统一源码的静态 SG/MG 矩阵仍待完成。单卡 90% 与多卡 85% 是优化目标，当前没有对应实测达成声明。

`first_pass/README_CN.md` 说明只读采集、源码选择与导出。T_first、T_ready 与整作业 wall time 分开，所有失败、取消和缺项继续保留。完整 formal 图表仍须通过各自准入合同。

静态 E1/E2 新增统一 native 测量适配器 `campaign/native_static_measurement/`，四函数与 P=1/2/4/8 统一选择 `Session.native_static`；runner 审核移至完整公共查询返回之后。协议终点变化明确记录，旧新计时不混合。runtime 源码 71ea 不变；CPU 通过不表示 GPU 正确性或加速已确认。
