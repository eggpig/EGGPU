# 文档与首轮工具后续增量（2026-11-06）

基底为已发布提交`62883ed29701e09d4e655c7a92575d5566613406`。EGGPU实现源码仍为71ea757，1162个源码文件逐字节不变。本增量不声明整个七图矩阵、第二次重复或90%/85%优化目标已完成。

Intro精确区分：validation_edges=0只省去新权重的全量合法性验证，完整权重重建与内容哈希仍有O(m)成本。C²graph文献口径也按准备后的主查询时间与含压缩的首次补充时间分别说明。首图SVG更新为此前已公布的同一71ea配对轨迹；其可移植标量输入保留原receipt SHA、完整/ready窗口和互斥阶段，不是新测量。旧构造例JSON保留原身份，不再作为当前图输入。

首轮工具新增同代E2/E4S/E6导出，并保留原始phase、n=1、完整质量与来源校验。旧源码先归历史；取消前的准备延期、实际CPU索引限制、算法失败和证据待复核分别记录。缓存作者ready保持原样，cached first单列且不进入冷启动比较。CPU文本缓存及唯一storage guard修订的范围、原件身份与公开路径适配均显式披露。

本增量没有复制滚动结果表、私有runtime state、数据集、机器二进制、凭据或任务PID。旧公开测量摘要保持可追溯；运行进度应从执行者自己的最新source-bound回执取得。公开文件投影与测试见`provenance/followup_tooling_projection.json`和`provenance/followup_public_validation.json`。
