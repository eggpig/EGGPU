# 当前论文写作与结果入口

| 内容 | 入口 | 证据性质 |
|---|---|---|
| Intro 初稿 | [INTRO_DRAFT_CN.md](intro/INTRO_DRAFT_CN.md) | P1–P6 问题、缺口、D-Maintain、C-Retain 与贡献；未填未完成性能结论 |
| 首页问题图 | [说明与图注](intro/SOURCE_AND_FIGURE_NOTES_CN.md)、[Python](intro/render_first_page.py)、[单栏 SVG](intro/first_page_problem_85mm.svg) | Constructed example，共享分配算术反例，不是测量 |
| Related Work | [两部分各十篇](RELATED_WORK_NARRATIVE_20261104_CN.md) | 原文主源及 CCF 目录版本有依据；cuGraph 等软件另列 |
| 实验章节设计 | [设计](PAPER_EXPERIMENT_DESIGN_20261104_CN.md)、[协议范围](EXPERIMENT_SCOPE_RECONCILIATION_20261105_CN.md) | 设计、实现、CPU 验证、GPU 资格、正式重复分别标识 |
| 新结果导出 | [严格导出器](../paper_results_current/README_CN.md) | 561 格预声明范围，十次重复、完整输出与源码门；当前没有完整正式结果 |

首页问题图是 Fig. 1。机制图 M1 与框架图 M2 由协作者绘制，保留位置，最终编号在排版时确定。`figures` 内旧机制图材料是历史来源，不使用旧的“Fig. 1=机制图”编号关系。

本次公开只有当前写作版本；没有发布合成性能测试图、原机器 UUID、部署凭据或机器绑定的资格回执。派生正式图以后最多维护 current/previous 两代，原始重复、失败及来源证据保留。

本检查点的验证范围和复验方式见 [PROTOCOLS_UPDATE_CN.md](../PROTOCOLS_UPDATE_CN.md)。
