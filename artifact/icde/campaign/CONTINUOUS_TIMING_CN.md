当前候选把作者系统的 `T_first` 改为一个连续区间：共享 canonical CSR 已在内存中，从父进程开始准备作者输入格式之前，到作者进程首次形成完整、拥有独立存储的 host 输出并完成同步之后。弱投影、重排、格式转换、实际序列化、进程启动、运行时初始化、图准备和首个查询均计入；输入预先核验、结果归档和独立 CPU oracle 不计入。没有把若干局部计时相加当作端到端时间。

各作者适配器新增同一个 CLOCK_MONOTONIC 完成事件，记录 PID、请求序号和输出长度。exec 启动器在同一 PID 中记录 boot id、time namespace 和时间偏移。父进程核查事件属于该次启动、恰好三次按序完成、完整顶点数一致，且全部处于父进程时间区间。T_ready 仍是第三次完整 native 查询的连续原有计时。此前的 native-first 和 layout+native-first 仅保留作分解指标，不能替代共同 T_first。

作者实现和固定编译参数保持原样。CGgraph legacy BFS 额外补一个每次请求的正确性重置：作者构造时 level=0，但 graphProcess 仅清零 ite，第二次调用继续使用上次 BFS 的 level。2026-11-04 的 v2 有向/无向 tiny 中，第一次输出准确，后两次分别多出 4/8 与 3/6 层。新适配器在原有完整调用计时区间内，将 GraphCG 的 BFS level 置零；SSSP、WCC、PR 与其它引擎不受影响。此处不改变遍历算法、并行调度或参数选择。

固定构建组是远端 `baselines/sg_clock_build_v4`，11 个单元中 8 个引用 v3 成功的观察点构建，3 个单独重建（CGgraph 整数/BFS 重置，SYgraph 两种入口修复编译器脚本 PATH）。sycl/core 环境都是已有环境，没有全局安装或修改。每函数二进制、作者 commit、数值类型、编译命令、补丁和动态库解析路径保留独立绑定。现代/legacy CGgraph、Gunrock Essentials/legacy CC 仍是明确的版本组，不能按图择快拼接。

CPU 合同已通过：14 项输入/CLI/版本路由、8 项真实 C++ 到 Python 的跨进程时钟和拒绝错误事件、21 项实际 ELF 动态库解析，共 43 项。CPU 编译不是 GPU 资格；本候选须重新跑全部 42 格有向/无向 tiny，之后按每图每函数 target 资格才允许正式十次重复。此前 v2 的 40/42 数值通过和 2 个 BFS 失败均保留，不能套用作此候选的源绑定证明。
