C²graph-P 在 E1 只比较共同 SSSP。使用作者 commit `97af1ebd935a03f5adc180935644e786a624bdbe` 的既有 P 单源 GPU 入口，保留以前明确记录的正确性修复；本次只增加完整 host 输出之后的 CLOCK_MONOTONIC 观察点。没有用 PM 双缓冲或多请求批时间除以请求数。

输入固定为七图各自 canonical CSR 和正整数 w16，BFS/PR 不冒充其支持函数。每个图固定三个 primary sources。每个 source 启动一个新图处理过程：转换作者 origin int32 权重、调用原有作者压缩器、加载压缩图、初始化运行时，然后 first/warmup/ready 三次独立 SSSP 求解。压缩图可以复用，距离数组、active flags、frontier、source compensation 和六个 GPU barrier scalar 每次按既有入口重新初始化。原始权重、顶点全集、孤立点和不可达输出均保留。

首请求是一个连续父起点到 native 完成点区间，包含格式准备、实际压缩、启动、H 传递补偿、执行、D2H 和 FP64/+inf 拥有输出；独立压缩审计、输出归档和 CPU oracle 在完成点之后。T_ready 是第三次 complete_call_seconds。所有结果逐全向量核验，每次原始执行记录保留，输出按内容寻址保存一次。

CPU 资格包括真实作者压缩器、真实 H 闭包 helper 与独立原图 Dijkstra 的有向/无向全源比较（14 个完整向量、98 个顶点值）、源参数和三次独立调用、ELF 装载、拒绝越权执行，以及跨进程时钟/错误事件和计划无副作用合同，共 15 项。GPU tiny/target/formal 资格必须由实验 owner 另行运行。本记录不声明 GPU 通过。
