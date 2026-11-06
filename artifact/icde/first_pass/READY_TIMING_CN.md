# 主图使用已有 ready 区间

E2/E4-S/E6报告顶层 `primary_metric=ready_seconds`，`supplementary_metric=first_seconds`。这是展示顺序修正，没有新测量，也没有改变worker、源码71、输入或数值门。

E4-S主值读取 `measurements.query_batch_seconds`：EGGPU从Session.create返回至全部query完整独立输出；KGraph从Compute入口（图、分区、pivot已准备）至全部FP64输出；C²graph-PM从正常图加载/上传和队列准备后的pm_batch_begin，计入逐源H-closure补偿、实际GPU/CPU协作及全部FP64输出。该批次不是第二次热批次；各系统按需准备仍在其实际query区间内。`first_complete_seconds`覆盖批次之前的准备，保留补充，不由差值构造主时间。

E6协议本来就独立记录first/warmup/ready三个完整workflow；只有first构建弱投影与后端。主值直接读取实际`T_ready_seconds`，与child.calls.ready.seconds一致，包含五算子、候选及胜者选择、交接和完整主机输出。每次重新算输出，未复用答案；缺少实际PASS target则不画数值。`T_first_seconds`另列补充，不能用阶段相减推造ready。

n=1无误差条，Time(s)统一单位。图、HTML、CSV与JSON保留真实资源及完整边界；原first和ready数值均保留。
