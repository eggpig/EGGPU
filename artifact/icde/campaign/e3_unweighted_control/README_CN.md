# E3独立无权PR负控制 v2

本入口只补权重不变性控制，不改冻结state_v4、源码、输入或80请求主trace。当前为CPU实现/测试完成，GPU资格和正式结果尚不存在。每个US/IN/OR配置先查询一次原BFS无权拓扑投影的PR，再进行20次真实FP64权重提交，每次立即重新调用同一无权PR；共20提交、21次真实PR调用、21个完整owning/readonly FP64向量。第16次更新恢复初始权重，但仍与第15次不同；每次提交都核权重身份变化及拓扑身份不变。

两臂为D0C1(persistent_full)和D1C1(lease_sparse)，C固定closure、无容量预算，ρ=1e-4 uniform。公共Native Session同时注册原weighted base与原BFS无权投影，绝不编辑manifest的weighted字段、不将带权图假装无权图。每次PR调用校验public contract、执行snapshot、正迭代数和收敛状态；返回数组必须相互独立，保留到Session关闭后再次核完整hash。

`old_readers=none`是唯一合法配置。现有`acquire_snapshot_lease`只保护已驻留的SSSP/weighted-PR FP64输入视图，纯无权PR轨迹没有这些视图；`per_update`会明确拒绝。加入带权查询来制造租约会改变本控制工作量，所以没有这样处理。真实旧读者/COW影响继续由E3主轨迹覆盖。

独立CPU oracle只读取canonical offsets/indices，构造单位边转移；不打开真实weights文件，不读取GPU答案作为参考。每次完整向量都核residual L1≤1e-6、mass error≤1e-6、reference L1≤1e-5；与base输出的L1差≤2e-8，另记录是否逐字节相同。2e-8覆盖FP64求解器的全局1e-9停止误差经0.85收缩界放大的量级，不将浮点归约末位误差当作权重依赖。

每次实际query记录native weighted-view full-H2D、scatter packet-H2D、dense-lease-H2D和COW-D2D计数。base允许初始准备成本；20次权重提交后的无权查询要求这些维护计数为0。原生未出现的counter key按冻结源码的零初始化map语义解释为0，并保留完整per-device原始metrics。总CUDA拷贝没有完整插桩，明确写`NOT_FULLY_INSTRUMENTED_DO_NOT_INFER_ZERO`；通信payload、共享数组新分配字节与权重维护计数分别保存，不混为同一“传输量”。

连续计时分两列：Session创建到21查询/20提交结束，以及Session创建后到工作量结束。CPU oracle、完整hash、无损NPZ落盘、最终inventory和Session close在计时外。所有完成输出先写内容寻址对象及append-only索引，再做质量检查；失败不删除已保存证据。磁盘门限为`21*n*8+2GiB`，不预支去重。主机门限为`4GiB+21*n*8+21*m*8+12*m+64*n`，覆盖不可变权重代际、结果及独立oracle的保守余量，不称为测得峰值。

## 计划与门禁

`pr_plan.py build --deployment <明确冻结deployment.json> --inputs <US inputs.json> <IN inputs.json> <OR inputs.json> --output <新计划目录> --results-root <新结果目录>`仅做CPU校验和计划生成。默认P1；可显式`--partitions 8`，但要求三张图对应P8输入全已准备，不由P1成功推断多卡可行。

P1计划为2tiny+6target，共8项资格。tiny用冻结TINY_A，但同样做21PR/20提交，以覆盖16步权重循环。每个真实图两臂各10独立重复，正式60次，固定按repeat交替臂顺序。每个任务由原全局GPU锁、16核、GPU空闲、主机/cgroup余量及根队列进程组超时约束。source/DSO/全部源码/冻结state_v4/support/本控制worker与planner/环境/UUID/输入均绑定；本控制tiny/target与主E3证明不能互换。

`formal_pending.json`不是正式队列。资格完成后`pr_plan.py admit-formal --bundle <计划目录> --output <新正式队列>`才会读回每个资格完整向量，核当前源码/配置与输入，导出60次队列。此时仍没有正式结果。队列只能由主代理提交。

## 分区集中轴：现有能力、缺口及实施成本

冻结`state_inputs.positions`仅有全局uniform和全局行连续clustered，不能把它们称为“精确集中到K个GPU分区”。worker的packet路径能消费任意完整签名的positions并按edge_cuts定位rank，故新增此轴只需独立CPU准备器和计划字段/CPU资格，不需改CUDA、D/C源码或主trace。

建议预注册P8、IN/OR、ρ=1e-4，`K={1,2,4,8}`表示实际承载修改弧的非空分区数。分区集合按固定seed对partition ID排序取前K，所有臂共用同一positions文件；不按测得时间或GPU负载选分区。每个输入记录逐rank修改数、实际K、最大占比、熵、成对弧和切边比例。ρ=0没有非空分区集中度；ρ=1通常只能K=8，均标明域限制。

IN是有向图：在固定集合各分区的edge interval内，按预注册等额配额抽样且总修改数固定19160。任一选定分区容量不足即显式INFEASIBLE，不悄悄扩大集合或改ρ。OR是无向图：必须按完整无向关系成对修改；若要严格K，候选关系两端的owner都必须属于选定集合。只挑单方向弧会破坏权重对称，不能采用。OR固定10634对、21268条存储弧；记录内部边候选容量和跨分区对数，容量不足同样拒绝。这个严格定义限制了OR候选结构，解释时必须注明，不能假定只改变设备负载而完全不改变选边分布。

CPU准备可按固定chunk流式扫描OR的u<v唯一关系，先计候选数，再按确定性选中序号写两方向位置；复用已资格的typed-search反向位置查找。无需新复制CSR/weights：8个输入的positions载荷约1.3MiB，另有小metadata。时间是每K至多两次O(A)扫描和被选反向查找；64MiB级chunk内存即可，实际准备时间和峰值要测后记录。这里不填造秒数。

若两D臂×两旧读者×两图×四K全覆盖，为32个目标配置、320次正式workflow，合计25600查询与6400提交；完整输出可用原内容寻址库但门限不预支复用。K=8的新配额样本与旧全局uniform样本不是同一输入，不能借旧target或正式结果顶替。此轴尚未实现/运行，要求仍保留在coverage待办；需主代理明确采用严格K合同后由独立准备器完成。

## v2部署修正

v2只增加正确的虚拟环境调用与身份绑定，21查询/20提交、数值阈值和传输合同不变。要求state_final_plan_v3 deployment的`python_invocation`及`python_runtime`，原解释器词法路径保留；ELF仅用于字节核验，不用其resolved path作argv[0]。本控制configuration绑定prefix/base_prefix和NumPy/SciPy实际文件，因此旧v1 proof不能混用。旧v1生成计划保留为错误部署历史，不得提交GPU。
