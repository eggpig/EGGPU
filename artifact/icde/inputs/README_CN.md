# 七图原始输入构建工具

本目录从冻结的原始发布文件生成 host CSR、w16 权重和三源点。没有 GPU 依赖，也不读取旧实验 CSR。当前已通过 19 项 CPU 格式、映射、搬迁、断点续传与缓存幂等检查；EP/OR/IN/UK/LJ/US 六图已从原始压缩缓存独立重建，完整数组/权重/源点哈希与当前规范输入相等，TW 仍在下载。[qualification.json](qualification.json)记录资格边界。另已在空目录实际下载 EP 并核对完整原文件哈希。完整七图的从零网络下载与构建尚未全部运行，不能据此宣称整套实验一键复现。

## 来源与格式

| 图 | 原发布源 | 构建规则 |
|---|---|---|
| EP | [SNAP soc-Epinions1](https://snap.stanford.edu/data/soc-Epinions1.html) | 压缩边文本；所有原始标签排序后映射到连续编号，保存 `source_labels.npy`，标签总数须等于发布节点数 |
| LJ | [SNAP soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html) | 压缩有向边文本；保留发布的完整连续节点域，去除环与重复有向弧 |
| US | [DIMACS10 clustering/road_usa](https://sites.cc.gatech.edu/dimacs10/archive/clustering.shtml)；[SuiteSparse 来源条目](https://sparse.tamu.edu/DIMACS10/road_usa) | 原发布 METIS 邻接行；保留每行含孤立点，one-based 减 1，拒绝环/重复弧，不额外对称化。来源链同图，不宣称 SuiteSparse 文件字节相同 |
| OR | [Network Repository soc-orkut](https://networkrepository.com/soc-orkut.php) | 官方 ZIP 内 `soc-orkut.mtx`，2,997,166 节点。只给原文件单 `%` 标头补一个 `%`，接受标头行末空白；按 symmetric Matrix Market 展开双向弧。不得替换为 SNAP Orkut |
| IN / UK / TW | [LAW indochina-2004](https://law.di.unimi.it/webdata/indochina-2004/)、[uk-2002](https://law.di.unimi.it/webdata/uk-2002/)、[twitter-2010](https://law.di.unimi.it/webdata/twitter-2010/) | 只使用原 basename 的 `.graph/.properties/.md5sums`，不选 transpose、natural 或高压缩变体；编译本目录 Java 源码，顺序导出完整节点域 |

`source_catalog.json` 固定原始文件 URL、长度和已观测 SHA256；LAW 同时核对发布者 MD5。TW 的完整 SHA256 尚未取得，必须通过已冻结的官方 MD5 才能通过来源资格。下载缓存不能静默替换；遇到内容变化直接拒绝。`--law-http` 显式使用发布者 HTTP 传输，仍须匹配官方 MD5 和可用的固定 SHA256；用于当前 HTTPS 不可用时，不更换镜像或图版本。

`expected_canonical.json` 记录已有六图的 offsets、indices、weights、sources 完整哈希及 EP 标签数组。重新生成必须逐项相同；收据和路径可以不同。TW 尚无已通过的规范输入参考，不伪造该项。规范输入的 id_mapping 明确说明标签规则。

## 前置依赖与空间

目标环境为 little-endian Linux x86_64，Python 3.12、支持 C++17/OpenMP 的 C++ 编译器、OpenSSL 开发头文件和 libcrypto、JDK 17（`java` 与 `javac`）。它们是显式前置依赖，工具不会修改系统软件或全局环境。已测科学包为 NumPy 2.5.4 / SciPy 1.18.1；官方 CPython 3.12 wheel URL 与 SHA256 在 `python-cp312-linux-x86_64.lock`，下载共 52,071,482 字节。其他 Python/CPU 架构应重新资格化，不能套用这些 wheel 哈希。

`java_dependencies.json` 固定 22 个 Maven Central JAR 的版本、URL、长度和 SHA256，共 33,477,194 字节；所有固定 URL 已做可访问与长度检查，并已从官方 Maven 重新取得全部 22 个 JAR，完整 SHA256 均匹配。`setup_tools.py` 对实际下载/缓存逐个验 SHA，然后用 `javac --release 17` 编译 `LawToCsr.java`，独立检查 Java 自测的完整输出，并记录编译器/运行时和全部产物哈希。JDK 使用调用者显式提供的路径，不再依赖旧实验目录。

`space_budget.json` 给出逐图数字，`preflight.py` 检查实际磁盘和 cgroup/主机内存余量：七图规范数组总计上界 **27,897,105,512 字节**，原压缩源 **3,531,981,894 字节**，加 JAR 与 6 GiB 余量，新数据目录预算上界为 **37,905,015,544 字节（约 35.30 GiB）**。TW 的最终去环后弧数尚未知，因此其 canonical 数字使用原始弧数上界，其余六图为精确 payload 数字。压缩文本流式读取，不另存全量解压文本。

主机内存准入采用 `64A + 128V + 2 GiB`，七图最大 **101,454,340,736 字节（约 94.49 GiB）**。这是保守准入公式，不是实测峰值。上述磁盘预算不包含单独安装的 JDK/Python 环境、旧版本、GPU 布局或正式重复输出；不要把它写成完整 artifact 需要的总空间。

## 新机器执行

先进入本目录。下列目录由调用者选择；全部输出置于自己的 artifact 工作目录。

```sh
python3.12 -m venv WORK/input-env
WORK/input-env/bin/python -m pip install --require-hashes -r python-cp312-linux-x86_64.lock

WORK/input-env/bin/python preflight.py --root WORK/inputs --enforce
WORK/input-env/bin/python setup_tools.py --tools WORK/input-tools --java /PATH/TO/JDK17/bin/java --javac /PATH/TO/JDK17/bin/javac
WORK/input-env/bin/python fetch.py --root WORK/inputs --law-http
WORK/input-env/bin/python prepare_inputs.py --root WORK/inputs --tools WORK/input-tools --java /PATH/TO/JDK17/bin/java --registry dataset_registry.json
```

预检是执行前置步骤；大图准备与正式性能计时应错开。构建工具目录须全新，不静默覆盖已绑定的工具链。`fetch.py`、`prepare_inputs.py` 和 `preflight.py` 均可通过 `--datasets US` 先处理单图。已有原始压缩缓存可用 `fetch.py --cache-import <JSON>` 指定文件名到路径的映射；只在完整内容校验后硬链接或复制，不能用旧 CSR 冒充原始重建。JAR 同理支持 `setup_tools.py --jar-cache <目录>`。

每个图输出 `canonical/<缩写>/canonical.json`、完整 CSR/权重、`sources.json`、`id_mapping.json` 和 `receipt.json`。只有 `PASS_CANONICAL_INPUT` 能供 worker 使用。失败目录保留，必须检查后为重试提供新输出目录，不原址覆盖共享数组。已通过输入整体搬迁时按新目录文件及原哈希验证，无需修改原收据的来源路径。

下载默认每文件 4 条连接；仅在原发布者持续正常响应时才显式选择 `--connections 8`。改变连接数仍沿用相同 URL/ETag 与原 ranges 收据，跳过已完成区块，不从头重下。下载对每图加独占锁，对全局 fetch_state 加共享更新锁，仅合并正在处理的图。不要同时运行旧的整表重写型下载器。数据源与工具构建时间不计入 E1 算法任务，但每个系统从共同 host CSR ready 开始需要的布局转换、注册和上传仍应计入 T_first。
