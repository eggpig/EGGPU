# 开发检查点增量：协议、输出恢复与论文资产

主实现继续保持 d84 的 1162 个文件字节不变。本次同步实验工具及写作资产；正式队列已经启动，完整结果尚未齐备，不声明达到单卡 90% 领先目标。2026-11-05 10:32 UTC，控制器负责人报告外部正式队列已有 2 条通过，300 条队列仍未完成。这个时间点不是静态总数；没有把私有机器回执改写成公开资格证明。

## 当前入口与范围

| 目录 | 用途 | 必须保留的边界 |
|---|---|---|
| `campaign/e4s_recoverable_outputs_v1` | 完整输出逐字节恢复、归档与正式导出检查 | 归档/维护在测量结束后；原输出、计时和失败保留；缺完整恢复证明拒绝正式导出 |
| `campaign/formal_admission_disposition_v1` | 完整矩阵逐格只读检查 | 不生成执行计划，不自动准入，不把 OOM/TO/INDEX/PLATFORM 算作胜利 |
| `campaign/watch_external_formals_v4.py` | 保留完整 SHA 的两字段/三字段 binding 兼容 | 默认检查；实际启动要求显式 launch 与完整资源/计划证明 |
| `campaign/state_protocol_completion_v1` | E3 三臂、E4 五臂、E3R 分组/交错控制 | 480 配置、664 资格及 4800 正式槽位是计划规模；新协议须重新资格 |
| `campaign/e6_protocol_completion_v2` | 两组扩展能力的完整 2×2 与同一 EGGPU 遍历尾部 | 60 资格已提交；450 正式槽位未准入；新旧超时/epoch 不混合 |
| `experimental/native_wcc_wide_v2` | WCC int64 offset 独立源码增量 | 20 文件 overlay 重建完整 6cce；80 资格已提交；不替换主源码 |
| `paper/` | 两部分各十篇主源、实验设计、中文 Intro 与 Python 首页图 | 首页图是共享 allocation 的确定性反例，不是性能测量；机制/框架图仅占位 |

## CPU 检查与可移植性

依赖调用者已有的 NumPy、SciPy 和 pytest。包不安装全局环境、不读取 SSH 配置、不绑定原机器 GPU。运行输出和 scratch 应位于仓库之外的不存在目录。

```sh
python -B artifact/icde/tools/reproduce.py test-followup-cpu --work /your/new-cpu-output
python -B artifact/icde/tools/reproduce.py state-protocol-test \
  --source Easy-Graph --scratch /your/new-state-scratch --output /your/new-state-output
```

本次投影检查：E4-S 29 项、watcher 20 项、E6 59 项、state 65 项通过；disposition 35 项通过，1 项依赖原私有冻结计划的数量检查明确跳过。该项可显式提供 `EGGPU_PRIVATE_FROZEN_PLAN_ROOT` 后重查，不能把默认 skip 写成原私有证据通过。WCC metadata/实际小图宽度读写/公开入口检查是独立的无 DSO 子集；真实编译 DSO 与 GPU 资格单列。

新 state 的 `prepare.py` 从调用者给定的原 deployment、coverage 和输入 binding 派生独立计划，保留 Python 虚拟环境调用路径和原 DSO 字节身份。不要把另一台机器的计划字符串替换路径后当作已资格计划。

E6 v2 的独立数值 oracle 仍须来自相同 canonical 输入、原生产者和完整 NPZ/receipt。先用包中原 E6 参考生产器在选择的 CPU 环境生成四个输入（tiny、EP、LJ、OR）的完整独立输出，再绑定其 producer、公式和所有 payload；v2 的 bridge 只改变消融菜单与 1800→3600 秒合同，不转移任何 GPU 计时/资格。`build_plans.py --original ... --campaign ... --output ...` 要求完整原 11 配置模板及实际 oracle；其 `--draft` 只生成非执行 `job_specs`。`fixtures/` 明确是 CPU 测试模板，不是可执行实测计划或 oracle。

E4-S `archive_outputs.py`、`maintain_e4s_outputs.py` 和 `export_eligible.py` 都使用调用者显式输入；恢复与导出必须重核完整字节。`decorate_formal_plan.py` 提供纯函数 `decorate(plan, contract, output)`，contract 要绑定本机 helper/wrapper/exporter/controller/supervisor 的实际字节、独立 object store 路径和所选 Python；它不调用 worker，也不把未准入计划变为可执行正式计划。完整矩阵检查工具的 `inspect` 不代表准入。

## 写作与图

`paper/RELATED_WORK_NARRATIVE_20261104_CN.md`、`paper/PAPER_EXPERIMENT_DESIGN_20261104_CN.md` 和 `paper/intro/INTRO_DRAFT_CN.md` 是当前正文。`paper/references_20261105/` 提供主源链接、DOI、CCF 来源与跨论文实验差异，不分发作者 PDF。CCF 官方下载被 WAF 拒绝的限制保留；镜像核验不能写成官方页面已成功访问。

首页反例输入、Python 源码和 SVG 在 `paper/intro/`。使用 Matplotlib 按脚本 `--help` 重画，数值来自明确的数组/持有者模型，无性能推断。机制图和框架图继续由协作者制作。
