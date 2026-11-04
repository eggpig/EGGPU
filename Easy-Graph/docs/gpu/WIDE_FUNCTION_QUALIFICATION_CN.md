# 宽函数路由的目标机资格与注册

`easygraph.egmgpu` 保留 16 个公开函数的 SINGLE / REQUEST_PARALLEL 适配、6 个 REPLICATED 适配、4 个 PARTITIONED 适配。`scripts/gpu/qualify.py` 验证的是高级 Session 的小图执行；它不会生成这些公开路由所需的 v3 index，也不会解锁 REPLICATED 或 AUTO。

新机器需生成绑定当前扩展 SHA-256、真实设备组合与拓扑的原始记录。此维护版提供严格加载和注册器，尚未整理一个覆盖全部宽函数的 probe 生成 CLI。首次迁移以核心 Session 资格为起点；宽函数回归单独运行并按以下合同保留材料。现有默认 index 路径是旧部署的占位入口，应始终显式传入本机新证据路径。

## 需要实测哪些材料

| 路由 | 原生入口/记录 | 完整性要求 |
|---|---|---|
| 16 个 SINGLE / REQUEST_PARALLEL | `cpp_egmgpu_single_execute`；`egmgpu.single-request-native-probe/v3` | 每个函数的原始结果与 CPU 参考；至少两个设备的单卡和请求分派；混合压力的实际分派、完成次数及 public API 检查 |
| 6 个 REPLICATED | `default_replicated_native_specs()` 中各 binding；按函数独立 raw-result | BC、Closeness、EffectiveSize、Efficiency、Constraint、Hierarchy 的完整参考、单卡结果、多卡重复结果及逐项误差；真实参与设备和调用计数 |
| 4 个 PARTITIONED | `default_partitioned_native_specs()`；`load_partitioned_binding_evidence` 接受的 index | BFS、PR、SSSP、WCC，各算法规定 case、方向/权重合同、后端覆盖、raw result 和来源文件散列；BFS frontier 与 SSSP policy 若启用，需独立匹配校准 |
| 部署与策略 | hardware / transport 原始 JSON | 设备 UUID/逻辑号/显存、实际 host 和 peer 传输测量、部署拓扑指纹；计划策略还需要同一机器上的费用校准 |

所有参考均应保存完整向量、分量或森林等实际输出。不能仅写 `passed: true`，也不能将未知设备数的资格扩大为 4/8 卡。加载器会重新验证记录、散列、数值 witness、完成次数和组成员。具体 schema 字段以 `single_bindings.py`、`native_bindings.py`、`partitioned_bindings.py` 的加载函数为准；只建立 JSON 外形不会获得真实资格。

## 显式注册示例

下列示例在已经完成相应实测、生成证据文件后使用。路径都是目标机实验产物；注册不会替调用者开展 GPU probe。

```python
from pathlib import Path
import json
from easygraph.egmgpu.single_bindings import (
    default_single_public_functions, register_probed_single_bindings)
from easygraph.egmgpu.native_bindings import register_probed_replicated_bindings
from easygraph.egmgpu.partitioned_bindings import register_probed_partitioned_bindings
from easygraph.egmgpu.deployment_planner import MeasuredDeployment

root = Path("qualification-wide")
hardware = json.loads((root / "hardware.json").read_text())
transport = json.loads((root / "transport.json").read_text())
deployment = MeasuredDeployment.from_measurements(hardware, transport, devices=(0, 1))
topology = deployment.topology_fingerprint
functions = default_single_public_functions()

register_probed_single_bindings(
    functions, root / "single-request-v3.json", topology_fingerprint=topology)
replicated = ("bc", "closeness", "effective_size", "efficiency", "constraint", "hierarchy")
register_probed_replicated_bindings(
    {name: functions[name] for name in replicated},
    {name: root / (name + "-raw-result.json") for name in replicated},
    topology_fingerprint=topology)
register_probed_partitioned_bindings(
    {name: functions[name] for name in ("bfs", "pagerank", "sssp", "wcc")},
    evidence_index=root / "partitioned-index.json", topology_fingerprint=topology)
```

`MeasuredDeployment.from_measurements` 接收已解析的 mapping，不接收文件名。其设备列表和 host/peer 带宽必须覆盖选定逻辑设备，单位必须与记录字段一致。拓扑指纹用于核对记录身份；实际设备绑定仍应在运行前检查。

以上是显式路由注册，不能借此把自定义回调升级为生产 AUTO。完整生产 bundle 的入口为 `bootstrap_production_bindings(single_evidence=..., replicated_evidence=..., partitioned_evidence_index=..., topology_fingerprint=...)`：它验证 16/6/4 完整矩阵后原子发布。AUTO 的放置选择另外需要实测 deployment 和适用费用校准。缺少某一部分证据时，可以继续使用已验证的核心 Session 或对应显式模式，并在能力表中记录实际通过范围。
