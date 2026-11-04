# GPU 维护入口

此目录对应上游 `pybind11` 分支 `a81b5dffa16b1cd2c6e05e1cb45fc8e4058d28f3` 的增量。整合保留当前上游 CPU 接口，并将单卡状态复用、分区多卡执行和公开函数调度放入同一包。普通 `import easygraph` 不启动 CUDA。高级 GPU 会话要求 Python ≥ 3.10；CPU 包原有 Python 版本范围保持不变。本轮实际验证环境为 Python 3.10。

## 从哪里维护

| 入口 | 作用 |
|---|---|
| `easygraph.gpu.Session` | 统一的版本请求、完整输出、生命周期和能力查询接口 |
| `easygraph/gpu/_native.py`、`_dynamic.py` | 单卡/多卡共用的分区快照执行、固定拓扑权重提交与保留策略 |
| `easygraph/gpu/_arena/` | 单卡有界 arena、依赖版本、按需准备与属性更新；内部历史目录名保留算法来源 |
| `easygraph/egmgpu/` | 原 16 个公开函数的显式路由、请求调度及兼容层 |
| `easygraph/utils/gpu_eggpu_backend.py` | 普通 EasyGraph 函数的单卡快速路径 |
| `gpu_easygraph/`、`cpp_easygraph/` | CUDA 内核、runtime 和 Python 原生绑定 |
| `scripts/gpu/build.py` | 外部目录构建和仅动态加载检查，不执行 GPU 算法 |
| `scripts/gpu/qualify.py` | 新机器上的小图完整输出资格入口，非性能实验 |
| `tests/gpu/` | 无 GPU 的状态、语义、接口和移植回归 |

不要把后端目录再复制为独立 EasyGraph 包；新增规则应通过此公共入口及独立后端实现进入维护树。

## 构建

选择目标环境的 Python、CMake、编译器和 CUDA 工具链。构建目录放在源码树外，所有二进制均在目标环境重新生成。

```bash
python scripts/gpu/build.py --build-dir ../build-cpu --jobs 1
python scripts/gpu/build.py --build-dir ../build-gpu --cuda --arena \
  --cuda-root "$CUDA_ROOT" --architecture 80 --jobs 1 \
  --nccl-include "$NCCL_INCLUDE_DIR" --nccl-library "$NCCL_LIBRARY"
```

`--architecture` 是目标 GPU 的 CUDA compute capability，例如 A100 为 `80`。可重复传入 `--cmake-arg=-DNAME=value`；构建脚本不下载或改装软件。NCCL 为可选通信构建依赖，传入时必须同时指定 include 和 library。构建记录包含命令、源文件散列、库散列和 `RTLD_NOW` 动态加载结果；动态加载检查不调用 CUDA 算子。

如果环境编译器的 C++ runtime 与系统版本不同，应只对当前命令设置该环境的 `LD_LIBRARY_PATH`，并以相同设置执行导入和测试。源码不写入本机 Conda 路径。将实际生成的扩展目录加入 `PYTHONPATH`，arena 使用 `EGGPU_ARENA_LIBRARY_DIR` 指向新构建库目录。

## 先在 CPU 准备，再在空闲 GPU 资格验证

```bash
python scripts/gpu/qualify.py --output ../qualify-inputs-p2 --devices 0,1 --prepare-only
python scripts/gpu/qualify.py --output ../qualify-native-p1 --devices 0 \
  --native-dso ../build-gpu/native/cpp_easygraph/cpp_easygraph.<python-abi>.so
python scripts/gpu/qualify.py --output ../qualify-native-p2 --devices 0,1 \
  --native-dso ../build-gpu/native/cpp_easygraph/cpp_easygraph.<python-abi>.so
python scripts/gpu/qualify.py --output ../qualify-arena-p1 --backend arena --devices 0 \
  --native-dso ../build-gpu/native/cpp_easygraph/cpp_easygraph.<python-abi>.so \
  --arena-library-dir ../build-gpu/arena
```

将 `<python-abi>` 替换为实际文件名。每次指定一个新的输出目录；设备号由机器现场选择。执行前检查所选设备的利用率与可用显存，随后用 GPU UUID 固定映射。脚本会保存设备身份、源文件与扩展散列、完整参考和实际向量、数值误差及会话关闭后的输出存活检查。默认要求 GPU 利用率不超过 10%、可用显存至少 2048 MiB。单卡 arena 只接受一个设备；native 接受 1/2/4/8 个设备，但参数可接受只代表执行合同，实际资格须由目标机运行建立。

native 检查 BFS、PR、SSSP、WCC、WPR，固定拓扑权重提交、旧快照保留及复访；arena 检查 BFS、SSSP、WPR、更新、过期请求拒绝及复访。PR/WPR 的参考误差门槛根据后端数值合同分别设置并写入收据。全部距离逐值比较，未达节点必须对应 `+inf`。此入口不将历史 GPU 性能记录迁移为新二进制的通过证明，也不为宽函数路由自动生成资格文件。

## 请求和生命周期

```python
from easygraph.gpu import Session

with Session.native({"graph": "partition-manifest.json"}, devices=(0, 1)) as session:
    request = session.request("graph", "bfs", {"source": 0})
    result = session.execute(request)
    capabilities = session.capabilities().to_dict()
    inventory = session.inventory()
# result.values 是独立持有、只读、完整的 host NumPy 输出。
```

一个原子请求声明读取的图状态、算法私有可写状态以及完成边界。请求捕获语义版本；执行前后都检查同一版本。单卡图已提交更新但尚未在 GPU 物化时，合法的准备动作不会改变语义身份。真实提交后，旧请求会被拒绝，必须重新创建请求。

原子请求不是通用算子融合器：BFS/SSSP 的 frontier、PR 的 rank、WCC 的 labels 保持各自独立。backend 负责 workspace 和完成同步，返回时完整结果已就绪。后续机制应通过状态依赖和完成合同复用数据，不能仅因函数名相近共享可写数值状态。

## 两种资源合同

arena 的预算覆盖一块预留 CUDA arena 中的图组件和 workspace，其 inventory 核对唯一 extent 与账本。native 的预算约束完成调用边界上可保留的分区状态；算法运行期工作区、通信缓冲和 allocator backing 另列，不能把这个数解释为全进程显存硬上限。两个后端共享描述和生命周期合同，各自保有实际内存所有权；当前没有跨后端共享同一 CUDA allocation。

普通公开函数调用与高级 Session 是两个明确入口。前者保持原有字典、图或列表返回形式；后者返回带节点顺序和数值合同的完整数组。调用者选择需要的接口，不依赖静默回退。

宽函数的独立资格材料与注册方法见 [WIDE_FUNCTION_QUALIFICATION_CN.md](WIDE_FUNCTION_QUALIFICATION_CN.md)。

## 当前验证边界

本维护版完成 CPU 回归、CPU 原生构建、CUDA/NCCL 主体及 arena 库编译和仅动态加载检查。GPU 实际运行需在获得空闲设备后执行上述资格命令。已有研究性能属于原实验快照，新维护版获得同等性能必须由新运行确认。宽函数层的源代码与接口保留，不等于所有模式在新构建上已重新完成资格。

上游 Python PR 与 C++ PR 在无向自环上的历史处理存在数值差别；此整合修复显式参数适配和 Python 迭代路径，不改变该既有 C++ 自环规则。涉及跨系统 PR 数值比较时，应使用声明的方向、自环、权重及收敛合同。
