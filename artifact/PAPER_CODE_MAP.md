# Design-to-code map

| Design element | Primary source locations |
|---|---|
| Opt-in dispatch and strict failure mode | `Easy-Graph/easygraph/utils/gpu_runtime.py`; dispatch helpers under `Easy-Graph/easygraph/functions/` |
| Python graph state and output conversion | `Easy-Graph/easygraph/utils/gpu_eggpu_backend.py` |
| Mutation generation and cache invalidation | `Easy-Graph/easygraph/classes/graph.py`, `directed_graph.py`, `multigraph.py`, `directed_multigraph.py`, and `operation.py` |
| Immutable large-graph CSR input | `Easy-Graph/easygraph/classes/eggpu_bulk_graph.py` |
| Native graph conversion and bindings | `Easy-Graph/cpp_easygraph/classes/graph_convert.cpp`; `Easy-Graph/cpp_easygraph/cpp_easygraph.cpp` |
| Device CSR registry and bounded reuse | `Easy-Graph/gpu_easygraph/common/device_graph_cache.h`; `device_graph_cache.cu`; `buffer_cache.h` |
| Adaptive host/device policies | `Easy-Graph/easygraph/utils/gpu_adaptive_policy.py`; `Easy-Graph/gpu_easygraph/common/adaptive_transfer_policy.*` |
| PageRank, BC, and Closeness | `Easy-Graph/gpu_easygraph/functions/centrality/` |
| LCC | `Easy-Graph/gpu_easygraph/functions/basic/cluster.cu` |
| WCC and SCC | `Easy-Graph/gpu_easygraph/functions/components/connected.cu` |
| MST, BFS, Dijkstra, Bellman-Ford, and SSSP | `Easy-Graph/gpu_easygraph/functions/path/` |
| KCore | `Easy-Graph/gpu_easygraph/functions/core/k_core.cu` |
| Structural-hole statistics and metrics | `Easy-Graph/gpu_easygraph/functions/structural_holes/`; `ego_edge_statistics.cu` |
| Dense and selective result construction | dense binding functions in `Easy-Graph/cpp_easygraph/functions/`; mapping/view classes in `gpu_eggpu_backend.py` |

The C++ entry points are registered in
`Easy-Graph/cpp_easygraph/cpp_easygraph.cpp`. The corresponding public API
wrappers remain under `Easy-Graph/easygraph/functions/`, which is the boundary
used by applications and by the correctness smoke.
