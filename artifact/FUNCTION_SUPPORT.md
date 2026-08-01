# Function support

The table maps the paper's 16 functions to the public EasyGraph call that
enters EGGPU. Multigraph inputs are not supported by these accelerated paths.

| Family | Paper name | Public call | Main input boundary |
|---|---|---|---|
| Centrality | PageRank | `eg.pagerank` | directed or undirected; optional weights |
| Centrality | BC | `eg.betweenness_centrality` | optional source subset and weights |
| Centrality | Closeness | `eg.closeness_centrality` | optional source subset and weights |
| Connectivity | LCC | `eg.clustering` | unweighted undirected graph; a bulk directed input requires an explicit undirected projection |
| Connectivity | WCC | `eg.weakly_connected_components` | directed graph |
| Connectivity | SCC | `eg.strongly_connected_components` | directed graph |
| Connectivity | KCore | `eg.k_core` | graph core-number vector |
| Path and spanning | MST | `eg.minimum_spanning_tree` | undirected weighted graph/forest |
| Path and spanning | BFS | `eg.multi_source_bfs` | unweighted source set |
| Path and spanning | Dijkstra | `eg.single_source_dijkstra` | one source, nonnegative weights |
| Path and spanning | BellmanFord | `eg.multi_source_bellman_ford` | weighted source set |
| Path and spanning | SSSP | `eg.multi_source_dijkstra` | weighted source set |
| Structural holes | EffectiveSize | `eg.effective_size` | optional node subset and weights |
| Structural holes | Efficiency | `eg.efficiency` | optional node subset and weights |
| Structural holes | Constraint | `eg.constraint` | optional node subset and weights |
| Structural holes | Hierarchy | `eg.hierarchy` | optional node subset and weights |

The path family also exposes single-source and multi-source aliases in
`easygraph.functions.path.path`. GPU dense results may implement the Python
`Mapping` protocol without being a mutable `dict`; `dict(result)` creates an
independent dictionary when needed.
