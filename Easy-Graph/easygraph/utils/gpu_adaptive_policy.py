import os
from dataclasses import dataclass


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().upper() in {"1", "TRUE", "ON", "YES", "AUTO"}


def env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return int(default)
    if value.strip().upper() == "AUTO":
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def policy_enabled():
    return env_bool("EASYGRAPH_GPU_ADAPTIVE_POLICY", True)


@dataclass(frozen=True)
class GraphStats:
    directed: bool
    node_count: int
    edge_slots: int
    max_degree: int
    nonzero_degree_nodes: int
    avg_degree: float


def build_graph_stats(G, adj_getter, stamp):
    directed, node_count, edge_slots, _ = stamp
    adj = adj_getter(G)
    max_degree = 0
    nonzero = 0
    for nbrs in adj.values():
        degree = len(nbrs)
        if degree > max_degree:
            max_degree = degree
        if degree > 0:
            nonzero += 1
    avg_degree = float(edge_slots) / float(node_count) if node_count else 0.0
    return GraphStats(
        directed=bool(directed),
        node_count=int(node_count),
        edge_slots=int(edge_slots),
        max_degree=int(max_degree),
        nonzero_degree_nodes=int(nonzero),
        avg_degree=avg_degree,
    )


def edge_slots(prepared):
    stats = prepared.get("graph_stats") if isinstance(prepared, dict) else None
    if isinstance(stats, GraphStats):
        return stats.edge_slots
    try:
        return int(prepared.get("stamp", (False, 0, 0, 0))[2])
    except Exception:
        return 0


def prefer_kcore_host(prepared):
    if not policy_enabled() or not env_bool("EASYGRAPH_GPU_ADAPTIVE_HOST", True):
        return False
    if not env_bool("EASYGRAPH_GPU_KCORE_HOST_ENABLE", False):
        return False
    stats = prepared.get("graph_stats") if isinstance(prepared, dict) else None
    slots = stats.edge_slots if isinstance(stats, GraphStats) else edge_slots(prepared)

    limit = os.environ.get("EASYGRAPH_GPU_KCORE_HOST_MAX_EDGE_SLOTS")
    if limit is not None and limit.strip().upper() != "AUTO":
        if slots <= env_int("EASYGRAPH_GPU_KCORE_HOST_MAX_EDGE_SLOTS", 0):
            return True
        return False

    if isinstance(stats, GraphStats):
        if (
            stats.node_count < env_int("EASYGRAPH_GPU_KCORE_HOST_MID_MAX_NODES", 500000)
            and slots >= env_int("EASYGRAPH_GPU_KCORE_HOST_MID_MIN_EDGE_SLOTS", 300000)
            and stats.max_degree >= env_int("EASYGRAPH_GPU_KCORE_HOST_MID_MIN_MAX_DEGREE", 2000)
        ):
            return True
        if stats.max_degree >= env_int("EASYGRAPH_GPU_KCORE_HOST_HUB_MAX_DEGREE", 50000):
            return True
        if (
            slots >= env_int("EASYGRAPH_GPU_KCORE_HOST_DENSE_MIN_EDGE_SLOTS", 500000)
            and stats.avg_degree >= float(os.environ.get("EASYGRAPH_GPU_KCORE_HOST_MIN_AVG_DEGREE", "8.0"))
        ):
            return True
    return False


def prefer_scc_host(prepared):
    if not policy_enabled() or not env_bool("EASYGRAPH_GPU_ADAPTIVE_HOST", True):
        return False
    if not env_bool("EASYGRAPH_GPU_SCC_HOST_ENABLE", False):
        return False
    stats = prepared.get("graph_stats") if isinstance(prepared, dict) else None
    slots = stats.edge_slots if isinstance(stats, GraphStats) else edge_slots(prepared)

    limit = os.environ.get("EASYGRAPH_GPU_SCC_HOST_MAX_EDGE_SLOTS")
    if limit is not None and limit.strip().upper() != "AUTO":
        if slots <= env_int("EASYGRAPH_GPU_SCC_HOST_MAX_EDGE_SLOTS", 0):
            return True
        return False

    if not isinstance(stats, GraphStats):
        return False
    return (
        stats.directed
        and stats.node_count <= env_int("EASYGRAPH_GPU_SCC_HOST_MAX_NODES", 600000)
        and slots >= env_int("EASYGRAPH_GPU_SCC_HOST_MIN_EDGE_SLOTS", 300000)
        and stats.max_degree >= env_int("EASYGRAPH_GPU_SCC_HOST_MIN_MAX_DEGREE", 2000)
        and stats.avg_degree >= float(os.environ.get("EASYGRAPH_GPU_SCC_HOST_MIN_AVG_DEGREE", "3.0"))
    )


def prefer_sssp_host(prepared):
    if not policy_enabled() or not env_bool("EASYGRAPH_GPU_ADAPTIVE_HOST", True):
        return False
    if not env_bool("EASYGRAPH_GPU_SSSP_HOST_ENABLE", False):
        return False
    stats = prepared.get("graph_stats") if isinstance(prepared, dict) else None
    if not isinstance(stats, GraphStats):
        return False
    if not stats.directed:
        return False
    if stats.node_count < env_int("EASYGRAPH_GPU_SSSP_HOST_MIN_NODES", 75000):
        return False
    if stats.edge_slots > env_int("EASYGRAPH_GPU_SSSP_HOST_MAX_EDGE_SLOTS", 2000000):
        return False
    return True


def prefer_dense_component_return(prepared, directed):
    if not policy_enabled():
        return False
    if not directed:
        return False
    if not env_bool("EASYGRAPH_GPU_COMPONENT_DENSE_RETURN", False):
        return False
    stats = prepared.get("graph_stats") if isinstance(prepared, dict) else None
    n = stats.node_count if isinstance(stats, GraphStats) else 0
    m = stats.edge_slots if isinstance(stats, GraphStats) else edge_slots(prepared)
    min_nodes = env_int("EASYGRAPH_GPU_COMPONENT_DENSE_MIN_NODES", 100000)
    min_edges = env_int("EASYGRAPH_GPU_COMPONENT_DENSE_MIN_EDGE_SLOTS", 500000)
    return n >= min_nodes or m >= min_edges
