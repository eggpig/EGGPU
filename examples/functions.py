"""Call the graph-analysis functions through the EasyGraph interface."""
import easygraph as eg


def main():
    graph = eg.Graph()
    graph.add_nodes_from(range(5))
    for u, v, w in [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 0, 3), (0, 2, 4)]:
        graph.add_edge(u, v, weight=w)
    directed = eg.DiGraph()
    directed.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges:
        directed.add_edge(u, v, **data)
        directed.add_edge(v, u, **data)
    calls = {
        "PageRank": lambda: dict(eg.pagerank(graph, max_iter=1000, tol=1e-8)),
        "Betweenness": lambda: eg.betweenness_centrality(graph),
        "Closeness": lambda: eg.closeness_centrality(graph),
        "LCC": lambda: dict(eg.clustering(graph)),
        "WCC": lambda: list(eg.connected_components(graph)),
        "SCC": lambda: list(eg.strongly_connected_components(directed)),
        "K-Core": lambda: eg.k_core(graph),
        "BFS": lambda: eg.bfs(graph, 0),
        "Dijkstra": lambda: eg.single_source_dijkstra(graph, 0),
        "Bellman-Ford": lambda: eg.multi_source_bellman_ford(graph, [0]),
        "SSSP": lambda: eg.multi_source_dijkstra(graph, [0]),
        "MST": lambda: list(eg.minimum_spanning_tree(graph).edges),
        "Effective size": lambda: dict(eg.effective_size(graph)),
        "Efficiency": lambda: dict(eg.efficiency(graph)),
        "Constraint": lambda: dict(eg.constraint(graph)),
        "Hierarchy": lambda: dict(eg.hierarchy(graph)),
    }
    for name, function in calls.items():
        print(f"{name}: {function()}")


if __name__ == "__main__":
    main()
