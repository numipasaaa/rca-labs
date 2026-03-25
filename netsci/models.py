"""Network model helpers for the Network Science Lab Course.

Thin wrappers around NetworkX graph generators with sensible defaults
and a fixed ``SEED`` for reproducibility.

Functions
---------
erdos_renyi(n, avg_degree, seed=SEED)
    Erdos-Renyi random graph with a target average degree.
watts_strogatz(n, k, p, seed=SEED)
    Connected Watts-Strogatz small-world graph.
barabasi_albert(n, m, seed=SEED)
    Barabasi-Albert preferential-attachment graph.
kleinberg_grid(n, r, p, q, seed=SEED)
    Kleinberg navigable small-world graph on a 2-D grid.
greedy_route(G, source, target, pos)
    Greedy geographic routing on a grid graph.
"""

import networkx as nx

from netsci.utils import SEED


def erdos_renyi(n, avg_degree, seed=SEED):
    """Generate an Erdos-Renyi random graph G(n, p).

    Parameters
    ----------
    n : int
        Number of nodes.
    avg_degree : float
        Target average degree.  The edge probability is computed as
        ``p = avg_degree / (n - 1)``.
    seed : int, default ``SEED``
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
    """
    p = avg_degree / (n - 1)
    return nx.erdos_renyi_graph(n, p, seed=seed)


def watts_strogatz(n, k, p, seed=SEED):
    """Generate a connected Watts-Strogatz small-world graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is joined with its ``k`` nearest neighbors in a ring.
    p : float
        Rewiring probability.
    seed : int, default ``SEED``
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
    """
    return nx.connected_watts_strogatz_graph(n, k, p, seed=seed)


def barabasi_albert(n, m, seed=SEED):
    """Generate a Barabasi-Albert preferential-attachment graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges to attach from a new node to existing nodes.
    seed : int, default ``SEED``
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def holme_kim(n, m, p_triangle=0.5, seed=SEED):
    """Generate a Holme-Kim power-law cluster graph.

    Extends the Barabási-Albert model by adding a "triad formation" step:
    after each preferential-attachment edge, with probability *p_triangle*
    a second edge is added to a neighbor of the target, closing a triangle.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges to attach from a new node to existing nodes.
    p_triangle : float, default 0.5
        Probability of adding a triangle after each PA edge.
    seed : int, default ``SEED``
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
    """
    return nx.powerlaw_cluster_graph(n, m, p_triangle, seed=seed)


def kleinberg_grid(n, r=2, p=1, q=1, seed=SEED):
    """Generate a Kleinberg navigable small-world graph on a 2-D grid.

    Parameters
    ----------
    n : int
        Grid side length (total nodes = n * n).
    r : int, default 2
        Clustering exponent for long-range links.  ``P(link) ~ d^(-r)``.
        r=0 gives uniform random links, r=2 is optimal for greedy routing.
    p : int, default 1
        Local link radius — each node connects to all grid neighbours
        within Manhattan distance *p*.
    q : int, default 1
        Number of long-range links per node.
    seed : int, default ``SEED``
        Random seed for reproducibility.

    Returns
    -------
    (nx.Graph, dict)
        ``G`` — the navigable small-world graph (converted to undirected).
        ``pos`` — mapping ``{node: (row, col)}`` for grid layout.
    """
    G_directed = nx.navigable_small_world_graph(n, p=p, q=q, r=r, seed=seed)
    G = G_directed.to_undirected()
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    return G, pos


def greedy_route(G, source, target, pos):
    """Greedy routing: at each step, forward to the neighbor closest to target.

    Uses Manhattan distance on node-tuple coordinates for the distance
    metric.  The *pos* parameter is accepted for API consistency but the
    distance is computed directly from the node tuples.

    Parameters
    ----------
    G : networkx.Graph
        A grid-based graph whose nodes are ``(row, col)`` tuples.
    source : tuple
        Starting node.
    target : tuple
        Destination node.
    pos : dict
        Node position mapping (unused — distance computed from tuples).

    Returns
    -------
    list or None
        The sequence of visited nodes, or ``None`` if the route gets
        stuck.
    """
    path = [source]
    current = source
    visited = {source}
    for _ in range(G.number_of_nodes()):
        if current == target:
            return path
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            return None  # stuck
        # Pick neighbor closest to target (Manhattan distance)
        best = min(neighbors, key=lambda n: abs(n[0]-target[0]) + abs(n[1]-target[1]))
        visited.add(best)
        path.append(best)
        current = best
    return None  # didn't reach target
