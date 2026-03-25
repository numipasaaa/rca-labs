"""Visualization functions for Lab 03 — Random Networks & Small Worlds.

Functions
---------
plot_shortest_path, plot_er_phase_panels, plot_er_phase_sweep,
plot_ws_rewiring_demo, plot_ws_ring_comparison, plot_ws_sweep,
plot_ws_k_sweep, plot_kleinberg_panels, plot_greedy_steps,
plot_greedy_paths, plot_kleinberg_sweep
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from netsci.utils import SEED
from netsci.viz._common import STYLE


# ---------------------------------------------------------------------------
# Shortest path visualization
# ---------------------------------------------------------------------------


def plot_shortest_path(G, source, target, pos=None, title=None):
    """Highlight the shortest path between two nodes on the graph.

    Parameters
    ----------
    G : networkx.Graph
    source, target : node
    pos : dict, optional
    title : str, optional
    """
    path = nx.shortest_path(G, source, target)
    path_edges = list(zip(path[:-1], path[1:]))

    if pos is None:
        pos = nx.spring_layout(G, seed=SEED)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Background edges (lowest layer)
        edges_bg = nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#dddddd", width=0.3, alpha=0.8,
        )
        if edges_bg is not None:
            edges_bg.set_zorder(1)

        # Base nodes
        nodes_bg = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=10, node_color="#4878CF", alpha=0.5,
        )
        nodes_bg.set_zorder(2)

        # Path edges on top of blue nodes
        edges_path = nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=path_edges, edge_color="#D65F5F", width=3.0,
        )
        if edges_path is not None:
            edges_path.set_zorder(3)

        # Path nodes on top of everything
        nodes_path = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=path, node_size=80, node_color="#D65F5F",
        )
        nodes_path.set_zorder(4)

        labels_art = nx.draw_networkx_labels(
            G,
            pos,
            {source: source, target: target},
            ax=ax,
            font_size=9,
            font_color="black",
            font_weight="bold",
        )
        for t in labels_art.values():
            t.set_zorder(5)

        ax.set_title(
            title
            or f"Only {len(path) - 1} hops between {source} and {target} "
            f"in {G.number_of_nodes()} nodes",
            fontsize=13,
        )
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# ER phase transition panels
# ---------------------------------------------------------------------------


def plot_er_phase_panels(n=100, phase_params=None, seed=SEED):
    """Three-panel ER phase transition at different average degrees.

    Parameters
    ----------
    n : int
    phase_params : list of (float, str), optional
        ``(avg_degree, label)`` tuples.  Defaults to 0.5, 1.0, 3.0.
    seed : int
    """
    if phase_params is None:
        phase_params = [
            (0.5, "Below threshold\navg degree ⟨k⟩ = 0.5"),
            (1.0, "At threshold\navg degree ⟨k⟩ = 1.0"),
            (3.0, "Above threshold\navg degree ⟨k⟩ = 3.0"),
        ]

    n_panels = len(phase_params)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (avg_d, label) in zip(axes, phase_params):
        p = avg_d / (n - 1)
        G_phase = nx.erdos_renyi_graph(n, p, seed=seed)
        pos = nx.spring_layout(G_phase, seed=seed, k=0.5)
        components = sorted(nx.connected_components(G_phase), key=len, reverse=True)
        giant = components[0]
        colors = ["#D65F5F" if node in giant else "#CCCCCC" for node in G_phase.nodes()]
        sizes = [40 if node in giant else 20 for node in G_phase.nodes()]
        nx.draw_networkx_nodes(G_phase, pos, ax=ax, node_color=colors, node_size=sizes)
        nx.draw_networkx_edges(
            G_phase, pos, ax=ax, edge_color="#999999", width=0.5, alpha=0.5
        )
        gc_frac = len(giant) / G_phase.number_of_nodes()
        ax.set_title(f"{label}\nLargest component: {gc_frac:.0%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "ER Phase Transition: Largest Component Emergence (red = largest component)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ER phase sweep plot
# ---------------------------------------------------------------------------


def plot_er_phase_sweep(avg_degrees, gcc_sizes):
    """Largest component fraction vs average degree with threshold line.

    Parameters
    ----------
    avg_degrees : array-like
    gcc_sizes : array-like
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(avg_degrees, gcc_sizes, "o-", markersize=5)
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="⟨k⟩ = 1 (theoretical threshold)")
        ax.set_xlabel("Average degree")
        ax.set_ylabel("Fraction in largest component")
        ax.set_title("ER Phase Transition")
        ax.legend()
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Watts-Strogatz rewiring demo
# ---------------------------------------------------------------------------


def plot_ws_rewiring_demo(n=12, k=4, p_values=None, seed=SEED):
    """Show WS rewiring at different p with original vs rewired edges.

    Parameters
    ----------
    n : int
    k : int
    p_values : list of float, optional
        Defaults to ``[0, 0.1, 1.0]``.
    seed : int
    """
    if p_values is None:
        p_values = [0, 0.1, 1.0]

    default_labels = {
        0: "p = 0\n(regular lattice)",
        0.1: "p = 0.1\n(small world)",
        1.0: "p = 1.0\n(random)",
    }

    n_panels = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    G_base = nx.watts_strogatz_graph(n, k, 0, seed=seed)
    base_edges = set(frozenset(e) for e in G_base.edges())

    for ax, p in zip(axes, p_values):
        G_demo = nx.watts_strogatz_graph(n, k, p, seed=seed)
        pos = nx.circular_layout(G_demo)

        regular = [(u, v) for u, v in G_demo.edges() if frozenset((u, v)) in base_edges]
        rewired = [
            (u, v) for u, v in G_demo.edges() if frozenset((u, v)) not in base_edges
        ]

        nx.draw_networkx_edges(
            G_demo, pos, edgelist=regular, ax=ax, edge_color="#999999", width=1.0
        )
        nx.draw_networkx_edges(
            G_demo,
            pos,
            edgelist=rewired,
            ax=ax,
            edge_color="#D65F5F",
            width=2.0,
            style="dashed",
        )
        nx.draw_networkx_nodes(G_demo, pos, ax=ax, node_color="#4878CF", node_size=120)
        label = default_labels.get(p, f"p = {p}")
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Watts-Strogatz Rewiring: gray = original, red dashed = rewired", fontsize=12
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# WS circular layout comparison
# ---------------------------------------------------------------------------


def plot_ws_ring_comparison(p_values, n=30, k=4, seed=SEED):
    """Circular-layout WS graphs at various rewiring probabilities.

    Parameters
    ----------
    p_values : list of float
    n, k : int
    seed : int
    """
    n_panels = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, p in zip(axes, p_values):
        G_ws = nx.watts_strogatz_graph(n, k, p, seed=seed)
        pos = nx.circular_layout(G_ws)
        nx.draw_networkx(
            G_ws,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=40,
            edge_color="#cccccc",
            width=0.5,
            with_labels=False,
            alpha=0.9,
        )
        C = nx.average_clustering(G_ws)
        ax.set_title(f"p = {p}\nC = {C:.2f}")
        ax.axis("off")

    fig.suptitle("Watts-Strogatz: from lattice (p=0) to random (p=1)", fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# WS sweep: C(p)/C(0) and L(p)/L(0) vs p
# ---------------------------------------------------------------------------


def plot_ws_sweep(p_sweep, C_list, L_list, title=None):
    """Plot normalized clustering and path length vs rewiring probability.

    Parameters
    ----------
    p_sweep : array-like
        Rewiring probabilities (log-spaced recommended).
    C_list, L_list : array-like
        Normalized values C(p)/C(0) and L(p)/L(0).
    title : str, optional
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(p_sweep, C_list, "o-", label="C(p) / C(0)", markersize=5)
        ax.plot(p_sweep, L_list, "s-", label="L(p) / L(0)", markersize=5)
        ax.set_xscale("log")
        ax.set_xlabel("Rewiring probability p")
        ax.set_ylabel("Normalized value")
        ax.set_title(title or "The Small-World Sweet Spot")
        ax.legend()
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        plt.show()


def plot_ws_k_sweep(k_values, p_sweep, gap_curves, areas):
    """Plot sweet-spot gap curves and area vs k for a Watts-Strogatz k sweep.

    Parameters
    ----------
    k_values : array-like
        Neighborhood sizes swept.
    p_sweep : array-like
        Rewiring probabilities used for each curve.
    gap_curves : dict[int, array-like]
        Mapping k -> C(p)/C(0) - L(p)/L(0) values.
    areas : dict[int, float]
        Mapping k -> integrated sweet-spot area.
    """
    log_p = np.log10(p_sweep)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for k_val in k_values:
            axes[0].plot(log_p, gap_curves[k_val], label=f"k={k_val}")
        axes[0].set_xlabel("log\u2081\u2080(p)")
        axes[0].set_ylabel("C(p)/C(0) \u2212 L(p)/L(0)")
        axes[0].set_title("Sweet spot gap at each k")
        axes[0].legend(fontsize=8)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)

        ks = list(k_values)
        axes[1].plot(ks, [areas[k] for k in ks], "o-", color="steelblue", linewidth=2)
        axes[1].set_xlabel("k (neighborhood size)")
        axes[1].set_ylabel("Sweet spot area")
        axes[1].set_title("Sweet spot area vs k")

        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Kleinberg grid panels
# ---------------------------------------------------------------------------


def plot_kleinberg_panels(grids, r_values, r_labels=None):
    """Show Kleinberg grids at different clustering exponents.

    Parameters
    ----------
    grids : list of (networkx.Graph, dict)
        ``(G, pos)`` tuples from ``models.kleinberg_grid``.
    r_values : list of float
    r_labels : list of str, optional
    """
    if r_labels is None:
        r_labels = [f"r = {r}" for r in r_values]

    n_panels = len(grids)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (G_k, pos_k), r_val, label in zip(axes, grids, r_values, r_labels):
        local_edges = []
        long_edges = []
        for u, v in G_k.edges():
            dist = abs(u[0] - v[0]) + abs(u[1] - v[1])
            if dist <= 1:
                local_edges.append((u, v))
            else:
                long_edges.append((u, v))

        nx.draw_networkx_edges(
            G_k, pos_k, edgelist=local_edges, ax=ax, edge_color="#CCCCCC", width=0.5
        )
        nx.draw_networkx_edges(
            G_k,
            pos_k,
            edgelist=long_edges,
            ax=ax,
            edge_color="#D65F5F",
            width=1.5,
            alpha=0.7,
        )
        nx.draw_networkx_nodes(G_k, pos_k, ax=ax, node_size=30, node_color="#4878CF")
        ax.set_title(f"{label}\n{len(long_edges)} long-range links", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Kleinberg Model: Local Grid + Long-Range Links (red)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Greedy routing step-by-step
# ---------------------------------------------------------------------------


def plot_greedy_steps(G_step, pos_step, path_step, decisions, target_step, n_panels=4):
    """Step-by-step greedy routing visualization.

    Parameters
    ----------
    G_step : networkx.Graph
    pos_step : dict
    path_step : list
        The greedy routing path.
    decisions : list of (current, chosen, neighbor_dists)
    target_step : node
    n_panels : int
    """
    n_panels = min(n_panels, len(decisions))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        current_node = decisions[idx][0]
        chosen = decisions[idx][1]
        neighbor_dists = decisions[idx][2]

        nx.draw_networkx_edges(G_step, pos_step, ax=ax, edge_color="#EEEEEE", width=0.3)
        nx.draw_networkx_nodes(
            G_step, pos_step, ax=ax, node_size=15, node_color="#DDDDDD"
        )

        partial_path = path_step[: idx + 1]
        if len(partial_path) > 1:
            p_edges = list(zip(partial_path[:-1], partial_path[1:]))
            nx.draw_networkx_edges(
                G_step,
                pos_step,
                edgelist=p_edges,
                ax=ax,
                edge_color="#999999",
                width=2.0,
            )
            nx.draw_networkx_nodes(
                G_step,
                pos_step,
                nodelist=partial_path[:-1],
                ax=ax,
                node_size=30,
                node_color="#999999",
            )

        nx.draw_networkx_nodes(
            G_step,
            pos_step,
            nodelist=[current_node],
            ax=ax,
            node_size=100,
            node_color="#D65F5F",
            edgecolors="black",
            linewidths=1.5,
        )

        candidates = list(neighbor_dists.keys())
        cand_colors = ["#6ACC65" if n == chosen else "#FFD700" for n in candidates]
        nx.draw_networkx_nodes(
            G_step,
            pos_step,
            nodelist=candidates,
            ax=ax,
            node_size=60,
            node_color=cand_colors,
            edgecolors="black",
            linewidths=0.8,
        )
        for n, d in neighbor_dists.items():
            ax.annotate(
                f"d={d}",
                xy=pos_step[n],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="#333333",
            )

        nx.draw_networkx_nodes(
            G_step,
            pos_step,
            nodelist=[target_step],
            ax=ax,
            node_size=100,
            node_color="#4878CF",
            edgecolors="black",
            linewidths=1.5,
        )

        ax.annotate(
            "",
            xy=pos_step[chosen],
            xytext=pos_step[current_node],
            arrowprops=dict(arrowstyle="-|>", color="#D65F5F", lw=2),
        )

        ax.set_title(
            f"Step {idx + 1}: at {current_node}\n"
            f"Choose {chosen} (d={neighbor_dists[chosen]})",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(
        "Greedy Routing Step-by-Step: red=current, green=chosen, blue=target",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Greedy routing path comparison
# ---------------------------------------------------------------------------


def plot_greedy_paths(grids, paths, r_labels=None):
    """Compare greedy routing paths at different exponents.

    Parameters
    ----------
    grids : list of (networkx.Graph, dict)
    paths : list of (list_or_None, source, target)
    r_labels : list of str, optional
    """
    if r_labels is None:
        r_labels = [f"Panel {i}" for i in range(len(grids))]

    n_panels = len(grids)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (G_k, pos_k), (path, source, target), label in zip(
        axes, grids, paths, r_labels
    ):
        nx.draw_networkx_edges(G_k, pos_k, ax=ax, edge_color="#EEEEEE", width=0.3)
        nx.draw_networkx_nodes(G_k, pos_k, ax=ax, node_size=20, node_color="#CCCCCC")

        if path is not None:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(
                G_k, pos_k, edgelist=path_edges, ax=ax, edge_color="#D65F5F", width=2.5
            )
            nx.draw_networkx_nodes(
                G_k, pos_k, nodelist=path, ax=ax, node_size=40, node_color="#D65F5F"
            )
            nx.draw_networkx_nodes(
                G_k,
                pos_k,
                nodelist=[source, target],
                ax=ax,
                node_size=100,
                node_color=["#6ACC65", "#4878CF"],
            )
            ax.set_title(f"{label}\nGreedy path: {len(path) - 1} hops", fontsize=11)
        else:
            ax.set_title(f"{label}\nGreedy routing STUCK", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Greedy Routing: green = source, blue = target, red = path",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Kleinberg sweep plot
# ---------------------------------------------------------------------------


def plot_kleinberg_sweep(r_sweep, mean_lengths):
    """Mean greedy path length vs clustering exponent r.

    Parameters
    ----------
    r_sweep : array-like
    mean_lengths : array-like
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(r_sweep, mean_lengths, "o-", markersize=6)
        ax.set_ylabel("Mean greedy path length")
        ax.set_title("Kleinberg Sweep: Optimal Routing at r = grid dimension")
        ax.axvline(
            2.0, color="red", linestyle="--", alpha=0.5, label="r = 2 (grid dimension)"
        )
        ax.set_xlabel("Clustering exponent r")
        ax.legend()
        fig.tight_layout()
        plt.show()
