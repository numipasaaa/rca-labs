"""Visualization functions for Lab 01 — Introduction to Networks.

Functions
---------
draw_graph_anatomy, draw_weighted_graph, plot_in_out_degree,
compare_layouts, draw_bipartite, draw_projection
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from netsci.utils import SEED
from netsci.viz._common import STYLE


# ---------------------------------------------------------------------------
# Annotated graph anatomy (tutorial helper)
# ---------------------------------------------------------------------------


def draw_graph_anatomy(G, seed=SEED):
    """Draw an annotated tutorial diagram labeling node, edge, and degree.

    Designed for a small (≤10 node) teaching graph with named nodes
    including 'Alice', 'Bob', and 'Carol'.
    """
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=seed)

        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#aaaaaa", width=2.0, alpha=0.5
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=600,
            edgecolors="white",
            linewidths=2.0,
            alpha=0.95,
        )

        label_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
        nx.draw_networkx_labels(
            G,
            label_pos,
            ax=ax,
            font_size=10,
            font_color="#2c3e50",
            font_weight="bold",
        )

        # Annotate "node" → first node, "edge" → midpoint, "degree" → second node
        nodes = list(G.nodes())
        n0, n1, n2 = nodes[0], nodes[1], nodes[2]

        ax.annotate(
            "node",
            xy=pos[n0],
            xytext=(pos[n0][0] - 0.35, pos[n0][1] + 0.30),
            fontsize=13,
            fontweight="bold",
            color="#e74c3c",
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2),
        )

        mid = ((pos[n1][0] + pos[n2][0]) / 2, (pos[n1][1] + pos[n2][1]) / 2)
        ax.annotate(
            "edge",
            xy=mid,
            xytext=(mid[0] + 0.35, mid[1] + 0.25),
            fontsize=13,
            fontweight="bold",
            color="#27ae60",
            arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2),
        )

        ax.annotate(
            f"degree = {G.degree(n1)}",
            xy=pos[n1],
            xytext=(pos[n1][0] + 0.38, pos[n1][1] - 0.28),
            fontsize=13,
            fontweight="bold",
            color="#8e44ad",
            arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=2),
        )

        ax.set_title(
            "Graph Anatomy: Nodes, Edges, and Degree",
            fontsize=15,
            fontweight="bold",
            pad=14,
        )
        ax.margins(0.15)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Weighted graph drawing
# ---------------------------------------------------------------------------


def draw_weighted_graph(
    G,
    weight_attr="weight",
    title=None,
    seed=SEED,
    top_labels=8,
    top_weight_edges=5,
    k=0.3,
):
    """Draw a node-link diagram with edge width/color proportional to weight.

    Labels are shown for top-degree nodes *and* nodes involved in the
    heaviest edges, so important connections are always visible.

    Parameters
    ----------
    G : networkx.Graph
    weight_attr : str, default "weight"
    title : str, optional
    seed : int
    top_labels : int
        Number of top-degree nodes to label.
    top_weight_edges : int
        Number of heaviest edges whose endpoints are also labeled.
    k : float
        Spring layout spacing parameter.
    """
    weights = [G[u][v][weight_attr] for u, v in G.edges()]
    max_w = max(weights)
    norm_widths = [0.3 + 4.0 * w / max_w for w in weights]
    edge_colors = [plt.cm.YlOrRd(0.2 + 0.7 * w / max_w) for w in weights]

    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 25 for n in G.nodes()]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7.5))
        pos = nx.spring_layout(G, seed=seed, k=k)

        nx.draw_networkx_edges(
            G, pos, ax=ax, width=norm_widths, edge_color=edge_colors, alpha=0.75
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color="#4878CF",
            alpha=0.90,
            edgecolors="white",
            linewidths=0.8,
        )

        # Label top-degree + top-weight-edge nodes
        top_by_degree = set(sorted(degrees, key=degrees.get, reverse=True)[:top_labels])
        top_edges = sorted(G.edges(data=weight_attr), key=lambda e: e[2], reverse=True)[
            :top_weight_edges
        ]
        top_by_weight = {u for u, v, w in top_edges} | {v for u, v, w in top_edges}
        label_nodes = top_by_degree | top_by_weight

        label_pos = {n: (pos[n][0], pos[n][1] + 0.04) for n in label_nodes}
        nx.draw_networkx_labels(
            G,
            label_pos,
            {n: n for n in label_nodes},
            ax=ax,
            font_size=9,
            font_weight="bold",
            font_color="#2c3e50",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )

        ax.set_title(title or "Weighted Graph", fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# In-degree / Out-degree comparison (directed graphs)
# ---------------------------------------------------------------------------


def plot_in_out_degree(G, title_prefix=""):
    """Plot side-by-side in-degree and out-degree histograms.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph.
    title_prefix : str
        Prefix for subplot titles (e.g. ``"Email"``).
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    prefix = f"{title_prefix} — " if title_prefix else ""

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        axes[0].hist(
            in_degrees,
            bins=50,
            edgecolor="white",
            alpha=0.85,
            color="#4878CF",
            linewidth=0.6,
        )
        axes[0].set_xlabel("In-degree", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title(
            f"{prefix}In-degree Distribution\n(How many emails received)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")

        axes[1].hist(
            out_degrees,
            bins=50,
            edgecolor="white",
            alpha=0.85,
            color="#D65F5F",
            linewidth=0.6,
        )
        axes[1].set_xlabel("Out-degree", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title(
            f"{prefix}Out-degree Distribution\n(How many emails sent)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")

        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Layout comparison (3-panel)
# ---------------------------------------------------------------------------


def compare_layouts(G, node_color=None, title=None):
    """Draw the same graph with spring, Kamada-Kawai, and circular layouts.

    Parameters
    ----------
    G : networkx.Graph
    node_color : color or list of colors, optional
    title : str, optional
        Overall suptitle.
    """
    n = G.number_of_nodes()
    _size = max(180, 300 - n * 3)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        layouts = {
            "Spring": nx.spring_layout(G, seed=SEED),
            "Kamada-Kawai": nx.kamada_kawai_layout(G),
            "Circular": nx.circular_layout(G),
        }

        for ax, (name, pos) in zip(axes, layouts.items()):
            nx.draw_networkx_edges(
                G, pos, ax=ax, edge_color="#aaaaaa", width=0.7, alpha=0.4
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=node_color or "#4878CF",
                node_size=_size,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.92,
            )
            if n <= 50:
                nx.draw_networkx_labels(
                    G, pos, ax=ax, font_size=7, font_color="#2c3e50"
                )
            ax.set_title(name, fontsize=14, fontweight="bold", pad=8)
            ax.axis("off")

        fig.suptitle(
            title or "Same Graph, Three Layouts", fontsize=16, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def draw_bipartite(
    G, top_nodes, bottom_nodes, title=None, top_label="Type A", bottom_label="Type B"
):
    """Draw a two-column bipartite layout.

    Parameters
    ----------
    G : networkx.Graph
        A bipartite graph.
    top_nodes : list
        Nodes to place in the left column (drawn as squares).
    bottom_nodes : list
        Nodes to place in the right column (drawn as circles).
    title : str, optional
        Plot title.
    top_label, bottom_label : str
        Legend labels for the two node types.
    """
    from matplotlib.lines import Line2D

    pos = {}
    for i, n in enumerate(top_nodes):
        pos[n] = (0, -i * 1.2)
    for i, n in enumerate(bottom_nodes):
        pos[n] = (2.0, -i * 0.9)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=top_nodes,
            ax=ax,
            node_color="#4878CF",
            node_shape="s",
            node_size=600,
            edgecolors="white",
            linewidths=2.0,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bottom_nodes,
            ax=ax,
            node_color="#D65F5F",
            node_size=500,
            edgecolors="white",
            linewidths=2.0,
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#aaaaaa", width=2.0, alpha=0.5
        )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=10,
            font_weight="bold",
            font_color="#2c3e50",
        )

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#4878CF",
                markersize=12,
                markeredgecolor="white",
                label=top_label,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#D65F5F",
                markersize=12,
                markeredgecolor="white",
                label=bottom_label,
            ),
        ]
        ax.legend(
            handles=legend_elements,
            fontsize=11,
            loc="upper right",
            framealpha=0.95,
            edgecolor="#cccccc",
        )
        ax.set_title(title or "", fontsize=14, fontweight="bold", pad=14)
        ax.margins(0.15)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


def draw_projection(G, bipartite_G=None, title=None):
    """Draw a one-mode projection, highlighting isolates in red.

    Parameters
    ----------
    G : networkx.Graph
        The projected (one-mode) graph.
    bipartite_G : networkx.Graph, optional
        The original bipartite graph.  If provided, shared neighbors are
        printed for each edge.
    title : str, optional
        Plot title.
    """
    # Print edge info
    if bipartite_G is not None:
        for u, v in G.edges():
            shared = set(bipartite_G.neighbors(u)) & set(bipartite_G.neighbors(v))
            print(f"  {u} -- {v}  (shared: {shared})")

    isolates = list(nx.isolates(G))
    if isolates:
        print(f"\nIsolated nodes (no connections): {isolates}")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G, seed=SEED, k=1.5)

        # Move isolates near the cluster center so they're visible
        if isolates:
            center = np.mean(list(pos.values()), axis=0)
            for iso in isolates:
                pos[iso] = center + np.array([0.4, -0.3])

        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="#aaaaaa", width=2.0, alpha=0.5
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=500,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.92,
        )
        if isolates:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=isolates,
                ax=ax,
                node_color="#D65F5F",
                node_size=500,
                edgecolors="white",
                linewidths=1.5,
                alpha=0.92,
            )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=10,
            font_color="#2c3e50",
            font_weight="bold",
        )

        ax.set_title(title or "", fontsize=13, fontweight="bold", pad=10)
        ax.margins(0.2)
        ax.axis("off")
        fig.tight_layout()
        plt.show()
