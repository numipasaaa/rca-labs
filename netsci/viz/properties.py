"""Visualization functions for Lab 02 — Network Properties.

Functions
---------
plot_centrality_comparison, draw_structural_roles, draw_clustering_concept,
draw_bridge, plot_neighbor_degree, plot_matrix_representations
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from netsci.utils import SEED
from netsci.viz._common import STYLE


# ---------------------------------------------------------------------------
# Centrality comparison (3-panel)
# ---------------------------------------------------------------------------


def plot_centrality_comparison(centralities, title=None, top_n=10):
    """Show top-*n* nodes for each centrality as horizontal bar charts.

    Parameters
    ----------
    centralities : list of (str, dict)
        List of ``(name, {node: value})`` tuples, one per panel.
    title : str, optional
        Overall suptitle.
    top_n : int, default 10
        Number of top nodes to display per measure.
    """
    colors = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66"]
    n_panels = len(centralities)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        # Collect all top nodes to highlight shared ones
        all_top_sets = []
        for _, cent in centralities:
            top_nodes = sorted(cent, key=cent.get, reverse=True)[:top_n]
            all_top_sets.append(set(top_nodes))
        shared = set.intersection(*all_top_sets) if all_top_sets else set()

        for idx, (ax, (name, cent)) in enumerate(zip(axes, centralities)):
            top_nodes = sorted(cent, key=cent.get, reverse=True)[:top_n]
            values = [cent[n] for n in top_nodes]
            labels = [str(n) for n in top_nodes]
            base_color = colors[idx % len(colors)]

            # Bold color for nodes unique to this measure, lighter for shared
            bar_colors = [base_color for _ in top_nodes]

            ax.barh(
                range(len(top_nodes)),
                values,
                color=bar_colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_yticks(range(len(top_nodes)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Score", fontsize=10)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=0.3, linestyle="--")

            # Mark nodes that appear in all top-N lists
            for i, n in enumerate(top_nodes):
                if n in shared:
                    ax.annotate(
                        "\u2605",
                        xy=(values[i], i),
                        fontsize=10,
                        ha="left",
                        va="center",
                        color="#2c3e50",
                    )

        suptitle = title or "Centrality Comparison"
        if shared:
            suptitle += f"   (\u2605 = in top {top_n} of all measures)"
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Structural roles concept diagram
# ---------------------------------------------------------------------------


def draw_structural_roles():
    """Draw a two-panel diagram contrasting a degree hub and a betweenness bridge."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # --- Panel 1: Degree Hub (wheel graph) ---
        G_hub = nx.wheel_graph(8)
        pos_hub = nx.spring_layout(G_hub, seed=SEED)
        colors_hub = ["#D65F5F" if n == 0 else "#4878CF" for n in G_hub.nodes()]
        sizes_hub = [500 if n == 0 else 200 for n in G_hub.nodes()]
        nx.draw_networkx(
            G_hub,
            pos_hub,
            ax=axes[0],
            node_color=colors_hub,
            node_size=sizes_hub,
            edge_color="#999999",
            width=1.2,
            with_labels=False,
        )
        axes[0].set_title("Degree Hub\nconnected to many nodes", fontsize=11)
        axes[0].axis("off")

        # --- Panel 2: Betweenness Bridge (two cliques joined by one node) ---
        G_bridge = nx.Graph()
        for i in range(5):
            for j in range(i + 1, 5):
                G_bridge.add_edge(i, j)
        for i in range(6, 11):
            for j in range(i + 1, 11):
                G_bridge.add_edge(i, j)
        G_bridge.add_edges_from([(2, 5), (5, 8)])

        pos_bridge = nx.spring_layout(G_bridge, seed=42)
        colors_bridge = ["#D65F5F" if n == 5 else "#4878CF" for n in G_bridge.nodes()]
        sizes_bridge = [500 if n == 5 else 200 for n in G_bridge.nodes()]
        nx.draw_networkx(
            G_bridge,
            pos_bridge,
            ax=axes[1],
            node_color=colors_bridge,
            node_size=sizes_bridge,
            edge_color="#999999",
            width=1.2,
            with_labels=False,
        )
        axes[1].set_title(
            "Betweenness Bridge\nconnects otherwise-separate groups",
            fontsize=11,
        )
        axes[1].axis("off")

        fig.suptitle(
            "Two Structural Roles in Networks",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Clustering coefficient concept diagram
# ---------------------------------------------------------------------------


def draw_clustering_concept():
    """Draw a three-panel diagram illustrating clustering coefficients.

    Shows the same center node (green, node 0) with 4 neighbors (purple)
    under three scenarios: all neighbor pairs connected (C=1.0), half
    connected (C=0.5), and none connected (C=0.0, a star).
    """
    # C = 1.0: all neighbor pairs connected
    G1 = nx.Graph()
    G1.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    )

    # C = 0.5: half the neighbor pairs connected
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)])

    # C = 0.0: star — no neighbors connected to each other
    G3 = nx.star_graph(4)

    graphs = [G1, G2, G3]
    c_vals = [nx.clustering(G, 0) for G in graphs]
    labels = [
        "All neighbor pairs connected",
        "Some neighbor pairs connected",
        "No neighbor pairs connected",
    ]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for ax, G, c_val, label in zip(axes, graphs, c_vals, labels):
            pos = nx.spring_layout(G, seed=SEED)
            colors = ["#6ACC65" if n == 0 else "#B47CC7" for n in G.nodes()]
            sizes = [400 if n == 0 else 250 for n in G.nodes()]
            nx.draw_networkx(
                G,
                pos,
                ax=ax,
                node_color=colors,
                node_size=sizes,
                edge_color="#999999",
                width=1.5,
                with_labels=True,
                font_size=9,
                font_color="white",
            )
            ax.set_title(f"C(green node) = {c_val:.2f}\n{label}", fontsize=11)
            ax.axis("off")

        fig.suptitle(
            "Clustering Coefficient: How Connected Are Your Neighbors?",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Bridge visualization
# ---------------------------------------------------------------------------


def draw_bridge(G, bridge_edge, cutoff=2, title=None):
    """Draw a bridge edge in its local neighborhood, showing the two sides.

    Extracts the subgraph within *cutoff* hops of the bridge endpoints,
    colors the two sides that would disconnect if the bridge were removed,
    and highlights the bridge edge in red.

    Parameters
    ----------
    G : networkx.Graph
        Any undirected NetworkX graph.
    bridge_edge : tuple
        The ``(u, v)`` edge that is a bridge.
    cutoff : int, default 2
        BFS radius around each endpoint to include in the subgraph.
    title : str, optional
        Plot title.
    """
    u, v = bridge_edge

    # Determine sides using the FULL graph so every node is classified correctly
    G_full_cut = G.copy()
    G_full_cut.remove_edge(u, v)
    side_u_full = nx.node_connected_component(G_full_cut, u)
    small_side_node = u if len(side_u_full) <= len(G) // 2 else v
    large_side_node = v if small_side_node == u else u
    small_side_full = (
        side_u_full
        if small_side_node == u
        else nx.node_connected_component(G_full_cut, v)
    )

    # Show the entire small side; use *cutoff* for the large side
    nearby_small = (
        set(
            nx.single_source_shortest_path_length(
                G, small_side_node, cutoff=max(cutoff, len(G))
            ).keys()
        )
        & small_side_full
    )
    nearby_large = set(
        nx.single_source_shortest_path_length(G, large_side_node, cutoff=cutoff).keys()
    )
    G_local = G.subgraph(nearby_small | nearby_large).copy()

    # Classify local nodes by their side in the full graph
    side_u = {n for n in G_local.nodes() if n in small_side_full}
    side_v = {n for n in G_local.nodes() if n not in small_side_full}

    # Layout: position each side independently, then shift left/right
    G_side_u = G_local.subgraph(side_u)
    G_side_v = G_local.subgraph(side_v)

    pos_u = nx.spring_layout(G_side_u, seed=SEED, center=(-1.5, 0), scale=0.8)
    pos_v = nx.spring_layout(G_side_v, seed=SEED, center=(1.5, 0), scale=0.8)
    pos = {**pos_u, **pos_v}

    # Color each side — bridge endpoints get their side's color
    node_colors = ["#4878CF" if n in side_u else "#6ACC65" for n in G_local.nodes()]

    # Separate intra-side edges from the bridge edge
    intra_edges = [
        (a, b)
        for a, b in G_local.edges()
        if (a, b) != bridge_edge and (b, a) != bridge_edge
    ]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        # Draw intra-side edges in gray
        nx.draw_networkx_edges(
            G_local,
            pos,
            ax=ax,
            edgelist=intra_edges,
            edge_color="#cccccc",
            width=1.2,
        )
        # Draw the bridge as a dashed red line spanning the gap
        nx.draw_networkx_edges(
            G_local,
            pos,
            ax=ax,
            edgelist=[bridge_edge],
            edge_color="#D65F5F",
            width=3.0,
            style="--",
        )

        nx.draw_networkx_nodes(
            G_local,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=200,
            edgecolors="white",
            linewidths=1.2,
            alpha=0.92,
        )
        nx.draw_networkx_labels(
            G_local,
            pos,
            ax=ax,
            font_size=7,
            font_color="#2c3e50",
        )

        ax.set_title(
            title
            or f"Bridge edge ({u}, {v}) \u2014 removing the dashed red edge splits blue from green",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Assortativity: average neighbor degree
# ---------------------------------------------------------------------------


def plot_neighbor_degree(networks, title=None):
    """Scatter plot of average neighbor degree vs node degree.

    Parameters
    ----------
    networks : list of (str, networkx.Graph)
        Each entry is ``(label, G)``; one subplot per network.
    title : str, optional
        Super-title for the figure.
    """
    n = len(networks)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
        if n == 1:
            axes = [axes]

        for ax, (name, G) in zip(axes, networks):
            knn = nx.average_neighbor_degree(G)
            degrees = dict(G.degree())

            x = list(degrees.values())
            y = [knn[node] for node in degrees]

            ax.scatter(
                x,
                y,
                s=15,
                alpha=0.5,
                color="#4878CF",
                edgecolors="white",
                linewidth=0.3,
            )
            ax.set_xlabel("Node degree k", fontsize=10)
            ax.set_ylabel("\u27e8k_nn\u27e9", fontsize=10)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.2, linestyle="--")

        fig.suptitle(
            title or "Average Neighbor Degree vs Node Degree",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Matrix representations
# ---------------------------------------------------------------------------


def plot_matrix_representations(G, title=None):
    """Heatmap of adjacency, degree, and Laplacian matrices.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Super-title for the figure.
    """
    # weight=None gives a binary 0/1 adjacency matrix (ignore edge weights)
    A = nx.to_numpy_array(G, weight=None)
    degrees_arr = np.array([G.degree(n) for n in G.nodes()])
    D = np.diag(degrees_arr)
    L = D - A

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        for ax, matrix, mtitle, cmap in zip(
            axes,
            [A, D, L],
            ["Adjacency (A)", "Degree (D)", "Laplacian (L = D \u2212 A)"],
            ["Blues", "Oranges", "RdBu_r"],
        ):
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                square=True,
                cbar_kws={"shrink": 0.7},
                xticklabels=False,
                yticklabels=False,
            )
            ax.set_title(mtitle, fontsize=12)

        fig.suptitle(
            title or "Three Matrix Representations",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        plt.show()
