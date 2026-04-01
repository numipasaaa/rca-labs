"""Visualization functions for Lab 05 — Communities.

Functions
---------
plot_community_concept, plot_louvain_steps, plot_algorithm_comparison,
plot_louvain_vs_lpa
"""

import matplotlib.pyplot as plt
import networkx as nx

from netsci.utils import SEED
from netsci.viz._common import STYLE

PALETTE = [
    "#4878CF",
    "#D65F5F",
    "#6ACC65",
    "#B47CC7",
    "#C4AD66",
    "#77BEDB",
    "#E8A06B",
    "#8C8C8C",
]


# ---------------------------------------------------------------------------
# Helper: map partition → node colours
# ---------------------------------------------------------------------------


def _partition_colors(G, communities):
    """Return a list of hex colours, one per node, based on community index."""
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = i
    return [PALETTE[node_to_comm[n] % len(PALETTE)] for n in G.nodes()]


# ---------------------------------------------------------------------------
# Community concept: clear communities vs random graph
# ---------------------------------------------------------------------------


def plot_community_concept():
    """Two-panel figure: clear community structure vs random graph."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: two 5-node cliques connected by a single bridge
    G_comm = nx.Graph()
    G_comm.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    )  # clique A
    G_comm.add_edges_from(
        [(5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
    )  # clique B
    G_comm.add_edge(4, 5)  # single bridge

    colors_comm = ["#4878CF"] * 5 + ["#D65F5F"] * 5
    pos_comm = nx.spring_layout(G_comm, seed=SEED)
    nx.draw_networkx(
        G_comm,
        pos_comm,
        ax=axes[0],
        node_color=colors_comm,
        node_size=250,
        edge_color="#999999",
        width=1.2,
        with_labels=True,
        font_size=9,
        font_color="white",
    )
    axes[0].set_title("Clear communities\n(dense within, sparse between)", fontsize=11)
    axes[0].axis("off")

    # Right: random graph with similar density
    G_rand = nx.erdos_renyi_graph(10, 0.45, seed=SEED)
    pos_rand = nx.spring_layout(G_rand, seed=SEED)
    nx.draw_networkx(
        G_rand,
        pos_rand,
        ax=axes[1],
        node_color="#8C8C8C",
        node_size=250,
        edge_color="#999999",
        width=1.2,
        with_labels=True,
        font_size=9,
        font_color="white",
    )
    axes[1].set_title("No community structure\n(edges spread uniformly)", fontsize=11)
    axes[1].axis("off")

    fig.suptitle("What makes a community?", fontsize=13)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Louvain step-by-step
# ---------------------------------------------------------------------------


def plot_louvain_steps(G):
    """Three-panel figure: initial state → first pass → final Louvain partition.

    Parameters
    ----------
    G : networkx.Graph
        Network to partition (e.g. Zachary karate club).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pos = nx.spring_layout(G, seed=SEED)

    # Panel 1: Each node in its own community (initial state)
    colors_init = [PALETTE[i % len(PALETTE)] for i in range(G.number_of_nodes())]
    nx.draw_networkx(
        G,
        pos,
        ax=axes[0],
        node_color=colors_init,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[0].set_title(
        f"Step 0: Each node = own community\n({G.number_of_nodes()} communities)",
        fontsize=11,
    )
    axes[0].axis("off")

    # Panel 2: First pass (high resolution to get intermediate-like result)
    comms_mid = nx.community.louvain_communities(G, resolution=2.0, seed=SEED)
    colors_mid = _partition_colors(G, comms_mid)
    nx.draw_networkx(
        G,
        pos,
        ax=axes[1],
        node_color=colors_mid,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    Q_mid = nx.community.modularity(G, comms_mid)
    axes[1].set_title(
        f"After first pass: {len(comms_mid)} communities\nQ = {Q_mid:.3f}", fontsize=11
    )
    axes[1].axis("off")

    # Panel 3: Final partition (default resolution)
    comms_final = nx.community.louvain_communities(G, resolution=1.0, seed=SEED)
    colors_final = _partition_colors(G, comms_final)
    nx.draw_networkx(
        G,
        pos,
        ax=axes[2],
        node_color=colors_final,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    Q_final = nx.community.modularity(G, comms_final)
    axes[2].set_title(
        f"Final: {len(comms_final)} communities\nQ = {Q_final:.3f}", fontsize=11
    )
    axes[2].axis("off")

    fig.suptitle(
        "Louvain Algorithm: From Individual Nodes to Communities",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Algorithm comparison: Ground Truth vs Girvan-Newman vs Louvain
# ---------------------------------------------------------------------------


def plot_algorithm_comparison(G, ground_truth_colors, gn_communities, louvain_communities,
                              Q_gt, Q_gn, Q_louvain):
    """Three-panel comparison: Ground Truth vs Girvan-Newman vs Louvain.

    Parameters
    ----------
    G : networkx.Graph
    ground_truth_colors : list[str]
        Hex colour for each node based on ground truth.
    gn_communities : list[set]
        Communities found by Girvan-Newman.
    louvain_communities : list[set]
        Communities found by Louvain.
    Q_gt, Q_gn, Q_louvain : float
        Modularity scores.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pos = nx.spring_layout(G, seed=SEED)

    # Panel 1: Ground truth
    nx.draw_networkx(
        G,
        pos,
        ax=axes[0],
        node_color=ground_truth_colors,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[0].set_title(f"Ground truth (2 factions)\nQ = {Q_gt:.3f}", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Girvan-Newman
    colors_gn = _partition_colors(G, gn_communities)
    nx.draw_networkx(
        G,
        pos,
        ax=axes[1],
        node_color=colors_gn,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[1].set_title(
        f"Girvan-Newman ({len(gn_communities)} communities)\nQ = {Q_gn:.3f}", fontsize=11
    )
    axes[1].axis("off")

    # Panel 3: Louvain
    colors_louvain = _partition_colors(G, louvain_communities)
    nx.draw_networkx(
        G,
        pos,
        ax=axes[2],
        node_color=colors_louvain,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[2].set_title(
        f"Louvain ({len(louvain_communities)} communities)\nQ = {Q_louvain:.3f}", fontsize=11
    )
    axes[2].axis("off")

    fig.suptitle(
        "Top-Down (Girvan-Newman) vs Bottom-Up (Louvain) vs Ground Truth",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Algorithm comparison: Ground Truth vs Louvain vs Label Propagation
# ---------------------------------------------------------------------------


def plot_louvain_vs_lpa(G, ground_truth_colors, louvain_communities, lpa_communities,
                        Q_gt, Q_louvain, Q_lpa):
    """Three-panel comparison: Ground Truth vs Louvain vs Label Propagation.

    Parameters
    ----------
    G : networkx.Graph
    ground_truth_colors : list[str]
        Hex colour for each node based on ground truth.
    louvain_communities : list[set]
        Communities found by Louvain.
    lpa_communities : list[set]
        Communities found by Label Propagation.
    Q_gt, Q_louvain, Q_lpa : float
        Modularity scores.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pos = nx.spring_layout(G, seed=SEED)

    # Panel 1: Ground truth
    nx.draw_networkx(
        G,
        pos,
        ax=axes[0],
        node_color=ground_truth_colors,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[0].set_title(f"Ground truth (2 factions)\nQ = {Q_gt:.3f}", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Louvain
    colors_louvain = _partition_colors(G, louvain_communities)
    nx.draw_networkx(
        G,
        pos,
        ax=axes[1],
        node_color=colors_louvain,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[1].set_title(
        f"Louvain ({len(louvain_communities)} communities)\nQ = {Q_louvain:.3f}",
        fontsize=11,
    )
    axes[1].axis("off")

    # Panel 3: Label Propagation
    colors_lpa = _partition_colors(G, lpa_communities)
    nx.draw_networkx(
        G,
        pos,
        ax=axes[2],
        node_color=colors_lpa,
        node_size=120,
        edge_color="#cccccc",
        width=0.5,
        with_labels=True,
        font_size=7,
        font_color="white",
    )
    axes[2].set_title(
        f"Label Propagation ({len(lpa_communities)} communities)\nQ = {Q_lpa:.3f}",
        fontsize=11,
    )
    axes[2].axis("off")

    fig.suptitle(
        "Ground Truth vs Louvain vs Label Propagation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
