"""Visualization helpers for the Network Science Lab Course.

Three opinionated, one-call visualization functions for common network
analysis plots.  Each function creates its own figure, applies the
``seaborn-v0_8-muted`` style for clean academic aesthetics, and calls
``plt.show()`` -- so a single function call in a notebook cell produces
a complete, publication-ready visualization.

Design philosophy:
  - **Standalone functions only** -- no ``ax`` parameter.  Students who
    need multi-panel figures compose them directly with ``plt.subplot()``.
  - **Sensible defaults** -- one call with just a graph ``G`` gives a
    useful plot; keyword arguments allow customization when needed.
  - **Clean academic style** -- white background, muted colors, minimal
    gridlines via the ``seaborn-v0_8-muted`` matplotlib style context.

Functions
---------
plot_degree_dist(G, title=None, log=False)
    Histogram of degree values, or log-log scatter for power-law detection.
plot_ccdf(G, title=None, fit_line=False)
    Complementary CDF of the degree distribution on log-log axes.
draw_graph(G, node_color=None, node_size=None, title=None, layout="spring")
    Spring-layout node-link diagram with muted colors and optional labels.
draw_graph_anatomy(G, seed=SEED)
    Annotated tutorial diagram labeling node, edge, and degree.
draw_weighted_graph(G, weight_attr="weight", title=None, seed=SEED, ...)
    Node-link diagram with edge width/color proportional to weight.
plot_adjacency(G, title=None, cmap="Blues", nodelist=None, group_labels=None, ...)
    Adjacency matrix with optional node reordering and group annotations.
plot_in_out_degree(G, title_prefix="", ...)
    Side-by-side in-degree / out-degree histograms for directed graphs.
compare_layouts(G, node_color=None, title=None)
    Three-panel comparison of spring, Kamada-Kawai, and circular layouts.
draw_bipartite(G, top_nodes, bottom_nodes, title=None, ...)
    Two-column bipartite layout with typed node shapes and legend.
draw_projection(G, bipartite_G=None, title=None)
    One-mode projection plot that highlights isolates in red.
plot_neighbor_degree(networks, title=None)
    Average-neighbor-degree vs node-degree scatter plots.
plot_matrix_representations(G, title=None)
    Heatmaps of the adjacency, degree, and Laplacian matrices.
draw_pyvis(G, node_color=None, title=None, height="500px", filename="graph.html")
    Interactive PyVis graph visualization saved to HTML and displayed inline.
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from netsci.utils import SEED

# ---------------------------------------------------------------------------
# Degree distribution
# ---------------------------------------------------------------------------


def plot_degree_dist(G, title=None, log=False):
    """Plot the degree distribution of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    log : bool, default False
        If *True*, produce a log-log scatter plot (useful for power-law
        detection).  Otherwise produce a histogram.
    """
    degrees = [d for _, d in G.degree()]

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        if log:
            deg_count = Counter(degrees)
            x, y = zip(*sorted(deg_count.items()))
            ax.scatter(
                x,
                y,
                s=40,
                alpha=0.8,
                color="#4878CF",
                edgecolors="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                title or "Degree Distribution (log-log)",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
        else:
            ax.hist(
                degrees,
                bins=range(min(degrees), max(degrees) + 2),
                edgecolor="white",
                alpha=0.85,
                color="#4878CF",
                linewidth=0.8,
            )
            ax.set_title(
                title or "Degree Distribution", fontsize=13, fontweight="bold", pad=10
            )

        ax.set_xlabel("Degree (k)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Complementary CDF
# ---------------------------------------------------------------------------


def plot_ccdf(G, title=None, fit_line=False):
    """Plot the complementary CDF of the degree distribution on log-log axes.

    The CCDF shows P(K >= k) for each degree value k.  On log-log axes a
    power-law distribution appears as a straight line with slope -(gamma-1).

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    fit_line : bool, default False
        If *True*, overlay an MLE power-law fit line.  The exponent is
        estimated as alpha = 1 + n / sum(ln(x_i / x_min)).
    """
    degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
    n = len(degrees)
    ccdf_y = np.arange(1, n + 1) / n
    ccdf_x = np.array(degrees)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ccdf_x, ccdf_y, s=20, alpha=0.7, edgecolors="white", linewidth=0.5)

        if fit_line:
            k_min = min(degrees)
            log_ratios = np.log(np.array(degrees) / k_min)
            alpha = 1 + n / log_ratios.sum()
            k_range = np.logspace(np.log10(k_min), np.log10(max(degrees)), 200)
            fit_y = (k_range / k_min) ** (-(alpha - 1))
            ax.plot(k_range, fit_y, "r--", lw=1.5, label=f"MLE fit  α = {alpha:.2f}")
            ax.legend(fontsize=9, framealpha=0.9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)", fontsize=11)
        ax.set_ylabel("P(K ≥ k)", fontsize=11)
        ax.set_title(
            title or "Degree CCDF (log-log)", fontsize=13, fontweight="bold", pad=10
        )
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Node-link graph drawing
# ---------------------------------------------------------------------------


def draw_graph(G, node_color=None, node_size=None, title=None, layout="spring"):
    """Draw a node-link diagram of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    node_color : color or list of colors, optional
        Node fill color(s).  Defaults to muted blue ``"#4878CF"``.
    node_size : int or list of ints, optional
        Node marker size(s).  Defaults to 80.
    title : str, optional
        Custom plot title.
    layout : {"spring", "kamada_kawai", "circular"}, default "spring"
        Layout algorithm.  ``"spring"`` uses a deterministic seed for
        reproducibility.
    """
    if G.number_of_nodes() > 500:
        print(
            f"Warning: graph has {G.number_of_nodes()} nodes. "
            "Drawing may be slow or unreadable."
        )

    layout_fns = {
        "spring": lambda: nx.spring_layout(G, seed=SEED),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
        "circular": lambda: nx.circular_layout(G),
    }
    pos = layout_fns.get(layout, layout_fns["spring"])()

    n_nodes = G.number_of_nodes()
    _node_size = node_size if node_size is not None else max(300 - n_nodes * 2, 40)

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw edges — with arrows for directed graphs
        edge_kw = dict(
            ax=ax,
            edge_color="#aaaaaa",
            width=0.7,
            alpha=0.5,
        )
        if G.is_directed():
            edge_kw.update(
                arrows=True,
                arrowstyle="-|>",
                arrowsize=12,
                connectionstyle="arc3,rad=0.1",
                min_source_margin=8,
                min_target_margin=8,
            )
        nx.draw_networkx_edges(G, pos, **edge_kw)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_color or "#4878CF",
            node_size=_node_size,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.92,
        )
        if n_nodes <= 50:
            nx.draw_networkx_labels(
                G,
                pos,
                ax=ax,
                font_size=max(10 - n_nodes // 10, 7),
                font_color="#2c3e50",
            )
        elif n_nodes <= 200:
            # Show labels for top-degree nodes only
            degrees = dict(G.degree())
            top_k = min(10, n_nodes // 5)
            top = sorted(degrees, key=degrees.get, reverse=True)[:top_k]
            labels = {n: str(n) for n in top}
            nx.draw_networkx_labels(
                G,
                pos,
                labels,
                ax=ax,
                font_size=9,
                font_weight="bold",
                font_color="#2c3e50",
            )
        ax.set_title(title or "", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Adjacency matrix heatmap
# ---------------------------------------------------------------------------


def plot_adjacency(
    G, title=None, cmap="Blues", nodelist=None, group_labels=None, group_colors=None
):
    """Plot a heatmap of the adjacency matrix of *G*.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    title : str, optional
        Custom plot title.
    cmap : str, default "Blues"
        Matplotlib/seaborn colormap name.  Ignored when *group_labels* is
        given (each block gets its own colour instead).
    nodelist : list, optional
        Order of nodes in the matrix rows/columns.  Sorting by community
        membership reveals block-diagonal structure.
    group_labels : list of (str, int), optional
        List of ``(label, count)`` tuples describing consecutive groups in
        *nodelist*.  E.g. ``[("Mr. Hi", 17), ("Officer", 17)]``.
        When provided the **heatmap cells themselves** are coloured by group:
        each on-diagonal block gets its own colour and cross-group edges are
        shown in grey, making the community structure immediately obvious.
    group_colors : list of str, optional
        One colour per group.  Defaults to ``["#D65F5F", "#4878CF", ...]``.
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    A = nx.to_numpy_array(G, nodelist=nodelist)
    A = (A > 0).astype(float)  # binary
    n = A.shape[0]

    default_colors = ["#D65F5F", "#4878CF", "#6ACC65", "#B47CC7", "#C4AD66"]

    with plt.style.context("seaborn-v0_8-muted"):
        fig, ax = plt.subplots(figsize=(7.5, 7))

        if group_labels is not None:
            colors = group_colors or default_colors

            # Build an RGBA image: background, cross-group edges, per-block edges
            bg = np.array([0.96, 0.96, 0.96, 1.0])  # light grey background
            cross_c = np.array([0.72, 0.72, 0.72, 1.0])  # grey for cross-group
            img = np.tile(bg, (n, n, 1))

            # Assign group index to each node
            group_idx = np.zeros(n, dtype=int)
            row = 0
            for gi, (_, count) in enumerate(group_labels):
                group_idx[row : row + count] = gi
                row += count

            for i in range(n):
                for j in range(n):
                    if A[i, j] > 0:
                        gi, gj = group_idx[i], group_idx[j]
                        if gi == gj:
                            # Intra-group edge → group colour
                            rgb = to_rgb(colors[gi % len(colors)])
                            img[i, j] = (*rgb, 1.0)
                        else:
                            # Cross-group edge → grey
                            img[i, j] = cross_c

            ax.imshow(img, interpolation="nearest", aspect="equal", origin="upper")

            # Draw block boundaries and bracket labels
            row = 0
            for gi, (label, count) in enumerate(group_labels):
                c = colors[gi % len(colors)]
                mid = row + count / 2 - 0.5

                # Label on the left
                ax.text(
                    -1.5,
                    mid,
                    label,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=c,
                    clip_on=False,
                )

                # Separator line after each group (except last)
                if gi < len(group_labels) - 1:
                    sep = row + count - 0.5
                    ax.axhline(sep, color="white", linewidth=2)
                    ax.axvline(sep, color="white", linewidth=2)

                row += count

            # Legend
            handles = [
                Patch(facecolor=colors[i % len(colors)], label=lbl)
                for i, (lbl, _) in enumerate(group_labels)
            ]
            handles.append(Patch(facecolor="#B8B8B8", label="Cross-group"))
            ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
                edgecolor="#cccccc",
            )

            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Plain heatmap (no groups)
            sns.heatmap(
                A,
                ax=ax,
                cmap=cmap,
                square=True,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                linewidths=0,
            )
            ax.set_ylabel("Node", fontsize=11)

        ax.set_xlabel("Node", fontsize=11)
        ax.set_title(
            title or "Adjacency Matrix", fontsize=13, fontweight="bold", pad=10
        )
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Annotated graph anatomy (tutorial helper)
# ---------------------------------------------------------------------------


def draw_graph_anatomy(G, seed=SEED):
    """Draw an annotated tutorial diagram labeling node, edge, and degree.

    Designed for a small (≤10 node) teaching graph with named nodes
    including 'Alice', 'Bob', and 'Carol'.
    """
    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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
    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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
    with plt.style.context("seaborn-v0_8-muted"):
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

    with plt.style.context("seaborn-v0_8-muted"):
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


# ---------------------------------------------------------------------------
# Interactive PyVis visualization
# ---------------------------------------------------------------------------

# Default color palette for community coloring (10 distinct colors).
_PYVIS_COLORS = [
    "#4878CF",
    "#6ACC65",
    "#D65F5F",
    "#B47CC7",
    "#C4AD66",
    "#77BEDB",
    "#E8A06B",
    "#8C8C8C",
    "#C572D1",
    "#64B5CD",
]


def draw_pyvis(G, node_color=None, title=None, height="500px", filename="graph.html"):
    """Render an interactive graph with PyVis and display inline.

    Parameters
    ----------
    G : networkx.Graph
        Any NetworkX graph.
    node_color : dict or list, optional
        Per-node colors.  A *dict* mapping ``node -> color_string`` or
        ``node -> int`` (community label mapped to a built-in palette).
        A *list* of color strings in ``G.nodes()`` order also works.
        When values are integers they are mapped to a 10-color palette.
    title : str, optional
        HTML heading displayed above the graph.
    height : str, default "500px"
        CSS height of the canvas.
    filename : str, default "graph.html"
        Path for the generated HTML file.
    """
    from IPython.display import HTML, display
    from pyvis.network import Network

    net = Network(height=height, width="100%", notebook=True, cdn_resources="in_line")
    if title:
        net.heading = title

    # Build color mapping ------------------------------------------------
    nodes = list(G.nodes())
    if node_color is None:
        color_map = {n: _PYVIS_COLORS[0] for n in nodes}
    elif isinstance(node_color, dict):
        color_map = {}
        for n in nodes:
            c = node_color.get(n, 0)
            if isinstance(c, (int, np.integer)):
                color_map[n] = _PYVIS_COLORS[int(c) % len(_PYVIS_COLORS)]
            else:
                color_map[n] = c
    else:
        # list / array — same order as G.nodes()
        color_map = {
            n: (
                _PYVIS_COLORS[int(c) % len(_PYVIS_COLORS)]
                if isinstance(c, (int, np.integer))
                else c
            )
            for n, c in zip(nodes, node_color)
        }

    # Add nodes and edges ------------------------------------------------
    for n in nodes:
        net.add_node(str(n), label=str(n), color=color_map[n])
    for u, v in G.edges():
        net.add_edge(str(u), str(v))

    # Save and display ---------------------------------------------------
    net.save_graph(filename)
    display(HTML(filename))


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

    with plt.style.context("seaborn-v0_8-muted"):
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
    with plt.style.context("seaborn-v0_8-muted"):
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
    with plt.style.context("seaborn-v0_8-muted"):
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
    with plt.style.context("seaborn-v0_8-muted"):
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
    with plt.style.context("seaborn-v0_8-muted"):
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
