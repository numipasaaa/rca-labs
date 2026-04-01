"""Core visualization functions shared across multiple labs.

Functions
---------
plot_degree_dist, plot_ccdf, draw_graph, plot_adjacency, draw_pyvis
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from netsci.utils import SEED
from netsci.viz._common import STYLE

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

    with plt.style.context(STYLE):
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

    with plt.style.context(STYLE):
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

    with plt.style.context(STYLE):
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

    with plt.style.context(STYLE):
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
# Interactive Plotly visualization
# ---------------------------------------------------------------------------

# Default color palette for community coloring (10 distinct colors).
_INTERACTIVE_COLORS = [
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


def draw_pyvis(G, node_color=None, title=None, height="500px", filename=None):
    """Render an interactive graph with Plotly (zoom, pan, hover).

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
        Title displayed above the graph.
    height : str or int, default "500px"
        Height of the figure (e.g. ``"500px"`` or ``500``).
    filename : str, optional
        Ignored (kept for backward compatibility with old PyVis calls).
    """
    import plotly.graph_objects as go

    pos = nx.spring_layout(G, seed=SEED)

    # Parse height to pixels
    if isinstance(height, str):
        h_px = int(height.replace("px", ""))
    else:
        h_px = int(height)

    # Build color mapping ------------------------------------------------
    nodes = list(G.nodes())
    if node_color is None:
        color_map = {n: _INTERACTIVE_COLORS[0] for n in nodes}
    elif isinstance(node_color, dict):
        color_map = {}
        for n in nodes:
            c = node_color.get(n, 0)
            if isinstance(c, (int, np.integer)):
                color_map[n] = _INTERACTIVE_COLORS[int(c) % len(_INTERACTIVE_COLORS)]
            else:
                color_map[n] = c
    else:
        color_map = {
            n: (
                _INTERACTIVE_COLORS[int(c) % len(_INTERACTIVE_COLORS)]
                if isinstance(c, (int, np.integer))
                else c
            )
            for n, c in zip(nodes, node_color)
        }

    # Edge traces --------------------------------------------------------
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#aaaaaa"),
        hoverinfo="none",
    )

    # Node traces --------------------------------------------------------
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_colors = [color_map[n] for n in nodes]
    node_labels = [str(n) for n in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=10, color=node_colors, line=dict(width=1, color="white")),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_labels,
        hoverinfo="text",
    )

    # Layout & display ---------------------------------------------------
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title or "", font=dict(size=14)),
            showlegend=False,
            height=h_px,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        ),
    )
    fig.show()
