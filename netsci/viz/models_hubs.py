"""Visualization functions for Lab 04 — Models & Hubs.

Functions
---------
plot_model_degree_comparison, plot_ba_growth, plot_model_comparison_grid,
plot_degree_overlay, plot_holme_kim_comparison, plot_ccdf_panels,
plot_ccdf_mle_panels, plot_ccdf_vs_theory, plot_robustness_sweep,
plot_robustness_concept
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from netsci.utils import SEED
from netsci.viz._common import STYLE


# ---------------------------------------------------------------------------
# ER & WS recap: degree distribution comparison
# ---------------------------------------------------------------------------


def plot_model_degree_comparison(graphs, title=None):
    """Log-log degree distribution panels for multiple networks.

    Parameters
    ----------
    graphs : list of (str, networkx.Graph)
        ``(label, G)`` tuples, one per panel.
    title : str, optional
        Super-title.
    """
    n_panels = len(graphs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (name, G) in zip(axes, graphs):
        degrees = [d for _, d in G.degree()]
        deg_count = Counter(degrees)
        x, y = zip(*sorted(deg_count.items()))
        ax.scatter(x, y, s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} (log-log)")

    fig.suptitle(title or "Degree Distribution Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# BA growth snapshots
# ---------------------------------------------------------------------------


def plot_ba_growth(G_grow, snapshots, m=2):
    """Show BA growth stages with node size proportional to degree.

    Parameters
    ----------
    G_grow : networkx.Graph
        A BA graph at least as large as ``max(snapshots)``.
    snapshots : list of int
        Node counts at which to show snapshots (e.g. ``[5, 10, 20]``).
    m : int
        The BA *m* parameter (for the title).
    """
    n_panels = len(snapshots)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, n_snap in zip(axes, snapshots):
        G_sub = G_grow.subgraph(range(n_snap)).copy()
        pos = nx.spring_layout(G_sub, seed=SEED)
        degrees = dict(G_sub.degree())
        sizes = [degrees[n] * 80 + 40 for n in G_sub.nodes()]
        nx.draw_networkx(
            G_sub,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=sizes,
            edge_color="#cccccc",
            width=1.0,
            with_labels=True,
            font_size=8,
            font_color="white",
        )
        ax.set_title(f"n = {n_snap}\nmax degree = {max(degrees.values())}", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"BA Growth (m={m}): node size \u2219 degree (early nodes become hubs)",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3x3 model comparison grid
# ---------------------------------------------------------------------------


def plot_model_comparison_grid(model_graphs):
    """3×3 grid: degree distribution, graph drawing, and statistics.

    Parameters
    ----------
    model_graphs : list of (str, networkx.Graph)
        Exactly three ``(label, G)`` tuples for the three rows.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    for row, (name, G) in enumerate(model_graphs):
        # Column 0: Degree distribution (log-log)
        ax = axes[row, 0]
        degrees = [d for _, d in G.degree()]
        deg_count = Counter(degrees)
        x, y = zip(*sorted(deg_count.items()))
        ax.scatter(x, y, s=20, alpha=0.7, edgecolors="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nDegree dist (log-log)")

        # Column 1: Graph drawing (subset of 100 nodes for readability)
        ax = axes[row, 1]
        nodes_sub = list(G.nodes())[:100]
        G_sub = G.subgraph(nodes_sub)
        pos = nx.spring_layout(G_sub, seed=SEED)
        nx.draw_networkx(
            G_sub,
            pos,
            ax=ax,
            node_color="#4878CF",
            node_size=20,
            edge_color="#cccccc",
            width=0.3,
            with_labels=False,
            alpha=0.8,
        )
        ax.set_title(f"{name}\nGraph (first 100 nodes)")
        ax.axis("off")

        # Column 2: Statistics
        ax = axes[row, 2]
        ax.axis("off")
        stats_text = (
            f"Nodes: {G.number_of_nodes()}\n"
            f"Edges: {G.number_of_edges()}\n"
            f"Avg degree: {np.mean(degrees):.1f}\n"
            f"Max degree: {max(degrees)}\n"
            f"Clustering: {nx.average_clustering(G):.4f}\n"
        )
        ax.text(
            0.1,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="center",
            fontfamily="monospace",
        )
        ax.set_title(f"{name}\nStatistics")

    fig.suptitle("Three Network Models Compared", fontsize=16, y=1.01)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Degree distribution overlay (real vs model)
# ---------------------------------------------------------------------------


def plot_degree_overlay(graphs, title=None):
    """Overlay log-log degree distributions of multiple graphs.

    Parameters
    ----------
    graphs : list of (networkx.Graph, str, str)
        ``(G, label, marker)`` tuples.
    title : str, optional
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for G, label, marker in graphs:
        degrees = [d for _, d in G.degree()]
        deg_count = Counter(degrees)
        x, y = zip(*sorted(deg_count.items()))
        ax.scatter(
            x,
            y,
            s=30,
            alpha=0.7,
            marker=marker,
            label=label,
            edgecolors="white",
            linewidth=0.3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("Count")
    ax.set_title(title or "Degree Distribution Overlay")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CCDF panels
# ---------------------------------------------------------------------------


def plot_ccdf_panels(graphs, title=None):
    """Side-by-side CCDF plots for multiple graphs.

    Parameters
    ----------
    graphs : list of (str, networkx.Graph)
        ``(label, G)`` tuples.
    title : str, optional
    """
    n_panels = len(graphs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (name, G) in zip(axes, graphs):
        degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
        n = len(degrees)
        ccdf_y = np.arange(1, n + 1) / n
        ax.scatter(degrees, ccdf_y, s=15, alpha=0.7, edgecolors="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("P(K ≥ k)")
        ax.set_title(name)

    fig.suptitle(
        title or "CCDF: Power Law \u2192 Straight Line, Exponential \u2192 Curved",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CCDF with MLE fit panels
# ---------------------------------------------------------------------------


def plot_ccdf_mle_panels(networks, fit_power_law, title=None):
    """CCDF panels with overlaid MLE power-law fit lines.

    Parameters
    ----------
    networks : list of (str, list[int])
        ``(label, degree_sequence)`` tuples.
    fit_power_law : callable
        ``fit_power_law(degrees, k_min=2) -> float`` returning alpha.
    title : str, optional
    """
    n_panels = len(networks)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (name, degs) in zip(axes, networks):
        degs_sorted = sorted([d for d in degs if d > 0], reverse=True)
        n = len(degs_sorted)
        ccdf_y = np.arange(1, n + 1) / n
        ax.scatter(degs_sorted, ccdf_y, s=15, alpha=0.6, edgecolors="white", linewidth=0.3)

        # MLE fit line
        k_min = 2
        alpha = fit_power_law(degs, k_min=k_min)
        k_range = np.logspace(np.log10(k_min), np.log10(max(degs)), 200)
        fit_y = (k_range / k_min) ** (-(alpha - 1))
        ax.plot(k_range, fit_y, "r--", lw=1.5, label=f"\u03b1 = {alpha:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("P(K \u2265 k)")
        ax.set_title(name)
        ax.legend(fontsize=9)

    fig.suptitle(title or "CCDF with MLE Power-Law Fit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CCDF vs theoretical distributions
# ---------------------------------------------------------------------------


def plot_ccdf_vs_theory(G, fit_power_law, poisson_cdf, title=None):
    """Compare empirical CCDF to power-law, exponential, and Poisson fits.

    Parameters
    ----------
    G : networkx.Graph
    fit_power_law : callable
    poisson_cdf : callable
        ``scipy.stats.poisson.cdf``.
    title : str, optional
    """
    fb_degs = sorted([d for _, d in G.degree() if d > 0], reverse=True)
    n = len(fb_degs)
    ccdf_y = np.arange(1, n + 1) / n

    k_range = np.logspace(np.log10(1), np.log10(max(fb_degs)), 300)

    # Power law fit
    alpha_fb = fit_power_law([d for _, d in G.degree()], k_min=2)
    ccdf_pl = (k_range / 2) ** (-(alpha_fb - 1))
    ccdf_pl = ccdf_pl / ccdf_pl[0]

    # Exponential fit (match mean)
    mean_deg = np.mean([d for _, d in G.degree()])
    ccdf_exp = np.exp(-k_range / mean_deg)
    ccdf_exp = ccdf_exp / ccdf_exp[0]

    # Poisson (what ER produces)
    ccdf_poisson = 1 - poisson_cdf(k_range, mean_deg)
    ccdf_poisson = np.maximum(ccdf_poisson, 1e-10)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            fb_degs,
            ccdf_y,
            s=15,
            alpha=0.5,
            label="Data",
            zorder=5,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.plot(k_range, ccdf_pl, "r--", lw=1.5, label=f"Power law (\u03b1={alpha_fb:.2f})")
        ax.plot(k_range, ccdf_exp, "g-.", lw=1.5, label="Exponential")
        ax.plot(k_range, ccdf_poisson, "m:", lw=1.5, label=f"Poisson (\u03bb={mean_deg:.1f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("P(K \u2265 k)")
        ax.set_title(title or "Data vs Theoretical Distributions")
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=1e-3)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Network robustness sweep
# ---------------------------------------------------------------------------


def plot_robustness_sweep(results, fractions, title=None):
    """Giant component fraction vs percentage of nodes removed.

    Parameters
    ----------
    results : dict[str, (list, list)]
        ``{network_name: (gcc_random, gcc_targeted)}``
    fractions : array-like
        Fraction values (0 to 0.5 typically).
    title : str, optional
    """
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, name in zip(axes, results):
        gcc_rand, gcc_targ = results[name]
        ax.plot(fractions * 100, gcc_rand, "o-", label="Random failure", markersize=4)
        ax.plot(fractions * 100, gcc_targ, "s-", label="Targeted attack", markersize=4)
        ax.set_xlabel("% of nodes removed")
        ax.set_ylabel("Giant component (fraction of N)")
        ax.set_title(f"{name}")
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

    fig.suptitle(
        title or "Network Robustness: Random Failure vs Targeted Attack",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Robustness concept: random vs targeted removal visualization
# ---------------------------------------------------------------------------


def plot_robustness_concept(G, frac_remove=0.10, title=None):
    """Two-panel visualization comparing random vs targeted hub removal.

    Parameters
    ----------
    G : networkx.Graph
    frac_remove : float
        Fraction of nodes to remove.
    title : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pos = nx.spring_layout(G, seed=SEED)
    rng = np.random.default_rng(SEED)
    n_remove = int(G.number_of_nodes() * frac_remove)
    nodes = list(G.nodes())
    N = G.number_of_nodes()

    # Panel 1: Random removal
    G_rand = G.copy()
    rand_remove = set(rng.choice(nodes, size=n_remove, replace=False))
    G_rand.remove_nodes_from(rand_remove)
    comps_rand = sorted(nx.connected_components(G_rand), key=len, reverse=True)
    colors_rand = []
    for n in G.nodes():
        if n in rand_remove:
            colors_rand.append("#FFFFFF")
        elif n in comps_rand[0]:
            colors_rand.append("#4878CF")
        else:
            colors_rand.append("#CCCCCC")
    sizes_rand = [0 if n in rand_remove else 30 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=axes[0], node_color=colors_rand, node_size=sizes_rand
    )
    edges_rand = [
        (u, v) for u, v in G.edges() if u not in rand_remove and v not in rand_remove
    ]
    nx.draw_networkx_edges(
        G, pos, ax=axes[0], edgelist=edges_rand, edge_color="#cccccc", width=0.3
    )
    gcc_rand = len(comps_rand[0]) / N
    axes[0].set_title(
        f"Random removal ({frac_remove:.0%} of nodes)\nLargest component: {gcc_rand:.0%} of network",
        fontsize=11,
    )
    axes[0].axis("off")

    # Panel 2: Targeted removal (top hubs)
    G_targ = G.copy()
    hubs_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    targ_remove = set(hubs_sorted[:n_remove])
    G_targ.remove_nodes_from(targ_remove)
    comps_targ = sorted(nx.connected_components(G_targ), key=len, reverse=True)
    colors_targ = []
    for n in G.nodes():
        if n in targ_remove:
            colors_targ.append("#FFFFFF")
        elif n in comps_targ[0]:
            colors_targ.append("#D65F5F")
        else:
            colors_targ.append("#CCCCCC")
    sizes_targ = [0 if n in targ_remove else 30 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=axes[1], node_color=colors_targ, node_size=sizes_targ
    )
    edges_targ = [
        (u, v) for u, v in G.edges() if u not in targ_remove and v not in targ_remove
    ]
    nx.draw_networkx_edges(
        G, pos, ax=axes[1], edgelist=edges_targ, edge_color="#cccccc", width=0.3
    )
    gcc_targ = len(comps_targ[0]) / N
    axes[1].set_title(
        f"Targeted hub removal ({frac_remove:.0%} of nodes)\nLargest component: {gcc_targ:.0%} of network",
        fontsize=11,
    )
    axes[1].axis("off")

    fig.suptitle(
        title or "Robustness Paradox: Random vs Targeted Removal",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Holme-Kim comparison: BA vs Holme-Kim vs real
# ---------------------------------------------------------------------------


def plot_holme_kim_comparison(graphs, title=None):
    """CCDF overlay + clustering bar chart for BA vs Holme-Kim vs real.

    Parameters
    ----------
    graphs : list of (str, networkx.Graph)
        ``(label, G)`` tuples (typically 3: real, BA, Holme-Kim).
    title : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: CCDF overlay
    ax = axes[0]
    for name, G in graphs:
        degrees = sorted([d for _, d in G.degree() if d > 0], reverse=True)
        n = len(degrees)
        ccdf_y = np.arange(1, n + 1) / n
        ax.scatter(degrees, ccdf_y, s=15, alpha=0.6, label=name,
                   edgecolors="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("P(K \u2265 k)")
    ax.set_title("Degree Distribution (CCDF)")
    ax.legend(fontsize=9)

    # Right: clustering bar chart
    ax = axes[1]
    names = [name for name, _ in graphs]
    clusterings = [nx.average_clustering(G) for _, G in graphs]
    colors = ["#4878CF", "#D65F5F", "#6ACC65"][:len(graphs)]
    ax.bar(names, clusterings, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Clustering Coefficient")
    ax.set_title("Clustering Comparison")
    for i, v in enumerate(clusterings):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle(title or "Closing the Gap: BA vs Holme-Kim",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Ultra-small property: path length vs N for ER and BA
# ---------------------------------------------------------------------------


def plot_ultra_small(results, title=None):
    """Average path length vs network size for ER and BA.

    Parameters
    ----------
    results : dict
        ``{"N": list[int], "ER": list[float], "BA": list[float]}``
    title : str, optional
    """
    N = np.array(results["N"])
    er = results["ER"]
    ba = results["BA"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N, er, "o-", label="Erdős–Rényi", markersize=7, linewidth=2)
    ax.plot(N, ba, "s-", label="Barabási–Albert", markersize=7, linewidth=2)

    # Reference curves (scaled to match at first point)
    ln_N = np.log(N)
    lnln_N = np.log(np.log(N))
    scale_ln = er[0] / ln_N[0]
    scale_lnln = ba[0] / lnln_N[0]
    ax.plot(N, scale_ln * ln_N, ":", color="#4878CF", alpha=0.5,
            label=r"$\sim \ln N$ (ER theory)")
    ax.plot(N, scale_lnln * lnln_N, ":", color="#D65F5F", alpha=0.5,
            label=r"$\sim \ln\ln N$ (BA theory)")

    ax.set_xscale("log")
    ax.set_xlabel("Network size (N)")
    ax.set_ylabel("Average shortest path length")
    ax.legend(fontsize=10)
    ax.set_title(title or "The Ultra-Small Property: Hubs as Superhighways")
    fig.tight_layout()
    plt.show()
