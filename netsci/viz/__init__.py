"""Visualization helpers for the Network Science Lab Course.

This package re-exports all plotting functions from its submodules so that
existing ``from netsci import viz; viz.plot_degree_dist(...)`` calls
continue to work unchanged.
"""

# Core (shared across labs)
from netsci.viz.core import (
    draw_graph,
    draw_pyvis,
    plot_adjacency,
    plot_ccdf,
    plot_degree_dist,
)

# Lab 01 — Introduction
from netsci.viz.intro import (
    compare_layouts,
    draw_bipartite,
    draw_graph_anatomy,
    draw_projection,
    draw_weighted_graph,
    plot_in_out_degree,
)

# Lab 02 — Properties
from netsci.viz.properties import (
    draw_bridge,
    draw_clustering_concept,
    draw_structural_roles,
    plot_centrality_comparison,
    plot_matrix_representations,
    plot_neighbor_degree,
)

# Lab 03 — Small Worlds
from netsci.viz.small_worlds import (
    plot_er_phase_panels,
    plot_er_phase_sweep,
    plot_greedy_paths,
    plot_greedy_steps,
    plot_kleinberg_panels,
    plot_kleinberg_sweep,
    plot_shortest_path,
    plot_ws_k_sweep,
    plot_ws_rewiring_demo,
    plot_ws_ring_comparison,
    plot_ws_sweep,
)

# Lab 04 — Models & Hubs
from netsci.viz.models_hubs import (
    plot_ba_growth,
    plot_ccdf_mle_panels,
    plot_ccdf_panels,
    plot_ccdf_vs_theory,
    plot_degree_overlay,
    plot_holme_kim_comparison,
    plot_model_comparison_grid,
    plot_model_degree_comparison,
    plot_robustness_concept,
    plot_robustness_sweep,
    plot_ultra_small,
)
