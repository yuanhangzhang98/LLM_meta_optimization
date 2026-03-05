"""
DAG visualization for MCGS design genealogy.

Creates a publication-quality figure showing the directed acyclic graph
of design evolution, with edges weighted by reference_weights.
Suitable for Nature Machine Intelligence style publications.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace_config


# ============================================================================ #
# Configuration
# ============================================================================ #

def setup_nature_style():
    """Configure matplotlib for Nature Machine Intelligence style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        "figure.dpi": 300,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        'axes.titlesize': 20,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        'lines.markersize': 8,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# ============================================================================ #
# Data Loading
# ============================================================================ #

def load_database(db_path: Path) -> Dict[str, Any]:
    """Load database JSON file."""
    with open(db_path, 'r') as f:
        return json.load(f)


def load_consensus_scores(db_path: Path) -> Dict[int, float]:
    """
    Load consensus objective scores from objectives_summary.csv.

    Uses workspace_config to find the results directory.

    Args:
        db_path: Path to database.json file (used as fallback)

    Returns:
        Dict mapping design_id -> consensus_score
    """
    # Try workspace_config results directory first
    results_dir = workspace_config.get_results_dir()
    csv_path = results_dir / 'objectives_summary.csv'

    if not csv_path.exists():
        # Fall back to relative to db_path
        csv_path = db_path.parent / 'objectives_summary.csv'

    if not csv_path.exists():
        return {}

    scores = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                design_id = int(row['design_id'])
                score = float(row['consensus_score'])
                scores[design_id] = score
            except (KeyError, ValueError):
                continue

    return scores


def build_graph(
    data: Dict[str, Any],
    consensus_scores: Optional[Dict[int, float]] = None
) -> nx.DiGraph:
    """
    Build networkx DiGraph from database designs.

    Node attributes: design_id, short_name, planner_round_id, timestamp, objective_id, objective
    Edge attributes: weight (from reference_weights)

    Edges go from parent -> child (direction of inheritance).

    Args:
        data: Database JSON data
        consensus_scores: Optional dict mapping design_id -> consensus_score.
                         If provided, used for objective attribute instead of median_step.
    """
    G = nx.DiGraph()

    for design in data['designs']:
        design_id = design['design_id']

        # Use consensus score if available, otherwise fall back to median_step at largest N
        if consensus_scores and design_id in consensus_scores:
            objective = consensus_scores[design_id]
        else:
            experiments = design.get('experiments', [])
            objective = None
            if experiments:
                by_n = sorted(experiments, key=lambda e: e.get('N', 0), reverse=True)
                if by_n and 'median_step' in by_n[0]:
                    objective = by_n[0]['median_step']

        G.add_node(
            design_id,
            short_name=design.get('short_name', ''),
            planner_round_id=design.get('planner_round_id', 0),
            timestamp=design.get('timestamp', ''),
            objective_id=design.get('objective_id', 0),
            objective=objective
        )

        # Add edges from parents to this design
        for ref in design.get('reference_weights', []):
            parent_id = ref['design_id']
            weight = ref['weight']
            # Edge direction: parent -> child
            G.add_edge(parent_id, design_id, weight=weight)

    return G


# ============================================================================ #
# Layout Algorithms
# ============================================================================ #

def hierarchical_layout(
    G: nx.DiGraph,
    horizontal_spacing: float = 1.0,
    vertical_spacing: float = 1.0
) -> Dict[int, Tuple[float, float]]:
    """
    Compute hierarchical layout based on planner_round_id.

    Each planner round forms a horizontal layer.
    Design 0 is at the top, later rounds are below.

    Returns:
        Dict mapping node_id -> (x, y) position
    """
    # Group nodes by planner_round_id
    layers: Dict[int, List[int]] = {}
    for node in G.nodes():
        round_id = G.nodes[node].get('planner_round_id', 0)
        if round_id not in layers:
            layers[round_id] = []
        layers[round_id].append(node)

    pos = {}
    sorted_rounds = sorted(layers.keys())

    for y_idx, round_id in enumerate(sorted_rounds):
        nodes_in_layer = sorted(layers[round_id])  # Sort by design_id
        n_nodes = len(nodes_in_layer)

        # Center the layer horizontally
        x_offset = -(n_nodes - 1) * horizontal_spacing / 2

        for x_idx, node in enumerate(nodes_in_layer):
            pos[node] = (x_offset + x_idx * horizontal_spacing, -y_idx * vertical_spacing)

    return pos


def _compact_layers(
    G: nx.DiGraph,
    node_depth: Dict[int, int],
    iterations: int = 3
) -> Dict[int, int]:
    """
    Compact sparse layers by moving nodes within their valid range.

    For each node, computes [min_layer, max_layer] based on parent/child
    constraints, then moves nodes from sparse to denser layers.

    Args:
        G: Directed graph
        node_depth: Current layer assignment (node -> depth)
        iterations: Number of compaction passes

    Returns:
        Updated node_depth dict
    """
    # Compute valid ranges for each node
    min_layer: Dict[int, int] = {}
    max_layer: Dict[int, float] = {}

    for node in G.nodes():
        parents = list(G.predecessors(node))
        children = list(G.successors(node))

        # Must be below all parents
        if parents:
            min_layer[node] = max(node_depth[p] for p in parents) + 1
        else:
            min_layer[node] = 0

        # Must be above all children
        if children:
            max_layer[node] = min(node_depth[c] for c in children) - 1
        else:
            max_layer[node] = float('inf')

    # Iterative compaction
    for _ in range(iterations):
        # Count nodes per layer
        layer_sizes: Dict[int, int] = {}
        for depth in node_depth.values():
            layer_sizes[depth] = layer_sizes.get(depth, 0) + 1

        if not layer_sizes:
            break

        avg_size = sum(layer_sizes.values()) / len(layer_sizes)
        max_depth = max(node_depth.values())

        # Try to move nodes from sparse layers
        for node in list(G.nodes()):
            current = node_depth[node]
            current_size = layer_sizes.get(current, 0)

            # Skip if already in dense layer
            if current_size >= avg_size:
                continue

            # Find best target within valid range
            best_target = current
            best_improvement = 0

            valid_max = min(max_layer[node], max_depth)
            for target in range(min_layer[node], int(valid_max) + 1):
                if target == current:
                    continue
                target_size = layer_sizes.get(target, 0)

                # Move if target is less crowded and we're sparse
                improvement = current_size - target_size - 1
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_target = target

            if best_target != current:
                # Update layer sizes
                layer_sizes[current] -= 1
                layer_sizes[best_target] = layer_sizes.get(best_target, 0) + 1
                node_depth[node] = best_target

    # Renumber layers to remove gaps
    used_depths = sorted(set(node_depth.values()))
    depth_map = {old: new for new, old in enumerate(used_depths)}
    node_depth = {node: depth_map[depth] for node, depth in node_depth.items()}

    return node_depth


def sugiyama_layout(
    G: nx.DiGraph,
    root: int = 0,
    horizontal_spacing: float = 1.0,
    vertical_spacing: float = 1.2,
    iterations: int = 4
) -> Dict[int, Tuple[float, float]]:
    """
    Sugiyama-style layered layout for DAGs.

    This is the gold standard algorithm for balanced layered graph visualization.
    It produces cleaner layouts than simple hierarchical approaches by:
    1. Assigning layers based on topological depth (longest path from root)
    2. Minimizing edge crossings using barycenter heuristic
    3. Centering children under their parents

    Args:
        G: Directed acyclic graph
        root: Root node ID (default: 0)
        horizontal_spacing: Horizontal distance between nodes
        vertical_spacing: Vertical distance between layers
        iterations: Number of barycenter passes for crossing minimization

    Returns:
        Dict mapping node_id -> (x, y) position
    """
    if G.number_of_nodes() == 0:
        return {}

    # Phase 1: Layer assignment by longest path from root
    # Using dynamic programming to find longest path to each node
    node_depth: Dict[int, int] = {}

    # Topological sort to process nodes in order
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Graph has cycles, fall back to BFS from root
        topo_order = list(nx.bfs_tree(G, root).nodes()) if root in G else list(G.nodes())

    # Initialize depths
    for node in G.nodes():
        node_depth[node] = 0

    # Compute longest path from any root (node with no predecessors)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        roots = [root] if root in G else [list(G.nodes())[0]]

    for node in topo_order:
        predecessors = list(G.predecessors(node))
        if predecessors:
            node_depth[node] = max(node_depth[p] for p in predecessors) + 1

    # Phase 1.5: Compact sparse layers
    node_depth = _compact_layers(G, node_depth, iterations=3)

    # Group nodes by depth into layers
    layers: Dict[int, List[int]] = {}
    for node, depth in node_depth.items():
        if depth not in layers:
            layers[depth] = []
        layers[depth].append(node)

    # Sort layers by depth
    sorted_depths = sorted(layers.keys())
    layer_list = [layers[d] for d in sorted_depths]

    # Phase 2: Crossing minimization using barycenter heuristic
    # Initialize order: sort by design_id within each layer
    for layer in layer_list:
        layer.sort()

    # Create position lookup for barycenter computation
    def get_layer_positions(layer_list):
        pos_in_layer = {}
        for layer_idx, layer in enumerate(layer_list):
            for idx, node in enumerate(layer):
                pos_in_layer[node] = (layer_idx, idx)
        return pos_in_layer

    # Barycenter heuristic: iterate top-down and bottom-up
    for _ in range(iterations):
        # Top-down pass
        for layer_idx in range(1, len(layer_list)):
            layer = layer_list[layer_idx]
            pos_in_layer = get_layer_positions(layer_list)

            # Compute barycenter for each node
            barycenters = []
            for node in layer:
                parents = list(G.predecessors(node))
                if parents:
                    parent_positions = [pos_in_layer[p][1] for p in parents if p in pos_in_layer]
                    if parent_positions:
                        bc = sum(parent_positions) / len(parent_positions)
                    else:
                        bc = pos_in_layer[node][1]
                else:
                    bc = pos_in_layer[node][1]
                barycenters.append((bc, node))

            # Sort by barycenter, keeping original order for ties
            barycenters.sort(key=lambda x: x[0])
            layer_list[layer_idx] = [node for _, node in barycenters]

        # Bottom-up pass
        for layer_idx in range(len(layer_list) - 2, -1, -1):
            layer = layer_list[layer_idx]
            pos_in_layer = get_layer_positions(layer_list)

            barycenters = []
            for node in layer:
                children = list(G.successors(node))
                if children:
                    child_positions = [pos_in_layer[c][1] for c in children if c in pos_in_layer]
                    if child_positions:
                        bc = sum(child_positions) / len(child_positions)
                    else:
                        bc = pos_in_layer[node][1]
                else:
                    bc = pos_in_layer[node][1]
                barycenters.append((bc, node))

            barycenters.sort(key=lambda x: x[0])
            layer_list[layer_idx] = [node for _, node in barycenters]

    # Phase 3: Coordinate assignment
    pos = {}
    for layer_idx, layer in enumerate(layer_list):
        n_nodes = len(layer)
        # Center the layer horizontally
        x_offset = -(n_nodes - 1) * horizontal_spacing / 2

        for x_idx, node in enumerate(layer):
            pos[node] = (x_offset + x_idx * horizontal_spacing, -layer_idx * vertical_spacing)

    return pos


def force_directed_layout(
    G: nx.DiGraph,
    k: Optional[float] = None,
    iterations: int = 100,
    seed: int = 42
) -> Dict[int, Tuple[float, float]]:
    """
    Compute force-directed layout for large graphs.

    Args:
        G: Graph
        k: Optimal node distance (None for auto)
        iterations: Number of iterations
        seed: Random seed for reproducibility
    """
    if k is None:
        k = 1.5 / np.sqrt(G.number_of_nodes())

    return nx.spring_layout(G, k=k, iterations=iterations, seed=seed)


def select_layout(
    G: nx.DiGraph,
    layout: str = 'auto'
) -> Dict[int, Tuple[float, float]]:
    """
    Select appropriate layout based on graph size and user preference.

    Args:
        G: Graph
        layout: 'auto', 'sugiyama', 'hierarchical', or 'force'
    """
    if layout == 'auto' or layout == 'sugiyama':
        # Sugiyama is the best default for DAGs
        return sugiyama_layout(G)
    elif layout == 'hierarchical':
        return hierarchical_layout(G)
    elif layout == 'force':
        return force_directed_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")


# ============================================================================ #
# Styling
# ============================================================================ #

def get_node_colors(
    G: nx.DiGraph,
    color_by: str = 'design_id',
    cmap_name: str = 'viridis_r'
) -> Tuple[List[float], Normalize, Any]:
    """
    Compute node colors based on coloring scheme.

    Args:
        G: Graph
        color_by: 'design_id', 'objective', or 'planner_round'
        cmap_name: Matplotlib colormap name

    Returns:
        (color_values, norm, cmap) tuple
    """
    nodes = list(G.nodes())
    cmap = plt.colormaps.get_cmap(cmap_name)

    if color_by == 'design_id':
        values = [n for n in nodes]
        vmin, vmax = min(values), max(values)
    elif color_by == 'objective':
        values = []
        for n in nodes:
            obj = G.nodes[n].get('objective')
            values.append(obj if obj is not None else float('nan'))
        vmin, vmax = 0, 1
    elif color_by == 'planner_round':
        values = [G.nodes[n].get('planner_round_id', 0) for n in nodes]
        vmin, vmax = min(values), max(values)
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    norm = Normalize(vmin=vmin, vmax=vmax)
    return values, norm, cmap


def get_edge_styles(
    G: nx.DiGraph,
    base_width: float = 0.5,
    max_width: float = 3.0,
    base_alpha: float = 0.3,
    max_alpha: float = 0.8
) -> Tuple[List[float], List[float]]:
    """
    Compute edge widths and alphas based on weights.

    Returns:
        (widths, alphas) lists in edge order
    """
    widths = []
    alphas = []

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        width = base_width + weight * (max_width - base_width)
        alpha = base_alpha + weight * (max_alpha - base_alpha)
        widths.append(width)
        alphas.append(alpha)

    return widths, alphas


def get_node_sizes(
    G: nx.DiGraph,
    base_size: float = 800,
    root_id: int = 0,
    highlight_ids: Optional[List[int]] = None
) -> List[float]:
    """
    Compute node sizes.

    Root and highlighted nodes are larger.
    """
    if highlight_ids is None:
        highlight_ids = []

    sizes = []
    for node in G.nodes():
        if node in highlight_ids:
            sizes.append(base_size * 1.8)
        else:
            sizes.append(base_size)

    return sizes


def get_visible_labels(
    G: nx.DiGraph,
    max_labels: Optional[int] = None,
    always_show: Optional[List[int]] = None
) -> Dict[int, str]:
    """
    Determine which labels to show.

    Args:
        G: Graph
        max_labels: Maximum number of labels to show. None = show all (default).
        always_show: Node IDs that should always be shown.

    Priority for filtering: always_show nodes, then by out-degree (most children).
    """
    if always_show is None:
        always_show = [0]  # Always show root

    n_nodes = G.number_of_nodes()

    # Show all labels by default
    if max_labels is None or n_nodes <= max_labels:
        return {n: str(n) for n in G.nodes()}

    # Prioritize nodes when limiting
    priority = []
    for node in G.nodes():
        if node in always_show:
            priority.append((node, float('inf')))
        else:
            out_deg = G.out_degree(node)
            priority.append((node, out_deg))

    priority.sort(key=lambda x: x[1], reverse=True)
    selected = [node for node, _ in priority[:max_labels]]

    return {n: str(n) for n in selected}


# ============================================================================ #
# Drawing
# ============================================================================ #

def draw_dag(
    G: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    ax: plt.Axes,
    color_by: str = 'design_id',
    highlight_ids: Optional[List[int]] = None,
    show_labels: bool = True,
    show_colorbar: bool = True,
    fig: Optional[plt.Figure] = None
) -> None:
    """
    Draw the DAG on matplotlib axes.

    Args:
        G: NetworkX DiGraph
        pos: Node positions
        ax: Matplotlib axes
        color_by: Coloring scheme
        highlight_ids: Node IDs to highlight
        show_labels: Whether to show node labels
        show_colorbar: Whether to show colorbar
        fig: Figure (required for colorbar)
    """
    if highlight_ids is None:
        highlight_ids = []

    nodes = list(G.nodes())

    # Get styling
    color_values, norm, cmap = get_node_colors(G, color_by)
    widths, alphas = get_edge_styles(G)
    sizes = get_node_sizes(G, highlight_ids=highlight_ids)

    # Draw edges with curved arrows for multi-parent relationships
    edge_color = '#505050'

    for (u, v, data), width, alpha in zip(G.edges(data=True), widths, alphas):
        # Count how many parents this child has
        n_parents = G.in_degree(v)
        parent_list = list(G.predecessors(v))

        if n_parents > 1 and u in parent_list:
            # Curved edge for multi-parent
            idx = parent_list.index(u)
            rad = 0.15 * (idx - (n_parents - 1) / 2)
            style = f'arc3,rad={rad}'
        else:
            style = 'arc3,rad=0'

        arrow = FancyArrowPatch(
            pos[u], pos[v],
            arrowstyle='-|>',
            connectionstyle=style,
            linewidth=width,
            alpha=alpha,
            color=edge_color,
            mutation_scale=12,
            zorder=1
        )
        ax.add_patch(arrow)

    # Prepare node colors
    node_colors = []
    for i, node in enumerate(nodes):
        val = color_values[i]
        if np.isnan(val) if isinstance(val, float) else False:
            node_colors.append('#cccccc')  # Gray for missing values
        else:
            node_colors.append(cmap(norm(val)))

    # Draw nodes
    # Special markers for root and highlighted nodes
    root_id = 0
    regular_nodes = [n for n in nodes if n not in highlight_ids]
    regular_pos = [pos[n] for n in regular_nodes]
    regular_colors = [node_colors[nodes.index(n)] for n in regular_nodes]
    regular_sizes = [sizes[nodes.index(n)] for n in regular_nodes]

    if regular_nodes:
        ax.scatter(
            [p[0] for p in regular_pos],
            [p[1] for p in regular_pos],
            c=regular_colors,
            s=regular_sizes,
            marker='o',
            edgecolors='white',
            linewidths=0.8,
            zorder=2
        )

    # Draw highlighted nodes with diamond marker
    for node in highlight_ids:
        if node in nodes and node != root_id:
            ax.scatter(
                pos[node][0], pos[node][1],
                c='#E64B35',  # Nature red
                s=sizes[nodes.index(node)],
                marker='D',
                edgecolors='white',
                linewidths=1.0,
                zorder=3
            )

    # Draw labels
    if show_labels:
        labels = get_visible_labels(G, always_show=[root_id] + highlight_ids)
        for node, label in labels.items():
            x, y = pos[node]
            # Determine label color based on node color value
            idx = nodes.index(node)
            val = color_values[idx]
            normalized_val = norm(val) if not (isinstance(val, float) and np.isnan(val)) else 0.5
            label_color = 'black' if normalized_val < 0.5 else 'white'
            ax.annotate(
                label,
                (x, y),
                fontsize=14,
                ha='center',
                va='center',
                color=label_color,
                fontweight='bold',
                zorder=4
            )

    # Add colorbar
    if show_colorbar and fig is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)

        label_map = {
            'design_id': 'Design ID',
            'objective': 'Consensus Score',
            'planner_round': 'Planner Round'
        }
        cbar.set_label(label_map.get(color_by, color_by), fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Clean up axes
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = []
    if highlight_ids:
        legend_elements.append(
            mpatches.Patch(facecolor='#E64B35', edgecolor='white', label='Highlighted')
        )

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            frameon=True,
            framealpha=0.9,
            edgecolor='none',
            fontsize=10
        )


def compute_figure_size(
    G: nx.DiGraph,
    layout: str = 'sugiyama',
    pos: Optional[Dict[int, Tuple[float, float]]] = None
) -> Tuple[float, float]:
    """Compute appropriate figure size based on graph characteristics."""
    n_nodes = G.number_of_nodes()

    if layout in ('sugiyama', 'hierarchical', 'auto'):
        if pos:
            # Use actual positions to compute bounds
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            x_range = max(x_coords) - min(x_coords) if x_coords else 1
            y_range = max(y_coords) - min(y_coords) if y_coords else 1

            # Scale to reasonable figure size
            width = max(10, min(20, x_range * 1.2 + 4))
            height = max(8, min(16, y_range * 1.2 + 4))
        else:
            # Estimate based on node count
            width = max(10, min(18, np.sqrt(n_nodes) * 1.5 + 4))
            height = max(8, min(14, np.sqrt(n_nodes) * 1.0 + 4))
    else:
        # Force-directed: roughly square
        size = max(10, min(14, np.sqrt(n_nodes) * 1.2))
        width, height = size, size * 0.85

    return (width, height)


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description='Visualize MCGS design genealogy as a DAG'
    )
    parser.add_argument(
        'database',
        type=Path,
        nargs='?',
        default=None,
        help='Path to database.json file (default: uses workspace config)'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        default=None,
        help="Workspace name for run isolation (e.g., 'run_22')"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output path (default: same dir as database, named dag_plot.png/pdf)'
    )
    parser.add_argument(
        '--layout',
        choices=['auto', 'sugiyama', 'hierarchical', 'force'],
        default='auto',
        help='Layout algorithm: sugiyama (balanced layers), hierarchical (by planner round), force (spring). Default: auto (uses sugiyama)'
    )
    parser.add_argument(
        '--color-by',
        choices=['design_id', 'objective', 'planner_round'],
        default='objective',
        help='Node coloring scheme (default: objective)'
    )
    parser.add_argument(
        '--highlight',
        type=int,
        nargs='+',
        default=None,
        help='Design IDs to highlight (larger nodes with diamond marker)'
    )
    parser.add_argument(
        '--filter-round',
        type=int,
        default=None,
        help='Only show designs up to this planner round'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide node labels'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=None,
        help='Figure size in inches (width height)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI for PNG (default: 300)'
    )

    args = parser.parse_args()

    # Set workspace for path resolution
    workspace_config.set_workspace(args.workspace)

    # Resolve database path
    if args.database is None:
        db_path = workspace_config.get_database_path()
    else:
        db_path = args.database

    # Load data
    print(f"Loading database from {db_path}...")
    data = load_database(db_path)
    print(f"Loaded {len(data['designs'])} designs")

    # Load consensus scores for objective coloring
    consensus_scores = load_consensus_scores(db_path)
    if consensus_scores:
        print(f"Loaded consensus scores for {len(consensus_scores)} designs")

    # Build graph
    G = build_graph(data, consensus_scores)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Filter by planner round if specified
    if args.filter_round is not None:
        nodes_to_remove = [
            n for n in G.nodes()
            if G.nodes[n].get('planner_round_id', 0) > args.filter_round
        ]
        G.remove_nodes_from(nodes_to_remove)
        print(f"Filtered to rounds 0-{args.filter_round}: {G.number_of_nodes()} nodes")

    # Compute layout
    layout_name = 'sugiyama' if args.layout == 'auto' else args.layout
    print(f"Computing {layout_name} layout...")
    pos = select_layout(G, args.layout)

    # Setup figure
    setup_nature_style()

    if args.figsize:
        figsize = tuple(args.figsize)
    else:
        figsize = compute_figure_size(G, args.layout, pos)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw
    print(f"Drawing DAG (color_by={args.color_by})...")
    draw_dag(
        G, pos, ax,
        color_by=args.color_by,
        highlight_ids=args.highlight,
        show_labels=not args.no_labels,
        show_colorbar=True,
        fig=fig
    )

    # Title
    ax.set_title('Design Genealogy DAG', fontsize=16, fontweight='bold', pad=10)

    plt.tight_layout()

    # Determine output paths
    if args.output:
        output_base = args.output.with_suffix('')
    else:
        output_base = db_path.parent / 'dag_plot'

    png_path = output_base.with_suffix('.png')
    pdf_path = output_base.with_suffix('.pdf')

    # Save
    fig.savefig(png_path, dpi=args.dpi, facecolor='white')
    fig.savefig(pdf_path, facecolor='white')

    print(f"\nFigures saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    plt.show()


if __name__ == '__main__':
    main()
