"""
Plot median_step vs. N scaling for all designs in the database.

Creates a publication-quality figure suitable for Nature Machine Intelligence,
showing the evolution of solver performance across design iterations with
power law fits for baseline and best designs.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace_config


def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * np.power(x, b)


def fit_power_law(x, y):
    """Fit power law using log-log linear regression for robustness."""
    log_x = np.log(x)
    log_y = np.log(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]  # exponent
    a = np.exp(coeffs[1])  # coefficient
    return a, b


def load_experiments(db_path):
    """Load all experiments from database."""
    with open(db_path, 'r') as f:
        data = json.load(f)

    experiments = []
    for design in data['designs']:
        design_id = design['design_id']
        for exp in design['experiments']:
            if 'N' in exp and 'median_step' in exp:
                experiments.append({
                    'design_id': design_id,
                    'N': exp['N'],
                    'median_step': exp['median_step']
                })

    return experiments


def load_benchmark_data(benchmark_path, design_id):
    """Load benchmark experiments and set correct design_id."""
    with open(benchmark_path, 'r') as f:
        data = json.load(f)

    experiments = []
    for exp in data['experiments']:
        if 'N' in exp and 'median_step' in exp:
            experiments.append({
                'design_id': design_id,
                'N': exp['N'],
                'median_step': exp['median_step']
            })
    return experiments


def load_reference_data(ref_path):
    """Load reference paper data from text file (space-separated N median_step)."""
    N_values = []
    step_values = []
    with open(ref_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                N_values.append(float(parts[0]))
                step_values.append(float(parts[1]))
    return np.array(N_values), np.array(step_values)


def setup_nature_style():
    """Configure matplotlib for Nature Machine Intelligence style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        'axes.titlesize': 24,
        "legend.fontsize": 20,
        "lines.linewidth": 2,
        'lines.markersize': 10,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def main():
    parser = argparse.ArgumentParser(description="Plot scaling analysis for designs")
    parser.add_argument("--workspace", type=str, default=None,
                       help="Workspace name (e.g., 'run_22')")
    parser.add_argument("--baseline-id", type=int, default=0,
                       help="Baseline design ID (default: 0)")
    parser.add_argument("--best-id", type=int, default=192,
                       help="Best design ID (default: 192)")
    parser.add_argument("--ref-data", type=str, default=None,
                       help="Path to reference data file")
    args = parser.parse_args()

    # Set workspace for path resolution
    workspace_config.set_workspace(args.workspace)

    baseline_id = args.baseline_id
    best_id = args.best_id

    # Configuration using workspace_config
    db_path = workspace_config.get_database_path()
    output_dir = workspace_config.get_database_dir()
    results_dir = workspace_config.get_results_dir()

    # Reference data path (workspace-independent)
    if args.ref_data:
        ref_data_path = Path(args.ref_data)
    else:
        ref_data_path = ROOT / 'latex_manuscript' / 'figures' / 'baseline_data_chesson' / '0_steps_0.txt'

    benchmark_baseline_path = results_dir / f'benchmark_solver_{baseline_id}_schedule_0.json'
    benchmark_best_path = results_dir / f'benchmark_solver_{best_id}_schedule_99999.json'

    # Load data
    print(f"Loading experiments from {db_path}...")
    experiments = load_experiments(db_path)
    print(f"Loaded {len(experiments)} experiments")

    # Load benchmark data and replace database experiments for baseline and best designs
    benchmark_baseline = load_benchmark_data(benchmark_baseline_path, baseline_id)
    benchmark_best = load_benchmark_data(benchmark_best_path, best_id)
    experiments = [e for e in experiments if e['design_id'] not in (baseline_id, best_id)]
    experiments.extend(benchmark_baseline)
    experiments.extend(benchmark_best)
    print(f"Loaded {len(benchmark_baseline)} benchmark experiments for design {baseline_id}")
    print(f"Loaded {len(benchmark_best)} benchmark experiments for design {best_id}")

    # Load reference paper data
    print(f"Loading reference data from {ref_data_path}...")
    ref_N, ref_steps = load_reference_data(ref_data_path)
    print(f"Loaded {len(ref_N)} reference data points")
    # ref_mask = ref_N <= 1000
    # ref_N = ref_N[ref_mask]
    # ref_steps = ref_steps[ref_mask]

    # Convert to arrays
    design_ids = np.array([e['design_id'] for e in experiments])
    Ns = np.array([e['N'] for e in experiments])
    median_steps = np.array([e['median_step'] for e in experiments])

    # Get unique design range for colormap
    max_design_id = design_ids.max()

    # Extract baseline and best design data
    baseline_mask = design_ids == baseline_id
    best_mask = design_ids == best_id
    other_mask = ~baseline_mask & ~best_mask

    baseline_N = Ns[baseline_mask]
    baseline_steps = median_steps[baseline_mask]
    best_N = Ns[best_mask]
    best_steps = median_steps[best_mask]

    # Fit power laws
    a_baseline, b_baseline = fit_power_law(baseline_N, baseline_steps)
    a_best, b_best = fit_power_law(best_N, best_steps)
    a_ref, b_ref = fit_power_law(ref_N, ref_steps)

    print(f"\nPower law fits:")
    print(f"  Reference paper: steps = {a_ref:.2f} * N^{b_ref:.3f}")
    print(f"  Baseline (design {baseline_id}): steps = {a_baseline:.2f} * N^{b_baseline:.3f}")
    print(f"  Best (design {best_id}): steps = {a_best:.2f} * N^{b_best:.3f}")
    print(f"  Exponent reduction: {b_baseline:.3f} -> {b_best:.3f} ({(1 - b_best/b_baseline)*100:.1f}% improvement)")

    # Setup figure
    setup_nature_style()
    fig, ax = plt.subplots()

    # Create colormap normalization
    norm = Normalize(vmin=0, vmax=max_design_id)
    cmap = plt.cm.viridis

    # Plot all other experiments as small dots with color gradient
    scatter = ax.scatter(
        Ns[other_mask],
        median_steps[other_mask],
        c=design_ids[other_mask],
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.5,
        edgecolors='none',
        zorder=1,
        rasterized=True  # Rasterize for smaller file size
    )

    # Plot baseline design with distinct style
    ax.scatter(
        baseline_N, baseline_steps,
        c='#666666',
        s=35,
        marker='o',
        edgecolors='white',
        linewidths=0.5,
        label=f'Baseline (design 0)',
        zorder=3
    )

    # Plot best design with distinct style
    ax.scatter(
        best_N, best_steps,
        c='#E64B35',  # Nature-style red
        s=45,
        marker='D',
        edgecolors='white',
        linewidths=0.5,
        label=f'Best (design {best_id})',
        zorder=3
    )

    # Plot reference paper data
    ax.scatter(
        ref_N, ref_steps,
        c='#4DBBD5',  # Nature-style cyan
        s=35,
        marker='s',
        edgecolors='white',
        linewidths=0.5,
        label='Reference',
        zorder=3
    )

    # Plot power law fits
    N_fit = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 100)

    ax.plot(
        N_fit, power_law(N_fit, a_baseline, b_baseline),
        color='#666666',
        linestyle='--',
        linewidth=1.5,
        label=f'$O(N^{{{b_baseline:.2f}}})$',
        zorder=2
    )

    ax.plot(
        N_fit, power_law(N_fit, a_best, b_best),
        color='#E64B35',
        linestyle='-',
        linewidth=1.5,
        label=f'$O(N^{{{b_best:.2f}}})$',
        zorder=2
    )

    ax.plot(
        N_fit, power_law(N_fit, a_ref, b_ref),
        color='#4DBBD5',
        linestyle=':',
        linewidth=1.5,
        label=f'$O(N^{{{b_ref:.2f}}})$ (ref)',
        zorder=2
    )

    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Problem Size $N$')
    ax.set_ylabel('Median Steps')

    # Add colorbar
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
        aspect=30,
        shrink=0.9,
    )
    cbar.set_label('Design ID', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Legend - positioned to avoid data
    legend = ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor='none',
        handletextpad=0.3,
        labelspacing=0.3,
        fontsize=14
    )
    legend.get_frame().set_linewidth(0)

    # Fine-tune layout
    plt.tight_layout()

    # Save figures
    png_path = output_dir / 'scaling_plot.png'
    pdf_path = output_dir / 'scaling_plot.pdf'

    fig.savefig(png_path, dpi=300, facecolor='white')
    fig.savefig(pdf_path, facecolor='white')

    print(f"\nFigures saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    plt.show()


if __name__ == '__main__':
    main()
