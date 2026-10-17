"""Regenerate both panels of Figure 3 from stored experiment and genealogy records.

Run: python reproduce/restore_design_figure.py
Dependencies: numpy, matplotlib, and the Graphviz `dot` executable.
No solver experiments, LLM calls, original figure assets, or network access.

The baseline uses its original 16-record benchmark. Design 340 uses all 16
campaign records and three fresh medians in one unweighted log-log fit;
repeated sizes remain separate observations. Only fresh measurements have
error bars (95% batch-bootstrap intervals). See the attachment's figuredata/figure_data/provenance.json.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT = Path(__file__).resolve().parent


def load_data(directory):
    return {
        'campaign': json.loads((directory / 'campaign_records.json').read_text()),
        'baseline': json.loads((directory / 'benchmark_solver_0_schedule_0.json').read_text()),
        'fresh': json.loads((directory / 'fresh_measurements.json').read_text()),
    }


def historical_points(experiments):
    return np.array([[row['N'], row['median_step']] for row in experiments
                     if 'N' in row and 'median_step' in row], dtype=float)


def power_law_fit(points):
    """Original plot_scaling.py regression and covariance calculation."""
    if not np.all(np.isfinite(points)) or np.any(points <= 0):
        raise ValueError('Power-law fits require finite positive measurements.')
    coefficients, covariance = np.polyfit(np.log(points[:, 0]),
                                         np.log(points[:, 1]), 1, cov=True)
    exponent, intercept = coefficients
    return float(np.exp(intercept)), float(exponent), covariance


def scaling_series(data):
    designs = data['campaign']['designs']
    baseline = historical_points(data['baseline']['experiments'])
    original340 = historical_points(next(d for d in designs if d['design_id'] == 340)['experiments'])
    fresh340 = np.array([[row['N'], row['estimate']['km_median']]
                         for row in data['fresh'] if row['solver'] == 'design340'], dtype=float)
    background = np.array([[d['design_id'], row['N'], row['median_step']]
                           for d in designs if d['design_id'] not in (0, 340)
                           for row in d['experiments']
                           if 'N' in row and 'median_step' in row], dtype=float)
    return baseline, original340, np.vstack([original340, fresh340]), background


def draw_scaling(fig, data):
    # Keep the approved right-panel geometry on the 864 x 360 point canvas.
    ax = fig.add_axes([(400 + 464 * .20) / 864, .18, 464 * .775 / 864, .775])
    baseline, original340, combined340, background = scaling_series(data)
    ax.scatter(background[:, 1], background[:, 2], c=background[:, 0],
               cmap='viridis', vmin=0, vmax=413, s=8, alpha=.48,
               linewidths=0, rasterized=True, zorder=1)
    colors = {'baseline': '#666666', 'design340': '#E64B35'}
    fits = {}
    for name, marker, points, fit_points in [
            ('baseline', 'o', baseline, baseline),
            ('design340', 'D', original340, combined340)]:
        coefficient, exponent, covariance = power_law_fit(fit_points)
        xs = np.geomspace(fit_points[:, 0].min(), fit_points[:, 0].max(), 200)
        log_x = np.log(xs)
        log_y = np.log(coefficient) + exponent * log_x
        if name == 'baseline':
            variance = (covariance[1, 1] + 2 * covariance[0, 1] * log_x
                        + covariance[0, 0] * log_x**2)
            sigma = np.sqrt(np.maximum(variance, 0))
            ax.fill_between(xs, np.exp(log_y - sigma), np.exp(log_y + sigma),
                            color=colors[name], alpha=.14, lw=0, zorder=1.4)
        ax.plot(xs, np.exp(log_y), color=colors[name],
                ls='--' if name == 'baseline' else '-', lw=1.6, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], s=32 if marker == 'o' else 43,
                   marker=marker, facecolor=colors[name], edgecolor='white', lw=.6, zorder=3)
        fits[name] = {'coefficient': coefficient, 'exponent': exponent,
                      'n_points': len(fit_points),
                      'range_N': [float(fit_points[:, 0].min()), float(fit_points[:, 0].max())]}
    for name, color, marker in [('tuned_baseline', '#111111', 's'),
                                 ('design340', colors['design340'], 'D')]:
        rows = [row for row in data['fresh'] if row['solver'] == name]
        xs = np.array([row['N'] for row in rows])
        ys = np.array([row['estimate']['km_median'] for row in rows])
        lo = np.array([row['estimate']['ci95'][0] for row in rows])
        hi = np.array([row['estimate']['ci95'][1] for row in rows])
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=marker,
                    ms=8.5 if name == 'tuned_baseline' else np.sqrt(43),
                    mec='white', mew=.65, mfc=color, ecolor=color,
                    elinewidth=1.6, capsize=3.3, zorder=5)
    handles = [
        Line2D([], [], marker='o', mfc=colors['baseline'], mec='white',
               color=colors['baseline'], ls='--', lw=1.3, ms=6),
        Line2D([], [], marker='s', mfc='#111111', mec='white', color='#111111', ls='none', ms=8),
        Line2D([], [], marker='D', mfc=colors['design340'], mec='white',
               color=colors['design340'], ls='-', lw=1.3, ms=6),
    ]
    labels = [rf'Baseline ($N^{{{fits["baseline"]["exponent"]:.2f}}}$)',
              'Tuned baseline', rf'Design 340 ($N^{{{fits["design340"]["exponent"]:.2f}}}$)']
    ax.legend(handles, labels, loc='upper left', frameon=True, facecolor='white',
              edgecolor='none', framealpha=.96, fontsize=10.6, handlelength=2,
              handletextpad=.6, labelspacing=.55, borderpad=.6)
    ax.set(xscale='log', yscale='log', xlim=(7.7, 3400), ylim=(4, 1.4e7),
           xlabel=r'Problem Size $N$', ylabel='Median Steps')
    ax.set_xticks([10, 100, 1000])
    ax.set_yticks([10, 100, 1000, 10000, 100000, 1000000, 10000000])
    fig.text((400 + 464 * .027) / 864, .925, 'b', fontsize=28, fontweight='bold')
    return {'fits': fits, 'background_points': len(background),
            'fresh_points_with_error_bars': len(data['fresh'])}


def main():
    from plot_genealogy import draw_genealogy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=SCRIPT / 'artifacts/data/figuredata/figure_data')
    parser.add_argument('--output', type=Path, default=SCRIPT / 'output/figure3.pdf')
    parser.add_argument('--summary', type=Path, help='Optional JSON record of regenerated counts and fits.')
    args = parser.parse_args()
    if not (args.data_dir / 'campaign_records.json').is_file():
        parser.error('Extract supporting_data.zip into reproduce/artifacts/, or provide --data-dir.')
    data = load_data(args.data_dir)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'axes.labelsize': 21, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .9})
    fig = plt.figure(figsize=(12, 5))
    draw_genealogy(fig, data['campaign'], panel=(0, 0, 400/864, 1))
    summary = draw_scaling(fig, data)
    designs = data['campaign']['designs']
    summary['genealogy'] = {'nodes': len(designs),
                            'weighted_edges': sum(len(d['reference_weights']) for d in designs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format='pdf', metadata={
        'Author': 'Yuan-Hang Zhang, Chesson Sipling, Massimiliano Di Ventra', 'Title': 'Algorithm discovery: genealogy and performance',
        'Creator': 'Reproducible figure script', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote {args.output}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
