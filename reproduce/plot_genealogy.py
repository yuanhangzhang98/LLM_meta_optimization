"""Draw Fig. 3a from recorded parent references and raw consensus scores.

Adapted from scripts/plot_dag.py. Graphviz ``dot`` recomputes the vertical DAG
layout; exact positions may vary with the Graphviz version. No coordinates or
experimental measurements are recovered from previous artwork.

Runtime dependencies: NumPy, Matplotlib, and the Graphviz ``dot`` executable.
"""
from collections import defaultdict
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


def graphviz_positions(records, dot='dot'):
    """Recompute dot/TB positions with the original script's layout options."""
    designs = records['designs']
    node_ids = [int(d['design_id']) for d in designs]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError('Duplicate design ID in genealogy records.')
    node_set = set(node_ids)
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for design in designs:
        child = int(design['design_id'])
        for reference in design.get('reference_weights', []):
            parent = int(reference['design_id'])
            weight = float(reference['weight'])
            if parent not in node_set:
                raise ValueError(f'Missing parent design {parent}.')
            outgoing[parent].append((child, weight))
            incoming[child].append(parent)
    # NetworkX's original iteration visits nodes, then their outgoing edges.
    edges = [(parent, child, weight) for parent in node_ids
             for child, weight in outgoing[parent]]
    source = ['strict digraph genealogy {',
              'graph [rankdir=TB, ranksep=0.5, nodesep=0.3];']
    source.extend(f'"{node}";' for node in node_ids)
    source.extend(f'"{parent}" -> "{child}" [weight="{weight}"];'
                  for parent, child, weight in edges)
    source.append('}')
    try:
        result = subprocess.run([dot, '-Tplain'], input='\n'.join(source),
                                text=True, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Graphviz dot is required to regenerate Fig. 3a.') from exc
    positions = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == 'node':
            positions[int(fields[1].strip('"'))] = (float(fields[2]), float(fields[3]))
    if set(positions) != node_set:
        raise ValueError('Graphviz did not return exactly the recorded nodes.')
    return node_ids, edges, incoming, positions


def draw_genealogy(fig, records, panel=(0.0, 0.0, 1.0, 1.0), dot='dot'):
    """Draw the genealogy in a figure-relative rectangular panel; return axes.

    A standalone 400 by 360 point figure has panel=(0,0,1,1). For an
    864 by 360 point combined figure use panel=(0,0,400/864,1).
    """
    node_ids, edges, incoming, positions = graphviz_positions(records, dot)
    scores = {int(k): float(v) for k, v in records['consensus_scores'].items()}
    if set(scores) != set(node_ids):
        raise ValueError('Every recorded node must have its raw consensus score.')
    x0, y0, width, height = panel
    ax = fig.add_axes([x0 + .065*width, y0 + .015*height, .71*width, .96*height])
    # Original plot areas were scaled down when the panel was placed in the
    # manuscript; scale paths and markers consistently to the 400x360pt panel.
    scale = fig.get_figheight()*72*height / 705.813
    base_size = 800 - len(node_ids) if len(node_ids) > 200 else 800
    inverse = 1 / np.maximum(np.array([scores[n] for n in node_ids]), .01)
    fraction = (inverse-inverse.min())/max(float(inverse.max()-inverse.min()), 1e-9)
    sizes = max(80, base_size) * (.15 + 4.85*fraction) * scale**2
    for parent, child, weight in edges:
        parents = incoming[child]
        radius = .15*(parents.index(parent)-(len(parents)-1)/2) if len(parents)>1 else 0
        ax.add_patch(FancyArrowPatch(
            positions[parent], positions[child], arrowstyle='-|>',
            connectionstyle=f'arc3,rad={radius}',
            linewidth=(.5+2.5*weight)*scale, alpha=.3+.5*weight,
            color='#505050', mutation_scale=12*scale,
            shrinkA=2*scale, shrinkB=2*scale, zorder=1))
    xy=np.array([positions[n] for n in node_ids])
    collection = ax.scatter(xy[:,0], xy[:,1], c=node_ids, cmap='viridis',
                            norm=Normalize(min(node_ids),max(node_ids)),
                            s=sizes, marker='o', edgecolors='white',
                            linewidths=.8*scale, zorder=2)
    ax.margins(.055)
    ax.axis('off')
    cax=fig.add_axes([x0+.84*width,y0+.12*height,.028*width,.76*height])
    cbar=fig.colorbar(collection,cax=cax)
    cbar.set_label('Design ID', fontsize=17)
    cbar.ax.tick_params(labelsize=14,width=.6,length=3)
    cbar.outline.set_linewidth(.6)
    # The original composite used a manual callout; anchor it to the new layout.
    if 340 in positions:
        ax.annotate('Best (design 340)', xy=positions[340], xycoords='data',
                    xytext=(.56,.035), textcoords='axes fraction',
                    fontsize=14, ha='center', va='bottom',
                    arrowprops={'arrowstyle':'->','color':'black','lw':1.0},
                    zorder=4)
    fig.text(x0+.035*width,y0+.925*height,'a',fontsize=28,fontweight='bold')
    return ax


def render_genealogy(records, output=None):
    """Return a standalone vector PDF panel, optionally writing it to disk."""
    fig=plt.figure(figsize=(400/72,360/72))
    draw_genealogy(fig,records)
    buffer=io.BytesIO()
    fig.savefig(buffer,format='pdf',metadata={'Author':'Yuan-Hang Zhang, Chesson Sipling, Massimiliano Di Ventra',
                'Title':'Design genealogy','Creator':'Recorded genealogy and Graphviz',
                'CreationDate':None,'ModDate':None})
    if output:
        Path(output).write_bytes(buffer.getvalue())
        fig.savefig(Path(output).with_suffix('.png'),dpi=150)
    plt.close(fig)
    return buffer.getvalue()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    render_genealogy(json.loads(args.data.read_text()),args.output)
