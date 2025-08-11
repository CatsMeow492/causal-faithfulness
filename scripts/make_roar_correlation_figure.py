#!/usr/bin/env python3
"""
Create a ROAR vs Faithfulness correlation scatter from a quick JSON export (Task 5.1 item 2).

Input: results/roar_corr_quick.json created by inline extraction.
Output: figures/roar_faithfulness_correlation.(png|pdf)
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Make ROAR correlation scatter figure")
    parser.add_argument("--input", type=str, default="results/roar_corr_quick.json")
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"Missing input: {p}")
        return
    with open(p, 'r') as f:
        entries = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    # Single-page grid
    n = len(entries)
    ncols = min(3, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    ax_flat = axes if isinstance(axes, list) else axes.flatten()

    for i, e in enumerate(entries):
        ax = ax_flat[i]
        x = e['roar_scores']
        y = e['faithfulness_scores']
        ax.scatter(x, y, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_title(f"{e['explainer_name']} (r={e.get('pearson_r', 0):.3f})")
        ax.set_xlabel('ROAR drop proxy')
        ax.set_ylabel('Faithfulness F')
        ax.set_ylim(0, 1.0)
    for j in range(len(entries), len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.suptitle('ROAR vs Faithfulness Correlation (Quick View)')
    plt.tight_layout()
    out_png = Path(args.outdir) / 'roar_faithfulness_correlation.png'
    out_pdf = Path(args.outdir) / 'roar_faithfulness_correlation.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', out_png, out_pdf)


if __name__ == "__main__":
    main()


