"""Draw separate platform and local charts from the published aggregate values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'docs/results_summary.json').read_text())
    rows = data['submissions']
    labels = [row['submission_id'] for row in rows]
    x = np.arange(len(rows))
    output = root / 'docs/figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout='constrained')
    for shift, key, label, color in [(-.16,'public_score_decimal','Public','#226a9d'),
                                      (.16,'private_score_decimal','Private','#c87336')]:
        values = [float(row[key]) for row in rows]
        bars = ax.bar(x+shift, values, width=.30, label=label, color=color)
        ax.bar_label(bars, fmt='%.4f', fontsize=10, padding=4)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, .85), ylabel='Competition score',
           title='Recorded competition results')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel('Author-recorded final results: 2026-09-21. Platform data unavailable for rerun.', fontsize=9)
    fig.savefig(output/'platform_results.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout='constrained')
    series=[('presence_decimal','Presence','#226a9d'),
            ('shape_diagnostic_decimal','Shape diagnostic','#789383'),
            ('reference_total_decimal','Reference total','#c87336')]
    for key,label,color in series:
        values=[float(row['local_holdout'][key]) for row in rows]
        ax.plot(x,values,marker='o',label=label,color=color,linewidth=2)
        for i,v in enumerate(values):ax.annotate(f'{v:.4f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=labels,ylim=(.42,1.08),ylabel='Local diagnostic value',
           title='Repeated development holdout — not independent test')
    ax.legend(loc='lower right',frameon=False,ncol=3,fontsize=9)
    ax.grid(axis='y',alpha=.2)
    fig.supxlabel('343 keys / 319 labeled; Shape subset: 89 dates, 9 sites. Full Shape and Total: NA.\nReference total = 0.6 Presence + 0.4 Shape diagnostic; final ensemble OOF unavailable.',fontsize=9)
    fig.savefig(output/'local_diagnostics.png',dpi=180)
    plt.close(fig)

    by_id = {row['submission_id']: row for row in rows}
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout='constrained')
    for shift, name, color in [(-.17, 'S029', '#226a9d'), (.17, 'S030', '#c87336')]:
        counts = by_id[name]['local_holdout']['confusion']
        bars = ax.bar(np.arange(2)+shift, [counts['fp'], counts['fn']],
                      width=.32, label=name, color=color)
        ax.bar_label(bars, padding=5, fontsize=14)
    ax.set(xticks=[0,1], xticklabels=['False positives / 199 negatives', 'False negatives / 120 positives'],
           ylim=(0,16), ylabel='Site-date count (lower is better)',
           title='S029 → S030: changing the Presence ensemble')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.supxlabel('Repeated development holdout: 343 keys / 319 labeled.\nCounts use final valid polygon presence; this is not a boundary-quality comparison.', fontsize=9)
    fig.savefig(output/'s030-error-comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10.8,4.8),layout='constrained')
    names = ['S030', 'S032']
    panels = [
        ('Repeated development holdout', [('Reference total', 'reference_total_decimal', '#789383')]),
        ('Competition evaluation', [('Public', 'public_score_decimal', '#226a9d'),
                                    ('Private', 'private_score_decimal', '#c87336')]),
    ]
    for ax, (title, series) in zip(axes, panels):
        for label, key, color in series:
            values = [float(by_id[name]['local_holdout'][key] if key=='reference_total_decimal'
                            else by_id[name][key]) for name in names]
            ax.plot([0,1],values,'o-',color=color,label=label,linewidth=2.5,markersize=8)
            for x, value in enumerate(values):
                offset = -18 if label == 'Private' else 10
                ax.annotate(f'{value:.6f}',(x,value),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=11)
        ax.set(xticks=[0,1],xticklabels=names,xlim=(-.4,1.4),ylim=(.65,.86),title=title)
        ax.grid(axis='y',alpha=.15)
        ax.legend(loc='lower left',frameon=False)
    axes[0].set_ylabel('Reference total (full Shape / Total: NA)')
    axes[1].set_ylabel('Platform score')
    fig.supxlabel('Different evaluation sets: compare candidates within each panel, not score levels across panels.\nSource: recorded final results, 2026-09-21. Holdout is not an independent test.',fontsize=9)
    fig.savefig(output/'local-platform-comparison.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
