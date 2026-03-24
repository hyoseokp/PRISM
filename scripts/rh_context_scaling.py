"""Figure: R_h vs Context Length (retrieval head fraction scaling).
Reads measured data from results/retrieval_heads/*.json."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_config import (OUT_DIR, COLOR_PRIMARY, COLOR_QUATERNARY,
                         COLOR_QUINARY, COLOR_THRESHOLD)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'retrieval_heads')


def load_qwen25_bf16():
    path = os.path.join(RESULTS_DIR, 'qwen25_7b_bf16_128k.json')
    with open(path) as f:
        d = json.load(f)
    ctxs, pcts, means = [], [], []
    for ctx_str in sorted((k for k in d if k.isdigit()), key=int):
        r = d[ctx_str]
        ctx = int(ctx_str)
        m = np.array(r['rh_matrix'])
        pct = 100 * (m > 0.3).sum() / m.size
        ctxs.append(ctx)
        pcts.append(round(pct, 1))
        means.append(round(float(m.mean()), 4))
    # Add 256K from extreme results if available
    ext_path = os.path.join(RESULTS_DIR, 'qwen25_7b_bf16_extreme.json')
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext = json.load(f)
        if 'contexts' in ext:
            for ctx_str, r in ext['contexts'].items():
                ctx = int(ctx_str)
                if ctx not in ctxs:
                    ctxs.append(ctx)
                    pcts.append(r.get('rh_pct', 0))
                    means.append(r.get('mean_rh', 0))
    return ctxs, pcts, means


def load_qwen3_bf16():
    path = os.path.join(RESULTS_DIR, 'step2_q3_rh.json')
    with open(path) as f:
        d = json.load(f)
    ctxs, pcts = [], []
    for ctx_str in sorted((k for k in d if k.isdigit()), key=int):
        r = d[ctx_str]
        ctx = int(ctx_str)
        if 'rh_matrix' in r:
            m = np.array(r['rh_matrix'])
            pct = 100 * (m > 0.3).sum() / m.size
        elif 'pct_t03' in r:
            pct = r['pct_t03']
        else:
            continue
        ctxs.append(ctx)
        pcts.append(round(pct, 1))
    return ctxs, pcts


def load_qwen25_4bit():
    path = os.path.join(RESULTS_DIR, 'step1_q25_rh.json')
    with open(path) as f:
        d = json.load(f)
    ctxs, pcts = [], []
    for ctx_str in sorted((k for k in d if k.isdigit()), key=int):
        r = d[ctx_str]
        ctx = int(ctx_str)
        if 'rh_matrix' in r:
            m = np.array(r['rh_matrix'])
            pct = 100 * (m > 0.3).sum() / m.size
        elif 'pct_t03' in r:
            pct = r['pct_t03']
        else:
            continue
        ctxs.append(ctx)
        pcts.append(round(pct, 1))
    return ctxs, pcts


def main():
    bf16_ctx, bf16_pct, bf16_mean = load_qwen25_bf16()
    q3_ctx, q3_pct = load_qwen3_bf16()
    q4_ctx, q4_pct = load_qwen25_4bit()

    ctx_labels = {2048:'2K', 4096:'4K', 8192:'8K', 16384:'16K', 32768:'32K',
                  65536:'64K', 131072:'128K', 262144:'256K'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 5.0),
                                    gridspec_kw={'height_ratios': [3, 2]})

    # Panel (a): R_h percentage
    ax1.plot(bf16_ctx, bf16_pct, 'o-', color=COLOR_PRIMARY,
             label='Qwen2.5-7B (bf16)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)
    ax1.plot(q3_ctx, q3_pct, 's--', color=COLOR_QUATERNARY,
             label='Qwen3-8B (bf16)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)
    ax1.plot(q4_ctx, q4_pct, '^:', color=COLOR_QUINARY,
             label='Qwen2.5-7B (4-bit)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)

    ax1.axhline(y=90, color=COLOR_THRESHOLD, linestyle='--', linewidth=0.8,
                alpha=0.7, label='90% threshold')

    ax1.set_xscale('log', base=2)
    ax1.set_xlim(1400, 380000)
    ax1.set_ylim(78, 101)
    ax1.set_ylabel(r'Retrieval Head Fraction $R_h$ (%)')
    ax1.set_xlabel('Context Length (tokens)')
    ax1.set_xticks(bf16_ctx)
    ax1.set_xticklabels([ctx_labels.get(c, str(c)) for c in bf16_ctx],
                         rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes,
             fontsize=10, fontweight='bold')

    # Panel (b): Mean R_h
    ax2.plot(bf16_ctx, bf16_mean, 'o-', color=COLOR_PRIMARY,
             label='Qwen2.5-7B (bf16)',
             markerfacecolor='white', markeredgewidth=1.0, zorder=5)

    ax2.set_xscale('log', base=2)
    ax2.set_xlim(1400, 380000)
    ax2.set_ylim(0.45, 1.0)
    ax2.set_ylabel(r'Mean $\overline{R}_h$')
    ax2.set_xlabel('Context Length (tokens)')
    ax2.set_xticks(bf16_ctx)
    ax2.set_xticklabels([ctx_labels.get(c, str(c)) for c in bf16_ctx],
                         rotation=45, ha='right')
    ax2.legend(loc='upper left')
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes,
             fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_rh_context_scaling.pdf')
    fig.savefig(path)
    print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
