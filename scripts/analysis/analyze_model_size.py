"""Model-size plots for rotMNIST: finetune vs overparameterisation vs ER.

Three strategies on the same axes, per angle:

- **Finetune** — fast momentum SGD (α=0), no memory, varying width.
- **Model size** — slow_ema (α=1) carry arm, varying width and horizon.
- **Experience replay** — ER at the Mammoth ``best_args`` LR (model er).

Each arm sits at its own LR: memory / fine-tune arms at the ReplaySGD
best LR, ER at the literature LR.

Per angle writes ``results/plan_analysis/rotmnist_angle<θ>/model_size/``:

- ``model_size.png`` — three-panel figure:
    * (a) ACC vs FRGT scatter, families colour-coded; Pareto front highlighted.
    * (b) ACC vs width (log-x), slow_ema lines per r, fast_only line, ER
      horizontal reference bands.
    * (c) FRGT vs width (log-x), same structure.
- ``model_size.csv`` — one row per cell with mean ± std and Pareto flag.

Run from the repo root:

  python scripts/analysis/analyze_model_size.py [--out_dir results/plan_analysis]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_pareto_best_lr as PP  # noqa: E402  (shared family colours)
from plot_labels import place_labels  # noqa: E402

ENTITY = '<wandb-entity>'
PROJECT = 'cl_optimizer_memory'
ANGLES = (9, 30)
WIDTHS = (100, 500, 1000)
HORIZONS = (1.0, 3.0, 10.0)
KS = (200, 500, 5120)
# Each arm at its own LR. Memory and fine-tune arms: the best LR of the
# ReplaySGD lr sweep. ER: the literature LR (Mammoth utils/best_args.py, model er).
LR_FINETUNE = {9: 0.1, 30: 0.3}
LR_SLOW_EMA = {9: 0.1, 30: 0.3}
LR_REPLAY = 0.2
ER_GROUP = 'plan_rotmnist_replaysgd_er'



def _num(cfg, key, default=None):
    v = cfg.get(key, default)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def fetch(angle):
    """Return a list of rows classified by (family, width, horizon, K)."""
    import wandb
    api = wandb.Api(timeout=60)
    rows = []

    def push(family, width, horizon, K, cfg, summary, run_id):
        acc = summary.get('ACC'); frgt = summary.get('FRGT')
        seed = cfg.get('seed')
        if acc is None or frgt is None or seed is None:
            return
        # A run can finish with NaN ACC; drop it.
        if not (np.isfinite(float(acc)) and np.isfinite(float(frgt))):
            return
        rows.append({
            'family': family, 'width': int(width),
            'horizon': horizon, 'K': K, 'seed': int(seed),
            'ACC': float(acc), 'FRGT': float(frgt), 'run_id': run_id,
        })

    # (1) Finetune w=100: keep only runs at LR_FINETUNE, so the cell is a
    # seed mean at a single LR.
    g = f'plan_rotmnist_ablation_a{angle}_lattice_control'
    for r in api.runs(f'{ENTITY}/{PROJECT}', filters={'group': g}, per_page=100):
        if r.state != 'finished':
            continue
        c = {k: v for k, v in r.config.items() if not k.startswith('_')}
        if abs(_num(c, 'lr') - LR_FINETUNE[angle]) > 1e-9:
            continue
        push('finetune', 100, None, None, c, r.summary, r.id)
    # (2) Finetune w=500 / w=1000: fast_only.
    g = f'plan_rotmnist_main_width_a{angle}_fast_only'
    for r in api.runs(f'{ENTITY}/{PROJECT}', filters={'group': g}, per_page=100):
        if r.state != 'finished':
            continue
        c = {k: v for k, v in r.config.items() if not k.startswith('_')}
        push('finetune', int(_num(c, 'width') or 100), None, None,
             c, r.summary, r.id)

    # (3) Model size w=100: α=1 slow_ema cells only, at LR_SLOW_EMA.
    g = f'plan_rotmnist_ablation_a{angle}_lattice_memory'
    for r in api.runs(f'{ENTITY}/{PROJECT}', filters={'group': g}, per_page=200):
        if r.state != 'finished':
            continue
        c = {k: v for k, v in r.config.items() if not k.startswith('_')}
        if c.get('ablation') != 'slow_ema':
            continue
        if float(c.get('alpha', 0) or 0) != 1.0:
            continue
        if abs(_num(c, 'lr') - LR_SLOW_EMA[angle]) > 1e-9:
            continue
        horizon = _num(c, 'horizon_ratio')
        if horizon not in HORIZONS:
            continue
        push('model_size', 100, horizon, None, c, r.summary, r.id)
    # (4) Model size w=500 / w=1000: slow_ema.
    g = f'plan_rotmnist_main_width_a{angle}_slow_ema'
    for r in api.runs(f'{ENTITY}/{PROJECT}', filters={'group': g}, per_page=100):
        if r.state != 'finished':
            continue
        c = {k: v for k, v in r.config.items() if not k.startswith('_')}
        horizon = _num(c, 'horizon_ratio')
        if horizon not in HORIZONS:
            continue
        push('model_size', int(_num(c, 'width') or 100), horizon, None,
             c, r.summary, r.id)

    # (5) Replay at the literature (Mammoth best_args, model er) LR. One run
    # per (K, seed): the latest finished one.
    latest = {}
    for r in api.runs(f'{ENTITY}/{PROJECT}', filters={'group': ER_GROUP}, per_page=200):
        if r.state != 'finished':
            continue
        c = {k: v for k, v in r.config.items() if not k.startswith('_')}
        if int(_num(c, 'angle') or -1) != angle or int(c.get('replay', 0) or 0) != 1:
            continue
        if abs(_num(c, 'lr') - LR_REPLAY) > 1e-9:
            continue
        K = int(_num(c, 'replay_buffer_size') or 0)
        if K not in KS:
            continue
        key = (K, int(c.get('seed')))
        if key not in latest or r.created_at > latest[key][0].created_at:
            latest[key] = (r, c)
    for (K, _), (r, c) in latest.items():
        push('replay', 100, None, K, c, r.summary, r.id)
    return rows


def _sample_std(x):
    """Sample std over seeds (ddof=1); 0 for a single seed."""
    return float(np.std(x, ddof=1)) if np.size(x) > 1 else 0.0


def aggregate(rows):
    buckets = defaultdict(lambda: {'ACC': [], 'FRGT': []})
    for r in rows:
        key = (r['family'], r['width'], r['horizon'], r['K'])
        buckets[key]['ACC'].append(r['ACC'])
        buckets[key]['FRGT'].append(r['FRGT'])
    cells = []
    for (family, width, horizon, K), v in buckets.items():
        acc = np.asarray(v['ACC']); frgt = np.asarray(v['FRGT'])
        cells.append({
            'family': family, 'width': width, 'horizon': horizon, 'K': K,
            'ACC_mean': float(acc.mean()), 'ACC_std': _sample_std(acc),
            'FRGT_mean': float(frgt.mean()), 'FRGT_std': _sample_std(frgt),
            'n_seeds': int(acc.size),
            'label': _label(family, width, horizon, K),
        })
    _mark_pareto(cells)
    return cells


def _label(family, width, horizon, K):
    if family == 'finetune':
        return f'finetune w={width}'
    if family == 'model_size':
        return f'slow_ema w={width}, r={horizon:g}'
    return f'replay K={K}'


def _mark_pareto(cells):
    for c in cells:
        c['pareto'] = True
    for i, ci in enumerate(cells):
        for j, cj in enumerate(cells):
            if i == j:
                continue
            if (cj['FRGT_mean'] <= ci['FRGT_mean']
                    and cj['ACC_mean'] >= ci['ACC_mean']
                    and (cj['FRGT_mean'] < ci['FRGT_mean']
                         or cj['ACC_mean'] > ci['ACC_mean'])):
                ci['pareto'] = False
                break


# -------- plotting ---------------------------------------------------------
# Same visual language as the Pareto and ablation figures (plot_pareto_best_lr):
# fine-tune black, slow memory in blues by r, ER in reds by K, legend columns
# with bold headers above the panels; panel (a) codes width by marker size.

WIDTH_SIZE = {100: 30, 500: 65, 1000: 115}
FAMILY = {'finetune': 'fine-tune', 'model_size': 'slow memory', 'replay': 'ER'}


def _style(c):
    """Pareto-figure view of a model-size cell (family, colour, marker)."""
    view = {'family': FAMILY[c['family']], 'horizon_ratio': c['horizon'], 'K': c['K'],
            'frgt': c['FRGT_mean'], 'acc': c['ACC_mean'],
            'frgt_std': c['FRGT_std'], 'acc_std': c['ACC_std'], 'width': c['width']}
    view['color'] = PP.family_color(view)
    view['marker'] = PP.FAMILY_MARKER[view['family']]
    return view


def _point_text(v):
    if v['family'] == 'ER':
        return f'K={v["K"]}'
    if v['family'] == 'slow memory':
        return f'r={v["horizon_ratio"]:g},w={v["width"]}'
    return f'FT w={v["width"]}'


def _draw_cells(ax, views, front, scale=1.0):
    items = []
    for v, on in zip(views, front):
        ax.errorbar(v['frgt'], v['acc'], xerr=v['frgt_std'], yerr=v['acc_std'], fmt='none',
                    ecolor=v['color'], alpha=0.35, lw=0.8, zorder=2)
        ax.scatter(v['frgt'], v['acc'], marker=v['marker'],
                   s=WIDTH_SIZE.get(v['width'], 60) * scale, color=v['color'],
                   edgecolors='black' if on else 'white', linewidths=1.4 if on else 0.6, zorder=3)
        items.append(dict(x=v['frgt'], y=v['acc'], text=_point_text(v), color='0.15', size=6.3))
    fr = sorted((v for v, on in zip(views, front) if on), key=lambda v: v['frgt'])
    ax.plot([v['frgt'] for v in fr], [v['acc'] for v in fr], color='0.6', lw=0.8, ls='--', zorder=1)
    ax.grid(alpha=0.25)
    return items


def _draw_pareto(ax, cells):
    """ACC vs forgetting; the fine-tune / slow-memory cluster gets a zoom inset."""
    views = [_style(c) for c in cells]
    front = [c['pareto'] for c in cells]
    items = _draw_cells(ax, views, front)
    ax.set_xlabel('forgetting FRGT (lower is better)')
    ax.set_ylabel('final average accuracy ACC')
    ax.set_title('(a) ACC vs forgetting', fontsize=9.5)
    return views, front, items


def _place_pareto_labels(fig, ax, views, front, items):
    from plot_ablations_best_lr import _inset_slot, _zoom_box, _zoom_useful
    zoom = _zoom_box(views, [])
    if not _zoom_useful(ax, zoom):
        place_labels(ax, items)
        return
    zx0, zx1, zy0, zy1, near = zoom
    inset = ax.inset_axes(_inset_slot(ax, views, zoom))
    keep = [i for i, v in enumerate(views) if v in near]
    inner = _draw_cells(inset, [views[i] for i in keep], [front[i] for i in keep], scale=0.8)
    inset.set_xlim(zx0, zx1)
    inset.set_ylim(zy0, zy1)
    inset.tick_params(labelsize=6)
    inset.xaxis.set_major_locator(MaxNLocator(4))
    inset.yaxis.set_major_locator(MaxNLocator(4))
    ax.indicate_inset_zoom(inset, edgecolor='0.5', alpha=0.8)
    outside = [it for it in items if not (zx0 <= it['x'] <= zx1 and zy0 <= it['y'] <= zy1)]
    place_labels(ax, outside, extra_obstacles=[inset.get_tightbbox(fig.canvas.get_renderer())])
    place_labels(inset, inner)


def _draw_scaling(ax, cells, y_key, y_label, title):
    """Metric vs width: slow memory per r, fine-tune, ER (w = 100) as horizontal bands."""
    for horizon in HORIZONS:
        pts = sorted((c for c in cells if c['family'] == 'model_size' and c['horizon'] == horizon),
                     key=lambda c: c['width'])
        if pts:
            colour = PP.family_color({'family': 'slow memory', 'horizon_ratio': horizon})
            ax.errorbar([c['width'] for c in pts], [c[f'{y_key}_mean'] for c in pts],
                        yerr=[c[f'{y_key}_std'] for c in pts], fmt='o-', color=colour,
                        mec='white', mew=0.6, lw=1.5, ms=6, capsize=2.5, zorder=3)
    pts = sorted((c for c in cells if c['family'] == 'finetune'), key=lambda c: c['width'])
    if pts:
        ax.errorbar([c['width'] for c in pts], [c[f'{y_key}_mean'] for c in pts],
                    yerr=[c[f'{y_key}_std'] for c in pts], fmt='s--', color=PP.FINE_TUNE_COLOR,
                    mec='white', mew=0.6, lw=1.4, ms=6, capsize=2.5, zorder=3)
    for K in KS:
        cell = next((c for c in cells if c['family'] == 'replay' and c['K'] == K), None)
        if cell is None:
            continue
        y, s = cell[f'{y_key}_mean'], cell[f'{y_key}_std']
        colour = PP.family_color({'family': 'ER', 'K': K})
        ax.axhspan(y - s, y + s, color=colour, alpha=0.15, lw=0)
        ax.axhline(y, color=colour, ls=':', lw=1.3, zorder=2)
    ax.set_xscale('log')
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.minorticks_off()
    ax.set_xlabel('MLP hidden width w')
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.25)


def _legend_columns(cells):
    def point(label, marker, color, size=6, edge='black', width=0.5, ls='none'):
        return plt.Line2D([], [], ls=ls, marker=marker, markersize=size, label=label, color=color,
                          markerfacecolor=color, markeredgecolor=edge, markeredgewidth=width)
    rs = sorted({c['horizon'] for c in cells if c['family'] == 'model_size'})
    ks = sorted({c['K'] for c in cells if c['family'] == 'replay'})
    return [
        ('fine-tune', [point('α = 0', 's', PP.FINE_TUNE_COLOR, ls='--')]),
        ('slow memory: colour = r',
         [point(f'r={r:g}', 'o', PP.family_color({'family': 'slow memory', 'horizon_ratio': r}), ls='-')
          for r in rs]),
        ('ER (w = 100): colour = K',
         [point(f'K={k}', '^', PP.family_color({'family': 'ER', 'K': k}), ls=':') for k in ks]),
        ('(a) marker size = width',
         [point(f'w={w}', 'o', '0.75', size=(WIDTH_SIZE[w] ** 0.5) * 0.9) for w in WIDTHS]),
        ('(a) Pareto front', [plt.Line2D([], [], color='0.6', ls='--', label='front'),
                              point('on the front', 'o', '0.85', width=1.4)]),
    ]


def render(angle, cells, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = 16.0, 5.9
    fig, (ax_p, ax_a, ax_f) = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                                           gridspec_kw={'width_ratios': [1.15, 1.0, 1.0]})
    views, front, items = _draw_pareto(ax_p, cells)
    _draw_scaling(ax_a, cells, 'ACC', 'final average accuracy ACC', '(b) ACC vs width')
    _draw_scaling(ax_f, cells, 'FRGT', 'forgetting FRGT', '(c) forgetting vs width')
    title_h = 0.19 + 0.12
    columns = _legend_columns(cells)
    rows = 1 + max(len(entries) for _, entries in columns)
    handles = []
    for head, entries in columns:
        col = [plt.Line2D([], [], ls='none', label=head)] + entries
        handles += col + [plt.Line2D([], [], ls='none', label='') for _ in range(rows - len(col))]
    leg = fig.legend(handles=handles, loc='upper center', ncol=len(columns), fontsize=8,
                     frameon=False, bbox_to_anchor=(0.5, 1 - title_h / fig_h),
                     columnspacing=2.0, handletextpad=0.6)
    for i, text in enumerate(leg.get_texts()):
        if i % rows == 0:
            text.set_fontweight('bold')
    fig.canvas.draw()
    leg_bottom = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.tight_layout(rect=(0, 0, 1, leg_bottom - 0.05 / fig_h))
    fig.suptitle(f'rotMNIST, {angle}° — ReplaySGD: model size', fontsize=10.5,
                 y=1 - 0.08 / fig_h)
    fig.canvas.draw()
    _place_pareto_labels(fig, ax_p, views, front, items)
    out_png = out_dir / 'model_size.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'wrote {out_png}')

    out_csv = out_dir / 'model_size.csv'
    with out_csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[
            'family', 'width', 'horizon', 'K', 'n_seeds',
            'ACC_mean', 'ACC_std', 'FRGT_mean', 'FRGT_std',
            'pareto', 'label',
        ])
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c['family'], c['width'],
                                              c['horizon'] or 0,
                                              c['K'] or 0)):
            w.writerow({k: c.get(k) for k in w.fieldnames})
    print(f'wrote {out_csv}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_dir', default='results/plan_analysis')
    ap.add_argument('--angles', nargs='+', type=int, default=list(ANGLES))
    opts = ap.parse_args()
    for angle in opts.angles:
        print(f'--- angle {angle} ---')
        rows = fetch(angle)
        cells = aggregate(rows)
        print(f'  {len(cells)} cells, {sum(c["pareto"] for c in cells)} on front')
        out = Path(opts.out_dir) / f'rotmnist_angle{angle}' / 'model_size'
        render(angle, cells, out)


if __name__ == '__main__':
    main()
