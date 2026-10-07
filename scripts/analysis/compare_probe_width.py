"""Staleness probe on rotMNIST slow_ema: learning rate vs width.

Three slow_ema run sets (``SETS``): w100_lowlr vs w100 changes only the
learning rate; w100 vs w1000 changes only the width. Every set is cut at the
last probe step the w=1000 runs reached.

Writes results/plan_analysis/rotmnist_angle<θ>/ablations/grad_staleness/
  probe_width_compare.png  — cos_dist, rel_E, proj_E/‖G_old‖, per-batch cos vs
                              tasks elapsed; one row per r, mean ± std over seeds
  probe_width_compare.csv  — per (set, r) means over the shared window
  probe_per_step_sets.csv  — per probe step, every run of every set
  probe_batch_cos_r3.{png,pdf} — standalone per-gradient staleness panel, log y
  w100_pinned_lr/          — cos_agg_vs_n, phi_by_pos, top_k_share for the
                              w100 set (analyze_grad_staleness renderers)

  python scripts/analysis/compare_probe_width.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_grad_staleness import (RESULTS_ROOT, _read_aggregate,  # noqa: E402
                                    fetch_configs, render_pos_figures, render_top_k,
                                    render_trajectory_figures, summarise_run)

L = 469
# (set id, width, W&B group template, colour, line style). Colour marks the
# learning rate (orange = lower, blue = higher); line style marks width.
SETS = [
    ('w100_lowlr', 100, 'plan_rotmnist_diagnostic_probe_slow_ema_a{a}', '#e6550d', '-'),
    ('w100', 100, 'plan_rotmnist_diagnostic_probe_slow_ema_a{a}_lrpin', '#08519c', '-'),
    ('w1000', 1000, 'plan_rotmnist_diagnostic_probe_width_slow_ema_a{a}', '#08519c', '--'),
]
HORIZONS = (1.0, 3.0)
METRICS = [
    ('cos_dist', r'cosine distance $(1-\cos(\mu^{st},\mu^{fr}))/2$', 'aggregate drift direction'),
    ('rel_E', r'$\|E\|/\|\mu^{fr}\|$', 'aggregate drift magnitude'),
    ('proj_E_n', r'$\langle G_{old},E\rangle/\|G_{old}\|$', 'drift along forgetting direction'),
    ('batch_cos', r'mean per-batch $\cos(g^{old}_s,g^{new}_s)$', 'per-gradient staleness'),
]
PER_STEP = ('cos_agg', 'cos_dist', 'rel_E', 'proj_E', 'proj_E_n', 'batch_cos')


def load(angle):
    """[(set id, width, r, seed, lr, run id, series dict)] for every run with probe output."""
    out = []
    for set_id, width, tmpl, *_ in SETS:
        group = tmpl.format(a=angle)
        cfgs = fetch_configs(group)
        for rid, c in cfgs.items():
            run_dir = RESULTS_ROOT / group / rid
            r, seed = c.get('horizon_ratio'), c.get('seed')
            if r is None or seed is None or not (run_dir / 'probe' / 'aggregate.csv').is_file():
                continue
            s = summarise_run(run_dir, float(r), int(seed))
            if s is None:
                continue
            agg = _read_aggregate(run_dir / 'probe' / 'aggregate.csv')
            held = np.asarray([row['held_out_norm'] for row in agg], dtype=np.float64)
            s['proj_E_n'] = s['proj_E'] / np.maximum(held, 1e-30)
            # pos_bin_cos is per fire × 10 position bins; mean over bins.
            s['batch_cos'] = np.nanmean(s['pos_bin_cos'], axis=1)
            out.append((set_id, width, float(r), int(seed), float(c.get('lr')), rid, s))
        print(f'  {group}: {sum(1 for o in out if o[0] == set_id)} runs')
    return out


def common_window(runs):
    """Largest n every w=1000 run reached."""
    return min(int(s['n'].max()) for set_id, *_, s in runs if set_id == 'w1000')


def set_runs(runs, set_id, r):
    return [s for sid, _, rr, _, _, _, s in runs if sid == set_id and rr == r]


def set_lr(runs, set_id):
    lrs = sorted({lr for sid, _, _, _, lr, _, _ in runs if sid == set_id})
    return '/'.join(f'{lr:g}' for lr in lrs)


def stack(rows, key, n_max):
    """(n grid, mean, sample std) over seeds, restricted to n ≤ n_max."""
    grid = np.asarray(sorted({int(n) for s in rows for n in s['n'] if n <= n_max}))
    mat = np.full((len(rows), grid.size), np.nan)
    for i, s in enumerate(rows):
        pos = {int(n): j for j, n in enumerate(s['n'])}
        for k, n in enumerate(grid):
            if n in pos:
                mat[i, k] = s[key][pos[n]]
    sd = np.nanstd(mat, axis=0, ddof=1) if len(rows) > 1 else np.zeros(grid.size)
    return grid, np.nanmean(mat, axis=0), sd


# Probe set names in the legends (sets A, B, C).
SET_NAME = {'w100_lowlr': 'A', 'w100': 'B', 'w1000': 'C'}
# The standalone panel (metric, r), drawn with a log y axis.
SINGLE = ('batch_cos', 3.0)


def _draw_metric(ax, runs, key, r, n_max):
    """Every set's mean ± sample std over seeds vs tasks elapsed; task boundaries dotted."""
    for set_id, width, _, colour, style in SETS:
        rows = set_runs(runs, set_id, r)
        if not rows:
            continue
        grid, mu, sd = stack(rows, key, n_max)
        x = grid / L
        ax.plot(x, mu, style, color=colour, lw=1.4)
        ax.fill_between(x, mu - sd, mu + sd, color=colour, alpha=0.15, lw=0)
    last = n_max / L
    for t in range(1, int(np.ceil(last)) + 1):
        ax.axvline(t, color='0.6', lw=0.5, ls=':', zorder=0)
    ax.grid(True, axis='y', alpha=0.3)
    if key == 'proj_E_n':
        ax.axhline(0.0, color='k', lw=0.5)
    if key == 'cos_dist':
        ax.set_ylim(0.0, 1.0)


def _set_handles(runs):
    return [plt.Line2D([], [], ls=style, color=colour, lw=1.6,
                       label=f'set {SET_NAME[set_id]}: w={width}, lr={set_lr(runs, set_id)}')
            for set_id, width, _, colour, style in SETS]


def render_metric(angle, runs, n_max, out_dir: Path, key, r, log_y=True):
    """One metric at one horizon as a standalone figure, ``probe_<key>_r<r>.{png,pdf}``."""
    ylabel, title = next((y, t) for k, y, t in METRICS if k == key)
    last = n_max / L
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _draw_metric(ax, runs, key, r, n_max)
    if log_y:
        ax.set_yscale('log')
        # Plain decimals on the log axis (0.02, 0.05, 0.1, ...), no 10^k labels.
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ylabel += ' (log scale)'
    ax.set_xlabel('tasks elapsed (dotted: task boundaries)', fontsize=9)
    ax.set_xticks(range(1, int(np.ceil(last)) + 1, 2))
    ax.set_xlim(right=last)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.legend(handles=_set_handles(runs), fontsize=8.5, loc='upper right', framealpha=0.95,
              edgecolor='none',
              title='colour = learning rate, line style = width\nmean ± std over 3 seeds',
              title_fontsize=8)
    ax.set_title(f'rotMNIST, {angle}° — ReplaySGD slow memory (r={r:g})\n{title}', fontsize=10)
    fig.tight_layout()
    stem = f'probe_{key}_r{r:g}'
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {out_dir}/{stem}.{{png,pdf}}')


def render(angle, runs, n_max, out_dir: Path):
    """Rows r = 1 / 3, columns the four metrics; legend above, task boundaries dotted."""
    fig_w, fig_h = 4.6 * len(METRICS), 3.9 * len(HORIZONS) + 0.9
    fig, axes = plt.subplots(len(HORIZONS), len(METRICS), figsize=(fig_w, fig_h),
                             squeeze=False, sharex=True)
    last = n_max / L
    for i, r in enumerate(HORIZONS):
        for j, (key, ylabel, title) in enumerate(METRICS):
            ax = axes[i][j]
            _draw_metric(ax, runs, key, r, n_max)
            ax.set_title(f'r={r:g}: {title}', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8.5)
            if i == len(HORIZONS) - 1:
                ax.set_xlabel('tasks elapsed n / L (dotted: task boundaries)', fontsize=9)
                ax.set_xticks(range(1, int(np.ceil(last)) + 1, 2))
    handles = _set_handles(runs)
    title_h = 0.19 + 0.12
    leg = fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=9,
                     frameon=False, bbox_to_anchor=(0.5, 1 - title_h / fig_h),
                     title='colour = learning rate, line style = width (mean ± std over 3 seeds)',
                     title_fontsize=8.5)
    fig.canvas.draw()
    leg_bottom = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.tight_layout(rect=(0, 0, 1, leg_bottom - 0.04 / fig_h))
    fig.suptitle(f'rotMNIST, {angle}° — ReplaySGD slow memory: staleness probe, learning rate '
                 f'vs width (tasks 1–{n_max // L})', fontsize=10.5, y=1 - 0.08 / fig_h)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'probe_width_compare.png', dpi=150)
    plt.close(fig)

    # Per (set, r): mean over the shared window (task ≥ 2, so the EMA has a
    # past task to be stale about), plus the proj_E > 0 fraction.
    summary = []
    for r in HORIZONS:
        for set_id, width, *_ in SETS:
            rows = set_runs(runs, set_id, r)
            if not rows:
                continue
            rec = {'angle': angle, 'set': set_id, 'width': width, 'r': r,
                   'n_seeds': len(rows), 'lr': float(set_lr(runs, set_id))}
            for key, *_ in METRICS + [('proj_E', None, None)]:
                vals = [np.nanmean(s[key][(s['n'] <= n_max) & (s['n'] > 2 * L)])
                        for s in rows]
                rec[f'{key}_mean'] = float(np.mean(vals))
                rec[f'{key}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rec['frac_projE_pos'] = float(np.mean([
                np.mean(s['proj_E'][(s['n'] <= n_max) & (s['n'] > L)] > 0) for s in rows]))
            summary.append(rec)
    with (out_dir / 'probe_width_compare.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    with (out_dir / 'probe_per_step_sets.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['set', 'width', 'lr', 'r', 'seed', 'run_id', 'n', 'task'] + list(PER_STEP))
        for set_id, width, r, seed, lr, rid, s in runs:
            for k, n in enumerate(s['n']):
                if n <= n_max:
                    w.writerow([set_id, width, lr, r, seed, rid, int(n), int(n) // L]
                               + [float(s[m][k]) for m in PER_STEP])

    # Per-batch figures for the w100 set.
    pinned = [s for sid, *_, s in runs if sid == 'w100']
    sub = out_dir / 'w100_pinned_lr'
    head = f'rotMNIST, {angle}° — staleness probe, set B (w=100, lr={set_lr(runs, "w100")})'
    render_trajectory_figures(pinned, sub, title=head)
    render_pos_figures(pinned, sub, title=f'{head}: per-batch metrics vs position in task')
    render_top_k(pinned, sub, title=f'{head}\nconcentration of forgetting mass across batches')
    print(f'wrote {out_dir}/probe_width_compare.{{png,csv}}, probe_per_step_sets.csv, {sub.name}/')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_dir', default='results/plan_analysis')
    ap.add_argument('--angles', nargs='+', type=int, default=[9, 30])
    opts = ap.parse_args()
    for angle in opts.angles:
        print(f'--- angle {angle} ---')
        runs = load(angle)
        lr_pin = {set_lr(runs, 'w100'), set_lr(runs, 'w1000')}
        if len(lr_pin) != 1:
            raise SystemExit(f'w=100 and w=1000 sets differ in LR: {lr_pin}')
        n_max = common_window(runs)
        out = Path(opts.out_dir) / f'rotmnist_angle{angle}' / 'ablations' / 'grad_staleness'
        for rec in render(angle, runs, n_max, out):
            print('  {set:>10} w={width:>4} r={r:g} lr={lr:g}  cos_dist={cos_dist_mean:.3f}  '
                  'rel_E={rel_E_mean:.3f}  projE/|G|={proj_E_n_mean:+.4f}  '
                  'batch_cos={batch_cos_mean:.3f}  P(projE>0)={frac_projE_pos:.2f}'
                  .format(**rec))
        render_metric(angle, runs, n_max, out, *SINGLE)


if __name__ == '__main__':
    main()
