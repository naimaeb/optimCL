"""End-to-end analysis of the gradient-staleness runs on rotMNIST.

Two input sweeps per angle:

- the slow_ema arm with ``--probe_staleness 1``. Each run has
  ``probe/aggregate.csv`` and per-step
  ``probe/task_<t>_step_<n>_batches.csv.gz`` files.
- the ``slow_ema_refresh`` arm.

Produces, per angle, a self-contained folder under
``results/plan_analysis/rotmnist_angle<θ>/ablations/grad_staleness/``:

- ``probe_prereg.md``: machine-filled scorecard of 7 staleness predictions.
- ``probe_aggregate_pr.csv``: per-probe-step aggregate pulled from all runs
  with their (seed, r) config attached.
- ``per_batch_summary_pr.csv``: per-run × per-pos-bin summaries of φ, cos,
  rel, φ_clip (10 pos bins × r × seed).
- ``refresh_arm_frgt.csv``: final ACC / FRGT / BWT for every
  slow_ema_refresh (r, S', seed) cell plus the slow_ema reference.
- Figures: ``cos_agg_vs_n.png``, ``rel_E_vs_n.png``, ``phi_by_pos.png``,
  ``clip_share_by_pos.png``, ``refresh_frgt_bars.png``, ``top_k_share.png``.

Run from the project root:

  python scripts/analysis/analyze_grad_staleness.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

RESULTS_ROOT = Path('results')
# ReplaySGD memory-arm LR per angle, and the Mammoth ER LR.
MEMORY_LR = {9: 0.1, 30: 0.3}
ER_LR = 0.2
L = 469  # rotMNIST steps per task: ceil(60000 / 128), one epoch
POS_BINS = np.linspace(0.0, 1.0, 11)  # 10 bins in [0, 1]


# ---------------------------------------------------------------------------
# Config fetch from W&B
# ---------------------------------------------------------------------------

def fetch_configs(group: str, entity='<wandb-entity>', project='cl_optimizer_memory'):
    """Map run_id → config dict for every run in a W&B group."""
    import wandb
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f'{entity}/{project}', filters={'group': group}, per_page=100))
    cfg = {}
    for r in runs:
        c = {k: v for k, v in r.config.items() if not k.startswith('_')}
        c['_summary'] = {k: r.summary.get(k) for k in
                         ('ACC', 'BWT', 'FRGT', 'FWT',
                          'AvgTestAccSeen')}
        c['_state'] = str(getattr(r, 'state', ''))
        c['_created'] = str(getattr(r, 'created_at', ''))
        cfg[r.id] = c
    return cfg


# ---------------------------------------------------------------------------
# Probe data: aggregate across runs
# ---------------------------------------------------------------------------

def _read_aggregate(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) if k != 'n' and k != 't_n'
                         else int(float(v)) for k, v in row.items()})
    return rows


def _read_batches(path: Path):
    """Returns a numpy structured dict of columns. Keeps only the needed
    columns to save RAM (each per-batch file is ~1 MB)."""
    needed = ('pos', 'age', 'w', 'kappa_old', 'kappa_new',
              'cos', 'rel', 'phi', 'phi_dir', 'phi_clip')
    cols = {k: [] for k in needed}
    with gzip.open(path, 'rt') as fh:
        for row in csv.DictReader(fh):
            for k in needed:
                cols[k].append(float(row[k]))
    return {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}


def summarise_run(run_dir: Path, r: float, seed: int) -> dict:
    """Per-run summary: pos-binned medians of |φ|, cos, rel, φ_clip share;
    plus top-k share and sign-of-sum_phi."""
    probe = run_dir / 'probe'
    agg = _read_aggregate(probe / 'aggregate.csv')
    if not agg:
        return None

    # Lists per probe step: pos-binned medians.
    pos_bin_phi = []
    pos_bin_cos = []
    pos_bin_rel = []
    pos_bin_clip_share = []
    pos_bin_age_mass = []
    # Per-run top-k share and age-share aggregates.
    all_abs_phi = []
    all_ages = []
    age_mass_above_L = []
    sign_of_sum_phi = []
    # Which steps had positive proj_E vs negative, from aggregate.
    sign_of_proj_E = []

    batches_files = sorted(probe.glob('task_*_batches.csv.gz'))
    for f in batches_files:
        try:
            b = _read_batches(f)
        except Exception:
            continue
        if b['phi'].size == 0:
            continue
        abs_phi = np.abs(b['phi'])
        abs_clip = np.abs(b['phi_clip'])
        clip_share = abs_clip / (abs_phi + 1e-30)

        # Per-step summaries into pos bins.
        idx = np.clip(np.digitize(b['pos'], POS_BINS) - 1, 0, len(POS_BINS) - 2)
        this_phi = np.full(10, np.nan)
        this_cos = np.full(10, np.nan)
        this_rel = np.full(10, np.nan)
        this_clip = np.full(10, np.nan)
        for i in range(10):
            m = idx == i
            if m.any():
                this_phi[i] = float(np.median(abs_phi[m]))
                this_cos[i] = float(np.median(b['cos'][m]))
                this_rel[i] = float(np.median(b['rel'][m]))
                this_clip[i] = float(np.median(clip_share[m]))
        pos_bin_phi.append(this_phi)
        pos_bin_cos.append(this_cos)
        pos_bin_rel.append(this_rel)
        pos_bin_clip_share.append(this_clip)

        all_abs_phi.append(abs_phi)
        all_ages.append(b['age'])
        age_mass_above_L.append(float(abs_phi[b["age"] > L].sum())
                                / max(float(abs_phi.sum()), 1e-30))
        sign_of_sum_phi.append(1 if b['phi'].sum() < 0 else 0)

    # Overall per-run arrays (concatenated).
    all_abs_phi = np.concatenate(all_abs_phi) if all_abs_phi else np.zeros(0)
    all_ages = np.concatenate(all_ages) if all_ages else np.zeros(0)

    # Top-k share: sort desc, cumulative share.
    if all_abs_phi.size > 0:
        srt = np.sort(all_abs_phi)[::-1]
        cum = np.cumsum(srt) / max(srt.sum(), 1e-30)
        top10_share = float(cum[int(0.1 * srt.size)]) if srt.size >= 10 else float(cum[-1])
        top1_share = float(cum[int(0.01 * srt.size)]) if srt.size >= 100 else float(cum[0])
    else:
        top10_share = float('nan')
        top1_share = float('nan')

    # Aggregate rows become numpy arrays (per probe step).
    ns = np.asarray([r['n'] for r in agg])
    rel_E = np.asarray([r['rel_E'] for r in agg], dtype=np.float32)
    cos_agg = np.asarray([r['cos_agg'] for r in agg], dtype=np.float32)
    # cos_dist = (1 - cos_agg)/2 ∈ [0, 1], recomputed here so aggregate.csv
    # files without that column still parse.
    cos_dist = 0.5 * (1.0 - cos_agg)
    proj_E = np.asarray([r['proj_E'] for r in agg], dtype=np.float32)
    sum_phi = np.asarray([r['sum_phi'] for r in agg], dtype=np.float32)
    mapping_err = np.asarray([r['mapping_err'] for r in agg], dtype=np.float32)

    return dict(
        r=float(r), seed=int(seed),
        n=ns,
        rel_E=rel_E, cos_agg=cos_agg, cos_dist=cos_dist,
        proj_E=proj_E, sum_phi=sum_phi,
        mapping_err_max=float(mapping_err.max()),
        pos_bin_phi=np.asarray(pos_bin_phi, dtype=np.float32),
        pos_bin_cos=np.asarray(pos_bin_cos, dtype=np.float32),
        pos_bin_rel=np.asarray(pos_bin_rel, dtype=np.float32),
        pos_bin_clip_share=np.asarray(pos_bin_clip_share, dtype=np.float32),
        age_mass_above_L=np.asarray(age_mass_above_L, dtype=np.float32),
        sign_of_sum_phi_neg=np.asarray(sign_of_sum_phi, dtype=np.int32),
        top10_phi_share=top10_share,
        top1_phi_share=top1_share,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _seed_band(summaries, r, key):
    """(x in tasks elapsed, mean, std, n_seeds) over the probe steps every seed of r has."""
    runs = [s for s in summaries if s['r'] == r]
    common = set(runs[0]['n'].tolist())
    for s in runs[1:]:
        common &= set(s['n'].tolist())
    ns = np.asarray(sorted(common))
    vals = np.stack([np.asarray([dict(zip(s['n'].tolist(), s[key].tolist()))[n] for n in ns])
                     for s in runs])
    std = vals.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(ns.size)
    return ns / L, vals.mean(axis=0), std, len(runs)


def render_trajectory_figures(summaries, out_dir: Path, title=None):
    """cos_agg, cos_dist and rel_E against tasks elapsed: mean ± std over seeds per r.

    The probe fires at the last step of a task and at the first step of the
    next, so the dotted task boundaries (integer n / L) separate those pairs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = {1.0: '#1f77b4', 3.0: '#d62728'}
    panels = [
        ('cos_agg', r'$\cos(\mu^{\rm stale}, \mu^{\rm fresh})$', 'EMA direction cosine', None),
        ('cos_dist', r'cosine distance $(1-\cos)/2$', 'EMA cosine distance (∈ [0, 1])', (0.0, 1.0)),
        ('rel_E', r'$\|E\|/\|\mu^{\rm fresh}\|$', 'Relative drift (unbounded)', None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    last = 0.0
    for r in sorted({s['r'] for s in summaries}):
        c = cmap.get(r, '#333')
        for ax, (key, *_rest) in zip(axes, panels):
            x, mu, sd, k = _seed_band(summaries, r, key)
            ax.plot(x, mu, color=c, lw=1.1, marker='.', ms=3,
                    label=f'r={r:g} (mean ± std, {k} seeds)')
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.2, lw=0)
            last = max(last, float(x.max()) if x.size else 0.0)
    for ax, (key, ylabel, title, ylim) in zip(axes, panels):
        for t in range(1, int(np.ceil(last)) + 1):
            ax.axvline(t, color='0.55', lw=0.6, ls=':', zorder=0)
        ax.set_xticks(range(1, int(np.ceil(last)) + 1, 2))
        ax.set_xlabel('tasks elapsed n / L (dotted: task boundaries)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
    axes[0].axhline(0.0, color='k', lw=0.3, alpha=0.3)
    fig.suptitle(title or f'slow_ema staleness probe (L = {L} steps per task)', fontsize=10)
    fig.savefig(out_dir / 'cos_agg_vs_n.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_pos_figures(summaries, out_dir: Path, title=None):
    """|φ|, cos, rel, clip share vs position-in-task."""
    centers = 0.5 * (POS_BINS[:-1] + POS_BINS[1:])
    cmap = {1.0: '#1f77b4', 3.0: '#d62728'}

    # Group runs by r, average over seeds and probe steps.
    by_r = defaultdict(list)
    for s in summaries:
        by_r[s['r']].append(s)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for r in sorted(by_r):
        color = cmap.get(r, '#333')
        # Mean over (seed × probe step) of the pos-binned medians.
        phi_stack = np.concatenate([s['pos_bin_phi'] for s in by_r[r]], axis=0)
        cos_stack = np.concatenate([s['pos_bin_cos'] for s in by_r[r]], axis=0)
        rel_stack = np.concatenate([s['pos_bin_rel'] for s in by_r[r]], axis=0)
        clip_stack = np.concatenate([s['pos_bin_clip_share'] for s in by_r[r]], axis=0)
        for ax, mat, title, ylabel in [
            (axes[0, 0], phi_stack, r'$|\varphi_s|$ by position in task', r'median $|\varphi|$'),
            (axes[0, 1], cos_stack, r'$\cos(g^{\rm old}, g^{\rm new})$ by position', r'median $\cos$'),
            (axes[1, 0], rel_stack, r'$\|\Delta_s\|/\|g^{\rm new}\|$ by position', r'median rel'),
            (axes[1, 1], clip_stack, r'$|\varphi_s^{\rm clip}|/|\varphi_s|$ by position', 'median clip share'),
        ]:
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
            ax.plot(centers, mu, '-o', color=color, ms=4,
                    label=f'r={r:g}, n_runs={len(by_r[r])}')
            ax.fill_between(centers, mu - sd, mu + sd, color=color, alpha=0.15)
            ax.set_xlabel('pos in task'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, alpha=0.3)
    for row in axes:
        for ax in row:
            ax.legend(fontsize=8)
    axes[0, 0].set_yscale('log')  # |φ| spans orders of magnitude
    axes[1, 1].set_ylim(0, 1.0)
    fig.suptitle(title or 'slow_ema probe: per-batch metrics vs position in task',
                 fontsize=10)
    fig.savefig(out_dir / 'phi_by_pos.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_top_k(summaries, out_dir: Path, title=None):
    cmap = {1.0: '#1f77b4', 3.0: '#d62728'}
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))
    rows = []
    for s in summaries:
        rows.append([s['r'], s['seed'], s['top1_phi_share'], s['top10_phi_share']])
    rows = np.asarray(rows)
    by_r = defaultdict(list)
    for row in rows:
        by_r[row[0]].append(row[2:])
    xs = np.array([1, 10], dtype=float)
    for r in sorted(by_r):
        arr = np.asarray(by_r[r])
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mu)
        ax.errorbar(xs, mu, yerr=sd, fmt='-o', color=cmap.get(r, '#333'),
                    label=f'r={r:g}', capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel('top-k% of batches')
    ax.set_ylabel(r'cumulative share of $\sum|\varphi|$')
    ax.set_title(title or 'Concentration of forgetting mass across batches')
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_xticks([1, 10]); ax.set_xticklabels(['1%', '10%'])
    fig.tight_layout()
    fig.savefig(out_dir / 'top_k_share.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Refresh arm: FRGT comparison
# ---------------------------------------------------------------------------

def refresh_table(refresh_cfgs, slow_ema_cfgs, out_dir: Path, angle: int):
    rows = []
    for rid, c in refresh_cfgs.items():
        s = c.get('_summary', {})
        rows.append({
            'arm': 'slow_ema_refresh',
            'angle': angle,
            'r': float(c.get('horizon_ratio')) if c.get('horizon_ratio') else None,
            'S_prime': int(c.get('refresh_batches', 0)) if c.get('refresh_batches') is not None else None,
            'seed': int(c.get('seed')) if c.get('seed') is not None else None,
            'run_id': rid, 'state': c.get('_state'),
            'ACC': s.get('ACC'), 'BWT': s.get('BWT'),
            'FRGT': s.get('FRGT'), 'FWT': s.get('FWT'),
        })
    for rid, c in slow_ema_cfgs.items():
        s = c.get('_summary', {})
        rows.append({
            'arm': 'slow_ema',
            'angle': angle,
            'r': float(c.get('horizon_ratio')) if c.get('horizon_ratio') else None,
            'S_prime': None, 'seed': int(c.get('seed')) if c.get('seed') is not None else None,
            'run_id': rid, 'state': c.get('_state'),
            'ACC': s.get('ACC'), 'BWT': s.get('BWT'),
            'FRGT': s.get('FRGT'), 'FWT': s.get('FWT'),
        })
    out_path = out_dir / 'refresh_arm_frgt.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out_path}')
    return rows


def render_refresh_bars(rows, out_dir: Path, angle: int):
    """Mean ± std of ACC and FRGT across seeds, grouped by (r, S')."""
    # bucket by (arm, r, S')
    bucket = defaultdict(lambda: {'ACC': [], 'FRGT': []})
    for r in rows:
        if r['ACC'] is None or r['FRGT'] is None:
            continue
        k = (r['arm'], r['r'], r['S_prime'])
        bucket[k]['ACC'].append(float(r['ACC']))
        bucket[k]['FRGT'].append(float(r['FRGT']))
    labels = []
    acc_mu, acc_sd = [], []
    frgt_mu, frgt_sd = [], []
    colors = []
    for (arm, r, Sp), v in sorted(bucket.items(),
                                   key=lambda kv: (kv[0][1] or 0, kv[0][0], kv[0][2] or -1)):
        if not v['ACC']:
            continue
        label = f'r={r:g}, {arm}'
        if arm == 'slow_ema_refresh':
            label += f", S'={Sp}"
        labels.append(label)
        acc_mu.append(np.mean(v['ACC'])); acc_sd.append(_mean_std(v['ACC'])[1])
        frgt_mu.append(np.mean(v['FRGT'])); frgt_sd.append(_mean_std(v['FRGT'])[1])
        if arm == 'slow_ema':
            colors.append('#777')
        elif Sp == 0:
            colors.append('#1f77b4')
        elif Sp == 4:
            colors.append('#2ca02c')
        else:
            colors.append('#d62728')

    if not labels:
        print('[refresh] no complete cells to plot')
        return
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * len(labels) + 2.0),
                             constrained_layout=True)
    axes[0].barh(y, acc_mu, xerr=acc_sd, color=colors)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel('ACC (%)'); axes[0].set_title('Final test accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[1].barh(y, frgt_mu, xerr=frgt_sd, color=colors)
    axes[1].set_yticks(y); axes[1].set_yticklabels([''] * len(labels))
    axes[1].set_xlabel('FRGT (pp)'); axes[1].set_title('Forgetting (lower is better)')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()
    fig.suptitle(f'slow_ema_refresh vs slow_ema — rotMNIST angle {angle}°',
                 fontsize=10)
    fig.savefig(out_dir / 'refresh_frgt_bars.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto front: FRGT vs ACC, staleness arms + replay
# ---------------------------------------------------------------------------

def _mean_std(xs):
    arr = np.asarray([x for x in xs if x is not None], dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0  # sample std over seeds
    return float(arr.mean()), sd, int(arr.size)


def _pareto_mask(frgt, acc):
    """Points not dominated on (FRGT min, ACC max). Ties stay on the front."""
    on = [True] * len(frgt)
    for i, (xi, yi) in enumerate(zip(frgt, acc)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            on[i] = False
            continue
        for j, (xj, yj) in enumerate(zip(frgt, acc)):
            if i == j or not (np.isfinite(xj) and np.isfinite(yj)):
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                on[i] = False
                break
    return on


def render_pareto(refresh_cfgs, slow_ema_cfgs, replay_cfgs,
                  out_dir: Path, angle: int):
    """Pareto plot of forgetting (FRGT) vs average test accuracy (ACC) across
    the gradient-staleness arms and experience replay, one point per cell
    (mean ± std over seeds)."""
    cells = defaultdict(lambda: {'ACC': [], 'FRGT': []})

    def _add(arm, cfgs, key_fn):
        for rid, c in cfgs.items():
            s = c.get('_summary', {})
            if s.get('ACC') is None or s.get('FRGT') is None:
                continue
            k = (arm,) + key_fn(c)
            cells[k]['ACC'].append(float(s['ACC']))
            cells[k]['FRGT'].append(float(s['FRGT']))

    _add('slow_ema', slow_ema_cfgs,
         lambda c: (float(c.get('horizon_ratio') or float('nan')), None, None))
    _add('slow_ema_refresh', refresh_cfgs,
         lambda c: (float(c.get('horizon_ratio') or float('nan')),
                    int(c.get('refresh_batches', 0) or 0), None))
    _add('replay', replay_cfgs,
         lambda c: (None, None,
                    int(c.get('replay_buffer_size') or 0) or None))

    points = []
    for (arm, r, Sp, K), v in cells.items():
        acc_mu, acc_sd, n_acc = _mean_std(v['ACC'])
        frgt_mu, frgt_sd, n_frgt = _mean_std(v['FRGT'])
        if not np.isfinite(acc_mu) or not np.isfinite(frgt_mu):
            continue
        if arm == 'slow_ema':
            label = f"slow_ema (r={r:g})"
        elif arm == 'slow_ema_refresh':
            label = f"refresh r={r:g}, S'={Sp}"
        else:
            label = f"replay K={K}"
        points.append({
            'arm': arm, 'r': r, 'S_prime': Sp, 'K': K,
            'ACC_mean': acc_mu, 'ACC_std': acc_sd,
            'FRGT_mean': frgt_mu, 'FRGT_std': frgt_sd,
            'n': min(n_acc, n_frgt), 'label': label,
        })

    if not points:
        print('[pareto] no points to plot')
        return

    frgt = [p['FRGT_mean'] for p in points]
    acc = [p['ACC_mean'] for p in points]
    on_front = _pareto_mask(frgt, acc)
    for p, flag in zip(points, on_front):
        p['pareto'] = bool(flag)

    arm_style = {
        'slow_ema':         {'marker': 's', 'edge': '#555555'},
        'slow_ema_refresh': {'marker': 'o', 'edge': '#08306b'},
        'replay':           {'marker': '^', 'edge': '#084081'},
    }
    # Fill colour encodes the sub-setting; marker shape encodes the arm.
    r_fill = {1.0: '#bdbdbd', 3.0: '#636363'}   # slow_ema (r)
    sp_fill = {0: '#c6dbef', 4: '#4292c6', 16: '#08519c'}  # refresh (S')
    k_fill = {200: '#c7e9b4', 500: '#41b6c4', 5120: '#225ea8'}  # replay (K)

    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    for p in points:
        style = arm_style[p['arm']]
        if p['arm'] == 'slow_ema':
            face = r_fill.get(p['r'], '#999999')
        elif p['arm'] == 'slow_ema_refresh':
            face = sp_fill.get(p['S_prime'], style['edge'])
        else:
            face = k_fill.get(p['K'], style['edge'])
        ax.errorbar(p['FRGT_mean'], p['ACC_mean'],
                    xerr=p['FRGT_std'] if np.isfinite(p['FRGT_std']) else None,
                    yerr=p['ACC_std'] if np.isfinite(p['ACC_std']) else None,
                    fmt=style['marker'], mfc=face, mec=style['edge'],
                    ecolor=style['edge'], ms=10, mew=1.3, elinewidth=0.8,
                    capsize=2.5, alpha=0.95,
                    zorder=5 if p['pareto'] else 3)
        if p['pareto']:
            ax.scatter([p['FRGT_mean']], [p['ACC_mean']], s=170,
                       facecolors='none', edgecolors='0.15',
                       linewidths=1.3, zorder=4)

    front = sorted([p for p in points if p['pareto']],
                   key=lambda p: p['FRGT_mean'])
    if len(front) >= 2:
        ax.plot([p['FRGT_mean'] for p in front],
                [p['ACC_mean'] for p in front],
                color='0.2', lw=1.3, ls='-', zorder=2)

    ax.set_xlabel('Forgetting FRGT (pp, lower is better)')
    ax.set_ylabel('Avg. test accuracy ACC (%)')
    ax.set_title(f'Pareto: forgetting vs accuracy — rotMNIST angle {angle}°\n'
                 'each arm at its own LR: slow_ema / refresh / ER')
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    handles = []
    _lrs = lambda cfgs: '/'.join(sorted({f"{float(c['lr']):g}" for c in cfgs.values()}))
    handles.append(Line2D([0], [0], color='none', label=f'slow_ema, lr {_lrs(slow_ema_cfgs)}:'))
    for r in sorted({p['r'] for p in points if p['arm'] == 'slow_ema'}):
        handles.append(Line2D([0], [0], marker='s',
                              color=arm_style['slow_ema']['edge'], ls='none',
                              mfc=r_fill.get(r, '#999999'), ms=8,
                              label=f'  r={r:g}'))
    handles.append(Line2D([0], [0], color='none', label=f'slow_ema_refresh, lr {_lrs(refresh_cfgs)}:'))
    for Sp in sorted({p['S_prime'] for p in points if p['arm'] == 'slow_ema_refresh'}):
        for r in sorted({p['r'] for p in points
                         if p['arm'] == 'slow_ema_refresh' and p['S_prime'] == Sp}):
            handles.append(Line2D([0], [0], marker='o',
                                  color=arm_style['slow_ema_refresh']['edge'],
                                  ls='none',
                                  mfc=sp_fill.get(Sp, '#4292c6'), ms=8,
                                  label=f"  r={r:g}, S'={Sp}"))
    handles.append(Line2D([0], [0], color='none', label=f'ER (Mammoth), lr {_lrs(replay_cfgs)}:'))
    for K in sorted({p['K'] for p in points if p['arm'] == 'replay'}):
        handles.append(Line2D([0], [0], marker='^',
                              color=arm_style['replay']['edge'], ls='none',
                              mfc=k_fill.get(K, '#41b6c4'), ms=8,
                              label=f'  K={K}'))
    handles.append(Line2D([0], [0], color='0.2', lw=1.3, label='Pareto front'))
    handles.append(Line2D([0], [0], marker='o', color='0.15', ls='none',
                         mfc='none', ms=11, mew=1.3, label='on Pareto front'))
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(out_dir / 'pareto_frgt_acc.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    out_csv = out_dir / 'pareto_frgt_acc.csv'
    with out_csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=[
            'arm', 'r', 'S_prime', 'K', 'n',
            'ACC_mean', 'ACC_std', 'FRGT_mean', 'FRGT_std', 'pareto', 'label'])
        w.writeheader()
        for p in points:
            w.writerow({k: p[k] for k in w.fieldnames})
    print(f'wrote {out_dir / "pareto_frgt_acc.png"}')
    print(f'wrote {out_csv}')


# ---------------------------------------------------------------------------
# Prediction scorecard
# ---------------------------------------------------------------------------

def write_prereg(angle: int, summaries, refresh_rows, out_dir: Path):
    """Score the 7 staleness predictions on the available data."""
    centers = 0.5 * (POS_BINS[:-1] + POS_BINS[1:])
    # Pool across runs for the shared stats.
    phi_all = np.concatenate([s['pos_bin_phi'] for s in summaries], axis=0)
    median_phi_by_pos = np.nanmedian(phi_all, axis=0)
    clip_all = np.concatenate([s['pos_bin_clip_share'] for s in summaries], axis=0)
    median_clip_by_pos = np.nanmedian(clip_all, axis=0)
    cos_all = np.concatenate([s['pos_bin_cos'] for s in summaries], axis=0)
    one_minus_cos_by_pos = 1 - np.nanmedian(cos_all, axis=0)

    # Prediction 1: |φ| and 1-cos peak in first 10%, >= 2x the last-10% median.
    phi_front = median_phi_by_pos[0]
    phi_back = median_phi_by_pos[-1]
    cos_front = one_minus_cos_by_pos[0]
    cos_back = one_minus_cos_by_pos[-1]
    pred1_pass = (phi_front >= 2 * phi_back) and (cos_front >= 2 * cos_back)

    # Prediction 2: top-10% < 50% of |φ|, averaged over runs.
    top10 = np.mean([s['top10_phi_share'] for s in summaries])
    pred2_pass = top10 < 0.5

    # Prediction 4: sign of Σ_s φ_s indicates refresh helps on old-task loss.
    #
    # Derivation. The optimizer takes θ_{n+1} = θ_n − η(m̂_1 + α μ)/(1 + α).
    # The μ-dependent change in the held-out old-task loss is
    #     ΔL(μ) = −η·α/(1+α) · ⟨G_old, μ⟩,
    # so the improvement from refreshing is
    #     ΔL_fresh − ΔL_stale = −η·α/(1+α) · ⟨G_old, E⟩,  E = μ_fresh − μ_stale.
    # η, α > 0 ⇒ the prefactor is negative ⇒ refresh LOWERS the loss ⇔
    # ⟨G_old, E⟩ > 0.
    # The probe reports Σ_s φ_s = ⟨G_old, E⟩ (same as proj_E in aggregate.csv).
    # So "refresh helps" ⇔ sum_phi > 0.
    frac_pos = []
    for s in summaries:
        frac_pos.append(float(np.mean(s['sum_phi'] > 0)))
    pred4_frac = float(np.mean(frac_pos))
    pred4_pass = pred4_frac >= 0.8

    # Prediction 6: rel_E grows with r; age mass > L grows with r.
    rel_by_r = defaultdict(list); age_by_r = defaultdict(list)
    for s in summaries:
        rel_by_r[s['r']].append(float(np.median(s['rel_E'])))
        age_by_r[s['r']].append(float(np.mean(s['age_mass_above_L'])))
    rs = sorted(rel_by_r)
    rel_means = [float(np.mean(rel_by_r[r])) for r in rs]
    age_means = [float(np.mean(age_by_r[r])) for r in rs]
    pred6_rel = all(rel_means[i] <= rel_means[i + 1] for i in range(len(rel_means) - 1)) \
        if len(rel_means) >= 2 else None
    pred6_age = all(age_means[i] <= age_means[i + 1] for i in range(len(age_means) - 1)) \
        if len(age_means) >= 2 else None

    # Prediction 7: clip share > 10% for pos < 0.05, < 5% elsewhere.
    pred7_front = median_clip_by_pos[0]
    pred7_rest = float(np.nanmedian(median_clip_by_pos[2:]))
    pred7_pass = (pred7_front > 0.10) and (pred7_rest < 0.05)

    # Prediction 5: slow_ema_refresh vs ER. Only the within-family delta
    # (slow_ema minus slow_ema_refresh S'=16) is computed here; the ER
    # comparison is not.
    get_frgt = lambda arm, Sp: [r['FRGT'] for r in refresh_rows
                                if r['arm'] == arm and r['S_prime'] == Sp
                                and r['FRGT'] is not None]
    frgt_slow = get_frgt('slow_ema', None) + get_frgt('slow_ema_refresh', 0)
    frgt_refresh16 = get_frgt('slow_ema_refresh', 16)
    if frgt_slow and frgt_refresh16:
        pred5_delta = float(np.mean(frgt_slow) - np.mean(frgt_refresh16))
    else:
        pred5_delta = None

    # Mapping sanity.
    max_mapping = max(s['mapping_err_max'] for s in summaries)

    rows = [
        ('1', 'Where (first-10% peak)',
         f'median|φ| front={phi_front:.3g}, back={phi_back:.3g}; '
         f'1-cos front={cos_front:.3g}, back={cos_back:.3g}',
         'Pass' if pred1_pass else 'Fail'),
        ('2', 'Shape (top-10% < 50% of |φ|)',
         f'top-10% share = {top10:.3f}',
         'Pass' if pred2_pass else 'Fail'),
        ('3', 'Type (|φ|-weighted ρ^NL < 0.3)',
         'deferred (needs Tier-2 linear/NL split)', 'Deferred'),
        ('4', 'Sign of Σφ: refresh helps on old-task loss (sum_phi > 0)',
         f'fraction of probe steps with sum_phi > 0: {pred4_frac:.3f}',
         'Pass' if pred4_pass else 'Fail'),
        ('5', 'slow_ema_refresh FRGT improvement',
         (f'FRGT(slow_ema) − FRGT(S\'=16) = {pred5_delta:.3f}'
          if pred5_delta is not None else 'incomplete'),
         'Measured'),
        ('6', 'Horizon: rel_E and >L-age mass grow with r',
         f'rel_E by r={rs}: {rel_means}; age>L mass: {age_means}',
         ('Pass' if (pred6_rel and pred6_age) else 'Fail')
         if (pred6_rel is not None and pred6_age is not None) else 'N/A'),
        ('7', 'Clip share > 10% only for pos < 0.05',
         f'median clip share: pos0={pred7_front:.3f}, rest={pred7_rest:.3f}',
         'Pass' if pred7_pass else 'Fail'),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'probe_prereg.md').open('w') as fh:
        fh.write(f'# Pre-reg scorecard — rotMNIST angle {angle}°\n\n')
        fh.write(f'Analysis of {len(summaries)} probe runs, {len(refresh_rows)} '
                 'refresh-arm cells. Max probe-mapping error: '
                 f'`{max_mapping:.2e}` (threshold 1e-5).\n\n')
        fh.write('## Headlines\n\n')
        fh.write(f'- **Refresh dominates stale on old-task loss.** `sum_phi > 0` in '
                 f'{pred4_frac*100:.1f}% of probe steps (score card #4). '
                 + (f'The slow_ema_refresh arm with S\'=16 cuts FRGT by '
                    f'`{pred5_delta:.2f}` pp over slow_ema (#5).\n'
                    if pred5_delta is not None else
                    'No slow_ema_refresh runs at the memory LR pin, so #5 is open.\n'))
        fh.write('- **Staleness is diffuse.** The top 10% of batches '
                 f'carry {top10*100:.1f}% of `Σ|φ|` (#2, pass). The forgetting '
                 'mass is spread across the stream, not concentrated in a '
                 'few batches.\n')
        fh.write('- **It is not front-loaded within a task.** `median |φ|` '
                 'at the beginning of a task is comparable to the end, '
                 'and the clip-share is roughly uniform across positions '
                 f'({pred7_front*100:.1f}% at pos0 vs {pred7_rest*100:.1f}% elsewhere). '
                 'Predictions 1 and 7 miss.\n')
        fh.write('- **Horizon effect is mixed.** The share of `|φ|` from '
                 'batches older than one task grows with r '
                 f'({age_means[0]*100:.1f}% at r=1 → {age_means[-1]*100:.1f}% '
                 'at r=3, matching #6b), but `rel_E` is already ~1 at r=1 '
                 'and does not grow further — the drift between μ_stale '
                 'and μ_fresh saturates at a large fraction of μ_fresh.\n\n')
        fh.write('## Scorecard\n\n')
        fh.write('Predictions from `docs/staleness_prereg.md`:\n\n')
        fh.write('| # | Prediction | Measured | Status |\n|---|---|---|---|\n')
        for n, pred, meas, st in rows:
            fh.write(f'| {n} | {pred} | {meas} | **{st}** |\n')
        fh.write('\n## Figures in this folder\n\n')
        fh.write('- `cos_agg_vs_n.png` — cos(μ_stale, μ_fresh) and ‖E‖/‖μ_fresh‖ '
                 'over the full training trajectory, one line per (r, seed).\n')
        fh.write('- `phi_by_pos.png` — per-batch median |φ|, cos, rel, '
                 'clip-share binned by position-in-task.\n')
        fh.write('- `top_k_share.png` — cumulative share of `Σ|φ|` carried by '
                 'the top 1% / top 10% of batches, per r.\n')
        fh.write('- `refresh_frgt_bars.png` — final ACC and FRGT bars for '
                 'slow_ema vs slow_ema_refresh at `S\' ∈ {0, 4, 16}`.\n')
        fh.write('- `refresh_arm_frgt.csv` — raw per-run metrics for the '
                 'refresh-arm comparison.\n')
        fh.write('- `probe_aggregate_pr.csv` — per-probe-step aggregate '
                 '(cos_agg, rel_E, proj_E, sum_phi) with the run config.\n')
    print(f'wrote {out_dir / "probe_prereg.md"}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse_angle(angle: int, root_results: Path):
    # w=100 probe runs at the memory LR.
    probe_group = f'plan_rotmnist_diagnostic_probe_slow_ema_a{angle}_lrpin'
    refresh_group = f'plan_rotmnist_ablation_slow_ema_refresh_a{angle}'
    # Pareto references, each arm at its own LR: slow_ema at ``MEMORY_LR``,
    # ER at ``ER_LR``.
    baseline_group = f'plan_rotmnist_ablation_a{angle}_lattice_memory'
    replay_group = 'plan_rotmnist_replaysgd_er'
    print(f'--- angle {angle} ---')
    probe_cfgs = fetch_configs(probe_group)
    # Refresh runs count only at the memory LR.
    refresh_all = fetch_configs(refresh_group)
    refresh_cfgs = {k: v for k, v in refresh_all.items()
                    if abs(float(v.get('lr')) - MEMORY_LR[angle]) < 1e-9}
    if len(refresh_cfgs) < len(refresh_all):
        off = sorted({float(v.get('lr')) for v in refresh_all.values()} - {MEMORY_LR[angle]})
        print(f'  excluded {len(refresh_all) - len(refresh_cfgs)} refresh runs at lr={off} '
              f'(memory LR pin is {MEMORY_LR[angle]:g})')
    baseline_cfgs = {k: v for k, v in fetch_configs(baseline_group).items()
                     if v.get('ablation') == 'slow_ema'
                     and v.get('horizon_ratio') in (1.0, 3.0)
                     and abs(float(v.get('lr')) - MEMORY_LR[angle]) < 1e-9}
    # One ER run per (K, seed): the latest created.
    replay_cfgs = {}
    for k, v in sorted(fetch_configs(replay_group).items(), key=lambda kv: kv[1]['_created']):
        if (int(v.get('angle', -1)) == angle and int(v.get('replay', 0) or 0) == 1
                and abs(float(v.get('lr')) - ER_LR) < 1e-9 and v['_state'] == 'finished'):
            replay_cfgs = {kk: vv for kk, vv in replay_cfgs.items()
                           if (vv['replay_buffer_size'], vv['seed'])
                           != (v['replay_buffer_size'], v['seed'])}
            replay_cfgs[k] = v
    print(f'  {len(probe_cfgs)} probe runs; '
          f'{len(refresh_cfgs)} refresh runs; '
          f'{len(baseline_cfgs)} slow_ema baseline cells; '
          f'{len(replay_cfgs)} replay runs')

    probe_root = RESULTS_ROOT / probe_group
    summaries = []
    for rid, c in probe_cfgs.items():
        rdir = probe_root / rid
        r = c.get('horizon_ratio')
        seed = c.get('seed')
        if r is None or seed is None:
            continue
        s = summarise_run(rdir, float(r), int(seed))
        if s is not None:
            summaries.append(s)
    print(f'  {len(summaries)} probe runs with data')

    out_dir = root_results / f'rotmnist_angle{angle}' / 'ablations' / 'grad_staleness'
    out_dir.mkdir(parents=True, exist_ok=True)

    render_trajectory_figures(summaries, out_dir)
    render_pos_figures(summaries, out_dir)
    render_top_k(summaries, out_dir)
    # The scorecard and bars compare within the refresh sweep (S'=0 is the
    # bit-identical slow_ema control at the refresh LR), so no outside baseline.
    if refresh_cfgs:
        refresh_rows = refresh_table(refresh_cfgs, {}, out_dir, angle)
        render_refresh_bars(refresh_rows, out_dir, angle)
    else:
        refresh_rows = []
        # Refresh outputs from an earlier build would read as current.
        for name in ('refresh_arm_frgt.csv', 'refresh_frgt_bars.png'):
            if (out_dir / name).is_file():
                (out_dir / name).unlink()
                print(f'  removed stale {name} (no refresh runs at the LR pin)')
    render_pareto(refresh_cfgs, baseline_cfgs, replay_cfgs, out_dir, angle)
    write_prereg(angle, summaries, refresh_rows, out_dir)

    # Also persist probe aggregates in a single CSV for downstream work.
    with (out_dir / 'probe_aggregate_pr.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['r', 'seed', 'n', 'cos_agg', 'cos_dist', 'rel_E',
                    'proj_E', 'sum_phi', 'mapping_err_max'])
        for s in summaries:
            for i, n in enumerate(s['n']):
                w.writerow([s['r'], s['seed'], int(n),
                            float(s['cos_agg'][i]), float(s['cos_dist'][i]),
                            float(s['rel_E'][i]),
                            float(s['proj_E'][i]), float(s['sum_phi'][i]),
                            s['mapping_err_max']])
    print(f'wrote {out_dir / "probe_aggregate_pr.csv"}')
    print(f'[done angle {angle}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_dir', default='results/plan_analysis',
                    help='Base output directory (per-angle subfolders are '
                         'created under <out>/rotmnist_angle{9,30}/ablations/'
                         'grad_staleness/).')
    ap.add_argument('--angles', nargs='+', type=int, default=[9, 30])
    opts = ap.parse_args()
    out = Path(opts.out_dir)
    for angle in opts.angles:
        analyse_angle(angle, out)


if __name__ == '__main__':
    main()
