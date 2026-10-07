"""Memory ablations per dataset at protocol LRs: one panel per (ablation, optimizer).

One figure per slice (rotMNIST split by angle; seq-CIFAR-10/100, CLEAR-10), or
per (slice, optimizer, arm) with ``arms``.
Panels: the ReplaySGD estimator arms (coverage × kernel with stale gradients,
the fresh arms, the refreshed slow EMA) and the AdEMAMix refreshed slow EMA.
The arm's cells are drawn only where every protocol seed finished;
references need at least two complete seeds. Configurations left out are listed
in the ``.md`` notes, not in the figure.
Each panel draws:

* the ablation's cells at the optimizer's α=0 best LR, filled, in greens by r
  (purple when the arm has no r), with the second axis (S, K, S' or C) in the
  label and the marker shape;
* references from ``plot_pareto_best_lr``, all at protocol LRs, as hollow
  markers in the Pareto colours: fine-tune (black), slow memory per r
  (blues), ER per K (reds).

Labels are placed by ``plot_labels.place_labels`` so they do not overlap, in
neutral ink: the marks carry the colour. A single-arm figure gets a legend that
lists the r colours, the second-axis shapes and the references it draws.

A cell is one group, one LR and one config, averaged over seeds only. Runs of
the arm at any other LR are counted in the ``.md`` notes, not plotted.

Writes ``results/plan_analysis/best_lr/ablations/<slice>_<optim>.png`` and ``.md``.

  python scripts/analysis/plot_ablations_best_lr.py [--refresh]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol_lr as P  # noqa: E402
import plot_pareto_best_lr as PP  # noqa: E402
from plot_labels import place_labels  # noqa: E402

NO_R_COLOR = '#6a3d9a'
# Second-axis shapes. The square and the triangle stay with the fine-tune and ER references.
SECOND_MARKER = ['o', 'D', 'P', 'X', 'v', 'h']
LABEL_INK, REF_INK = '0.15', '0.45'
# A reference (fine-tune, slow memory, ER) is drawn as mean ± std, so it needs at
# least two complete seeds; one with fewer is left out.
REF_MIN_SEEDS = 2

# (optim, arm key, title, ablation value, extra predicate)
PANELS = [
    ('replaysgd', 'fifo_uniform_stale', 'FIFO window × uniform kernel, stale',
     'fifo_uniform_stale', None),
    ('replaysgd', 'reservoir_geometric_stale',
     'reservoir × geometric kernel, stale (subsampled EMA target)',
     'reservoir_geometric_stale', None),
    ('replaysgd', 'reservoir_uniform_stale', 'reservoir × uniform kernel, stale (r = ∞)',
     'reservoir_uniform_stale', None),
    ('replaysgd', 'fifo_geometric_fresh', 'FIFO × geometric, fresh (whole window)',
     'fifo_geometric_fresh', lambda r: not r['K']),
    ('replaysgd', 'fifo_geometric_fresh_K', 'FIFO × geometric, fresh, budget K',
     'fifo_geometric_fresh', lambda r: bool(r['K'])),
    ('replaysgd', 'reservoir_uniform_fresh', 'reservoir × uniform, fresh (ER buffer as μ)',
     'reservoir_uniform_fresh', None),
    ('replaysgd', 'slow_ema_refresh', 'slow EMA + refresh correction',
     'slow_ema_refresh', None),
    ('ademamix', 'slow_ema_refresh', 'slow EMA + refresh correction',
     'slow_ema_refresh', None),
]


def arm_groups(dataset, angle, optim, arm):
    """Groups holding this arm, or (None, reason) when this slice has none."""
    if dataset == 'rotmnist':
        a = f'_a{angle}'
        if optim == 'ademamix':
            return ['plan_rotmnist_ademamix_ablation_slow_ema_refresh'], None
        return {
            'fifo_uniform_stale': [f'plan_rotmnist_ablation{a}_lattice_memory'],
            'reservoir_geometric_stale': [f'plan_rotmnist_ablation{a}_lattice_memory'],
            'reservoir_uniform_stale': [f'plan_rotmnist_ablation{a}_reservoir_uniform_stale'],
            # _v2 runs clip the recomputed memory gradient like the current one.
            'fifo_geometric_fresh': [f'plan_rotmnist_ablation{a}_fifo_geometric_fresh_v2'],
            'fifo_geometric_fresh_K': [f'plan_rotmnist_ablation{a}_fifo_geometric_fresh_K'],
            'reservoir_uniform_fresh': [f'plan_rotmnist_ablation{a}_reservoir_uniform_fresh'],
            'slow_ema_refresh': [f'plan_rotmnist_ablation_slow_ema_refresh{a}'],
        }[arm], None
    if optim == 'ademamix':
        return [f'plan_{dataset}_ademamix_ablation_slow_ema_refresh'], None
    if arm == 'fifo_uniform_stale':
        return None, 'not run: the M = 2τ gradient FIFO does not fit at ResNet-18 scale'
    if arm == 'slow_ema_refresh':
        return None, 'not run: ReplaySGD refresh is rotMNIST-only'
    return {
        'reservoir_geometric_stale': [f'plan_{dataset}_ablation_lattice_memory',
                                      f'plan_{dataset}_ablation_reservoir_geometric_stale_c100'],
        'reservoir_uniform_stale': [f'plan_{dataset}_ablation_reservoir_uniform_stale'],
        'fifo_geometric_fresh': [f'plan_{dataset}_ablation_fifo_geometric_fresh'],
        'fifo_geometric_fresh_K': [f'plan_{dataset}_ablation_fifo_geometric_fresh_K'],
        'reservoir_uniform_fresh': [f'plan_{dataset}_ablation_reservoir_uniform_fresh'],
    }[arm], None


def all_groups():
    groups = PP.all_groups()
    for dataset, angle in P.SLICES:
        for optim, arm, *_ in PANELS:
            gs, _ = arm_groups(dataset, angle, optim, arm)
            groups += gs or []
    return list(dict.fromkeys(groups))


def second_axis(arm, c):
    """(name, value) of the arm's non-r axis, or None."""
    if arm == 'fifo_geometric_fresh':
        return 'S', c['memory_samples'] or 'B'
    if arm in ('fifo_geometric_fresh_K', 'reservoir_uniform_fresh'):
        return 'K', c['K']
    if arm == 'slow_ema_refresh':
        return "S'", c['refresh_batches']
    if arm in ('reservoir_uniform_stale', 'reservoir_geometric_stale'):
        return 'C', c['buffer_size']
    return None


def second_key(sec):
    """Sort key for second-axis values: numeric, with unset (S = B) first."""
    if sec is None or isinstance(sec[1], str) or sec[1] is None:
        return (0, 0)
    return (1, sec[1])


def label(arm, c):
    bits = []
    if c['horizon_ratio'] is not None and arm != 'reservoir_uniform_stale':
        bits.append(f'r={c["horizon_ratio"]:g}')
    sec = second_axis(arm, c)
    if sec and arm != 'reservoir_geometric_stale':
        bits.append(f'{sec[0]}={sec[1]}')
    if c['group'].endswith('_c100'):
        bits.append('relaunch')
    return ','.join(bits) or arm


def cell_color(c):
    """Ablation cells: greens by r, purple when the arm has no horizon."""
    if c['horizon_ratio'] is None or c.get('arm') == 'reservoir_uniform_stale':
        return NO_R_COLOR
    return PP._shade(plt.cm.Greens, c['horizon_ratio'], 0.3, 10.0)


def ref_label(c):
    """Reference point label, prefixed so it does not read as an ablation cell."""
    text = PP.point_label(c['family'], c)
    return {'slow memory': f'slow {text}', 'ER': f'ER {text}'}.get(c['family'], text)


def arm_legend(arm, refs, cells, seconds):
    """Legend columns [(header, handles)] for one arm: the r colours and second-axis
    shapes it draws, then the fine-tune / slow-memory and the ER references."""
    def point(label, marker, color, hollow=False):
        style = (dict(markerfacecolor='none', markeredgecolor=color, markeredgewidth=1.2)
                 if hollow else dict(markerfacecolor=color, markeredgecolor='black',
                                     markeredgewidth=0.5))
        return plt.Line2D([], [], ls='none', marker=marker, markersize=6, label=label, **style)

    columns = []
    rs = sorted({c['horizon_ratio'] for c in cells if cell_color(c) != NO_R_COLOR})
    if rs:
        columns.append(('ablation: colour = r',
                        [point(f'r={r:g}', 'o', cell_color({'horizon_ratio': r, 'arm': arm}))
                         for r in rs]))
    elif cells:
        columns.append(('ablation', [point('no horizon r', 'o', NO_R_COLOR)]))
    if seconds and seconds[0] is not None:
        name = seconds[0][0]
        columns.append((f'ablation: shape = {name}',
                        [point(f'{name}={v}', SECOND_MARKER[i % len(SECOND_MARKER)], '0.6')
                         for i, (_, v) in enumerate(seconds)]))
    core = [c for c in refs if c['family'] in ('fine-tune', 'slow memory')]
    if core:
        core.sort(key=lambda c: (c['family'] != 'fine-tune', c['horizon_ratio'] or 0))
        columns.append(('references (hollow)',
                        [point('fine-tune' if c['family'] == 'fine-tune'
                               else f'slow memory r={c["horizon_ratio"]:g}',
                               PP.FAMILY_MARKER[c['family']], PP.family_color(c), hollow=True)
                         for c in core]))
    er = sorted((c for c in refs if c['family'] == 'ER'), key=lambda c: c['K'] or 0)
    if er:
        columns.append(('ER references (hollow)',
                        [point(ref_label(c), PP.FAMILY_MARKER['ER'], PP.family_color(c), hollow=True)
                         for c in er]))
    return columns


def legend_handles():
    hollow = dict(ls='none', markerfacecolor='none', markeredgewidth=1.2)
    return [plt.Line2D([], [], marker='s', color=PP.FINE_TUNE_COLOR, label='fine-tune (ref.)', **hollow),
            plt.Line2D([], [], marker='o', color=plt.cm.Blues(0.75), label='slow memory (ref., by r)', **hollow),
            plt.Line2D([], [], marker='^', color=plt.cm.Reds(0.75), label='ER (ref., by K)', **hollow),
            plt.Line2D([], [], marker='o', ls='none', color=plt.cm.Greens(0.75), markeredgecolor='black',
                       label='ablation cell (darker = larger r)'),
            plt.Line2D([], [], marker='o', ls='none', color=NO_R_COLOR, markeredgecolor='black',
                       label='ablation cell, no r'),
            plt.Line2D([], [], marker='D', ls='none', color='0.6', markeredgecolor='black',
                       label="shape = second axis (S, K, S', C)")]


def _ref_complete(c):
    """At least ``REF_MIN_SEEDS`` protocol seeds complete."""
    return len(set(c['seeds']) & set(P.SEEDS)) >= REF_MIN_SEEDS


def references(runs_by_group, dataset, angle, optim, protocol):
    p = protocol[(dataset, angle, optim)]
    refs = []
    for family, tiers in PP.sources(dataset, angle, optim).items():
        flr = p['er_lr'] if family == 'ER' else p['lr']
        cs, _ = PP.family_cells(runs_by_group, dataset, angle, optim, family, tiers, flr)
        for c in cs:
            c['family'] = family
        refs += [c for c in cs if _ref_complete(c)]
    return refs


def _draw_points(ax, refs, cells, arm, seconds, scale=1.0):
    """Hollow references and filled arm cells; returns label items for every point."""
    items = []
    for c in refs:
        color = PP.family_color(c)
        # Seed std of the reference, lighter than the arm's own error bars.
        ax.errorbar(c['frgt'], c['acc'], xerr=c['frgt_std'], yerr=c['acc_std'],
                    fmt='none', ecolor=color, alpha=0.3, lw=0.7, zorder=1)
        ax.scatter(c['frgt'], c['acc'], marker=PP.FAMILY_MARKER[c['family']], s=40 * scale,
                   facecolors='none', edgecolors=color, linewidths=1.1, alpha=0.9, zorder=1)
        # Reference labels give way to the arm's labels and are dropped if no spot is free.
        items.append(dict(x=c['frgt'], y=c['acc'], text=ref_label(c),
                          color=REF_INK, size=5.8, priority=0, required=False))
    for c in sorted(cells, key=lambda c: (c['horizon_ratio'] or 0, second_key(second_axis(arm, c)))):
        color = cell_color(c)
        marker = SECOND_MARKER[seconds.index(second_axis(arm, c)) % len(SECOND_MARKER)]
        ax.errorbar(c['frgt'], c['acc'], xerr=c['frgt_std'], yerr=c['acc_std'],
                    fmt='none', ecolor=color, alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(c['frgt'], c['acc'], marker=marker, s=48 * scale, color=color,
                   edgecolors='black', linewidths=0.5, zorder=3)
        items.append(dict(x=c['frgt'], y=c['acc'], text=label(arm, c), color=LABEL_INK,
                          size=6.5, priority=1))
    return items


def _zoom_box(refs, cells):
    """Data box around the fine-tune / slow-memory references and the points near them."""
    core = [c for c in refs if c['family'] in ('fine-tune', 'slow memory')]
    if not core:
        return None
    xs, ys = [c['frgt'] for c in core], [c['acc'] for c in core]
    sx, sy = max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)
    near = [c for c in refs + cells
            if min(xs) - 1.5 * sx <= c['frgt'] <= max(xs) + 1.5 * sx
            and min(ys) - 1.5 * sy <= c['acc'] <= max(ys) + 1.5 * sy]
    xs, ys = [c['frgt'] for c in near], [c['acc'] for c in near]
    px, py = max(0.15 * (max(xs) - min(xs)), 0.6), max(0.15 * (max(ys) - min(ys)), 0.6)
    return min(xs) - px, max(xs) + px, min(ys) - py, max(ys) + py, near


def _zoom_useful(ax, zoom):
    """True when the zoom box is small on both axes (at most half of each range and
    at most 35% of one), so an inset magnifies it rather than repeating the panel."""
    if zoom is None:
        return False
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fx, fy = (zoom[1] - zoom[0]) / (x1 - x0), (zoom[3] - zoom[2]) / (y1 - y0)
    return fx <= 0.5 and fy <= 0.5 and min(fx, fy) <= 0.35


# Inset slots in axes fractions: four corners, each in a large and a small size.
# Left and bottom slots keep clear of the main spines for the inset's tick labels.
_INSET_SLOTS = [(0.52, 0.46, 0.45, 0.50), (0.08, 0.10, 0.44, 0.46),
                (0.52, 0.10, 0.45, 0.46), (0.08, 0.46, 0.44, 0.50),
                (0.58, 0.53, 0.39, 0.43), (0.08, 0.10, 0.38, 0.40),
                (0.58, 0.10, 0.39, 0.40), (0.08, 0.53, 0.38, 0.43)]
# A marker this close to a slot (fraction of the axis range) counts as covered:
# its label would land on the inset frame or its tick labels.
_SLOT_MARGIN = 0.05


def _inset_slot(ax, points, box):
    """Slot that leaves the zoomed box visible and covers the fewest markers
    (with a margin); among equals, the larger slot."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    mx, my = _SLOT_MARGIN * (x1 - x0), _SLOT_MARGIN * (y1 - y0)
    best = None
    for slot in _INSET_SLOTS:
        sx0, sy0 = x0 + slot[0] * (x1 - x0), y0 + slot[1] * (y1 - y0)
        sx1, sy1 = sx0 + slot[2] * (x1 - x0), sy0 + slot[3] * (y1 - y0)
        covered = sum(sx0 - mx <= c['frgt'] <= sx1 + mx and sy0 - my <= c['acc'] <= sy1 + my
                      for c in points)
        hits_box = not (box[1] < sx0 or box[0] > sx1 or box[3] < sy0 or box[2] > sy1)
        rank = (hits_box, covered, -slot[2] * slot[3])
        if best is None or rank < best[0]:
            best = (rank, slot)
    return best[1]


def plot_slice(dataset, angle, protocol, runs_by_group, out_dir, optims=P.OPTIMS, stem=None,
               arms=None):
    """Draw the panels of ``optims`` (and ``arms``, if given) for one slice; returns the arm cells.

    Panels where the fine-tune / slow-memory cluster is small next to the axis
    range get an inset that zooms on it. Labels of points in the zoomed box go
    in the inset; the rest stay on the panel and keep clear of the inset.

    Titles name the dataset and the ablation only; learning rates and run counts
    go in the figure description. Configurations at the protocol LR that lack a
    finished seed are listed in the ``.md`` notes only. A single-panel figure has
    one title, at the top.
    """
    acc_key, frgt_key, ro = P.readout(dataset)
    panels = [p for p in PANELS if p[0] in optims and (arms is None or p[1] in arms)]
    ncol = min(4, len(panels))
    nrow = (len(panels) + ncol - 1) // ncol
    # A single arm gets a taller figure for its per-value legend.
    fig_w, fig_h = max(5.0 * ncol, 7.0), (5.2 if len(panels) == 1 else 4.5 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    out_cells, drawn = [], []
    md = [f'# {P.slice_title(dataset, angle)} — ablations at protocol LRs', '']
    refs_by_optim = {o: references(runs_by_group, dataset, angle, o, protocol) for o in P.OPTIMS}
    single = len(panels) == 1
    head = None  # the single panel's title, moved to the figure title
    for ax, (optim, arm, title, ablation, pred) in zip(axes, panels):
        lr = protocol[(dataset, angle, optim)]['lr']
        ax.grid(alpha=0.25)
        refs = refs_by_optim[optim]
        md.append(f'## {P.OPTIM_NAME[optim]} — {arm} (lr={lr:g})')
        groups, reason = arm_groups(dataset, angle, optim, arm)
        panel = f'{P.OPTIM_NAME[optim]}: {title}'
        if groups is None:
            if single:
                head = panel
            else:
                ax.set_title(panel, fontsize=8.5)
            ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha='center', va='center',
                    fontsize=7.5, wrap=True, color='0.3')
            md += [f'- {reason}', '']
            continue
        recs = [r for g in groups for r in runs_by_group.get(g, [])
                if r['optim'] == optim and r['angle'] == angle and r['ablation'] == ablation
                and (pred is None or pred(r))]
        at_lr = [r for r in recs if P.lr_eq(r['lr'], lr)]
        off = sorted({r['lr'] for r in recs if not P.lr_eq(r['lr'], lr)})
        all_cs = P.cells(at_lr, dataset)
        cs = [c for c in all_cs if P.has_all_seeds(c)]
        dropped = [c for c in all_cs if not P.has_all_seeds(c)]
        seconds = sorted({second_axis(arm, c) for c in cs}, key=second_key)
        for c in cs:
            c.update(dataset=dataset, arm=arm, label=label(arm, c))
            md.append(f'- `{label(arm, c)}` lr={c["lr"]:g} group=`{c["group"]}` → '
                      f'{acc_key}={c["acc"]:.2f}±{c["acc_std"]:.2f} '
                      f'{frgt_key}={c["frgt"]:.2f}±{c["frgt_std"]:.2f} (n={c["n"]}, seeds {c["seeds"]})')
        for c in dropped:
            md.append(f'- not plotted, seeds incomplete: `{label(arm, c)}` group=`{c["group"]}` '
                      f'(complete seeds {c["seeds"]})')
        out_cells += cs
        # Configurations at the LR without every seed finished (some may have none).
        shown = {tuple(c[k] for k in P.CELL_KEYS) for c in cs}
        unfinished = {}
        for r in at_lr:
            if P.cell_key(r) not in shown:
                unfinished.setdefault(P.cell_key(r), r)
        names = sorted(label(arm, r) for r in unfinished.values())
        if names:
            md.append(f'- not shown, seeds unfinished: {"; ".join(names)}')
        n_run = len({r['id'] for r in at_lr})
        n_done = sum(c['n'] for c in cs)
        status = f'lr={lr:g}: {len(cs)} cells, {n_done}/{n_run} runs used'
        if dropped:
            status += f'; {len(dropped)} cells without all seeds not plotted'
        if off:
            n_off = sum(1 for r in recs if not P.lr_eq(r['lr'], lr))
            status += f'; excluded {n_off} runs at lr={", ".join(f"{x:g}" for x in off)}'
        if not recs:
            status = 'no runs'
        if single:
            head = panel
        else:
            ax.set_title(panel, fontsize=8.5)
        md += [f'- {status}', '']
        drawn.append((ax, refs, cs, arm, seconds, _draw_points(ax, refs, cs, arm, seconds)))
    for ax in axes[len(panels):]:
        ax.set_axis_off()
    sfx = f' ({ro})' if ro else ''
    for ax in axes[-ncol:]:
        ax.set_xlabel(f'forgetting {frgt_key}{sfx}')
    for ax in axes[::ncol]:
        ax.set_ylabel(f'{acc_key}{sfx}')
    title = f'{P.slice_title(dataset, angle)} — {head if single else "memory ablations"}'
    # Wrap to the figure width so single-panel figures keep the whole title.
    lines = textwrap.wrap(title, width=int(12.5 * fig_w))
    title_h = 0.19 * len(lines) + 0.12
    rows = None
    if single and drawn:
        # One column per legend group, filled top to bottom; headers in bold.
        _, refs0, cs0, arm0, seconds0, _ = drawn[0]
        columns = arm_legend(arm0, refs0, cs0, seconds0)
        rows = 1 + max(len(entries) for _, entries in columns)
        handles = []
        for head, entries in columns:
            col = [plt.Line2D([], [], ls='none', label=head)] + entries
            handles += col + [plt.Line2D([], [], ls='none', label='')
                              for _ in range(rows - len(col))]
        ncol_leg = len(columns)
    else:
        handles = legend_handles()
        ncol_leg = len(handles) if fig_w >= 12 else 3 if fig_w >= 7 else 2
    leg = fig.legend(handles=handles, loc='upper center', ncol=ncol_leg, fontsize=7.5,
                     frameon=False, bbox_to_anchor=(0.5, 1 - title_h / fig_h),
                     columnspacing=1.6, handletextpad=0.5)
    if rows:
        for i, text in enumerate(leg.get_texts()):
            if i % rows == 0:
                text.set_fontweight('bold')
    # Panels start right under the legend, measured once it is drawn.
    fig.canvas.draw()
    leg_bottom = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.tight_layout(rect=(0, 0, 1, leg_bottom - 0.05 / fig_h))
    # Added after the layout: tight_layout would reserve room for it a second time.
    fig.suptitle('\n'.join(lines), fontsize=9.5, y=1 - 0.08 / fig_h)
    fig.canvas.draw()  # final (shared) limits before choosing the zooms
    renderer = fig.canvas.get_renderer()
    for ax, refs, cs, arm, seconds, items in drawn:
        zoom = _zoom_box(refs, cs)
        # Zoom only where the arm has cells and the cluster is small on the axes.
        if not cs or not _zoom_useful(ax, zoom):
            place_labels(ax, items)
            continue
        zx0, zx1, zy0, zy1, near = zoom
        inset = ax.inset_axes(_inset_slot(ax, refs + cs, zoom))
        inner = _draw_points(inset, [c for c in refs if c in near], [c for c in cs if c in near],
                             arm, seconds, scale=0.9)
        inset.set_xlim(zx0, zx1)
        inset.set_ylim(zy0, zy1)
        inset.tick_params(labelsize=6)
        inset.xaxis.set_major_locator(MaxNLocator(4))
        inset.yaxis.set_major_locator(MaxNLocator(4))
        inset.grid(alpha=0.25)
        ax.indicate_inset_zoom(inset, edgecolor='0.5', alpha=0.8)
        outside = [it for it in items
                   if not (zx0 <= it['x'] <= zx1 and zy0 <= it['y'] <= zy1)]
        # The inset's tick labels sit outside its axes box; keep labels off them too.
        place_labels(ax, outside, extra_obstacles=[inset.get_tightbbox(renderer)])
        place_labels(inset, inner)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f'{P.slice_id(dataset, angle)}_{"_".join(optims)}'
    fig.savefig(out_dir / f'{stem}.png', dpi=150)
    plt.close(fig)
    (out_dir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {out_dir / stem}.png')
    return out_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--refresh', action='store_true', help='refetch every group from W&B')
    args = ap.parse_args()
    runs = P.load_runs(all_groups(), refresh=args.refresh)
    runs_by_group = {}
    for r in runs:
        runs_by_group.setdefault(r['group'], []).append(r)
    protocol = P.protocol_table()
    for dataset, angle in P.SLICES:
        for optim in P.OPTIMS:
            plot_slice(dataset, angle, protocol, runs_by_group, P.OUT_ROOT / 'ablations',
                       optims=(optim,))


if __name__ == '__main__':
    main()
