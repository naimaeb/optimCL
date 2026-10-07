"""Pareto front (ACC vs forgetting) per dataset and optimizer, protocol LRs only.

One figure per (slice, optimizer); rotMNIST is split by angle. Single panel in the
style of the ablation figures (column legend above the axes, zoom inset on the
fine-tune / slow-memory cluster); learning rates are left to the table summary.
Each figure holds:

* fine-tune (α = 0) at the α=0 best LR,
* slow memory (α = 1, one point per r) at the same LR,
* ER (one point per K) at the protocol ER LR (``protocol_lr.er_lr``).

Colours: fine-tune black, slow memory blues by r, ER reds by K. Labels are
placed by ``plot_labels.place_labels`` so they do not overlap.

The protocol and the cell rule live in ``protocol_lr.py``. Every point is the
mean ± std over seeds of a single cell (one group, one LR, one config), drawn
only when every seed of its sweep finished (unfinished ones are flagged) and
its ACC is above chance (chance-level cells are left out and listed).
Each family reads its first source tier with complete runs at the protocol
LR. A family with none is listed in the figure's corner, with the LRs it does
have, instead of being substituted.

Writes ``results/plan_analysis/best_lr/pareto/<slice>_<optim>.png`` and ``.md``, plus
``results/plan_analysis/best_lr/protocol_lrs.md``.

  python scripts/analysis/plot_pareto_best_lr.py [--refresh]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import math

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol_lr as P  # noqa: E402
from plot_labels import place_labels  # noqa: E402

FAMILY_MARKER = {'fine-tune': 's', 'slow memory': 'o', 'ER': '^'}
FAMILIES = ('fine-tune', 'slow memory', 'ER')
FINE_TUNE_COLOR = '#1a1a1a'


def _shade(cmap, value, lo, hi):
    """Colour from ``cmap`` by log position of ``value`` in [lo, hi] (light to dark)."""
    if value is None:
        return cmap(0.7)
    t = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return cmap(0.4 + 0.55 * min(1.0, max(0.0, t)))


def family_color(c):
    """Fine-tune black; slow memory in blues by r; ER in reds by K."""
    if c['family'] == 'slow memory':
        return _shade(plt.cm.Blues, c['horizon_ratio'], 0.3, 10.0)
    if c['family'] == 'ER':
        return _shade(plt.cm.Reds, c['K'], 200, 5120)
    return FINE_TUNE_COLOR


def family_handles():
    return [plt.Line2D([], [], color=FINE_TUNE_COLOR, marker='s', ls='none', label='fine-tune'),
            plt.Line2D([], [], color=plt.cm.Blues(0.75), marker='o', ls='none',
                       label='slow memory (darker = larger r)'),
            plt.Line2D([], [], color=plt.cm.Reds(0.75), marker='^', ls='none',
                       label='ER (darker = larger K)')]


def _is_ft(r):
    return (r['alpha'] or 0.0) == 0.0 and not r['replay']


def _is_mem(r):
    # Slow memory is reported at α = 1 only.
    return (r['alpha'] or 0.0) == 1.0 and not r['replay'] and r['ablation'] in (None, 'slow_ema')


def _is_er(r):
    return bool(r['replay']) and (r['alpha'] or 0.0) == 0.0


def sources(dataset, angle, optim):
    """{family: [tier, ...]}; a tier is a list of (group, predicate)."""
    a = '' if angle is None else f'_a{angle}'
    if dataset == 'rotmnist' and optim == 'replaysgd':
        return {
            'fine-tune': [[(f'plan_rotmnist_replaysgd_baseline{a}', _is_ft)],
                          [(f'plan_rotmnist_ablation{a}_lattice_control',
                            lambda r: r['ablation'] == 'fast_only')]],
            'slow memory': [[(f'plan_rotmnist_replaysgd_memory{a}', _is_mem)],
                            [(f'plan_rotmnist_ablation{a}_lattice_memory', _is_mem)]],
            'ER': [[('plan_rotmnist_replaysgd_er_v2', _is_er)],
                   [('plan_rotmnist_replaysgd_er', _is_er)]],
        }
    mem_tiers = [[(f'plan_{dataset}_{optim}_memory', _is_mem)]]
    if optim == 'replaysgd':
        tier = [(f'plan_{dataset}_ablation_lattice_memory', _is_mem)]
        if dataset == 'seqcifar100':
            tier.append(('plan_seqcifar100_ablation_slow_ema_r10_reseed', _is_mem))
        mem_tiers.append(tier)
    if (dataset, optim) == ('clear10', 'ademamix'):
        # Normalised-mix group; keep β4 = β3 as in the other AdEMAMix memory groups.
        mem_tiers.append([('plan_clear10_ademamix_mixnorm',
                           lambda r: _is_mem(r) and r['beta4_from_horizon'] == 1)])
    return {
        'fine-tune': [[(f'plan_{dataset}_{optim}_baseline', _is_ft)],
                      [(f'plan_{dataset}_{optim}_lr', _is_ft)]],
        'slow memory': mem_tiers,
        'ER': [[(f'plan_{dataset}_{optim}_er_v2', _is_er)],
               [(f'plan_{dataset}_{optim}_er', _is_er)]],
    }


def all_groups():
    groups = []
    for dataset, angle in P.SLICES:
        for optim in P.OPTIMS:
            groups.append(f'plan_{dataset}_{optim}_lr')
            for tiers in sources(dataset, angle, optim).values():
                groups += [g for tier in tiers for g, _ in tier]
    return list(dict.fromkeys(groups))


def family_cells(runs_by_group, dataset, angle, optim, family, tiers, lr):
    """(cells at lr from the first tier that has any, note or None)."""
    seen_lrs = set()
    for tier in tiers:
        recs = []
        for group, pred in tier:
            for r in runs_by_group.get(group, []):
                if r['angle'] != angle or r['optim'] != optim or not pred(r):
                    continue
                if P.complete(r, dataset):
                    seen_lrs.add(r['lr'])
                if P.lr_eq(r['lr'], lr):
                    recs.append(r)
        cs = P.cells(recs, dataset)
        if cs:
            return cs, None
    other = ', '.join(f'{x:g}' for x in sorted(seen_lrs)) or 'none'
    return [], (f'{P.OPTIM_NAME[optim]} {family}: no complete runs at lr={lr:g} '
                f'(complete runs exist at: {other})')


def point_label(family, c):
    if family == 'slow memory':
        bits = [f'r={c["horizon_ratio"]:g}']
        if c['alpha'] not in (None, 1.0):
            bits.append(f'α={c["alpha"]:g}')
        return ','.join(bits)
    if family == 'ER':
        return f'K={c["K"]}'
    return 'FT'


LABEL_INK = '0.15'  # as in the ablation figures


# Cells accepted with fewer seeds: (group, r) -> minimum number of complete seeds.
PARTIAL_OK = {('plan_seqcifar10_ademamix_memory', 10.0): 2,
              ('plan_rotmnist_replaysgd_er', None): 2}
# A cell whose mean ACC is within this many points of chance learned nothing; it is
# left out of the figure (and its front) and listed in the tables instead.
CHANCE_MARGIN = 1.0
CHANCE_ACC = {
    ('rotmnist', 'ACC'): 10.0, ('rotmnist_cyclic', 'ACC'): 10.0,
    ('clear10', 'ACC'): 100.0 / 11, ('core50', 'ACC'): 2.0,
    ('seqcifar10', 'ACC'): 10.0, ('seqcifar10', 'ACC_TIL'): 50.0,
    ('seqcifar100', 'ACC'): 1.0, ('seqcifar100', 'ACC_TIL'): 10.0,
}


def _is_complete(c):
    """Every seed the source sweep grids has a complete run (``PARTIAL_OK`` lists exceptions)."""
    need = PARTIAL_OK.get((c['group'], c['horizon_ratio']))
    if need is not None:
        return c['n'] >= need
    if c['group'].endswith('_lr'):
        return {42, 123} <= set(c['seeds'])
    return P.has_all_seeds(c)


def shown_cells(runs_by_group, dataset, angle, optim, family, tiers, lr):
    """(complete cells, labels of configurations left out, note).

    Reads the tiers in order and stops at the first one with a complete cell.
    Configurations at ``lr`` without every seed finished, in the tiers read, are
    returned as labels so the figure can flag them.
    """
    pending = []
    for tier in tiers:
        recs = [r for group, pred in tier for r in runs_by_group.get(group, [])
                if r['angle'] == angle and r['optim'] == optim and pred(r) and P.lr_eq(r['lr'], lr)]
        if not recs:
            continue
        done = [c for c in P.cells(recs, dataset) if _is_complete(c)]
        done_keys = {tuple(c[k] for k in P.CELL_KEYS) for c in done}
        for r in recs:
            text = f'{family} {point_label(family, r)}' if family != 'fine-tune' else family
            if P.cell_key(r) not in done_keys and text not in pending:
                pending.append(text)
        if done:
            # A configuration shown from another group of the tier is not flagged.
            shown = {f'{family} {point_label(family, c)}' if family != 'fine-tune' else family
                     for c in done}
            return done, [t for t in pending if t not in shown], None
    note = None if pending else f'{family}: no runs at lr={lr:g}'
    return [], pending, note


def _draw(ax, pts, front, fontsize, scale=1.0):
    """Points, error bars and the front line; returns label items for ``place_labels``."""
    items = []
    for c, on in zip(pts, front):
        color = family_color(c)
        ax.errorbar(c['frgt'], c['acc'], xerr=c['frgt_std'], yerr=c['acc_std'],
                    fmt='none', ecolor=color, alpha=0.35, lw=0.9, zorder=2)
        ax.scatter(c['frgt'], c['acc'], marker=FAMILY_MARKER[c['family']], s=60 * scale,
                   color=color, edgecolors='black' if on else 'white',
                   linewidths=1.4 if on else 0.6, zorder=3)
        items.append(dict(x=c['frgt'], y=c['acc'], text=point_label(c['family'], c),
                          color=LABEL_INK, size=fontsize, priority=1))
    fr = sorted((c for c, on in zip(pts, front) if on), key=lambda c: c['frgt'])
    ax.plot([c['frgt'] for c in fr], [c['acc'] for c in fr], color='0.6', lw=0.8,
            ls='--', zorder=1)
    ax.grid(alpha=0.25)
    return items


def pareto_legend(pts):
    """Legend columns [(header, handles)]: fine-tune, slow memory by r, ER by K, the front."""
    def point(label, marker, color, edge='black', width=0.5):
        return plt.Line2D([], [], ls='none', marker=marker, markersize=6, label=label,
                          markerfacecolor=color, markeredgecolor=edge, markeredgewidth=width)
    columns = []
    if any(c['family'] == 'fine-tune' for c in pts):
        columns.append(('fine-tune', [point('α = 0', 's', FINE_TUNE_COLOR)]))
    mem = sorted({c['horizon_ratio'] for c in pts if c['family'] == 'slow memory'})
    if mem:
        columns.append(('slow memory: colour = r',
                        [point(f'r={r:g}', 'o', family_color({'family': 'slow memory', 'horizon_ratio': r}))
                         for r in mem]))
    ks = sorted({c['K'] for c in pts if c['family'] == 'ER'})
    if ks:
        columns.append(('ER: colour = K',
                        [point(f'K={k}', '^', family_color({'family': 'ER', 'K': k})) for k in ks]))
    columns.append(('Pareto front', [plt.Line2D([], [], color='0.6', ls='--', label='front'),
                                     point('on the front', 'o', '0.85', edge='black', width=1.4)]))
    return columns


def plot_slice(dataset, angle, protocol, runs_by_group, out_dir, optim, stem=None):
    """Draw one slice for one optimizer; returns (plotted cells, flags).

    Single panel in the ablation-figure style: the title names the dataset and
    optimizer (learning rates go in the table summary), a column legend sits
    above the axes, and the fine-tune / slow-memory cluster gets a zoom inset
    when it is small next to the axis range. Only cells with every seed finished
    are drawn; the rest are listed in the ``.md`` notes. Cells at chance are
    named under the title.
    """
    from plot_ablations_best_lr import _inset_slot, _zoom_box, _zoom_useful  # shared zoom layout
    from matplotlib.ticker import MaxNLocator

    acc_key, frgt_key, ro = P.readout(dataset)
    fig_w, fig_h = 7.0, 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    p = protocol[(dataset, angle, optim)]
    md = [f'# {P.slice_title(dataset, angle)} — {P.OPTIM_NAME[optim]} Pareto front',
          f'lr={p["lr"]:g}; ER lr={p["er_lr"]:g} ({p["er_source"]})', '']
    flags, all_pts = [], []
    for family, tiers in sources(dataset, angle, optim).items():
        flr = p['er_lr'] if family == 'ER' else p['lr']
        cs, pending, note = shown_cells(runs_by_group, dataset, angle, optim, family, tiers, flr)
        flags += pending + ([note] if note else [])
        for c in sorted(cs, key=lambda c: (c['horizon_ratio'] or 0, c['alpha'] or 0, c['K'] or 0)):
            c.update(optim=optim, family=family, dataset=dataset, label=point_label(family, c))
            all_pts.append(c)
            md.append(f'- {family} `{c["label"]}` lr={c["lr"]:g} group=`{c["group"]}` '
                      f'→ {acc_key}={c["acc"]:.2f}±{c["acc_std"]:.2f} '
                      f'{frgt_key}={c["frgt"]:.2f}±{c["frgt_std"]:.2f} (n={c["n"]}, seeds {c["seeds"]})')
    md += [f'- not shown: {f}' for f in flags]
    # Chance-level cells are dropped from the figure and its front; the tables list them.
    chance = CHANCE_ACC[(dataset, acc_key)]
    for c in all_pts:
        c['at_chance'] = c['acc'] <= chance + CHANCE_MARGIN
    drawn = [c for c in all_pts if not c['at_chance']]
    dropped = [f'{c["family"]} {c["label"]}' if c['family'] != 'fine-tune' else c['family']
               for c in all_pts if c['at_chance']]
    md += [f'- not shown, at chance ({acc_key} ≤ {chance:g} + {CHANCE_MARGIN:g}): {d}' for d in dropped]
    front = P.pareto_mask(drawn) if drawn else []
    items = _draw(ax, drawn, front, 6.5)
    sfx = f' ({ro})' if ro else ''
    ax.set_xlabel(f'forgetting {frgt_key}{sfx} (lower is better)')
    ax.set_ylabel(f'final average accuracy {acc_key}{sfx}')
    lines = textwrap.wrap(f'{P.slice_title(dataset, angle)} — {P.OPTIM_NAME[optim]}',
                          width=int(12.5 * fig_w))
    # Configurations with unfinished seeds are listed in the .md notes, not the title.
    if dropped:
        lines += textwrap.wrap('not shown, at chance: ' + '; '.join(dropped), width=int(12 * fig_w))
    title_h = 0.19 * len(lines) + 0.12
    # One legend column per group, filled top to bottom; headers in bold.
    columns = pareto_legend(drawn)
    rows = 1 + max(len(entries) for _, entries in columns)
    handles = []
    for head, entries in columns:
        col = [plt.Line2D([], [], ls='none', label=head)] + entries
        handles += col + [plt.Line2D([], [], ls='none', label='') for _ in range(rows - len(col))]
    leg = fig.legend(handles=handles, loc='upper center', ncol=len(columns), fontsize=7.5,
                     frameon=False, bbox_to_anchor=(0.5, 1 - title_h / fig_h),
                     columnspacing=1.6, handletextpad=0.5)
    for i, text in enumerate(leg.get_texts()):
        if i % rows == 0:
            text.set_fontweight('bold')
    fig.canvas.draw()
    leg_bottom = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.tight_layout(rect=(0, 0, 1, leg_bottom - 0.05 / fig_h))
    fig.suptitle('\n'.join(lines), fontsize=9.5, y=1 - 0.08 / fig_h)
    fig.canvas.draw()
    zoom = _zoom_box(drawn, []) if drawn else None
    # Zoom only when the fine-tune / slow-memory cluster is small on the axes.
    if not _zoom_useful(ax, zoom):
        place_labels(ax, items)
    else:
        zx0, zx1, zy0, zy1, near = zoom
        inset = ax.inset_axes(_inset_slot(ax, drawn, zoom))
        inner_pts = [c for c in drawn if c in near]
        inner = _draw(inset, inner_pts, [on for c, on in zip(drawn, front) if c in near],
                      6.5, scale=0.9)
        inset.set_xlim(zx0, zx1)
        inset.set_ylim(zy0, zy1)
        inset.tick_params(labelsize=6)
        inset.xaxis.set_major_locator(MaxNLocator(4))
        inset.yaxis.set_major_locator(MaxNLocator(4))
        ax.indicate_inset_zoom(inset, edgecolor='0.5', alpha=0.8)
        outside = [it for it in items if not (zx0 <= it['x'] <= zx1 and zy0 <= it['y'] <= zy1)]
        # The inset's tick labels sit outside its axes box; keep labels off them too.
        place_labels(ax, outside, extra_obstacles=[inset.get_tightbbox(fig.canvas.get_renderer())])
        place_labels(inset, inner)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f'{P.slice_id(dataset, angle)}_{optim}'
    fig.savefig(out_dir / f'{stem}.png', dpi=150)
    plt.close(fig)
    (out_dir / f'{stem}.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {out_dir / stem}.png  ({len(drawn)} points, {len(flags)} unfinished, '
          f'{len(dropped)} at chance)')
    return all_pts, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--refresh', action='store_true', help='refetch every group from W&B')
    args = ap.parse_args()
    groups = all_groups()
    runs = P.load_runs(groups, refresh=args.refresh)
    runs_by_group = {}
    for r in runs:
        runs_by_group.setdefault(r['group'], []).append(r)
    protocol = P.protocol_table()
    P.write_protocol_md(protocol, P.OUT_ROOT / 'protocol_lrs.md')
    for dataset, angle in P.SLICES:
        for optim in P.OPTIMS:
            plot_slice(dataset, angle, protocol, runs_by_group, P.OUT_ROOT / 'pareto', optim)


if __name__ == '__main__':
    main()
