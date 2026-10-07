"""Protocol learning rates and run loading for the best-LR figures.

Protocol:

* Fine-tuning and slow memory use the best LR of the α=0 ``lr`` sweep for
  that (dataset, angle, optimizer): highest mean final ACC over complete
  seeds, ties to more seeds, then to the smaller LR. Split CIFAR picks on
  Task-IL ACC.
* ER with SGD (ReplaySGD) uses the ER LR of Mammoth ``utils/best_args.py``
  (model ``er``) where it exists, else (CLEAR-10) the α=0 best LR. Avalanche's
  ``examples/clear.py`` trains Naive fine-tuning, so it is not an ER value.
* ER with AdEMAMix: the literature values are for SGD. On rotMNIST and
  CLEAR-10 it uses the best LR of the AdEMAMix ER LR sweep
  (``ADEMAMIX_ER_SWEEP``, averaged over cyclic and one-pass rotMNIST); on
  seq-CIFAR it uses the α=0 best LR.

Cells are keyed by group and every config value except the seed, so a cell
never mixes learning rates, groups or protocol variants.

Reads W&B group by group and caches the records in
``results/plan_analysis/cache/protocol_runs.json``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ENTITY = '<wandb-entity>'
PROJECT = 'cl_optimizer_memory'
REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / 'results' / 'plan_analysis' / 'cache' / 'protocol_runs.json'
OUT_ROOT = REPO / 'results' / 'plan_analysis' / 'best_lr'

SPLIT = frozenset({'seqcifar10', 'seqcifar100'})
OPTIMS = ('replaysgd', 'ademamix')
OPTIM_NAME = {'replaysgd': 'ReplaySGD', 'ademamix': 'AdEMAMix'}
DATASET_NAME = {'rotmnist': 'rotMNIST', 'seqcifar10': 'seq-CIFAR-10',
                'seqcifar100': 'seq-CIFAR-100', 'clear10': 'CLEAR-10'}
# (dataset, angle); rotMNIST angles are never pooled.
SLICES = (('rotmnist', 9), ('rotmnist', 30), ('seqcifar10', None),
          ('seqcifar100', None), ('clear10', None))

# Mammoth utils/best_args.py, model er, SGD (identical across K = 200/500/5120;
# seq-cifar100 lists K = 500/2000 only, both 0.1).
MAMMOTH_ER_LR = {
    ('rotmnist', 'replaysgd'): 0.2,
    ('seqcifar10', 'replaysgd'): 0.1,
    ('seqcifar100', 'replaysgd'): 0.1,
}

# AdEMAMix ER LR sweep; its best LR sets AdEMAMix ER on these datasets.
# Elsewhere (seq-CIFAR) AdEMAMix ER uses the α=0 best LR.
ADEMAMIX_ER_SWEEP = 'plan_rotmnist_cyclic_lr_er'
ADEMAMIX_ER_SWEEP_DATASETS = frozenset({'rotmnist', 'clear10'})

CONFIG_KEYS = ('dataset_name', 'optim', 'angle', 'stream_type', 'lr', 'alpha', 'horizon_ratio',
               'beta1', 'beta4', 'beta4_from_horizon', 'ablation', 'replay',
               'replay_buffer_size', 'memory_examples', 'memory_samples',
               'buffer_size', 'refresh_batches', 'width', 'seed', 'warmup_tasks',
               'lr_scheduler')
SUMMARY_KEYS = ('ACC', 'FRGT', 'BWT', 'ACC_TIL', 'FRGT_TIL', 'BWT_TIL')
# Everything that defines a cell, apart from the seed.
CELL_KEYS = ('group', 'optim', 'angle', 'stream_type', 'lr', 'lr_scheduler', 'alpha', 'horizon_ratio', 'beta1',
             'beta4', 'ablation', 'replay', 'K', 'memory_samples', 'buffer_size',
             'refresh_batches', 'width')


def slice_id(dataset, angle):
    return f'{dataset}_angle{angle}' if angle is not None else dataset


def slice_title(dataset, angle):
    name = DATASET_NAME.get(dataset, dataset)
    return f'{name}, {angle}°' if angle is not None else name


def readout(dataset):
    """(acc key, forgetting key, label) of the headline readout."""
    if dataset in SPLIT:
        return 'ACC_TIL', 'FRGT_TIL', 'Task-IL'
    return 'ACC', 'FRGT', ''


def bwt_key(dataset):
    return 'BWT_TIL' if dataset in SPLIT else 'BWT'


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _record(run, group):
    cfg = {k: run.config.get(k) for k in CONFIG_KEYS}
    rec = {'id': run.id, 'group': group, 'state': str(run.state),
           'created': str(run.created_at)}
    rec.update(cfg)
    for k in ('lr', 'alpha', 'horizon_ratio', 'beta1', 'beta4'):
        rec[k] = _num(rec[k])
    for k in ('angle', 'replay', 'replay_buffer_size', 'memory_examples',
              'memory_samples', 'buffer_size', 'refresh_batches', 'width', 'seed'):
        v = _num(rec[k])
        rec[k] = None if v is None else int(round(v))
    rec['dataset'] = rec.pop('dataset_name')
    if rec['dataset'] != 'rotmnist':
        rec['angle'] = None
    # Budget in examples: the reservoir for ER, memory_examples for fresh arms.
    rec['K'] = (rec['replay_buffer_size'] if rec['replay'] else rec['memory_examples']) or None
    summary = run.summary
    rec['summary'] = {k: _num(summary.get(k)) for k in SUMMARY_KEYS}
    return rec


def load_runs(groups, refresh=False):
    """Records of every run in ``groups`` (all states), cached per group."""
    blob = json.loads(CACHE.read_text()) if CACHE.is_file() else {'groups': {}}
    todo = [g for g in dict.fromkeys(groups) if refresh or g not in blob['groups']]
    if todo:
        import wandb
        api = wandb.Api(timeout=180)
        for g in todo:
            runs = list(api.runs(f'{ENTITY}/{PROJECT}', filters={'group': g}, per_page=300))
            blob['groups'][g] = [_record(r, g) for r in runs]
            print(f'fetched {g}: {len(runs)} runs')
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(blob))
    return [rec for g in dict.fromkeys(groups) for rec in blob['groups'].get(g, [])]


def complete(rec, dataset):
    """Finished, with a finite headline ACC and forgetting."""
    acc, frgt, _ = readout(dataset)
    return (rec['state'] == 'finished' and rec['summary'].get(acc) is not None
            and rec['summary'].get(frgt) is not None)


# Protocol seeds. A cell missing any of them still has seeds running or unfinished.
SEEDS = (42, 123, 7)


def has_all_seeds(cell):
    """True when every protocol seed has a complete run in ``cell``."""
    return set(SEEDS) <= set(cell['seeds'])


def lr_eq(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(abs(a), abs(b))


def best_lr(dataset, optim, angle=None, refresh=False):
    """(best lr, [(lr, mean ACC, n complete, n runs)]) from the α=0 lr sweep."""
    acc_key = readout(dataset)[0]
    recs = [r for r in load_runs([f'plan_{dataset}_{optim}_lr'], refresh)
            if r['angle'] == angle and (r['alpha'] or 0.0) == 0.0]
    by_lr = defaultdict(list)
    for r in recs:
        by_lr[r['lr']].append(r)
    table, best = [], None
    for lr in sorted(by_lr):
        accs = [r['summary'][acc_key] for r in
                latest_per_seed([r for r in by_lr[lr] if complete(r, dataset)])]
        mean = sum(accs) / len(accs) if accs else None
        table.append((lr, mean, len(accs), len(by_lr[lr])))
        if mean is not None:
            rank = (mean, len(accs), -lr)
            if best is None or rank > best[0]:
                best = (rank, lr)
    return (best[1] if best else None), table


def ademamix_er_lr(refresh=False):
    """(best lr, [(lr, mean ACC over streams)]) of the AdEMAMix ER LR sweep.

    Each stream's mean over complete seeds counts once, so the two streams
    weigh the same.
    """
    recs = [r for r in load_runs([ADEMAMIX_ER_SWEEP], refresh) if r['replay']
            and complete(r, 'rotmnist_cyclic')]
    by_lr = defaultdict(lambda: defaultdict(list))
    for r in latest_per_seed(recs):
        by_lr[r['lr']][r.get('stream_type')].append(r['summary']['ACC'])
    table = [(lr, _mean([_mean(v) for v in streams.values()])) for lr, streams in sorted(by_lr.items())]
    best = max(table, key=lambda row: (row[1], -row[0]))[0] if table else None
    return best, table


def er_lr(dataset, optim, memory_lr, refresh=False):
    """(ER lr, source label) under the protocol."""
    lit = MAMMOTH_ER_LR.get((dataset, optim))
    if lit is not None:
        return lit, 'Mammoth best_args'
    if optim == 'ademamix' and dataset in ADEMAMIX_ER_SWEEP_DATASETS:
        lr, _ = ademamix_er_lr(refresh)
        if lr is not None:
            return lr, 'AdEMAMix ER LR sweep (cyclic rotMNIST)'
    if optim == 'ademamix':
        return memory_lr, 'α=0 best (rate the AdEMAMix ER runs were launched at)'
    return memory_lr, 'α=0 best (no literature ER LR)'


def latest_per_seed(recs):
    """One run per (cell, seed): the latest created, so reruns replace failures."""
    keep = {}
    for r in sorted(recs, key=lambda r: r['created']):
        keep[(cell_key(r), r['seed'])] = r
    return list(keep.values())


# Arms that read each memory knob. Elsewhere the knob is an inert argparse
# default that not every run logs (e.g. buffer_size=500, refresh_batches=4),
# so it must not split a cell.
_BUFFERED = frozenset({'fifo_uniform_stale', 'reservoir_uniform_stale',
                       'reservoir_geometric_stale', 'fifo_geometric_fresh',
                       'reservoir_uniform_fresh'})
_KNOB_USERS = {
    'buffer_size': frozenset({'reservoir_uniform_stale', 'reservoir_geometric_stale'}),
    'memory_samples': _BUFFERED,
    'refresh_batches': frozenset({'slow_ema_refresh'}),
}


def cell_key(rec):
    out = []
    for k in CELL_KEYS:
        v = rec.get(k)
        if k in _KNOB_USERS and rec.get('ablation') not in _KNOB_USERS[k]:
            v = None
        if k == 'memory_samples' and v == 0:
            v = None  # 0 means S = B, the same draw as unset
        if v == '':
            v = None  # unset strings (stream_type) may be logged as ''
        out.append(v)
    return tuple(out)


def cells(recs, dataset):
    """Mean ± sample std over complete seeds per cell, never across cells."""
    acc_key, frgt_key, _ = readout(dataset)
    grouped = defaultdict(list)
    for r in latest_per_seed([r for r in recs if complete(r, dataset)]):
        grouped[cell_key(r)].append(r)
    out = []
    for key, rs in grouped.items():
        accs = [r['summary'][acc_key] for r in rs]
        frgts = [r['summary'][frgt_key] for r in rs]
        bwts = [r['summary'].get(bwt_key(dataset)) for r in rs]
        bwts = bwts if all(b is not None for b in bwts) else None
        out.append({
            **dict(zip(CELL_KEYS, key)),
            'n': len(rs), 'seeds': sorted(r['seed'] for r in rs),
            'runs': [r['id'] for r in rs],
            'acc': _mean(accs), 'acc_std': _std(accs),
            'frgt': _mean(frgts), 'frgt_std': _std(frgts),
            'bwt': _mean(bwts) if bwts else None, 'bwt_std': _std(bwts) if bwts else None,
        })
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pareto_mask(points):
    """Non-dominated points for (max acc, min frgt)."""
    mask = []
    for p in points:
        dominated = any(q['acc'] >= p['acc'] and q['frgt'] <= p['frgt']
                        and (q['acc'] > p['acc'] or q['frgt'] < p['frgt'])
                        for q in points)
        mask.append(not dominated)
    return mask


def protocol_table(refresh=False):
    """{(dataset, angle, optim): {'lr', 'er_lr', 'er_source', 'table'}}."""
    out = {}
    for dataset, angle in SLICES:
        for optim in OPTIMS:
            lr, table = best_lr(dataset, optim, angle, refresh)
            elr, src = er_lr(dataset, optim, lr, refresh)
            out[(dataset, angle, optim)] = {'lr': lr, 'er_lr': elr,
                                            'er_source': src, 'table': table}
    return out


def write_protocol_md(protocol, path):
    lines = ['# Protocol learning rates (from the α=0 lr sweeps on W&B)', '',
             '| slice | optimizer | readout | best LR | ER LR | LR sweep: mean ACC (complete/total) |',
             '|---|---|---|---|---|---|']
    for (dataset, angle, optim), p in protocol.items():
        grid = '; '.join(f'{lr:g}: {"—" if m is None else f"{m:.2f}"} ({n}/{t})'
                         for lr, m, n, t in p['table'])
        edge = ''
        if p['lr'] is not None and p['table']:
            lrs = [row[0] for row in p['table']]
            if p['lr'] in (min(lrs), max(lrs)):
                edge = ' (grid edge)'
        lines.append(f'| {slice_title(dataset, angle)} | {OPTIM_NAME[optim]} | '
                     f'{readout(dataset)[0]} | {p["lr"]}{edge} | {p["er_lr"]} ({p["er_source"]}) | {grid} |')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
