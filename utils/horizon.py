"""Slow-EMA coefficients from the task length L and the horizon ratio r.

Recipe: L = ceil(N_task / batch) * epochs, τ = r L, β = τ / (1 + τ).
r = τ/L is the mean memory age in tasks. The same β applies to every slow
channel (β3 and, when requested, β4). A grid in β is not comparable across
benchmarks because the same coefficient reaches a different number of past
tasks once L changes.

Use the exact β = rL/(1+rL). The large-L limit e^{-1/r} is only for reading
off how much kernel mass sits on previous tasks.
"""

from __future__ import annotations

import math
from statistics import median

# n_per_task is what L depends on. For class-IL it is n_train // n_tasks,
# NOT n_train. Keep n_tasks here so the two can never drift apart.
DATASET_DETAILS = {
    # --- domain-IL: every task sees the full training set ---
    'mnist':            {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'rotmnist':         {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'rotmnist_gradual': {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'rotmnist_cyclic':  {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'permmnist':        {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'permcontinuous':   {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'labelpermmnist':   {'n_train':  60_000, 'n_classes':  10, 'per_task': 'full'},
    'cifar10':          {'n_train':  50_000, 'n_classes':  10, 'per_task': 'full'},
    'cifar100':         {'n_train':  50_000, 'n_classes': 100, 'per_task': 'full'},

    # --- class-IL: training set is PARTITIONED across tasks ---
    'seqcifar10':  {'n_train':  50_000, 'n_classes':  10, 'per_task': 'split',
                    'n_tasks':  5, 'classes_per_task':  2},
    'seqcifar100': {'n_train':  50_000, 'n_classes': 100, 'per_task': 'split',
                    'n_tasks': 10, 'classes_per_task': 10},
    'seqtinyimg':  {'n_train': 100_000, 'n_classes': 200, 'per_task': 'split',
                    'n_tasks': 10, 'classes_per_task': 20},

    # --- session-based: tasks are predefined acquisition sessions ---
    # NI protocol: 8 training sessions x 50 objects x ~300 frames.
    # 164,866 images / 11 sessions => ~14,987, not exactly 15,000.
    'core50': {'n_train': 119_894, 'n_classes': 50, 'per_task': 'sessions',
               'n_tasks': 8, 'n_per_task': 14_987},

    # --- synthetic: size comes from the CLI ---
    'quadratics': {'per_task': 'arg', 'n_arg': 'quad_n_train'},
    'bitflip':    {'per_task': 'arg', 'n_arg': 'quad_n_train'},
}

# Reference horizons. Only r values whose kernel reaches previous tasks: at a
# boundary the previous-task mass is e^{-1/r}(1-e^{-1/r}), which is ~0 for
# r ≤ 0.1 (within-task memory) and first becomes non-negligible at r = 0.3.
# usable_horizon_ratios drops r > T/4 (the EMA is not yet ≥98% filled by end of
# a T-task stream).
HORIZON_RATIOS = (0.3, 1.0, 3.0, 10.0)


def n_per_task(name, args=None, n_tasks=None):
    d = DATASET_DETAILS[name]
    kind = d['per_task']
    if kind == 'full':
        return d['n_train']
    if kind == 'split':
        return d['n_train'] // (n_tasks or d['n_tasks'])
    if kind == 'sessions':
        return d['n_per_task']
    if kind == 'arg':
        return getattr(args, d['n_arg'])
    raise ValueError(kind)


def task_length(name, batch_size, epochs, args=None, n_tasks=None, drop_last=False):
    """L: optimizer updates per task."""
    n = n_per_task(name, args, n_tasks)
    steps_per_epoch = n // batch_size if drop_last else -(-n // batch_size)
    return steps_per_epoch * epochs


def median_task_length(sizes, batch_size, epochs, drop_last=False):
    """L for ragged tasks (CLEAR buckets, uneven sessions): median over tasks.

    The horizon ratio is defined with this median. Per-task ages inside a task
    still use that task's own length.
    """
    if not sizes:
        raise ValueError('median_task_length requires at least one task size')
    lengths = []
    for n in sizes:
        n = int(n)
        steps = n // batch_size if drop_last else -(-n // batch_size)
        lengths.append(steps * int(epochs))
    return float(median(lengths))


def r_max(n_tasks):
    """Largest r whose EMA is ≥98% filled after T tasks: T/r ≥ 4."""
    return float(n_tasks) / 4.0


def usable_horizon_ratios(n_tasks, ratios=HORIZON_RATIOS):
    cap = r_max(n_tasks)
    return [float(r) for r in ratios if float(r) <= cap + 1e-12]


def beta_from_horizon(r, L):
    """Exact map. Returns (β, τ) with τ = r L and β = τ / (1 + τ)."""
    tau = float(r) * float(L)
    beta = 0.0 if tau == 0.0 else tau / (1.0 + tau)
    return beta, tau


def past_task_mass(r):
    """Kernel mass on all previous tasks at a boundary, e^{-1/r}."""
    r = float(r)
    if r <= 0.0:
        return 0.0
    return math.exp(-1.0 / r)


def resolve_task_length(name, batch_size, epochs, train_sizes, args=None, n_tasks=None,
                        drop_last=False):
    """DATASET_DETAILS L when the dataset is listed, else the median measured L.

    Prints a warning when the two disagree by more than 5% (validation split
    or drop_last). The listed value is the one used for β.
    """
    measured = median_task_length(train_sizes, batch_size, epochs, drop_last=drop_last)
    if name not in DATASET_DETAILS:
        print(
            f'{name}: L = median steps/task = {measured:g} '
            f'(not in DATASET_DETAILS; ragged tasks use the median).'
        )
        return measured
    L = float(task_length(
        name, batch_size, epochs, args=args, n_tasks=n_tasks, drop_last=drop_last))
    if L > 0.0 and abs(measured - L) / L > 0.05:
        print(
            f'WARNING: {name}: median steps/task is {measured:g}, '
            f'DATASET_DETAILS gives L={L:g} (>5% apart). '
            f'β uses DATASET_DETAILS. A validation split or drop_last '
            f'would explain the gap.'
        )
    else:
        print(f'{name}: L={L:g} steps/task (DATASET_DETAILS; measured median {measured:g}).')
    return L


def apply_slow_horizon(args, L):
    """Write β3 (and β4 when requested) from --horizon_ratio and L.

    ``--beta4_from_horizon 1`` sets β4 = β3, so the slow second moment reaches
    the same past gradients as the slow first moment. ``0`` sets β4 = 0
    (the slow metric sees only the current gradient). A positive ``--beta4``
    passed next to ``--horizon_ratio`` is also replaced: a raw β4 is a step
    count, not a task horizon.

    ``channel_variant=dual_only`` has already cleared β4 and is left alone.
    No-op when ``--horizon_ratio`` is omitted.
    """
    args.cfg_L = float(L)
    r = getattr(args, 'horizon_ratio', None)
    if r is None:
        beta = float(getattr(args, 'beta3', 0.0) or 0.0)
        args.cfg_tau = (beta / (1.0 - beta)) if 0.0 < beta < 1.0 else 0.0
        args.cfg_horizon_ratio = None
        args.cfg_past_task_mass = None
        return

    r = float(r)
    if r < 0.0:
        raise ValueError(f'horizon_ratio must be ≥ 0, got {r}')
    cap = r_max(getattr(args, 'tasks', 0) or 0)
    if cap > 0.0 and r > cap + 1e-12:
        stream = str(getattr(args, 'stream_type', '') or '')
        if stream == 'cyclic':
            print(
                f'NOTE: cyclic stream keeps r={r:g} above r_max=T/4={cap:g}. '
                f'The EMA is not 98% full in one-pass steps; the design target '
                f'is the within-cycle kernel (uniform near r=24).'
            )
        else:
            print(
                f'WARNING: horizon_ratio r={r:g} exceeds r_max=T/4={cap:g}. '
                f'The EMA is still transient at the end of the stream '
                f'(filled fraction 1-exp(-T/r) < 0.98).'
            )

    beta, tau = beta_from_horizon(r, L)
    args.beta3 = beta
    past = past_task_mass(r)
    match = getattr(args, 'beta4_from_horizon', None)
    if getattr(args, 'channel_variant', None) == 'dual_only':
        args.beta4 = None
    elif match is not None and int(match) == 1:
        args.beta4 = beta
    elif match is not None and int(match) == 0:
        args.beta4 = 0.0
    else:
        b4 = getattr(args, 'beta4', None)
        if b4 is not None and float(b4) > 0.0:
            args.beta4 = beta

    args.cfg_tau = tau
    args.cfg_horizon_ratio = r
    args.cfg_past_task_mass = past
    b4_show = getattr(args, 'beta4', None)
    b4_txt = 'None' if b4_show is None else f'{float(b4_show):.10f}'
    print(
        f'horizon: L={float(L):g} r={r:g} tau={tau:.6g} '
        f'beta3={beta:.10f} beta4={b4_txt} '
        f'past_task_mass={past:.6g} current_task_mass={1.0 - past:.6g}'
    )
