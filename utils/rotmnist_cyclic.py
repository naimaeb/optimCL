"""Ping-pong cyclic rot-MNIST and its monotone one-pass control.

Both streams change the rotation by ``angle_delta`` at every task boundary.
Cyclic ping-pongs inside a bounded amplitude; one-pass marches upward and
must stay below 360° so it does not wrap into a cycle. The matched-drift
assert is the comparison: if it fails, the streams differ in per-step drift
as well as amplitude.
"""

from __future__ import annotations

import numpy as np


def angle_for_task(k, stream_type, angle_delta, cycle_slots):
    """Deterministic angle in degrees. ``k`` is the 0-indexed stream position."""
    k = int(k)
    angle_delta = int(angle_delta)
    cycle_slots = int(cycle_slots)
    if stream_type == 'cyclic':
        h = cycle_slots // 2
        return angle_delta * (h - abs(h - (k % cycle_slots)))
    if stream_type == 'onepass':
        return angle_delta * k
    raise ValueError(stream_type)


def angle_schedule(tasks, stream_type, angle_delta, cycle_slots):
    """Angles for positions ``0 .. tasks-1``, checked against the stream invariant."""
    tasks = int(tasks)
    angles = [
        angle_for_task(k, stream_type, angle_delta, cycle_slots)
        for k in range(tasks)
    ]
    validate_angle_schedule(angles, stream_type, tasks, angle_delta, cycle_slots)
    return angles


def validate_angle_schedule(angles, stream_type, tasks, angle_delta, cycle_slots):
    """Raise ``AssertionError`` when the two streams are no longer comparable."""
    angles = [int(a) for a in angles]
    tasks = int(tasks)
    angle_delta = int(angle_delta)
    cycle_slots = int(cycle_slots)
    if tasks < 1:
        raise AssertionError('tasks must be positive')
    if len(angles) != tasks:
        raise AssertionError('angle list length must equal tasks')
    if angle_delta <= 0:
        raise AssertionError('angle_delta must be positive')

    if stream_type == 'cyclic':
        if cycle_slots < 2:
            raise AssertionError('cycle_slots must be >= 2')
        if tasks % cycle_slots != 0:
            raise AssertionError('T must be a whole number of cycles')
        # A single cycle has an empty second slice, so the two-slice check
        # applies once a second cycle exists. Every cycle is still compared
        # with the first, including the T = P case.
        cycle = angles[:cycle_slots]
        if tasks >= 2 * cycle_slots and angles[:cycle_slots] != angles[cycle_slots:2 * cycle_slots]:
            raise AssertionError('cyclic schedule must repeat every cycle_slots tasks')
        for start in range(0, tasks, cycle_slots):
            chunk = angles[start:start + cycle_slots]
            if chunk != cycle:
                raise AssertionError(
                    'cyclic schedule must repeat the first cycle: '
                    f'angles[{start}:{start + cycle_slots}]={chunk} != {cycle}'
                )
        amplitude = angle_delta * (cycle_slots // 2)
        if max(angles) - min(angles) != amplitude:
            raise AssertionError(
                f'cyclic amplitude must be angle_delta * (cycle_slots // 2) = {amplitude}'
            )
    elif stream_type == 'onepass':
        if len(set(angles)) != tasks:
            raise AssertionError('onepass must not repeat an angle')
        if max(angles) >= 360:
            raise AssertionError('onepass would wrap and become accidentally cyclic')
    else:
        raise ValueError(stream_type)

    # Matched per-boundary drift. This is the whole comparison.
    if not all(abs(angles[i + 1] - angles[i]) == angle_delta for i in range(tasks - 1)):
        raise AssertionError('per-boundary drift must equal angle_delta on every boundary')


def bind_cyclic_schedule(args):
    """Resolve and store the angle list on ``args``. Idempotent."""
    if getattr(args, 'cyclic_angles', None) is not None:
        return args.cyclic_angles
    stream = str(getattr(args, 'stream_type', '') or '')
    if stream not in ('cyclic', 'onepass'):
        raise RuntimeError(
            'rotmnist_cyclic ping-pong requires --stream_type cyclic or onepass'
        )
    delta = int(getattr(args, 'angle_delta', 0) or 0)
    if delta <= 0:
        delta = int(getattr(args, 'angle', 0) or 0)
    if delta <= 0:
        raise RuntimeError(
            'rotmnist_cyclic stream requires --angle_delta (or --angle) > 0'
        )
    slots = int(getattr(args, 'cycle_slots', 0) or 0)
    tasks = int(args.tasks)
    angles = angle_schedule(tasks, stream, delta, slots)
    args.cyclic_angles = angles
    args.cfg_stream_type = stream
    args.cfg_angle_delta = delta
    args.cfg_cycle_slots = slots
    args.cfg_amplitude_deg = int(max(angles) - min(angles))
    args.cfg_n_distinct_angles = len(set(angles))
    print(
        f'rotmnist_cyclic {stream}: delta={delta} slots={slots} '
        f'amplitude={args.cfg_amplitude_deg} distinct={args.cfg_n_distinct_angles} '
        f'angles={angles}'
    )
    return angles


def apply_replay_per_angle(args):
    """Set reservoir capacity from exemplars per distinct angle, when requested.

    ``--replay_per_angle m`` stores ``K = m * n_distinct_angles`` so coverage
    stays constant as the one-pass stream adds angles. Fixed ``--replay_buffer_size``
    is left alone; the exemplars-per-angle ratio is still recorded.
    """
    m = int(getattr(args, 'replay_per_angle', 0) or 0)
    n = getattr(args, 'cfg_n_distinct_angles', None)
    replay_on = bool(getattr(args, 'replay', 0))
    if m > 0:
        if not replay_on:
            raise RuntimeError('--replay_per_angle requires --replay 1')
        if not n:
            raise RuntimeError(
                '--replay_per_angle requires rotmnist_cyclic with a resolved angle schedule'
            )
        args.replay_buffer_size = int(m) * int(n)
        args.cfg_exemplars_per_angle = float(m)
        args.cfg_K_schedule = 'scaled'
        print(
            f'Scaled reservoir: m={m} exemplars/angle x {int(n)} distinct angles '
            f'= K={args.replay_buffer_size}'
        )
        return
    if replay_on and n:
        k = int(getattr(args, 'replay_buffer_size', 0) or 0)
        args.cfg_exemplars_per_angle = float(k) / float(n)
        args.cfg_K_schedule = 'fixed'


def angle_accuracy_metrics(R, angles):
    """Per-position and per-distinct-angle accuracy on the final row of ``R``.

    ``R[i][j]`` is accuracy on task position ``j`` after finishing position ``i``.
    Positions that share an angle stay separate in ``R``; the per-angle number
    is their mean, and ``spread`` is max minus min.
    """
    matrix = np.asarray(R, dtype=float)
    final = matrix[-1]
    angles = [int(a) for a in angles]
    if len(angles) != final.shape[0]:
        raise ValueError(
            f'angle list length {len(angles)} does not match R columns {final.shape[0]}'
        )
    lines = ['# final-row accuracy by task position (not collapsed by angle)']
    metrics = {}
    for i, ang in enumerate(angles):
        acc = float(final[i])
        lines.append(f'pos {i} angle {ang} acc {acc:.4f}')
        metrics[f'pos_final_acc/{i}'] = acc
    lines.append('# mean over positions that share an angle')
    for ang in sorted(set(angles)):
        idx = [i for i, a in enumerate(angles) if a == ang]
        vals = final[idx]
        acc = float(np.nanmean(vals))
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        lines.append(
            f'angle {ang} n {len(idx)} acc {acc:.4f} spread {spread:.4f} '
            f'positions {idx}'
        )
        metrics[f'angle_acc/{ang}'] = acc
        metrics[f'angle_acc_spread/{ang}'] = spread
    return '\n'.join(lines) + '\n', metrics
