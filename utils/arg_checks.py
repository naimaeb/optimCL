"""Reject naive.py argument combinations where one flag silently overrides another.

Several parameters can be set two ways (a raw coefficient or a derived one), and
naive.py resolves the pair by precedence. Passing both is almost always a
mistake, so naive.py refuses it. The flags stay in utils/args.py; only naive.py
enforces these rules.
"""
from __future__ import annotations

# Deprecated for naive.py: accepted by the shared parser, rejected by check_naive_args.
DEPRECATED_FLAGS = {
    'cycle_period': 'legacy sawtooth; use --stream_type with --cycle_slots',
    'channel_variant': 'set --alpha / --beta4 directly (metric_only is plain Adam)',
}


def explicit_flags(argv):
    """Destinations given on the command line, from ``--name value`` or ``--name=value``."""
    names = set()
    for token in argv:
        if token.startswith('--') and len(token) > 2:
            names.add(token[2:].split('=', 1)[0].replace('-', '_'))
    return names


def _positive(value):
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def conflicts(values, given):
    """Messages for overlapping flags.

    ``values`` maps destination -> parsed value; ``given`` is the set of
    destinations the user passed explicitly (defaults do not count).
    """
    both = lambda a, b: a in given and b in given
    out = []
    if both('beta3', 'horizon_ratio'):
        out.append('--beta3 and --horizon_ratio: the horizon ratio sets beta3 = rL/(1+rL); pass one')
    if both('beta4', 'beta4_from_horizon'):
        out.append('--beta4 and --beta4_from_horizon: beta4_from_horizon sets beta4; pass one')
    if 'warmup_tasks' in given and _positive(values.get('warmup_tasks')):
        for flag in ('beta3_warmup', 'alpha_warmup'):
            if flag in given:
                out.append(f'--{flag} and --warmup_tasks > 0: warmup_tasks sets both warmups '
                           'from the task length; pass one')
    if 'replay_per_angle' in given and _positive(values.get('replay_per_angle')) \
            and 'replay_buffer_size' in given:
        out.append('--replay_buffer_size and --replay_per_angle > 0: K = m * n_distinct_angles; pass one')
    if values.get('dataset_name') == 'rotmnist_cyclic' and both('angle', 'angle_delta') \
            and _positive(values.get('angle_delta')):
        out.append('--angle and --angle_delta on rotmnist_cyclic: angle_delta is the step; pass one')
    if both('cycle_period', 'stream_type') and values.get('stream_type'):
        out.append('--cycle_period and --stream_type: cycle_period is ignored when stream_type is set')
    variant = values.get('channel_variant')
    if variant == 'dual_only' and 'beta4' in given:
        out.append('--channel_variant dual_only clears --beta4; pass one')
    if variant == 'metric_only' and 'alpha' in given:
        out.append('--channel_variant metric_only sets alpha=0; pass one')
    if values.get('ablation') == 'replay' and 'replay' in given and not values.get('replay'):
        out.append('--ablation replay with --replay 0: the replay ablation needs the reservoir')
    return out


def check_naive_args(args, argv):
    """Raise ``SystemExit`` with every conflict; print deprecation notices."""
    given = explicit_flags(argv)
    values = vars(args)
    for flag, why in DEPRECATED_FLAGS.items():
        if flag in given:
            print(f'WARNING: --{flag} is deprecated for naive.py ({why}).')
    if 'use_multihead_model' in given and values.get('use_multihead_model'):
        print('WARNING: --use_multihead_model is ignored; apply_il_protocol forces a single head.')
    problems = conflicts(values, given)
    if problems:
        raise SystemExit('Conflicting arguments:\n  ' + '\n  '.join(problems))
