"""Run setup: seeds, device, per-task LR scheduler, W&B init, run names and metric axes."""
import os
import random

import numpy as np
import torch
import wandb


def make_lr_scheduler(args, optimizer):
    """Per-task scheduler. Default cosine when the dataset file defines none."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
    name = getattr(args, 'lr_scheduler', 'cosine') or 'cosine'
    if name == 'none':
        return LambdaLR(optimizer, lambda _: 1.0)
    if name == 'multistep':
        raw = getattr(args, 'lr_milestones', '35,45')
        if isinstance(raw, (list, tuple)):
            milestones = [int(x) for x in raw]
        else:
            milestones = [int(x.strip()) for x in str(raw).split(',') if x.strip()]
        gamma = float(getattr(args, 'lr_gamma', 0.1))
        print(f'LR scheduler: MultiStepLR milestones={milestones} gamma={gamma}')
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    t_max = max(1, int(args.epochs_per_task))
    print(f'LR scheduler: CosineAnnealingLR T_max={t_max}')
    return CosineAnnealingLR(optimizer, T_max=t_max)


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_device():
    """CUDA if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_wandb(entity, project, job_type, group, name, config):
    """Attach to the W&B sweep run when WANDB_SWEEP_ID is set.

    Do not pass a custom ``Settings()`` object: an empty Settings on wandb 0.19+
    can drop the API key / force ``offline-run-*`` and the sweep UI stays empty
    even though the agent fetched configs. Directory-level ``wandb offline``
    leftovers are overridden unless the caller explicitly set WANDB_MODE.
    """
    os.environ.setdefault('WANDB_MODE', 'online')
    os.environ.setdefault('WANDB__SERVICE_WAIT', '300')
    os.environ.setdefault('WANDB_CONSOLE', 'off')
    return wandb.init(
        project=project,
        job_type=job_type,
        entity=entity,
        group=group,
        name=name,
        config=config,
    )


# Short labels for swept parameters in run names; anything else keeps its name.
_RUN_NAME_ALIASES = {
    'horizon_ratio': 'r',
    'beta4_from_horizon': 'b4h',
    'replay_buffer_size': 'K',
    'replay_per_angle': 'm',
    'memory_samples': 'S',
    'memory_examples': 'K',
    'stream_type': 'stream',
}


def _run_name_value(value):
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def sweep_run_name(run, args):
    """``<key>=<value>_...`` over the parameters this run's sweep varies.

    Under a W&B sweep every run gets the same ``--wandb_name``. The varied keys
    come from the sweep config itself (entries with more than one value), so
    the name follows the YAML without any extra flag. ``seed`` goes last.
    Returns None outside a sweep or when the sweep config cannot be read.
    """
    sweep_id = getattr(run, 'sweep_id', None)
    if run is None or not sweep_id:
        return None
    try:
        sweep = wandb.Api().sweep(f'{run.entity}/{run.project}/{sweep_id}')
        params = sweep.config.get('parameters', {})
    except Exception as exc:  # network / permission: keep the default name
        print(f'WARNING: could not read sweep {sweep_id} for the run name ({exc})')
        return None
    varied = [k for k, spec in params.items()
              if isinstance(spec, dict) and len(spec.get('values', []) or []) > 1]
    varied.sort(key=lambda k: k == 'seed')
    # Several keys may map to the same short alias (e.g. replay_buffer_size and
    # memory_examples both abbreviate to K). Within a single sweep, that would
    # emit two identical tokens, hiding which varied; refuse rather than lie.
    seen = {}
    for k in varied:
        alias = _RUN_NAME_ALIASES.get(k, k)
        if alias in seen:
            raise RuntimeError(
                f'sweep varies {k!r} and {seen[alias]!r}, which both alias to '
                f'{alias!r} in run names; give one of them a distinct alias.'
            )
        seen[alias] = k
    parts = [f'{_RUN_NAME_ALIASES.get(k, k)}={_run_name_value(getattr(args, k, None))}'
             for k in varied]
    return '_'.join(parts) or None


def define_task_metrics(num_tasks=None):
    """Pin per-task CL metrics to `crt_task` as the X-axis.

    Without this, W&B plots each metric against its internal monotonically-
    increasing `_step` counter, which advances once per `wandb.log()` call.
    Scripts log at different frequencies per task (e.g. per-epoch + per-task-eval,
    or just per-task-eval), so two runs with the same number of tasks but
    different epochs_per_task end up with different X-axes and their curves
    cannot be directly compared.

    After this call, any metric matching the patterns below will use the
    logical task index (`crt_task`, 1..N) as its X-axis, making curves from any
    combination of script / optimizer / epochs_per_task directly comparable.

    W&B note: `wandb.define_metric` only accepts ``*`` as a *suffix* glob — it
    rejects patterns like ``task*_TestAcc`` with the wildcard in the middle.
    We therefore register the per-task metrics (``task{i}of{N}_TestAcc`` etc.)
    explicitly when ``num_tasks`` is provided.
    """
    wandb.define_metric('crt_task')
    wandb.define_metric('global_epoch')
    for pattern in (
        'AvgTestAcc*',
        'AvgTrainAcc*',
        'fisher/*',
        'FRGT',
        'replay/*',
        'eval/*',
        'eval_loss/*',
        'eval_TIL/*',
        'eval_loss_TIL/*',
        'guard/*',
    ):
        wandb.define_metric(pattern, step_metric='crt_task')
    # AdEMAMix tracking is logged per-epoch, so the half-life ramp, per-task EMA
    # mass fractions, and per-task retained-gradient counts evolve on a much
    # finer time-scale than `crt_task` (one point per task). Pin them to the
    # optimizer's own global gradient counter so the curves show smoothly in
    # W&B with one point per minibatch gradient consumed. `ademamix/step` is
    # exported by AdEMAMix.get_tracking_metrics().
    wandb.define_metric('ademamix/step')
    wandb.define_metric('ademamix/*', step_metric='ademamix/step')
    # ReplaySGD ablation lattice: one point per optimizer step.
    wandb.define_metric('memory/step')
    wandb.define_metric('memory/*', step_metric='memory/step')

    # Single X-axis for train loss/acc across all tasks and epochs.
    wandb.define_metric('sequence/*', step_metric='global_epoch')
    # Per-layer optimizer-geometry diagnostics (see `OptimGeometryTracker`).
    # Logged once per training epoch -- pin to `global_epoch` so curves are
    # comparable across runs with different epochs_per_task / task counts.
    wandb.define_metric('optim_geom/*', step_metric='global_epoch')
    # Geometric drift and catastrophic interference diagnostics.
    # Evaluated periodically during the new task's training epochs.
    # First-moment: drift/cosine_sim, drift/log_scale_ratio.
    # Second-moment: drift/cosine_sim_sq, drift/log_scale_ratio_sq.
    # See `GeometricDriftTracker`.
    wandb.define_metric('drift/*', step_metric='global_epoch')
    # Gradient-noise proxy. Logged at most once per epoch
    # (see --track_grad_noise / --grad_noise_every_epochs).
    wandb.define_metric('grad_noise/*', step_metric='global_epoch')
    # Quadratic-benchmark-only diagnostics: v_t^s-vs-curvature check and
    # distance-to-task-0-optimum. Logged every optimizer step, so pin to the
    # same fine-grained step counter as `ademamix/*` rather than `global_epoch`.
    # See `QuadraticMemoryDiagnostics`.
    wandb.define_metric('quad_diag/*', step_metric='ademamix/step')

    if num_tasks is not None:
        for i in range(1, num_tasks + 1):
            for suffix in (
                'TestAcc', 'TestLoss', 'TrainAcc', 'TrainLoss',
                'NextTestAcc', 'NextTestLoss',
            ):
                wandb.define_metric(
                    f'task{i}of{num_tasks}_{suffix}',
                    step_metric='crt_task',
                )
