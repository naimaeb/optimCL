"""naive.py

Continual-learning training loop with a standard or memory-augmented optimizer.

Trains a model sequentially on each task, carrying the optimizer state across
task boundaries. Optional ``--replay 1`` adds a reservoir of past-task examples
mixed into each step — a data-memory baseline to compare against optimizer
memory (AdEMAMix ``beta3`` / ``beta4``).

Supported datasets
------------------
  See --dataset_name for the full list.
  CLEAR10 / CLEAR100  (raw images, streaming or iid protocol)
  rotmnist / permmnist / seqcifar10 / seqcifar100 / ... (get_rotated_tasks)
  quadratics / bitflip (convex regression; model=linreg)

RotMNIST / PermMNIST / CORe50-NI / CLEAR (Domain-IL): after each task the model is
evaluated on **all** tasks, filling the upper triangle (zero-shot on future tasks).
Split CIFAR / TinyImageNet train **Class-IL** (unmasked CE on ``Linear(*, C)``).
The same run writes Class-IL ``R-test-acc.txt`` and Task-IL ``R-test-acc-til.txt``
(masked argmax at eval only). Shared-label streams stay Domain-IL (one head, no mask).
Summary metrics **BWT / FWT / forgetting** follow Mammoth (FwT uses per-task random-init test accuracies
stored in ``mammoth_random_baseline.txt``, and the Task-IL FWT the masked
``mammoth_random_baseline_til.txt``; forgetting is also logged to wandb
after every task from task 2 onward as ``FRGT`` vs ``crt_task``). Training
loss/accuracy are logged under ``sequence/*`` vs ``global_epoch`` for one curve
across the full run.

Under a W&B sweep the run is named after the swept values
(``<key>=<value>_..._seed=<seed>``); see ``sweep_run_name``.

``--dataset_name rotmnist_cyclic --stream_type cyclic`` ping-pongs the rotation
angle in steps of ``--angle_delta`` over ``--cycle_slots`` tasks per cycle;
``--stream_type onepass`` is the monotone control with the same |Δ|.
``--replay_per_angle m`` sets the reservoir to ``m`` exemplars per distinct angle.

Usage
-----
  python naive.py \\
      --dataset_path ./data \\
      --dataset_name rotmnist --angle 10 \\
      --model mlp --width 100 --bias 1 \\
      --optim sgd --lr 1e-2 --weight_decay 0 \\
      --tasks 10 --epochs_per_task 5 --batch_size 256 \\
      --use_multihead_model 0 --seed 0 \\
      --output_folder ./results/<name> \\
      --wandb_entity <entity> --wandb_project <project> \\
      --wandb_group <group> --wandb_job_type train \\
      --wandb_name <name>

  Add ``--replay 1 --replay_buffer_size 1000`` for the replay baseline: past-task
  examples are stored in a reservoir and mixed into the current-task loss.
"""

import os
import sys
from os.path import join

import numpy as np
import torch
import wandb
from torch.nn.functional import cross_entropy as CE, mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimizers.replay_memory import ABLATION_ESTIMATORS, IndexedDataset
from optimizers.sam import is_active as _sam_is_active
from utils.arg_checks import check_naive_args
from utils.args import get_arg_parse
from utils.cl_metrics import (
    CLEARMetric,
    cl_chance_level,
    cl_metric_avg_acc,
    cl_metric_backward_transfer,
    cl_metric_forgetting,
    cl_metric_forgetting_after,
    cl_metric_forward_transfer,
    cl_r_summaries,
    cl_summaries_wandb,
    infer_memory_arm,
    optimizer_family_name,
    tau3_over_L,
)
from utils.diagnostics import GeometricDriftTracker, OptimGeometryTracker, QuadraticMemoryDiagnostics
from utils.evaluation import mammoth_random_results_baseline, save_mammoth_random_baseline, test
from utils.factory import double_mix_alpha, get_model, get_optimizer
from utils.gradient_noise import estimate_gradient_noise
from utils.horizon import apply_slow_horizon, resolve_task_length
from utils.il_protocol import NUM_CLASSES, apply_il_protocol, task_il_block_size
from utils.quadratic_tasks import get_quadratic_tasks
from utils.replay import ReservoirReplayBuffer
from utils.rotmnist_cyclic import angle_accuracy_metrics, apply_replay_per_angle
from utils.run_setup import (
    define_task_metrics,
    get_device,
    make_lr_scheduler,
    set_all_seeds,
    setup_wandb,
    sweep_run_name,
)
from utils.tasks import get_clear_tasks, get_rotated_tasks

NAN_EXIT_CODE = 666


def _is_clear(dataset_name):
    return 'clear' in dataset_name


def _is_quadratics(dataset_name):
    return dataset_name == 'quadratics'


def _is_regression(dataset_name):
    return dataset_name in ('quadratics', 'bitflip')


def _nan_matrix(n):
    """``n×n`` R matrix; NaN marks entries that are never evaluated."""
    return [[float('nan')] * n for _ in range(n)]


def _nanmean_or_nan(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float('nan')


def _prefix_square(matrix, n):
    """Leading n×n block of an R matrix (metrics after the first n tasks)."""
    return np.asarray([row[:n] for row in matrix[:n]], dtype=float)


def _chance_level_for(args, n_tasks, readout='primary'):
    """Random-guess accuracy for this dataset / readout (Domain-, Class-, or Task-IL)."""
    return cl_chance_level(
        NUM_CLASSES.get(args.dataset_name, 10),
        n_tasks,
        classes_per_task=task_il_block_size(args),
        readout=readout,
    )


def _log_cl_r_summaries(log_dict, matrix_acc, matrix_loss, n_seen, args,
                        til_acc=None, til_loss=None):
    """Write chance-normalised LA/ACC/BWT/ρ/FRGT/W(H) for the first n_seen tasks.

    Accuracy → ``eval/*``; raw CE loss → ``eval_loss/*``. Split datasets also
    log the masked Task-IL readout (``eval_TIL/*``) and the head-bias gap
    ``ACC_TIL − ACC_classIL``.
    """
    if n_seen < 1:
        return
    R = _prefix_square(matrix_acc, n_seen)
    L = _prefix_square(matrix_loss, n_seen)
    acc_sum = cl_r_summaries(R, chance=_chance_level_for(args, n_seen), acc_is_percent=True)
    loss_sum = cl_r_summaries(
        L, chance=0.0, higher_is_better=False, acc_is_percent=False)
    log_dict.update(cl_summaries_wandb('eval', acc_sum))
    log_dict.update(cl_summaries_wandb('eval_loss', loss_sum))
    if til_acc is None:
        return
    Rt = _prefix_square(til_acc, n_seen)
    til_sum = cl_r_summaries(
        Rt, chance=_chance_level_for(args, n_seen, 'til'), acc_is_percent=True)
    log_dict.update(cl_summaries_wandb('eval_TIL', til_sum))
    if til_loss is not None:
        log_dict.update(cl_summaries_wandb(
            'eval_loss_TIL',
            cl_r_summaries(
                _prefix_square(til_loss, n_seen),
                chance=0.0, higher_is_better=False, acc_is_percent=False)))
    if acc_sum.get('ACC') == acc_sum.get('ACC') and til_sum.get('ACC') == til_sum.get('ACC'):
        log_dict['eval/head_bias_gap'] = til_sum['ACC'] - acc_sum['ACC']


def _save_r_matrices(folder, acc, loss, til_acc=None, til_loss=None):
    """Unrounded R dumps: acc in percent, loss as mean CE (not ×100)."""
    np.save(join(folder, 'R_test_acc.npy'), np.asarray(acc, dtype=np.float64))
    np.save(join(folder, 'R_test_loss.npy'), np.asarray(loss, dtype=np.float64))
    if til_acc is not None:
        np.save(join(folder, 'R_test_acc_taskil.npy'), np.asarray(til_acc, dtype=np.float64))
    if til_loss is not None:
        np.save(join(folder, 'R_test_loss_taskil.npy'), np.asarray(til_loss, dtype=np.float64))


def _parse_probe_points(spec, task_length_L):
    """Convert ``'boundary,0.25,0.5,end'`` into a set of task_steps in [1, L].

    ``boundary`` → 0.0 (first step of a task), ``end`` → 1.0 (last step).
    """
    if task_length_L <= 0:
        return set()
    out = set()
    for part in str(spec).split(','):
        part = part.strip().lower()
        if not part:
            continue
        if part == 'boundary':
            frac = 0.0
        elif part == 'end':
            frac = 1.0
        else:
            frac = float(part)
        step = int(round(frac * task_length_L))
        step = max(1, min(task_length_L, step))
        out.add(step)
    return out


def _build_probe_heldout(task_datasets, crt_task_id, per_task, batch_size, seed, device):
    """Deterministic draw of ``per_task`` examples from each PAST task's test
    split, chunked into (task_id, x, y) batches for the probe's held-out loss.
    """
    rng = torch.Generator().manual_seed(int(seed) * 10_007 + int(crt_task_id))
    batches = []
    for past_t in range(crt_task_id):
        _, test_ds = task_datasets[past_t]
        n = len(test_ds)
        take = min(int(per_task), n)
        if take <= 0:
            continue
        perm = torch.randperm(n, generator=rng)[:take].tolist()
        images, labels = [], []
        for i in perm:
            image, label = test_ds[i][:2]
            images.append(image)
            labels.append(label if torch.is_tensor(label) else int(label))
        # chunk into batch_size-sized tiles
        for start in range(0, take, int(batch_size)):
            end = min(start + int(batch_size), take)
            x = torch.stack(images[start:end])
            if torch.is_tensor(labels[start]):
                y = torch.stack(labels[start:end])
            else:
                y = torch.tensor(labels[start:end], dtype=torch.long)
            batches.append((past_t, x, y))
    return batches


def _recompute_refresh_correction(args, model, optimizer, task_datasets, device):
    """Sampled correction for slow_ema_refresh.

    Default (StepLog stores G): for each of the S' sampled past batches,
    forward+backward at θ_n, clip, and compute Δ_k = g_new_k - G_{s_k}. The
    correction is the (possibly importance-weighted) mean of Δ_k — the
    control-variate estimator of μ_fresh − μ_stale.

    Fallback (StepLog has store_grad=False, ResNet-scale models): G_{s_k} is
    not stored. We instead form
        C_raw = Σ_k iw_k · g_new_k / Σ_k iw_k   (unbiased for μ_fresh)
        C     = C_raw − μ_stale
    using the optimizer's maintained slow EMA. Variance is higher (the
    control-variate noise cancellation is gone), but the estimator is still
    unbiased, and the optimizer's effective_slow = exp_avg_slow + C evaluates
    to the same quantity as in the CV case (μ_fresh at large n).

    Returns None when the arm doesn't need it (task 0, empty log, or S'=0).
    """
    draws = optimizer.sample_refresh_steps()
    if not draws:
        optimizer.set_refresh_correction(None)
        return None
    uses_cv = all(d['G_old'] is not None for d in draws)
    clip_grad = float(getattr(args, 'clip_grad', 0.0) or 0.0)
    acc = None
    weight_sum = 0.0
    saved = [p.grad.detach().clone() if p.grad is not None else None
             for p in model.parameters()]
    for draw in draws:
        task_id = int(draw['task'])
        idx_arr = draw['idx']
        iw = float(draw['weight'])
        # Rebuild (x, y) exactly like _recompute_memory_grad does: fetch each
        # example from task_datasets[task_id][0] (the per-task TRAIN dataset,
        # pre-rotated at construction) and stack.
        images, labels = [], []
        for i in idx_arr.tolist():
            image, label = task_datasets[task_id][0][int(i)][:2]
            images.append(image)
            labels.append(label if torch.is_tensor(label) else int(label))
        x = torch.stack(images).to(device, non_blocking=True)
        y = (torch.stack(labels).to(device, non_blocking=True)
             if torch.is_tensor(labels[0])
             else torch.tensor(labels, dtype=torch.long, device=device))
        model.zero_grad(set_to_none=True)
        CE(model(x, task_id=0, out='logits'), y).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip_grad if clip_grad > 0 else float('inf'))
        g_new = torch.cat([p.grad.detach().reshape(-1).float().cpu()
                           for p in model.parameters() if p.requires_grad])
        if uses_cv:
            G_old = draw['G_old']  # fp32 flat tensor on CPU
            term = (g_new - G_old) * iw
        else:
            term = g_new * iw
        acc = term if acc is None else (acc + term)
        weight_sum += iw
    # Restore prior grads.
    for param, grad in zip(model.parameters(), saved):
        param.grad = grad
    C = acc / float(weight_sum)
    if not uses_cv:
        # Subtract the exact bias-corrected μ_stale so the optimizer's
        # effective_slow = exp_avg_slow + C still targets μ_fresh.
        mu_stale = optimizer.mu_stale_flat()
        if mu_stale.numel() != C.numel():
            raise RuntimeError(
                f'mu_stale_flat ({mu_stale.numel()}) and refresh correction '
                f'({C.numel()}) have different flat sizes')
        C = C - mu_stale
    optimizer.set_refresh_correction(C)
    return C


def _recompute_memory_grad(model, pairs, task_datasets, device, clip_grad=0):
    """Mean cross-entropy gradient of gathered past examples, at the current weights.

    Clipped with the same rule as the current-batch gradient.
    """
    images = []
    labels = []
    for task_id, index in pairs.tolist():
        image, label = task_datasets[int(task_id)][0][int(index)][:2]
        images.append(image)
        labels.append(label if torch.is_tensor(label) else int(label))
    x = torch.stack(images).to(device, non_blocking=True)
    if torch.is_tensor(labels[0]):
        y = torch.stack(labels).to(device, non_blocking=True)
    else:
        y = torch.tensor(labels, dtype=torch.long, device=device)
    saved = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    CE(model(x, task_id=0, out='logits'), y).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        clip_grad if clip_grad > 0 else float("inf"),
    )
    fresh = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    for param, grad in zip(model.parameters(), saved):
        param.grad = grad
    return fresh, grad_norm


def _supervised_loss(model, x, y, regression):
    """Mean loss and number of correct predictions (regression: Σ exp(-err²)).

    Split datasets train Class-IL (unmasked CE) so the Class-IL and Task-IL
    evaluations share one run.
    """
    y_hat = model(x, task_id=0, out='logits')
    n = x.shape[0]
    if regression:
        if y_hat.dim() > 1:
            y_hat = y_hat.squeeze(-1)
        loss_sum = mse_loss(y_hat, y, reduction='sum')
        correct = torch.exp(-(y_hat - y) ** 2).sum().item()
        return loss_sum / n, correct
    loss_sum = CE(y_hat, y, reduction='sum')
    correct = (torch.argmax(y_hat, 1) == y).sum().item()
    return loss_sum / n, correct


def _load_task_datasets(args):
    """Per-task (train, test) datasets. Loaders from the builders are dropped.

    Lengths are needed before the optimizer exists: ``--horizon_ratio`` turns
    them into L, then into β3 / β4.
    """
    quad_specs = None
    if _is_clear(args.dataset_name):
        task_loaders = get_clear_tasks(args)
    elif _is_regression(args.dataset_name):
        task_loaders, quad_specs = get_quadratic_tasks(args)
    else:
        task_loaders = get_rotated_tasks(args)
    task_datasets = [(tr_ld.dataset, te_ld.dataset) for (tr_ld, te_ld) in task_loaders]
    del task_loaders
    return task_datasets, quad_specs


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _apply_channel_variant(args):
    """``--channel_variant`` override of α / β4, before anything reads them.

    Selects a named variant of AdEMAMix. ``both`` keeps --alpha / --beta4 as passed.

    The slow second moment enters the denominator scaled by the same α as the
    slow first moment, so α=0 switches both off. ``metric_only`` is therefore
    plain Adam.
    """
    variant = getattr(args, 'channel_variant', None)
    if variant == 'dual_only':
        args.beta4 = None
    elif variant == 'metric_only':
        print('WARNING: --channel_variant metric_only sets α=0, which also removes '
              'the slow second moment (it is weighted by α). This run is Adam.')
        args.alpha = 0.0


def _ademamix_channel_variant(args):
    """Which AdEMAMix memory channels actually reach the update.

    Computed from the final α / β4 (after --channel_variant and the horizon),
    so grouping works however the variant was requested. α=0 removes both
    slow channels; β4=None or 0 leaves the slow second moment without memory.
    """
    if args.optim != 'ademamix':
        return None
    try:
        has_dual = float(args.alpha) != 0.0
    except (TypeError, ValueError):
        has_dual = False
    beta4 = getattr(args, 'beta4', None)
    has_metric = has_dual and beta4 is not None and float(beta4) > 0.0
    if has_dual and has_metric:
        return 'both'
    if has_dual:
        return 'dual_only'
    return 'neither'


def _resolve_protocol(args):
    """Load the tasks and fix every derived hyperparameter before W&B sees args.

    β3/β4 come from the task horizon, warmup from warmup_tasks × L, and the
    scaled ER capacity from the angle schedule.
    """
    task_datasets, quad_specs = _load_task_datasets(args)
    task_length_L = resolve_task_length(
        args.dataset_name,
        int(args.batch_size),
        int(args.epochs_per_task),
        [len(train_ds) for train_ds, _ in task_datasets],
        args=args,
        n_tasks=args.tasks,
    )
    apply_slow_horizon(args, task_length_L)

    # Warmup length tracks the task horizon: warmup_tasks tasks' worth of steps.
    # Counts global optimizer steps (as in AdEMAMix), so warmup_tasks=1 warms the
    # slow term (β3: β1→β3, α: 0→α) over task 0. Uses the measured L, so ragged
    # streams like CLEAR get a sensible value from their median task length.
    warmup_tasks = float(getattr(args, 'warmup_tasks', 0.0) or 0.0)
    if warmup_tasks > 0 and task_length_L:
        warmup_steps = int(round(warmup_tasks * task_length_L))
        args.beta3_warmup = warmup_steps
        args.alpha_warmup = warmup_steps
        args.cfg_warmup_tasks = warmup_tasks
        args.cfg_warmup_steps = warmup_steps

    # Scaled ER (K = m * n_distinct_angles) must land before W&B snapshots
    # the config and before the reservoir is allocated.
    apply_replay_per_angle(args)
    # Total optimizer steps = L * n_tasks. Used by StepLog to preallocate the
    # per-step history buffer for the staleness probe and slow_ema_refresh.
    args.cfg_total_steps = int(task_length_L or 0) * int(args.tasks)
    return task_datasets, quad_specs, task_length_L


def _config_update(args, task_length_L, base_output_folder, replay_enabled):
    """Derived config written to W&B next to the CLI arguments."""
    K = int(args.replay_buffer_size) if replay_enabled else 0
    cfg = {
        'base_output_folder': base_output_folder,
        'run_output_folder': args.output_folder,
        'ademamix_channel_variant': _ademamix_channel_variant(args),
        'replay': int(replay_enabled),
        'replay_buffer_size': K,
        'cfg/family': optimizer_family_name(args.optim),
        'cfg/arm': infer_memory_arm(args),
        'cfg/K': K,
        'cfg/minibatch_size': int(args.batch_size),
        'cfg/L': float(getattr(args, 'cfg_L', task_length_L)),
        'cfg/tau': float(getattr(args, 'cfg_tau', float('nan'))),
        'cfg/horizon_ratio': getattr(args, 'cfg_horizon_ratio', None),
        'cfg/past_task_mass': getattr(args, 'cfg_past_task_mass', None),
        'cfg/beta3': float(args.beta3),
        'cfg/beta4': None if getattr(args, 'beta4', None) is None else float(args.beta4),
        'cfg/tau3_over_L': tau3_over_L(args),
        'cfg/replay_method': 'er' if replay_enabled else 'none',
        'guard/state_carried': 1,
    }
    ablation = getattr(args, 'ablation', None)
    if ablation:
        cfg.pop('cfg/arm', None)
        cfg['cfg/ablation'] = ablation
        switches = ABLATION_ESTIMATORS.get(ablation)
        if switches is not None:
            if switches[:2] == ('reservoir', 'uniform'):
                # Uniform draws over a whole-history reservoir: no horizon.
                cfg['cfg/tau'] = None
                cfg['cfg/horizon_ratio'] = None
                cfg['cfg/tau3_over_L'] = None
            cfg['cfg/memory_support'] = switches[0]
            cfg['cfg/memory_weighting'] = switches[1]
            cfg['cfg/memory_gradients'] = switches[2]
            cfg['cfg/buffer_size'] = int(getattr(args, 'buffer_size', 0) or 0)
            draws = int(getattr(args, 'memory_samples', 0) or 0)
            cfg['cfg/memory_samples'] = draws if draws > 0 else int(args.batch_size)
            budget = int(getattr(args, 'memory_examples', 0) or 0)
            if switches[2] == 'fresh':
                # Example budget, ER's currency; 0 = the whole FIFO window.
                cfg['cfg/memory_examples'] = budget
                cfg['cfg/K'] = budget
    if getattr(args, 'cyclic_angles', None) is not None:
        slots = int(getattr(args, 'cfg_cycle_slots', 0) or 0)
        L_now = float(getattr(args, 'cfg_L', task_length_L))
        cfg.update({
            'cfg/stream_type': args.cfg_stream_type,
            'cfg/angle_delta': int(args.cfg_angle_delta),
            'cfg/cycle_slots': slots,
            'cfg/amplitude_deg': int(args.cfg_amplitude_deg),
            'cfg/n_distinct_angles': int(args.cfg_n_distinct_angles),
            'cfg/C': float(slots) * L_now if slots else None,
            'cfg/angles': [int(a) for a in args.cyclic_angles],
            'cfg/exemplars_per_angle': getattr(args, 'cfg_exemplars_per_angle', None),
            'cfg/K_schedule': getattr(args, 'cfg_K_schedule', None),
        })
    return cfg


def _build_trackers(args, model, optimizer, quad_specs):
    """Optional diagnostics; each entry is None when switched off."""
    # Per-layer momentum / squared-gradient / preconditioner tracker. Snapshots
    # state[p][momentum_buffer|exp_avg|exp_avg_fast|exp_avg_slow|exp_avg_sq]
    # **before** optimizer.step(), so the buffers still hold m_{t-1}, v_{t-1}.
    # See `OptimGeometryTracker` in utils.diagnostics for the metric definitions.
    geom = OptimGeometryTracker(model, optimizer) if bool(args.track_optim_geom) else None

    # Geometric-drift tracker — only meaningful with AdEMAMix (reads exp_avg_slow).
    # At the end of each task, captures the slow-momentum snapshot and the
    # half-life data buffer; then during the *next* task evaluates
    # CosineSim(g_actual, m_slow_T) and ||g_actual - m_slow_T|| / ||m_slow_T||.
    track_drift = bool(getattr(args, 'track_drift', 0))
    if track_drift and args.optim != 'ademamix':
        print(
            'WARNING: --track_drift=1 has no effect unless --optim ademamix '
            '(requires exp_avg_slow in optimizer state). Disabling drift tracker.'
        )
        track_drift = False
    drift = (
        GeometricDriftTracker(
            model=model,
            optimizer=optimizer,
            beta3=args.beta3,
            batch_size=args.batch_size,
            eval_every=max(1, int(getattr(args, 'drift_eval_every', 1))),
            track_ref=bool(getattr(args, 'track_drift_ref', 0)),
        )
        if track_drift else None
    )

    # Quadratic-benchmark-only diagnostics: v_t^s-vs-curvature check and
    # distance-to-task-0-optimum. See `QuadraticMemoryDiagnostics` in
    # utils.diagnostics.
    quad = (
        QuadraticMemoryDiagnostics(model=model, optimizer=optimizer)
        if (_is_quadratics(args.dataset_name) and args.optim == 'ademamix' and quad_specs is not None)
        else None
    )
    return geom, drift, quad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _unpack_batch(batch, recomputes_gradients, device):
    """``(x, y, heads, index)``; heads is the per-example task id when present."""
    index = None
    if recomputes_gradients:
        if len(batch) == 4:
            x, y, heads, index = batch
        elif len(batch) == 3:
            x, y, index = batch
            heads = None
        else:
            raise RuntimeError('recomputed gradients expect the dataset index on each training batch')
    elif len(batch) == 3:
        x, y, heads = batch
    else:
        x, y = batch
        heads = None
    if heads is not None:
        heads = heads.to(device, non_blocking=True)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), heads, index


def _replay_loss(model, replay_buffer, args, crt_task_id, loss, n_cur, regression, device):
    """Mix one reservoir minibatch from past tasks into ``loss`` (example-weighted)."""
    replay = replay_buffer.sample_batch(
        args.batch_size, single_task=False, max_task_id=crt_task_id - 1)
    if replay is None:
        return loss
    xr, yr, _ = replay
    xr = xr.to(device, non_blocking=True)
    yr = yr.to(device, non_blocking=True)
    y_hat_r = model(xr, task_id=0, out='logits')
    if regression:
        if y_hat_r.dim() > 1:
            y_hat_r = y_hat_r.squeeze(-1)
        loss_r = mse_loss(y_hat_r, yr)
    else:
        loss_r = CE(y_hat_r, yr)
    n_rep = float(xr.shape[0])
    return (loss * n_cur + loss_r * n_rep) / (n_cur + n_rep)


def _train_task(crt_task_id, args, model, optimizer, task_datasets, replay_buffer,
                trackers, quad_specs, loader_kw, device, counters,
                probe=None, heldout_batches=None, probe_task_steps=None,
                task_length_L=0, boundary_probe=None):
    """All epochs of one task. ``counters`` carries global step / epoch across tasks."""
    geom_tracker, drift_tracker, quad_diag_tracker = trackers
    regression = _is_regression(args.dataset_name)
    recomputes = getattr(optimizer, 'recomputes_gradients', False)
    # slow_ema_refresh + the probe both need the per-step batch indices, even
    # though they do not go through the StreamMemory recompute path.
    uses_step_log = getattr(optimizer, 'step_log', None) is not None
    needs_index = recomputes or uses_step_log
    geom_every = max(1, int(args.optim_geom_every))
    # Gradient-noise proxy. See utils/gradient_noise.py. Cheap-ish (one extra
    # vmapped forward+backward on a single minibatch) but not free, so gated
    # behind --track_grad_noise and only evaluated every
    # --grad_noise_every_epochs epochs.
    track_grad_noise = bool(getattr(args, 'track_grad_noise', 0))
    grad_noise_every = max(1, int(getattr(args, 'grad_noise_every_epochs', 1)))

    train_ds, test_ds = task_datasets[crt_task_id]
    loader_ds = IndexedDataset(train_ds) if needs_index else train_ds
    # Just-in-time DataLoader: only one task's workers are alive at a time.
    train_loader = DataLoader(loader_ds, shuffle=True, batch_size=args.batch_size, **loader_kw)
    model.train()

    # Reset the learning rate and start a fresh per-task scheduler.
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    scheduler = make_lr_scheduler(args, optimizer)

    if quad_diag_tracker is not None:
        quad_diag_tracker.on_task_start(quad_specs[crt_task_id])

    # Task-boundary probe: sample the held-out probe batch from this task's
    # TEST split (avoids training-batch overlap) and compute the within-task
    # task_steps at which the probe should fire. Called BEFORE training this
    # task so H[s] exists when later probes reference it.
    boundary_probe_steps = {}       # dict: task_steps -> task_fraction
    boundary_probe_mid_step = None  # single task_steps value for midpoint snapshot
    if boundary_probe is not None:
        boundary_probe.on_task_start(crt_task_id, test_ds, device)
        # Total batches in this task across all epochs. L_task = ceil(N_train/B) * E.
        batches_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
        L_task = batches_per_epoch * int(args.epochs_per_task)
        boundary_probe_steps = boundary_probe.batch_ids_for_task(
            L_task, boundary_probe.probe_task_fractions)
        # Which task_steps corresponds to the 0.5 fraction (midpoint snapshot)?
        for step_key, frac in boundary_probe_steps.items():
            if abs(frac - 0.5) < 1e-6:
                boundary_probe_mid_step = step_key
                break

    task_id_for_head = 0 if not args.use_multihead_model else crt_task_id
    task_steps = 0

    for epoch in range(args.epochs_per_task):
        train_loss, train_size, train_correct = 0., 0, 0
        progress = tqdm(train_loader)

        # Only record/log optimizer geometry on a subset of epochs to cut
        # the per-step recording overhead and W&B history volume. The first
        # and last epoch of each task are always included so task-boundary
        # transitions are captured.
        last_epoch = epoch == args.epochs_per_task - 1
        log_geom_this_epoch = geom_tracker is not None and (epoch % geom_every == 0 or last_epoch)
        log_grad_noise_this_epoch = track_grad_noise and (epoch % grad_noise_every == 0 or last_epoch)
        grad_noise_sigma_sq = None

        for batch_id, batch in enumerate(progress):
            task_steps += 1
            counters['step'] += 1
            x, y, heads, index = _unpack_batch(batch, needs_index, device)
            crt_batch_size = x.shape[0]
            train_size += crt_batch_size

            if hasattr(optimizer, 'set_task_context'):
                optimizer.set_task_context(crt_task_id, args.tasks)
            optimizer.zero_grad()

            # SAM: two-pass gradient when --rho_sam > 0 and --sam_mode fires
            # for this task. Replaces the standard loss.backward() +
            # clip_grad_norm_ pair; leaves p.grad holding the perturbed
            # gradient so the downstream memory/step path consumes it as
            # usual.
            sam_now = (float(getattr(args, 'rho_sam', 0.0) or 0.0) > 0.0
                       and _sam_is_active(getattr(args, 'sam_mode', 'none'),
                                           crt_task_id))
            if sam_now:
                from optimizers.sam import sam_step
                def _recompute_loss():
                    l, _ = _supervised_loss(model, x, y, regression)
                    if replay_buffer is not None and crt_task_id > 0:
                        l = _replay_loss(model, replay_buffer, args, crt_task_id,
                                         l, float(crt_batch_size), regression, device)
                    return l
                sam_info = sam_step(model, _recompute_loss,
                                    rho=float(args.rho_sam),
                                    clip_grad=float(args.clip_grad or 0.0))
                loss_val = sam_info['loss']
                grad_norm = sam_info['raw_norm']
                train_correct += 0  # SAM uses the perturbed loss; the
                # per-batch train accuracy is from the first-pass logits above
                # (not surfaced by sam_step) — treat batch_correct as 0 for
                # SAM batches to avoid double-counting.
            else:
                loss, batch_correct = _supervised_loss(model, x, y, regression)
                train_correct += batch_correct
                if replay_buffer is not None and crt_task_id > 0:
                    loss = _replay_loss(model, replay_buffer, args, crt_task_id, loss,
                                        float(crt_batch_size), regression, device)
                # Stop before a non-finite loss reaches the weights or the optimizer state.
                if not torch.isfinite(loss):
                    print(f'Exited because loss is {loss.item()} '
                          f'(task {crt_task_id + 1}, epoch {epoch + 1}, batch {batch_id + 1})')
                    sys.exit(NAN_EXIT_CODE)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_grad if args.clip_grad > 0 else float("inf"),
                )
                loss_val = float(loss.item())
            step_log = {
                f'grad/task={crt_task_id}/step': task_steps,
                f'grad/task={crt_task_id}/norm': grad_norm,
                'grad/global/step': counters['step'],
                'grad/global/norm': grad_norm,
            }
            if needs_index:
                optimizer.set_batch_indices(index)
            # StepLog bookkeeping for the probe / refresh arm.
            if uses_step_log:
                raw_norm = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                clip_c = float(args.clip_grad or 0.0)
                kappa = min(1.0, clip_c / raw_norm) if (clip_c > 0 and raw_norm > 0) else 1.0
                optimizer.set_clip_info(raw_norm=raw_norm, kappa=kappa,
                                        loss=float(loss_val))
            if getattr(optimizer, 'needs_recomputed_gradient', lambda: False)():
                if regression:
                    raise RuntimeError('recomputed memory gradients are classification only')
                pairs = optimizer.sample_stored_examples()
                if pairs is not None:
                    memory_grad, memory_grad_norm = _recompute_memory_grad(
                        model, pairs, task_datasets, device, args.clip_grad)
                    optimizer.set_recomputed_gradient(memory_grad)
                    step_log['grad/memory/norm'] = memory_grad_norm
            # slow_ema_refresh: sampled correction C = mean(g_new - G_old).
            if getattr(optimizer, 'needs_refresh_batches', lambda: False)():
                if regression:
                    raise RuntimeError('slow_ema_refresh is classification only')
                _recompute_refresh_correction(args, model, optimizer, task_datasets, device)
            if replay_buffer is not None:
                replay_buffer.add_batch(x, y, heads if heads is not None else crt_task_id)
            if log_geom_this_epoch:
                geom_tracker.record()
            if 'double' in args.optim:
                optimizer.step(alpha=double_mix_alpha(args.alpha, crt_task_id))
            else:
                optimizer.step()
            if getattr(optimizer, 'ablation_enabled', False):
                memory_metrics = optimizer.pop_memory_metrics()
                if memory_metrics:
                    step_log.update(memory_metrics)
            wandb.log(step_log)
            # Staleness probe trigger: only task_id>=1, only at designated
            # per-task fractions, only on slow_ema with --probe_staleness.
            if (probe is not None and crt_task_id >= 1
                    and probe_task_steps is not None
                    and task_steps in probe_task_steps):
                probe.run(
                    n=int(counters['step']), task_id=int(crt_task_id),
                    model=model, optimizer=optimizer,
                    task_datasets=task_datasets, heldout_batches=heldout_batches,
                    device=device,
                )

            # Task-boundary probe mid-task snapshot (fires once per task at
            # the 0.5 fraction, for every task s including task 0 — the
            # snapshot is used by probes in later tasks).
            if (boundary_probe is not None
                    and boundary_probe_mid_step is not None
                    and task_steps == boundary_probe_mid_step):
                boundary_probe.on_task_midpoint(crt_task_id, model)

            # Task-boundary probe fire: at every scheduled task_fraction of
            # every task t >= 1. Also fires at the end-of-task fraction 1.0
            # BEFORE on_task_end, so the probe sees theta_now at the final
            # training step of this task (not after the next task has begun).
            if (boundary_probe is not None
                    and crt_task_id >= 1
                    and task_steps in boundary_probe_steps):
                boundary_probe.on_probe_step(
                    n=int(counters['step']),
                    t=int(crt_task_id),
                    task_fraction=float(boundary_probe_steps[task_steps]),
                    model=model, device=device,
                )

            if quad_diag_tracker is not None:
                quad_metrics = quad_diag_tracker.step_metrics()
                if quad_metrics:
                    quad_metrics['ademamix/step'] = optimizer._global_step
                    wandb.log(quad_metrics)

            if log_grad_noise_this_epoch and batch_id == 0:
                # One measurement per qualifying epoch, on its first minibatch.
                grad_noise_sigma_sq = estimate_gradient_noise(
                    model, x, y, task_id=task_id_for_head, regression=regression)

            train_loss += float(loss_val) * crt_batch_size
            progress.set_description(
                f'[Task {crt_task_id+1}/{args.tasks}]'
                f'[Epoch {epoch+1}]'
                f'[Batch {batch_id+1}]'
                f'[loss={train_loss/train_size:.6f}, '
                f'acc={train_correct/train_size*100:.2f}%]')

        scheduler.step()  # per-epoch schedule
        train_loss /= train_size
        train_acc = train_correct / train_size * 100.

        # Log everything for this epoch in a SINGLE wandb.log() commit so
        # all per-epoch metrics share one internal step. Splitting this into
        # several wandb.log() calls puts `sequence/*`, `ademamix/*` and
        # `optim_geom/*` on different internal steps, and since each family
        # is pinned to its own step_metric the workspace panels end up
        # sparse/empty depending on the selected x-axis. Their step metrics
        # (`global_epoch`, `ademamix/step`) must be present in the same
        # commit as the values they index.
        epoch_log = {
            'global_epoch': counters['epoch'],
            'crt_task': crt_task_id + 1,
            'sequence/TrainLoss': train_loss,
            'sequence/TrainAcc': train_acc,
            'epoch': epoch,
            f'task{crt_task_id+1}of{args.tasks}_TrainLoss': train_loss,
            f'task{crt_task_id+1}of{args.tasks}_TrainAcc': train_acc,
        }
        if hasattr(optimizer, 'get_tracking_metrics'):
            epoch_log.update(optimizer.get_tracking_metrics(prefix='ademamix'))
        if log_geom_this_epoch:
            geom_metrics = geom_tracker.flush(prefix='optim_geom')
            if geom_metrics:
                epoch_log.update(geom_metrics)
        if drift_tracker is not None:
            drift_metrics = drift_tracker.evaluate(prefix='drift')
            if drift_metrics:
                epoch_log.update(drift_metrics)
        if grad_noise_sigma_sq is not None:
            epoch_log['grad_noise/sigma_sq'] = grad_noise_sigma_sq
        if replay_buffer is not None:
            epoch_log['replay/num_examples'] = len(replay_buffer)
            epoch_log['replay/n_seen'] = replay_buffer.n_seen
        wandb.log(epoch_log)
        counters['epoch'] += 1

        if (args.min_train_loss > 0) and (train_loss <= args.min_train_loss):
            break
    # Task-boundary probe end-of-task snapshot. Fires AFTER the full task
    # (and after any within-task probe at fraction 1.0 already ran); records
    # theta_end[s], N[s], S[s] for use by probes in later tasks.
    if boundary_probe is not None:
        boundary_probe.on_task_end(crt_task_id, model, int(counters['step']))

    torch.save(model.state_dict(), join(args.output_folder, f'chkpt-task{crt_task_id+1}of{args.tasks}.pt'))
    # Dump args.json next to the first checkpoint so from-checkpoint analysis
    # can reconstruct the model via get_model(args) without a W&B API call.
    if crt_task_id == 0:
        import json as _json
        args_path = join(args.output_folder, 'args.json')
        if not os.path.exists(args_path):
            serialisable = {}
            for k, v in vars(args).items():
                try:
                    _json.dumps(v)
                    serialisable[k] = v
                except (TypeError, ValueError):
                    serialisable[k] = repr(v)
            with open(args_path, 'w') as fh:
                _json.dump(serialisable, fh, indent=2, sort_keys=True)
    del train_loader

    # Snapshot slow-momentum and half-life buffer for the *next* task's drift
    # evaluation. Skipped after the final task since there is no following
    # task to evaluate against.
    if drift_tracker is not None and crt_task_id < args.tasks - 1:
        drift_tracker.on_task_end(
            train_dataset=train_ds,
            task_id_for_head=task_id_for_head,
            regression=regression,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_range(args, crt_task_id):
    """Test tasks evaluated after training task ``crt_task_id``.

    - CLEAR + streaming: seen buckets (retention / forgetting, in_domain,
      backward_transfer) plus the next bucket t+1 (CLEAR next-domain metric).
    - CLEAR + iid, and every other stream: all tasks, seen and not yet
      trained, for the full R matrix and FWT.
    """
    if _is_clear(args.dataset_name) and args.clear_eval_protocol == 'streaming':
        tasks = list(range(crt_task_id + 1))
        if (crt_task_id + 1) < args.tasks:
            tasks.append(crt_task_id + 1)
        return tasks
    return list(range(args.tasks))


def _evaluate_after_task(crt_task_id, args, model, task_datasets, R, task_il_cpt, loader_kw):
    """Fill row ``crt_task_id`` of every R matrix and log the per-task summaries."""
    regression = _is_regression(args.dataset_name)
    model.eval()
    for prev_task_id in _eval_range(args, crt_task_id):
        _, prev_test_ds = task_datasets[prev_task_id]
        loader = DataLoader(prev_test_ds, shuffle=False, batch_size=args.batch_size, **loader_kw)
        correct, loss_sum, size = test(model, loader, task_id=0, regression=regression)
        mean_loss = loss_sum / size
        R['acc'][crt_task_id][prev_task_id] = correct / size * 100.0
        # R-test-loss.txt and task*_TestLoss keep a ×100 scale;
        # the .npy dumps and eval_loss/* use the raw mean loss.
        R['loss_x100'][crt_task_id][prev_task_id] = mean_loss * 100.0
        R['loss'][crt_task_id][prev_task_id] = mean_loss
        if task_il_cpt:
            til_correct, til_loss_sum, til_size = test(
                model, loader, task_id=0, regression=regression,
                task_il_cpt=task_il_cpt, mask_task_id=prev_task_id)
            R['acc_til'][crt_task_id][prev_task_id] = til_correct / til_size * 100.0
            R['loss_til'][crt_task_id][prev_task_id] = til_loss_sum / til_size
        print(f'Evaluated on task {prev_task_id+1}/{args.tasks}')
        del loader

    # Evolution of accuracy across training — one point per training task.
    row_all = np.array(R['acc'][crt_task_id], dtype=float)
    tag = f'task{crt_task_id+1}of{args.tasks}'
    log_dict = {
        'crt_task': crt_task_id + 1,
        f'{tag}_TestAcc': R['acc'][crt_task_id][crt_task_id],
        f'{tag}_TestLoss': R['loss_x100'][crt_task_id][crt_task_id],
        'AvgTestAccSeen': _nanmean_or_nan(row_all[:crt_task_id + 1]),
    }
    if task_il_cpt:
        row_til = np.array(R['acc_til'][crt_task_id], dtype=float)
        log_dict[f'{tag}_TestAcc_TIL'] = R['acc_til'][crt_task_id][crt_task_id]
        log_dict['AvgTestAccSeen_TIL'] = _nanmean_or_nan(row_til[:crt_task_id + 1])
    if _is_clear(args.dataset_name):
        log_dict['AvgTestAccAll'] = _nanmean_or_nan(row_all)
        if args.clear_eval_protocol == 'streaming':
            nxt = crt_task_id + 1
            has_next = nxt < args.tasks
            log_dict[f'{tag}_NextTestAcc'] = R['acc'][crt_task_id][nxt] if has_next else float('nan')
            log_dict[f'{tag}_NextTestLoss'] = R['loss_x100'][crt_task_id][nxt] if has_next else float('nan')
    n_seen = crt_task_id + 1
    if n_seen >= 2:
        log_dict['FRGT'] = cl_metric_forgetting_after(R['acc'], n_seen)
    log_dict['guard/state_carried'] = 1
    _log_cl_r_summaries(
        log_dict, R['acc'], R['loss'], n_seen, args,
        til_acc=R.get('acc_til'), til_loss=R.get('loss_til'))
    wandb.log(log_dict)


# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------

def _write_matrix(w, matrix, n, width, digits):
    for i in range(n):
        for j in range(n):
            w.write(f'{round(matrix[i][j], digits):>{width}} ')
        w.write('\n')


def _mammoth_summary(matrix, random_rows):
    """Unrounded Mammoth ACC / BWT / FRGT / FWT of a final R matrix (percent)."""
    return dict(
        ACC=float(cl_metric_avg_acc(matrix)),
        BWT=float(cl_metric_backward_transfer(matrix)),
        FRGT=float(cl_metric_forgetting(matrix)),
        FWT=float(cl_metric_forward_transfer(matrix, random_rows)),
    )


def _write_summary_file(path, matrix, n, summary):
    with open(path, 'w') as w:
        _write_matrix(w, matrix, n, width=5, digits=2)
        w.write('\n')
        w.write(f"avg-acc: {summary['ACC']:.4f}\n")
        w.write(f"bwt: {summary['BWT']:.4f}\n")
        w.write(f"frgt: {summary['FRGT']:.4f}\n")
        w.write(f"fwt: {summary['FWT']:.4f}\n")


def _write_final_results(args, R, random_rows, random_rows_til):
    folder = args.output_folder
    n = args.tasks

    final = _mammoth_summary(R['acc'], random_rows)
    _write_summary_file(join(folder, 'R-test-acc.txt'), R['acc'], n, final)
    r_summaries = {}
    _log_cl_r_summaries(
        r_summaries, R['acc'], R['loss'], n, args,
        til_acc=R.get('acc_til'), til_loss=R.get('loss_til'))
    wandb.log(dict(**final, **r_summaries))
    _save_r_matrices(folder, R['acc'], R['loss'],
                     til_acc=R.get('acc_til'), til_loss=R.get('loss_til'))

    if getattr(args, 'cyclic_angles', None) is not None:
        # Position-indexed R is the headline readout. Per-angle
        # means are extra: positions that share an angle should agree.
        report, angle_metrics = angle_accuracy_metrics(R['acc'], args.cyclic_angles)
        with open(join(folder, 'angle_final_acc.txt'), 'w') as ang_f:
            ang_f.write(report)
        np.save(join(folder, 'angles.npy'), np.asarray(args.cyclic_angles, dtype=np.int64))
        print(report)
        wandb.log(angle_metrics)

    if R.get('acc_til') is not None:
        til = _mammoth_summary(R['acc_til'], random_rows_til)
        _write_summary_file(join(folder, 'R-test-acc-til.txt'), R['acc_til'], n, til)
        wandb.log({f'{k}_TIL': v for k, v in til.items()})

    with open(join(folder, 'R-test-loss.txt'), 'w') as w:
        _write_matrix(w, R['loss_x100'], n, width=8, digits=4)

    # ---- CLEAR paper metrics (only for CLEAR datasets) ----
    if _is_clear(args.dataset_name):
        clear_m = CLEARMetric().get_metrics(np.array(R['acc']))
        print(f'\nCLEAR metrics: {clear_m}')
        with open(join(folder, 'clear_metrics.txt'), 'w') as w:
            for k, v in clear_m.items():
                w.write(f'{k}: {v:.4f}\n')
        wandb.log({
            'CLEAR_in_domain':   clear_m['in_domain'],
            'CLEAR_next_domain': clear_m['next_domain'],
            'CLEAR_accuracy':    clear_m['accuracy'],
            'CLEAR_fwt':         clear_m['forward_transfer'],
            'CLEAR_bwt':         clear_m['backward_transfer'],
        })


# ---------------------------------------------------------------------------

def main():
    args = get_arg_parse()
    check_naive_args(args, sys.argv[1:])
    if not args.clip_grad > 0:
        print('WARNING: --clip_grad is 0, so gradients are not clipped '
              '(every plan sweep uses --clip_grad 1.0).')

    if bool(getattr(args, 'pretrained', 0)):
        raise RuntimeError(
            'From-scratch protocol: --pretrained 1 is not allowed '
            '(no ImageNet / CLIP / bucket-0 MoCo weights).'
        )

    apply_il_protocol(args)
    task_il_cpt = task_il_block_size(args)
    _apply_channel_variant(args)
    set_all_seeds(args.seed)

    # Lengths first, so β3/β4 are the task-horizon coefficients before W&B
    # snapshots the config and before the optimizer is built.
    task_datasets, quad_specs, task_length_L = _resolve_protocol(args)

    run = setup_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        job_type=args.wandb_job_type,
        group=args.wandb_group,
        name=args.wandb_name,
        config=args)
    run_name = sweep_run_name(run, args)
    if run_name:
        run.name = run_name
    define_task_metrics(num_tasks=args.tasks)

    # Per-run subfolder so parallel sweep workers do not overwrite each other's
    # checkpoints / R-*.txt / clear_metrics.txt / mammoth_random_baseline.txt.
    base_output_folder = args.output_folder
    run_id = run.id if run is not None else 'offline'
    args.output_folder = join(base_output_folder, run_id)
    os.makedirs(args.output_folder, exist_ok=True)

    if getattr(args, 'ablation', None) == 'replay' and not getattr(args, 'replay', 0):
        args.replay = 1
    replay_enabled = bool(getattr(args, 'replay', 0)) and int(getattr(args, 'replay_buffer_size', 0)) > 0
    replay_buffer = ReservoirReplayBuffer(args.replay_buffer_size, seed=args.seed) if replay_enabled else None
    if replay_enabled:
        print(f'Replay reservoir enabled: capacity={args.replay_buffer_size} examples')
    wandb.config.update(
        _config_update(args, task_length_L, base_output_folder, replay_enabled),
        allow_val_change=True)

    device = get_device()
    model = get_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({'n_params': int(n_params)}, allow_val_change=True)
    print(f'Model {args.model} n_params={n_params}')

    optimizer = get_optimizer(args, model_params=model.parameters(), model_modules=model.modules())
    trackers = _build_trackers(args, model, optimizer, quad_specs)

    # persistent_workers=True is required when num_workers>0: otherwise each
    # epoch rebuilds a multiprocessing iterator (queues + POSIX semaphores in
    # /dev/shm). That can leak until SemLock raises FileNotFoundError
    # around epoch ~50. Quadratics stay in-process (num_workers=0).
    regression = _is_regression(args.dataset_name)
    dl_workers = 0 if regression else 4
    loader_kw = dict(num_workers=dl_workers, pin_memory=not regression,
                     persistent_workers=dl_workers > 0)

    task_test_datasets = [te_ds for _, te_ds in task_datasets]
    random_rows = mammoth_random_results_baseline(
        model, args, task_test_datasets, args.use_multihead_model)
    save_mammoth_random_baseline(args.output_folder, random_rows)
    random_rows_til = None
    if task_il_cpt:
        # DataLoader iterators draw a seed from the global RNG; fork it so this
        # extra pass leaves the training shuffle exactly as without it.
        with torch.random.fork_rng(devices=[]):
            random_rows_til = mammoth_random_results_baseline(
                model, args, task_test_datasets, args.use_multihead_model, task_il_cpt=task_il_cpt)
        save_mammoth_random_baseline(
            args.output_folder, random_rows_til, filename='mammoth_random_baseline_til.txt')

    # R[i][j]: test metric on task j after training task i. NaN = not evaluated
    # (e.g. CLEAR streaming beyond the next bucket).
    R = {name: _nan_matrix(args.tasks) for name in ('acc', 'loss', 'loss_x100')}
    if task_il_cpt:
        R['acc_til'] = _nan_matrix(args.tasks)
        R['loss_til'] = _nan_matrix(args.tasks)

    counters = {'step': 0, 'epoch': 0}  # global optimizer step / epoch across tasks

    # Staleness probe setup (slow_ema arm, --probe_staleness 1).
    probe = None
    probe_task_steps = None
    heldout_batches_by_task = None
    if (getattr(args, 'probe_staleness', 0) and args.ablation == 'slow_ema'
            and getattr(optimizer, 'step_log', None) is not None):
        from optimizers.stale_probe import StaleProbe
        # Pick the probe's loss match the training loss (CE by default; MSE
        # for the regression streams like `quadratics`).
        if _is_regression(args.dataset_name):
            def _mean_mse(y_hat, y):
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze(-1)
                return mse_loss(y_hat, y, reduction='mean')
            def _sum_mse(y_hat, y):
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze(-1)
                return mse_loss(y_hat, y, reduction='sum')
            _loss_mean, _loss_sum = _mean_mse, _sum_mse
        else:
            _loss_mean, _loss_sum = None, None  # defaults = CE
        probe = StaleProbe(
            out_dir=args.output_folder, beta3=float(args.beta3),
            task_length_L=int(task_length_L or 0),
            batch_size=int(args.batch_size), clip_c=float(args.clip_grad or 0.0),
            chunk_size=int(args.probe_chunk), run_sigma=bool(int(args.probe_sigma)),
            stride=int(args.probe_stride), window_steps=int(args.probe_window),
            n_tasks=int(args.tasks), seed=int(args.seed),
            dtype=str(args.steplog_dtype),
            loss_mean_fn=_loss_mean, loss_sum_fn=_loss_sum,
        )
        probe_task_steps = _parse_probe_points(args.probe_points, int(task_length_L or 0))
        print(f'[probe] enabled: {len(probe_task_steps)} probe points per task '
              f'at task_steps={sorted(probe_task_steps)} (L={task_length_L})')

    # Task-boundary + mid-task probe setup. Independent of stale_probe above;
    # requires slow_ema arm so the EMA kernel W_s(n) is well-defined.
    boundary_probe = None
    if (getattr(args, 'task_boundary_probe', 0) and args.ablation == 'slow_ema'):
        from optimizers.task_boundary_probe import TaskBoundaryProbe
        if _is_regression(args.dataset_name):
            def _mean_mse_bp(y_hat, y):
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze(-1)
                return mse_loss(y_hat, y, reduction='mean')
            def _sum_mse_bp(y_hat, y):
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze(-1)
                return mse_loss(y_hat, y, reduction='sum')
            _bp_mean, _bp_sum = _mean_mse_bp, _sum_mse_bp
        else:
            _bp_mean, _bp_sum = None, None  # defaults = CE
        fractions = TaskBoundaryProbe.parse_fractions(
            getattr(args, 'probe_task_fractions', '0.25,0.5,0.75,1.0'))
        boundary_probe = TaskBoundaryProbe(
            out_dir=args.output_folder, beta3=float(args.beta3),
            n_tasks=int(args.tasks),
            probe_batch_size=int(getattr(args, 'probe_batch_size', 128)),
            probe_task_fractions=fractions,
            seed=int(args.seed),
            dtype=str(getattr(args, 'probe_snapshot_dtype', 'bf16')),
            loss_mean_fn=_bp_mean, loss_sum_fn=_bp_sum,
        )
        print(f'[boundary_probe] enabled: fractions={fractions}, '
              f'B_probe={args.probe_batch_size}, dtype={args.probe_snapshot_dtype}')

    for crt_task_id in range(args.tasks):
        heldout_batches = None
        if probe is not None and crt_task_id >= 1:
            heldout_batches = _build_probe_heldout(
                task_datasets, crt_task_id, args.probe_heldout, args.batch_size,
                int(args.seed), device)
        _train_task(crt_task_id, args, model, optimizer, task_datasets, replay_buffer,
                    trackers, quad_specs, loader_kw, device, counters,
                    probe=probe, heldout_batches=heldout_batches,
                    probe_task_steps=probe_task_steps,
                    task_length_L=int(task_length_L or 0),
                    boundary_probe=boundary_probe)
        _evaluate_after_task(crt_task_id, args, model, task_datasets, R, task_il_cpt, loader_kw)

    _write_final_results(args, R, random_rows, random_rows_til)


if __name__ == '__main__':
    # wandb.finish() with no exit code marks the run finished. A crash in
    # main() must pass a non-zero code or the sweep treats the cell as done.
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else (0 if code is None else 1)
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        wandb.finish(exit_code=exit_code)
