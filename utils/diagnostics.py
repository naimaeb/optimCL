"""Training diagnostics: Fisher summary, optimizer geometry, geometric drift, quadratic memory."""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy as CE, mse_loss
from torch.utils.data import DataLoader, Subset


def fisher_diagnostics(fisher, probe_grad=None):
    """Probe a DenseCoreMFAC Fisher to check whether damping dominates.

    The preconditioner returns ``F^{-1} g = partA - partB`` where
        partA = (1/damp) * g         (pure damp / identity contribution)
        partB = G^T M(G g)           (empirical-Fisher correction, M built
                                      from the m-step buffer ``G``)
    so ``F^{-1} ≈ (1/damp) I`` iff ``||partB|| << ||partA||``.

    Equivalent spectral view: the empirical Fisher ``F_hat = G^T G / m`` has
    the same nonzero eigenvalues as ``fisher.dots / m`` (since
    ``fisher.dots = G G^T``). If ``lambda_max(F_hat) << damp`` the damp floor
    completely swamps the curvature signal.

    Args
    ----
    fisher : DenseCoreMFAC instance (already filled with gradients).
    probe_grad : optional 1-D tensor. If provided, probe the action of F^{-1}
        on this gradient; otherwise only spectral diagnostics are returned.

    Returns
    -------
    dict of scalars, all keys prefixed ``fisher/`` so they can be filtered in
    W&B and are pinned to the ``crt_task`` step-metric by define_task_metrics().
    """
    import torch as _torch

    out = {}
    with _torch.no_grad():
        out['fisher/damp']         = float(fisher.damp)
        out['fisher/grads_count']  = int(min(fisher.grads_count, fisher.m))

        m = float(fisher.m)
        eig = _torch.linalg.eigvalsh(fisher.dots.float())       # ascending
        eig_max = float(eig[-1].item()) / m
        eig_mean = float(eig.mean().item()) / m                 # = trace / m
        out['fisher/F_hat_eig_max']     = eig_max
        out['fisher/F_hat_trace']       = float(eig.sum().item()) / m
        out['fisher/eig_max_over_damp'] = eig_max / fisher.damp
        out['fisher/trace_over_damp']   = (eig_mean * m) / fisher.damp

        if probe_grad is not None:
            g = probe_grad.detach().to(fisher.dev).flatten()
            u = fisher.precondition(g).flatten()

            pa = fisher.wandb_data.get('norm_partA', None)
            pb = fisher.wandb_data.get('norm_partB', None)
            if pa is not None and pb is not None:
                pa_f = float(pa.item() if hasattr(pa, 'item') else pa)
                pb_f = float(pb.item() if hasattr(pb, 'item') else pb)
                out['fisher/partA_norm']       = pa_f
                out['fisher/partB_norm']       = pb_f
                out['fisher/partB_over_partA'] = pb_f / (pa_f + 1e-20)

            g_norm = float(g.norm(p=2).item())
            u_norm = float(u.norm(p=2).item())
            out['fisher/g_norm']                 = g_norm
            out['fisher/u_norm']                 = u_norm
            # ||u|| / (||g||/damp) == 1 iff F^{-1} acts exactly like (1/damp) I on g.
            out['fisher/u_norm_over_id_action']  = u_norm / ((g_norm / fisher.damp) + 1e-20)
            # Cosine between raw and preconditioned gradient. 1 = no rotation (pure scaling).
            cos_denom = g_norm * u_norm + 1e-20
            out['fisher/cos_g_u'] = float(_torch.dot(g, u).item() / cos_denom)
    return out


class OptimGeometryTracker:
    """Per-layer momentum / squared-gradient / preconditioner diagnostics.

    Captured **between** ``loss.backward()`` and ``optimizer.step()`` so the
    optimizer state holds ``m_{t-1}, v_{t-1}`` (the previous step's buffers)
    while ``p.grad`` already holds the current ``g_t``. After the step the
    buffers are updated to ``m_t, v_t`` and the comparison is no longer
    meaningful.

    Three metric families, all reported layer-wise (one entry per parameter
    tensor known to ``model.named_parameters()``) and globally (concatenating
    every tracked tensor into one vector):

    1. **Momentum dynamics** ``m_{t-1}`` vs ``g_t``

       * Cosine similarity
         ``cos = <m_{t-1}, g_t> / (||m_{t-1}|| ||g_t||)``
       * Magnitude ratio ``||m_{t-1}|| / ||g_t||``

       The momentum buffer is found by inspecting the optimizer state for
       (in priority order) ``momentum_buffer`` (SGD), ``exp_avg`` (Adam,
       AdamW) or ``exp_avg_fast`` (AdEMAMix's fast EMA). When the optimizer
       also has a slow EMA (AdEMAMix's ``exp_avg_slow``) the same two
       metrics are reported under the ``mslow`` suffix.

    2. **Squared-gradient dynamics** ``v_{t-1}`` vs ``g_t^2``

       * Cosine similarity (alignment of the empirical Fisher diagonal)
         ``cos = <v_{t-1}, g_t^2> / (||v_{t-1}|| ||g_t^2||)``
       * Surprise factor ``||g_t^2|| / ||v_{t-1}||``

       ``v_{t-1}`` is read from ``exp_avg_sq``.

    3. **Effective step-size penalty**

       ``ratio = ||g_t / (sqrt(v_{t-1}) + eps)|| / ||g_t||``

       How much the preconditioner is suppressing the raw gradient on this
       layer. ``ratio ≈ 1/sqrt(typical_v)`` if ``v`` is roughly constant;
       ``ratio << 1`` means strong suppression, ``ratio >> 1`` (rare)
       means amplification by directions where ``v`` is far below ``eps^2``.

    Metrics are computed each training step (call :meth:`record` after
    backward, before ``optimizer.step()``), accumulated in running sums, and
    :meth:`flush` returns the mean over the accumulated window (cleared after
    each flush). Skipped when the state buffer is uninitialised or exactly
    zero (first observed step for that parameter).

    Notes
    -----
    * All work happens under ``torch.no_grad`` -- no autograd graph is built
      and the call has zero effect on the upcoming optimizer step. Tensors
      are only ``.float()``-cast when their dtype isn't already a floating
      type (to avoid silent fp16 underflow on cosine denominators without
      paying for an unnecessary copy in the common fp32 case).
    * Optimizers whose ``param_groups`` are split across instances (e.g.
      layer/parameter-wise MFAC, blockadam) are still tracked: we iterate
      ``optimizer.param_groups`` directly, so anything stored under
      ``optimizer.state[p]`` is visible.
    * If the optimizer is a ``list`` (used in some Muon configurations) we
      track each sub-optimizer separately and merge the per-layer entries.
    """

    _MOM_KEYS = ('momentum_buffer', 'exp_avg', 'exp_avg_fast')
    _SLOW_KEY = 'exp_avg_slow'
    _V_KEY = 'exp_avg_sq'

    def __init__(self, model, optimizer, default_eps=1e-8):
        self._optims = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        self._default_eps = float(default_eps)
        self._id_to_name = {id(p): name for name, p in model.named_parameters()}
        self._sums = {}
        self._counts = {}

    def _name_for(self, p):
        return self._id_to_name.get(id(p), f'param_id{id(p)}')

    @staticmethod
    def _cos(a, b):
        an = a.norm()
        bn = b.norm()
        denom = an * bn
        if float(denom) <= 0.0:
            return None
        return float(torch.clamp(torch.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def _ratio(num, den):
        if float(den) <= 0.0:
            return None
        return float(num / den)

    def _add(self, key, value):
        if value is None or not np.isfinite(value):
            return
        self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._counts[key] = self._counts.get(key, 0) + 1

    @staticmethod
    def _as_flat_float(t):
        """Detached, flattened view in fp32 (no copy when already fp32)."""
        t = t.detach()
        if not t.is_floating_point() or t.dtype != torch.float32:
            t = t.float()
        return t.reshape(-1)

    @torch.no_grad()
    def record(self):
        """Snapshot per-layer geometry metrics for the *current* batch.

        Pure observation -- no autograd graph is built and no optimizer
        state is mutated. Call once per training iteration, **after**
        ``loss.backward()`` and **before** ``optimizer.step()``: at that
        point ``p.grad`` already holds ``g_t`` while the optimizer's
        running buffers still hold ``m_{t-1}, v_{t-1}`` from the previous
        iteration.
        """
        global_g = []
        global_g_for_m = []
        global_g_for_v = []
        global_m = []
        global_v = []
        global_gsq = []
        eps_used = self._default_eps

        for opt in self._optims:
            for group in opt.param_groups:
                eps_used = float(group.get('eps', self._default_eps))
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = opt.state.get(p, {})
                    if not state:
                        continue

                    name = self._name_for(p)
                    g = self._as_flat_float(p.grad)
                    g_norm = g.norm()
                    if float(g_norm) <= 0.0:
                        continue

                    mom = None
                    for k in self._MOM_KEYS:
                        v = state.get(k, None)
                        if isinstance(v, torch.Tensor):
                            mom = v
                            break
                    if isinstance(mom, torch.Tensor):
                        m_prev = self._as_flat_float(mom)
                        m_norm = m_prev.norm()
                        if float(m_norm) > 0.0:
                            self._add(f'{name}/cos_m_g', self._cos(m_prev, g))
                            self._add(f'{name}/mag_ratio_m_g', self._ratio(m_norm, g_norm))
                            global_m.append(m_prev)
                            global_g_for_m.append(g)

                    slow = state.get(self._SLOW_KEY, None)
                    if isinstance(slow, torch.Tensor):
                        s_prev = self._as_flat_float(slow)
                        s_norm = s_prev.norm()
                        if float(s_norm) > 0.0:
                            self._add(f'{name}/cos_mslow_g', self._cos(s_prev, g))
                            self._add(f'{name}/mag_ratio_mslow_g', self._ratio(s_norm, g_norm))

                    v_buf = state.get(self._V_KEY, None)
                    if isinstance(v_buf, torch.Tensor):
                        v_prev = self._as_flat_float(v_buf)
                        v_norm = v_prev.norm()
                        g_sq = g * g
                        g_sq_norm = g_sq.norm()
                        if float(v_norm) > 0.0 and float(g_sq_norm) > 0.0:
                            self._add(f'{name}/cos_v_gsq', self._cos(v_prev, g_sq))
                            self._add(f'{name}/surprise', self._ratio(g_sq_norm, v_norm))
                            precond = g / (v_prev.sqrt() + eps_used)
                            self._add(
                                f'{name}/precond_ratio',
                                self._ratio(precond.norm(), g_norm),
                            )
                            global_v.append(v_prev)
                            global_g_for_v.append(g)
                            global_gsq.append(g_sq)

                    global_g.append(g)

        if not global_g:
            return

        g_flat = torch.cat(global_g)
        g_norm = g_flat.norm()
        if float(g_norm) <= 0.0:
            return

        if global_m and global_g_for_m:
            m_flat = torch.cat(global_m)
            g_m_flat = torch.cat(global_g_for_m)
            g_m_norm = g_m_flat.norm()
            if float(m_flat.norm()) > 0.0 and float(g_m_norm) > 0.0 and m_flat.numel() == g_m_flat.numel():
                self._add('global/cos_m_g', self._cos(m_flat, g_m_flat))
                self._add('global/mag_ratio_m_g', self._ratio(m_flat.norm(), g_m_norm))

        if global_v and global_g_for_v and global_gsq:
            v_flat = torch.cat(global_v)
            g_v_flat = torch.cat(global_g_for_v)
            gsq_flat = torch.cat(global_gsq)
            v_norm = v_flat.norm()
            g_v_norm = g_v_flat.norm()
            if float(v_norm) > 0.0 and float(g_v_norm) > 0.0 and v_flat.numel() == gsq_flat.numel() == g_v_flat.numel():
                self._add('global/cos_v_gsq', self._cos(v_flat, gsq_flat))
                self._add('global/surprise', self._ratio(gsq_flat.norm(), v_norm))
                precond = g_v_flat / (v_flat.sqrt() + eps_used)
                self._add(
                    'global/precond_ratio',
                    self._ratio(precond.norm(), g_v_norm),
                )

    def flush(self, prefix='optim_geom'):
        """Return mean of accumulated metrics since the last flush.

        Empty dict if no steps were recorded since the last flush (e.g. when
        the optimizer is still in its first iteration and state buffers have
        not yet been populated). Clears the running sums.
        """
        out = {}
        for key, total in self._sums.items():
            count = self._counts.get(key, 0)
            if count > 0:
                out[f'{prefix}/{key}'] = total / count
        self._sums.clear()
        self._counts.clear()
        return out


class GeometricDriftTracker:
    """Track geometric drift and catastrophic interference at task boundaries.

    Compares the AdEMAMix slow-momentum buffer (``exp_avg_slow``) captured at
    the end of task *k* against the gradient of the *current* model evaluated
    on the half-life data window of that same slow EMA, during the training of
    task *k+1*.

    Scalar metrics are produced each evaluation epoch and returned as a
    dict suitable for merging into the per-epoch ``wandb.log`` call.

    EMA-based comparison (always on) — current statistic vs the optimizer's
    accumulated EMA snapshot from the task boundary:

    * ``drift/cosine_sim``       – CosineSim(g_actual, m_slow_T)
    * ``drift/log_scale_ratio``  – log(||g_actual|| / ||m_slow_T||)
    * ``drift/cosine_sim_sq``       – CosineSim(g_actual², v_sq_T)
    * ``drift/log_scale_ratio_sq``  – log(||g_actual²|| / ||v_sq_T||)

    Reference-based comparison (opt-in via ``track_ref``) — current gradient vs
    the gradient on the *same buffer* recomputed at the *frozen task-boundary
    weights* ``theta_T``.  This isolates pure parameter drift (same data, same
    fixed past weights) and costs one extra stored gradient vector:

    * ``drift/cosine_sim_ref``        – CosineSim(g_actual, g_ref_T)
    * ``drift/log_scale_ratio_ref``   – log(||g_actual|| / ||g_ref_T||)
    * ``drift/cosine_sim_ref_sq``     – CosineSim(g_actual², g_ref_T²)
    * ``drift/log_scale_ratio_ref_sq``– log(||g_actual²|| / ||g_ref_T²||)

    Where:

    * ``m_slow_T`` is the bias-corrected snapshot of ``exp_avg_slow`` captured
      right after the last optimizer step of task *k*.
    * ``g_actual`` is the gradient evaluated on the previous task's sequential
      half-life window *using the current model weights* (no optimizer step
      taken, no gradient leakage into the training loop).

    The half-life window contains the final
    ``N_window = ceil(ln 0.5 / ln beta3) × batch_size`` samples from the
    task-*k* dataset, taken in their original sequential order — the exact
    data points whose gradients have at least half-weight in ``exp_avg_slow``.
    Batches are stored on CPU and the mean-buffer gradient is accumulated
    with one ``backward()`` per mini-batch so peak GPU memory stays comparable
    to a training step (retaining every activation graph and calling
    ``backward()`` once can OOM for large ``beta3`` on CLEAR ResNets).

    Typical usage in a training script::

        drift_tracker = GeometricDriftTracker(
            model=model, optimizer=optimizer,
            beta3=args.beta3, batch_size=args.batch_size,
            eval_every=args.drift_eval_every,
        )

        # ── end of task k (after the epoch loop, before eval) ──
        drift_tracker.on_task_end(
            train_dataset=train_ds,
            task_id_for_head=task_id_for_head,
            regression=regression,
        )

        # ── inside the task k+1 epoch loop ──
        drift_metrics = drift_tracker.evaluate()
        if drift_metrics:
            epoch_log.update(drift_metrics)

    Parameters
    ----------
    model:
        The network being trained.
    optimizer:
        The AdEMAMix optimizer whose ``state[p]['exp_avg_slow']`` buffers hold
        the slow EMA.
    beta3:
        The *final* (fully-warmed-up) slow-EMA decay rate.  Used for both the
        half-life window calculation and the bias-correction divisor
        ``1 − beta3**step_T``.  During warmup this is an approximation.
    batch_size:
        Training batch size ``B``; used to convert half-life steps into samples.
    eval_every:
        Evaluate once every ``eval_every`` epochs during the new task.
        Default 1 (every epoch).
    track_ref:
        When ``True``, also snapshot ``g_ref_T = grad(buffer, theta_T)`` at the
        task boundary and log the reference-based ``*_ref`` metrics.  Costs one
        extra full-size gradient vector and one backward pass per task boundary.
        Default ``False``.
    device:
        Target device.  Inferred from model parameters when ``None``.
    """

    _SLOW_KEY = 'exp_avg_slow'
    _SQ_KEY   = 'exp_avg_sq'       # fast second moment (always present in AdEMAMix)
    _SQ_SLOW_KEY = 'exp_avg_sq_slow'  # slow second moment (only when --beta4 is set)

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta3: float,
        batch_size: int,
        eval_every: int = 1,
        track_ref: bool = False,
        device=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.beta3 = float(beta3)
        self.batch_size = int(batch_size)
        self.eval_every = max(1, int(eval_every))
        self.track_ref = bool(track_ref)

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.device = device

        self._m_slow_T = None
        self._v_sq_T   = None   # second-moment snapshot at task boundary
        self._g_ref_T  = None   # grad(buffer, theta_T) snapshot (opt-in)
        self._half_life_buffer = None
        self._task_id_for_head = 0
        self._regression = False
        self._epoch_in_new_task = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def half_life_window_size(self) -> int:
        """Number of samples in the slow-EMA half-life window.

        ``N_window = ceil( half_life_beta3 ) × batch_size``

        Reads ``half_life_beta3`` directly from the optimizer's own tracking
        snapshot (``optimizer._latest_tracking['half_life_beta3']``) so that
        the warmed-up beta3 value is used automatically — consistent with how
        AdEMAMix itself reports this quantity via ``get_tracking_metrics()``.
        Falls back to computing ``ceil(ln 0.5 / ln beta3)`` from the
        configured final beta3 when the tracking dict is not yet populated
        (e.g. before the first optimizer step).
        """
        tracking = getattr(self.optimizer, '_latest_tracking', {})
        hl = tracking.get('half_life_beta3')
        if hl is not None and hl > 0:
            h_steps = math.ceil(hl)
        else:
            h_steps = math.ceil(math.log(0.5) / math.log(self.beta3))
        return h_steps * self.batch_size

    def on_task_end(
        self,
        train_dataset,
        task_id_for_head: int = 0,
        regression: bool = False,
        step_T: int = None,
    ) -> None:
        """Snapshot the slow-momentum buffer and capture the half-life data.

        Call this **once**, after the last ``optimizer.step()`` of task *k*
        and before the training loop of task *k+1* begins.

        Parameters
        ----------
        train_dataset:
            The ``torch.utils.data.Dataset`` for the task that just ended.
            The tracker slices the final ``N_window`` samples in sequential
            order (no shuffling).
        task_id_for_head:
            Head index used when calling ``model(x, task_id=..., out='logits')``.
            Must match the head used during task-*k* training.
        regression:
            If ``True``, use ``mse_loss``; otherwise use ``cross_entropy``.
        step_T:
            Global optimiser step count at the task boundary, used for the
            bias-correction factor ``1 − beta3**step_T``.  When ``None`` the
            tracker reads ``optimizer._global_step`` (set by AdEMAMix), then
            falls back to per-param ``state['step']``.
        """
        self._epoch_in_new_task = 0
        self._task_id_for_head = task_id_for_head
        self._regression = regression

        # Resolve step_T
        if step_T is None:
            step_T = getattr(self.optimizer, '_global_step', None)
        if step_T is None:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    st = self.optimizer.state.get(p, {})
                    if 'step' in st:
                        step_T = int(st['step'])
                        break
                if step_T is not None:
                    break
        if step_T is None or step_T <= 0:
            raise RuntimeError(
                "Cannot infer step_T from optimizer state. "
                "Pass step_T explicitly or ensure at least one optimizer step was taken."
            )

        self._half_life_buffer = self._build_half_life_buffer(train_dataset)
        self._m_slow_T = self._extract_slow_momentum(int(step_T))
        self._v_sq_T   = self._extract_second_moment(int(step_T))

        # Reference gradient at the frozen task-boundary weights theta_T.
        # The model parameters are still theta_T at this point (on_task_end is
        # called right after the last optimizer.step() of task k), so this is
        # exactly grad(buffer, theta_T) -- the clean past-parameter baseline.
        if self.track_ref:
            self._g_ref_T = self._compute_g_actual()
        else:
            self._g_ref_T = None

    def evaluate(self, prefix: str = 'drift') -> dict:
        """Compute drift metrics for the current epoch if the eval window is due.

        Returns a dict on evaluation epochs, or ``{}`` otherwise.  Safe to call
        before ``on_task_end`` (returns ``{}`` on the first task).

        EMA-based keys (always present):
          ``drift/cosine_sim``        – direction alignment, first moment
          ``drift/log_scale_ratio``   – log(||g||/||m_slow||), first moment
          ``drift/cosine_sim_sq``     – direction alignment, second moment
          ``drift/log_scale_ratio_sq``– log(||g²||/||v_sq||), second moment

        Reference-based keys (only when ``track_ref=True``):
          ``drift/cosine_sim_ref``        – vs grad(buffer, theta_T)
          ``drift/log_scale_ratio_ref``   – vs grad(buffer, theta_T)
          ``drift/cosine_sim_ref_sq``     – squared, vs grad(buffer, theta_T)²
          ``drift/log_scale_ratio_ref_sq``– squared, vs grad(buffer, theta_T)²
        """
        if self._m_slow_T is None or self._half_life_buffer is None:
            return {}

        self._epoch_in_new_task += 1
        if self._epoch_in_new_task % self.eval_every != 0:
            return {}

        g_actual = self._compute_g_actual()
        metrics = self._compute_metrics(g_actual, self._m_slow_T, prefix=prefix)

        # Element-wise square, computed once and reused by the second-moment
        # comparisons below.
        g_sq = None

        def _suffixed(g, ref, suffix):
            """Compute metrics(g, ref) and append ``suffix`` to each metric name."""
            for k, val in self._compute_metrics(g, ref, prefix=prefix).items():
                _, metric_name = k.split('/', 1)
                metrics[f'{prefix}/{metric_name}{suffix}'] = val

        # --- EMA second-moment comparison: g_actual^2 vs v_sq_T ---
        if self._v_sq_T is not None:
            g_sq = g_actual ** 2
            _suffixed(g_sq, self._v_sq_T, '_sq')

        # --- Reference (past-parameter) comparison: vs grad(buffer, theta_T) ---
        if self._g_ref_T is not None:
            _suffixed(g_actual, self._g_ref_T, '_ref')
            if g_sq is None:
                g_sq = g_actual ** 2
            _suffixed(g_sq, self._g_ref_T ** 2, '_ref_sq')

        return metrics

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _build_half_life_buffer(self, dataset) -> list:
        """Collect the final ``N_window`` samples from ``dataset`` in order.

        Returns a list of ``(x, y)`` tuples kept on **CPU**.  Batches are
        moved to ``self.device`` only inside ``_compute_g_actual``, so a
        large ``beta3`` window (CLEAR 224×224, often the whole bucket) does
        not sit on the GPU alongside the model and optimizer state.
        """
        N_window = self.half_life_window_size()
        n_total = len(dataset)
        start_idx = max(0, n_total - N_window)
        indices = list(range(start_idx, n_total))

        tail_loader = DataLoader(
            Subset(dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,   # one-shot capture; workers would just add fork overhead
            pin_memory=False,
            drop_last=False,
        )

        buffer = []
        with torch.no_grad():
            for batch in tail_loader:
                # Task-IL train sets yield (x, y, task_id). The drift gradient
                # matches Class-IL training, which is unmasked CE on (x, y).
                x, y = batch[0], batch[1]
                buffer.append((x.detach().cpu(), y.detach().cpu()))
        return buffer

    def _extract_slow_momentum(self, step_T: int) -> torch.Tensor:
        """Flatten, bias-correct, and concatenate ``exp_avg_slow`` buffers.

        Returns a 1-D fp32 tensor detached from the computation graph.
        """
        parts = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p)
                if state is None:
                    continue
                slow = state.get(self._SLOW_KEY)
                if not isinstance(slow, torch.Tensor):
                    continue
                parts.append(slow.detach().float().flatten())

        if not parts:
            raise KeyError(
                f"Slow-momentum key '{self._SLOW_KEY}' not found in optimizer state. "
                "Ensure at least one optimizer step was taken before on_task_end()."
            )

        flat = torch.cat(parts).to(self.device)

        # EMA bias correction: exp_avg_slow is initialised at 0, so without
        # correction early values are systematically under-estimated.
        bias_correction = 1.0 - self.beta3 ** step_T
        if bias_correction <= 0.0:
            raise ValueError(
                f"Bias correction is non-positive (step_T={step_T}, beta3={self.beta3}). "
                "Ensure step_T > 0."
            )
        return (flat / bias_correction).detach()

    def _extract_second_moment(self, step_T: int) -> 'torch.Tensor | None':
        """Flatten, bias-correct, and concatenate the second-moment EMA buffers.

        Prefers ``exp_avg_sq_slow`` (when ``--beta4`` is set) over the fast
        ``exp_avg_sq`` so the snapshot time-horizon matches ``m_slow_T``.
        Falls back to ``exp_avg_sq`` (always present in AdEMAMix).

        Returns a 1-D fp32 tensor detached from the computation graph, or
        ``None`` if neither buffer is found (non-AdEMAMix optimizer).
        """
        # Determine which key and beta to use for bias correction.
        # Prefer slow second moment when available.
        group0 = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
        beta4 = group0.get('beta4', None)
        beta2 = group0.get('betas', (None, 0.999, None))[1] if 'betas' in group0 else 0.999

        use_slow_sq = beta4 is not None
        key  = self._SQ_SLOW_KEY if use_slow_sq else self._SQ_KEY
        beta = float(beta4) if use_slow_sq else float(beta2)

        parts = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p)
                if state is None:
                    continue
                buf = state.get(key)
                if not isinstance(buf, torch.Tensor):
                    # Fall back to fast second moment if slow not found
                    buf = state.get(self._SQ_KEY)
                if not isinstance(buf, torch.Tensor):
                    continue
                parts.append(buf.detach().float().flatten())

        if not parts:
            return None

        flat = torch.cat(parts).to(self.device)
        bias_correction = 1.0 - beta ** step_T
        if bias_correction <= 0.0:
            return flat.detach()
        return (flat / bias_correction).detach()

    def _compute_g_actual(self) -> torch.Tensor:
        """Evaluate the gradient on the half-life buffer with current weights.

        The model is set to ``eval()`` so BatchNorm/Dropout are frozen.
        Gradients are zeroed before and after so nothing leaks into the
        surrounding training loop.

        Each mini-batch is moved to ``self.device``, backward'd, and freed
        before the next batch.  Peak activation memory is therefore one
        training step, not the full ``beta3`` half-life window.  Per-batch
        losses are weighted by ``batch_size / N`` so the accumulated
        ``.grad`` matches a mean loss over the whole buffer.

        Returns a 1-D fp32 gradient vector (no computation graph attached).
        """
        was_training = self.model.training
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        total_samples = sum(x.shape[0] for x, _ in self._half_life_buffer)
        if total_samples == 0:
            parts = [
                torch.zeros(p.numel(), device=self.device)
                for group in self.optimizer.param_groups
                for p in group['params']
            ]
            if was_training:
                self.model.train()
            return (
                torch.cat(parts) if parts
                else torch.zeros(0, device=self.device)
            )

        for x, y in self._half_life_buffer:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            y_hat = self.model(x, task_id=self._task_id_for_head, out='logits')
            if self._regression:
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze(-1)
                loss = mse_loss(y_hat, y)
            else:
                loss = CE(y_hat, y)
            # Weight by batch fraction so the gradient matches a mean-loss
            # evaluated over the full buffer regardless of the last-batch size.
            batch_weight = x.shape[0] / total_samples
            (loss * batch_weight).backward()

        parts = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    parts.append(p.grad.detach().float().flatten())
                else:
                    parts.append(torch.zeros(p.numel(), device=self.device))

        g_actual = torch.cat(parts)

        # Zero immediately — must not leak into the caller's training loop.
        self.optimizer.zero_grad(set_to_none=True)

        if was_training:
            self.model.train()

        return g_actual

    @staticmethod
    def _compute_metrics(
        g: torch.Tensor,
        m: torch.Tensor,
        prefix: str = 'drift',
    ) -> dict:
        """Compute cosine similarity and log scale ratio.

        ``cosine_sim``     – direction alignment, scale-invariant.
        ``log_scale_ratio`` – log(||g|| / ||m||): positive means g has grown
                              larger than the snapshot, negative means it
                              shrank.  Zero when magnitudes match exactly.
                              Avoids the blow-up of ||g−m||/||m|| when the
                              model converges (small ||m||) and then drifts.

        Returns a plain-Python-float dict safe to pass directly to
        ``wandb.log`` or merge into an ``epoch_log`` dict.
        """
        g = g.float()
        m = m.float()

        norm_g = float(torch.linalg.norm(g))
        norm_m = float(torch.linalg.norm(m))

        if norm_g < 1e-12 or norm_m < 1e-12:
            cosine_sim = 0.0
        else:
            cosine_sim = float(
                torch.clamp(torch.dot(g, m) / (norm_g * norm_m), -1.0, 1.0)
            )

        if norm_g >= 1e-12 and norm_m >= 1e-12:
            log_scale_ratio = math.log(norm_g / norm_m)
        elif norm_m < 1e-12:
            log_scale_ratio = float('inf')
        else:
            log_scale_ratio = float('-inf')

        return {
            f'{prefix}/cosine_sim': cosine_sim,
            f'{prefix}/log_scale_ratio': log_scale_ratio,
        }


class QuadraticMemoryDiagnostics:
    """Per-step diagnostics for the quadratic benchmark.

    Quadratics-only (needs ``QuadraticTaskSpec.Q`` / ``.dist_to_opt``), AdEMAMix-only
    (reads ``exp_avg_sq_slow`` from optimizer state). Near-free: a couple of extra
    tensor ops per optimizer step, no extra forward/backward passes.

    Slow second moment vs. curvature: compares the bias-corrected slow second
    moment ``v_t^s`` against the *current* task's constant true curvature
    ``diag(H_i)`` (``H_i = spec.Q``), every step.

    Distance to the first optimum: tracks ``||theta_t - theta_A*||`` continuously
    across *all* tasks (not just task A), where task A is the first task seen.

    Usage
    -----
        tracker = QuadraticMemoryDiagnostics(model, optimizer)
        for crt_task_id in range(args.tasks):
            tracker.on_task_start(specs[crt_task_id])
            for x, y in train_loader:
                ...
                optimizer.step()
                wandb.log(tracker.step_metrics())
    """

    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.optimizer = optimizer
        self._current_spec = None
        self._spec_task0 = None
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.device = device

    def on_task_start(self, spec) -> None:
        """Call once at the start of every task, including task 0."""
        self._current_spec = spec
        if self._spec_task0 is None:
            self._spec_task0 = spec

    def step_metrics(self, prefix: str = 'quad_diag') -> dict:
        metrics = {}

        # --- distance to task-0 (task A) optimum, tracked through every task ---
        if self._spec_task0 is not None and hasattr(self.model, 'get_w'):
            w = self.model.get_w()
            metrics[f'{prefix}/dist_to_task0_opt'] = self._spec_task0.dist_to_opt(w)

        # --- v_t^s (bias-corrected) vs. diag(H) of the *current* task ---
        if self._current_spec is not None:
            group0 = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
            beta4 = group0.get('beta4', None)
            if beta4 is not None:
                parts = []
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.optimizer.state.get(p, {})
                        buf = state.get('exp_avg_sq_slow')
                        step = state.get('step')
                        if not isinstance(buf, torch.Tensor) or not step:
                            continue
                        bias_correction4 = 1.0 - float(beta4) ** step
                        if bias_correction4 <= 0.0:
                            continue
                        parts.append((buf.detach().float() / bias_correction4).flatten())

                if parts:
                    v_slow = torch.cat(parts).to(self.device)
                    H_diag = self._current_spec.Q.diagonal().float().to(self.device)
                    if v_slow.numel() == H_diag.numel():
                        cs = GeometricDriftTracker._compute_metrics(v_slow, H_diag, prefix=prefix)
                        metrics[f'{prefix}/cos_vslow_vs_H'] = cs[f'{prefix}/cosine_sim']
                        metrics[f'{prefix}/log_scale_ratio_vslow_vs_H'] = cs[f'{prefix}/log_scale_ratio']

        return metrics
