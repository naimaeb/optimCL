import torch

from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler
from .replay_memory import ABLATION_ESTIMATORS, StreamMemory
from .step_log import StepLog

# Bias-corrected mix. ``replay`` keeps the plain (non-ablation) step and lets
# naive.py mix a reservoir minibatch into the loss. ``fast_only`` bias-corrects the fast
# momentum and leaves the memory estimate at zero.
_BIAS_CORRECTED_ABLATIONS = frozenset({
    'slow_ema',
    'slow_ema_refresh',
    'slow_ema_reset',
    'fifo_geometric_fresh',
    'reservoir_geometric_stale',
    'fifo_uniform_stale',
    'reservoir_uniform_stale',
    'reservoir_uniform_fresh',
    'fast_only',
})
_BUFFERED_ABLATIONS = frozenset(
    name for name, (support, _, _) in ABLATION_ESTIMATORS.items() if support != 'none'
)
# Arms that keep a slow EMA (as opposed to buffered recompute). slow_ema_refresh
# shares the same m_slow trajectory as slow_ema; the refresh correction C sits
# on top of the bias-corrected slow EMA. slow_ema_reset is the same arm with
# m_slow zeroed at every task boundary.
_SLOW_EMA_ARMS = frozenset({'slow_ema', 'slow_ema_refresh', 'slow_ema_reset'})
# Arms that zero m_slow and the slow-bias-correction step counter on each new
# task — the "reset" control.
_SLOW_EMA_RESET_ARMS = frozenset({'slow_ema_reset'})


def _flatten(tensors):
    return torch.cat([t.detach().reshape(-1).float() for t in tensors])


def _unflatten(flat, params):
    expected = sum(p.numel() for p in params)
    if flat.numel() != expected:
        raise RuntimeError(
            f'flat vector has {flat.numel()} entries but params need {expected}; '
            'the recompute path and the step path likely built different param lists '
            '(requires_grad vs. grad is not None)'
        )
    pieces = []
    offset = 0
    for param in params:
        n = param.numel()
        pieces.append(flat[offset:offset + n].view_as(param))
        offset += n
    return pieces


def _global_norm(tensors):
    if not tensors:
        return 0.0
    total = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    for tensor in tensors:
        total = total + tensor.detach().float().square().sum()
    return float(torch.sqrt(total).item())


class ReplaySGD(torch.optim.Optimizer):
    """SGD with a fast momentum (β1) plus a slow momentum (β3) mixed by α.

    Update is (m_fast(β1) + α m_slow(β3)) / (1 + α): α = 0 is plain
    momentum SGD; α > 0 mixes in the slow EMA without changing the step
    length. Matches the normalised AdEMAMix first-moment mix.
    Optional warmup (β3: β1→β3, α: 0→α) mirrors AdEMAMix and only touches the slow term.

    ``ablation`` selects a memory estimator. Those ablations bias-correct the
    fast momentum and keep ``mu = 0`` on task 0. ``slow_ema`` bias-corrects the
    slow EMA; the buffered ablations resample stored or recomputed gradients.
    The plain (non-bias-corrected) step runs when ``ablation`` is omitted.
    """

    def __init__(self, params, lr, weight_decay, beta1, beta3, alpha, nesterov=False,
                 beta3_warmup=None, alpha_warmup=None, ablation=None, tau=0.0,
                 buffer_size=500, batch_size=128, memory_samples=0,
                 memory_dtype='bf16', seed=0, memory_examples=0,
                 total_steps=0, steplog_dtype='fp32', steplog_store_theta=False,
                 steplog_store_grad=True,
                 probe_staleness=False, refresh_batches=0,
                 refresh_sampling='systematic', refresh_importance='none'):
        super().__init__(params, dict(
            lr=lr, weight_decay=weight_decay, beta1=beta1, beta3=beta3, alpha=alpha,
            nesterov=nesterov, beta3_warmup=beta3_warmup, alpha_warmup=alpha_warmup))
        self.optim_step = 0
        self._task_id = 0
        # The global step at which the current task started. slow_ema_reset
        # uses it to compute the per-task bias correction (1 − β3^{n − n0}).
        self._task_boundary_step = 0
        self._batch_indices = None
        self._external_mu = None
        self._sample_was_empty = False
        self._memory_metrics = None
        self.ablation = ablation or None
        self.ablation_enabled = self.ablation in _BIAS_CORRECTED_ABLATIONS
        self.memory = None
        self.beta3 = float(beta3)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        if self.ablation_enabled:
            if beta3_warmup or alpha_warmup:
                raise ValueError('these ablations do not use beta3/alpha warmup')
            if nesterov:
                raise ValueError('these ablations do not use Nesterov')
        if self.ablation in _BUFFERED_ABLATIONS:
            support, weighting, gradients = ABLATION_ESTIMATORS[self.ablation]
            draws = int(memory_samples) if int(memory_samples) > 0 else int(batch_size)
            self.memory = StreamMemory(
                support, weighting, gradients, beta3, tau, buffer_size,
                batch_size, draws, dtype=memory_dtype, seed=seed,
                capacity_examples=memory_examples)
        # StepLog: per-step history of (idx, G) + bookkeeping. Allocated when
        # the probe is requested on slow_ema, or always for slow_ema_refresh
        # (which needs it to form the correction C each step).
        self.probe_staleness = bool(probe_staleness)
        self.refresh_batches = int(refresh_batches)
        self.refresh_sampling = str(refresh_sampling)
        self.refresh_importance = str(refresh_importance)
        self._external_correction = None  # set by naive.py per step
        self.step_log = None
        want_log = (self.probe_staleness and self.ablation == 'slow_ema') or \
                   self.ablation == 'slow_ema_refresh'
        if want_log:
            if int(total_steps) <= 0:
                raise ValueError('StepLog needs total_steps > 0')
            self.step_log = StepLog(
                capacity=int(total_steps), batch_size=int(batch_size),
                dtype=str(steplog_dtype), seed=int(seed),
                store_theta=bool(steplog_store_theta),
                store_grad=bool(steplog_store_grad))
        # slow_ema_refresh fallback flag: True iff this arm will have to form
        # its correction without the stored G control variate. Set when the
        # StepLog is on and store_grad was switched off (ResNet-scale models).
        self.refresh_uses_cv = bool(
            self.step_log is None or self.step_log.store_grad)
        if self.ablation:
            print(
                f"=== ReplaySGD ablation:{self.ablation} "
                f"estimator={ABLATION_ESTIMATORS.get(self.ablation)} tau={float(tau):g} ==="
            )
        else:
            print("=== ReplaySGD Optimizer ===")

    @property
    def uses_slow_ema(self):
        # Both the plain slow_ema and slow_ema_refresh arms maintain m_slow.
        return self.ablation in _SLOW_EMA_ARMS

    @property
    def uses_slow_ema_refresh(self):
        return self.ablation == 'slow_ema_refresh'

    @property
    def recomputes_gradients(self):
        # Buffered fresh arms (fifo_geometric_fresh, reservoir_uniform_fresh)
        # route an aggregated recomputed gradient through _external_mu.
        # slow_ema_refresh follows a separate route (_external_correction) and
        # does not set this flag; its StreamMemory is None.
        return self.memory is not None and self.memory.gradients == 'fresh'

    def needs_refresh_batches(self):
        """True when naive.py should recompute S' past batches this step."""
        return (
            self.uses_slow_ema_refresh
            and self._task_id >= 1
            and self.refresh_batches > 0
            and self.step_log is not None
            and self.step_log.n_pushed > 0
        )

    def sample_refresh_steps(self):
        """Return [(step, task, idx_array, G_old_flat)] of length S' for the
        upcoming step. ``s`` is sampled from the geometric w_s over all stored
        past steps (optionally with age importance correction).
        """
        if not self.needs_refresh_batches():
            return []
        n = self.optim_step + 1  # the probe sees w_s for s <= n (n = next step)
        if self.refresh_importance == 'age':
            steps_pool, q, w = self.step_log.importance_weights(
                n, self.beta3, self.tau)
            if steps_pool.size == 0:
                return []
            # Importance sample from q (either systematic or iid), rescale by w/q.
            import numpy as np
            if self.refresh_sampling == 'systematic':
                from .replay_memory import systematic_indices
                picks = systematic_indices(q, int(self.refresh_batches), self.step_log.rng)
            else:
                picks = self.step_log.rng.choice(
                    steps_pool.size, size=int(self.refresh_batches),
                    replace=True, p=q)
            selected_steps = steps_pool[picks]
            iw = (w[picks] / q[picks]).astype('float32')
        else:
            selected_steps = self.step_log.sample(
                n, self.beta3, int(self.refresh_batches),
                sampling=self.refresh_sampling)
            iw = None
        out = []
        for i, s in enumerate(selected_steps):
            rec = self.step_log.get(int(s))
            weight = float(iw[i]) if iw is not None else 1.0
            out.append({
                'step': rec['step'],
                'task': rec['task'],
                'idx': rec['idx'],
                'G_old': rec['G'],
                'weight': weight,
            })
        return out

    def set_refresh_correction(self, C_flat):
        """Stash the mean-weighted correction C = Σ_k (g_new_k - G_{s_k})/S'
        (optionally reweighted by iw) for consumption in the next step. When
        the StepLog runs without stored G, naive.py passes the control-variate-
        free form C = Σ_k iw_k g_new_k / Σ_k iw_k − μ_stale instead; the sum
        with exp_avg_slow is the same either way."""
        if C_flat is None:
            self._external_correction = None
            return
        self._external_correction = C_flat.detach().cpu().float()

    @torch.no_grad()
    def mu_stale_flat(self):
        """Current bias-corrected slow EMA, concatenated across parameters.
        Returns a CPU fp32 flat tensor matching the (requires_grad, grad is
        not None) order used by the refresh plumbing in step(). The optimizer
        step() uses m_slow / corr3 as μ_stale once the first .step() has
        allocated state['m_slow']; before that this returns a zero vector."""
        if not self.uses_slow_ema:
            raise RuntimeError('mu_stale_flat only defined for slow-EMA arms')
        beta3 = float(self.param_groups[0]['beta3'])
        n_slow = max(1, int(self.optim_step) - (
            int(self._task_boundary_step)
            if self.ablation in _SLOW_EMA_RESET_ARMS else 0))
        corr3 = 1.0 - (beta3 ** n_slow) if 0.0 < beta3 < 1.0 else 1.0
        if corr3 == 0.0:
            corr3 = 1.0
        pieces = []
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                m_slow = state.get('m_slow')
                if m_slow is None:
                    pieces.append(torch.zeros(p.numel(), dtype=torch.float32))
                else:
                    pieces.append(
                        (m_slow.detach().float() / corr3).reshape(-1).cpu())
        return torch.cat(pieces) if pieces else torch.zeros(0)

    @property
    def logs_reservoir_ess(self):
        # Report ESS whenever the draw uses a non-trivial kernel: all reservoir
        # arms, and every fresh FIFO arm (thinned or whole-window geometric).
        return self.memory is not None and (
            self.memory.support == 'reservoir' or self.memory.gradients == 'fresh')

    def set_task_context(self, task_id, n_tasks=None):
        prev = int(self._task_id)
        new = int(task_id)
        self._task_id = new
        # slow_ema_reset: zero m_slow and restart the per-task step counter at
        # every task boundary so the bias correction uses k = steps-in-task,
        # not the global optim_step. The reset happens on the *transition*,
        # i.e. only when the task id changes (and only after task 0 starts).
        if self.ablation in _SLOW_EMA_RESET_ARMS and new != prev:
            self._task_boundary_step = int(self.optim_step)
            for group in self.param_groups:
                for p in group['params']:
                    st = self.state.get(p)
                    if st is not None and 'm_slow' in st:
                        st['m_slow'].zero_()

    def set_batch_indices(self, indices):
        if indices is None:
            self._batch_indices = None
            return
        self._batch_indices = torch.as_tensor(indices, dtype=torch.long).view(-1).cpu()

    def set_recomputed_gradient(self, grads):
        """Per-parameter recomputed gradient, already detached, same order as ``params``."""
        params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        if len(grads) != len(params):
            raise RuntimeError('recomputed gradient does not match the parameter list')
        self._external_mu = _flatten(grads)

    def needs_recomputed_gradient(self):
        return (
            self.recomputes_gradients
            and self._task_id >= 1
            and self.memory.filled > 0
        )

    def sample_stored_examples(self):
        """``(task, index)`` rows for the step that is about to run."""
        if not self.needs_recomputed_gradient():
            return None
        pairs = self.memory.sample_pairs(self.optim_step + 1)
        self._sample_was_empty = pairs is None
        return pairs

    def pop_memory_metrics(self):
        metrics = self._memory_metrics
        self._memory_metrics = None
        return metrics

    @torch.no_grad()
    def step(self):
        self.optim_step += 1
        if self.ablation_enabled:
            self._step_ablation()
            return
        self._step_legacy()

    def _step_legacy(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta_fast = group["beta1"]
            beta3_final = group["beta3"]
            alpha_final = group["alpha"]
            nesterov = group["nesterov"]
            beta3_warmup = group["beta3_warmup"]
            alpha_warmup = group["alpha_warmup"]

            # Falsy warmup (None or 0) => no warmup: use the configured values.
            if alpha_warmup:
                alpha = linear_warmup_scheduler(self.optim_step, alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
            else:
                alpha = alpha_final
            if beta3_warmup:
                beta_slow = linear_hl_warmup_scheduler(self.optim_step, beta_end=beta3_final, beta_start=beta_fast, warmup=beta3_warmup)
            else:
                beta_slow = beta3_final

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["m_slow"] = torch.zeros_like(p)
                    state["m_fast"] = torch.zeros_like(p)
                    if nesterov:
                        state["n_slow"] = torch.zeros_like(p)
                        state["n_fast"] = torch.zeros_like(p)

                g = p.grad
                m_slow = state["m_slow"]
                m_fast = state["m_fast"]

                m_slow.lerp_(g, 1 - beta_slow)
                m_fast.lerp_(g, 1 - beta_fast)

                if nesterov:
                    slow = state["n_slow"].copy_(g).lerp_(m_slow, beta_slow)
                    fast = state["n_fast"].copy_(g).lerp_(m_fast, beta_fast)
                else:
                    slow = m_slow
                    fast = m_fast

                p.mul_(1 - lr * wd).add_((fast + alpha * slow) / (1.0 + alpha), alpha=-lr)

    def _step_ablation(self):
        """Bias-corrected mix. ``mu`` is 0 until task 1, including the slow EMA."""
        n = self.optim_step
        task_id = self._task_id
        params = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        if not params:
            self._external_mu = None
            self._batch_indices = None
            return
        grads = [p.grad for p in params]
        group = self.param_groups[0]
        beta1 = float(group['beta1'])
        beta3 = float(group['beta3'])
        alpha = float(group['alpha'])
        lr = group['lr']
        wd = group['weight_decay']
        if self.ablation != 'fast_only':
            if not 0.0 < beta3 < 1.0:
                raise AssertionError(f'beta3 must lie in (0, 1), got {beta3}')
        corr1 = 1.0 - (beta1 ** n)
        # slow_ema_reset: the slow EMA restarts at every task boundary, so the
        # bias correction must use the per-task step count k = n − n_task_start.
        n_slow = n - int(self._task_boundary_step) \
            if self.ablation in _SLOW_EMA_RESET_ARMS else n
        corr3 = 1.0 - (beta3 ** n_slow) if 0.0 < beta3 < 1.0 else 1.0
        if corr1 == 0.0:
            raise AssertionError('fast-momentum bias correction is undefined')

        mu_list = self._memory_mu(params, grads, n, task_id)
        if task_id == 0 and _global_norm(mu_list) != 0.0:
            raise AssertionError('memory must be zero during task 0')

        # Snapshot θ_s BEFORE any param mutation this step. g_n was computed
        # at these weights by naive.py's loss.backward(); the probe and the
        # refresh correction want θ_s = pre-step weights so g_new_s = ∇ℓ(B_s;
        # θ_s) at age 0 agrees with the stored G_s = ∇ℓ(B_s; θ_s). Only
        # taken when the StepLog needs θ.
        theta_snapshot_flat = None
        if self.step_log is not None and self.step_log.store_theta:
            theta_snapshot_flat = torch.cat([
                p.detach().reshape(-1).float().cpu() for p in params
            ])

        # Refresh correction C (slow_ema_refresh only). Added on top of the
        # bias-corrected slow EMA. task 0: muted along with μ_stale.
        C_pieces = None
        if self.uses_slow_ema_refresh and task_id >= 1 \
                and self._external_correction is not None:
            C_pieces = _unflatten(
                self._external_correction.to(grads[0].device), params)

        m1_hats = []
        for i, (param, grad, mu) in enumerate(zip(params, grads, mu_list)):
            state = self.state[param]
            if 'm_fast' not in state:
                state['m_fast'] = torch.zeros_like(param)
                if self.uses_slow_ema:
                    state['m_slow'] = torch.zeros_like(param)
            m_fast = state['m_fast']
            m_fast.lerp_(grad, 1.0 - beta1)
            m1_hat = m_fast / corr1
            if self.uses_slow_ema:
                m_slow = state['m_slow']
                m_slow.lerp_(grad, 1.0 - beta3)
                mu = torch.zeros_like(grad) if task_id == 0 else m_slow / corr3
                if C_pieces is not None:
                    mu = mu + C_pieces[i]
            direction = (m1_hat + alpha * mu) / (1.0 + alpha)
            param.mul_(1 - lr * wd).add_(direction, alpha=-lr)
            m1_hats.append(m1_hat)
            if self.uses_slow_ema:
                mu_list[i] = mu

        self._push_memory(params, grads, n, task_id)
        self._push_step_log(grads, n, task_id, theta_snapshot_flat)
        self._record_metrics(n, alpha, m1_hats, mu_list)
        self._external_mu = None
        self._external_correction = None
        self._sample_was_empty = False
        self._batch_indices = None

    def _memory_mu(self, params, grads, n, task_id):
        zeros = [torch.zeros_like(g) for g in grads]
        if self.memory is None or task_id < 1:
            return zeros
        if self.recomputes_gradients:
            if self._external_mu is None:
                if self.memory.filled > 0 and not self._sample_was_empty:
                    raise RuntimeError('recomputed memory gradient was not set')
                return zeros
            return _unflatten(self._external_mu.to(grads[0].device), params)
        flat = self.memory.sample_grad(n)
        if flat is None:
            return zeros
        return _unflatten(flat.to(device=grads[0].device), params)

    def _push_memory(self, params, grads, n, task_id):
        if self.memory is None:
            return
        if self.recomputes_gradients:
            if self._batch_indices is None:
                raise RuntimeError('current batch indices were not set')
            self.memory.push_indices(self._batch_indices, task_id, n)
            return
        self.memory.push_grad(_flatten(grads).cpu(), n)

    def _push_step_log(self, grads, n, task_id, theta_snapshot_flat=None):
        """Record (idx, G, kappa, raw_norm) for the step that just ran.

        Called on every step including task 0 whenever a StepLog exists —
        age-0 recomputes are then drawable at step n+1. Raw norm and clip
        factor are reconstructed from the already-clipped gradient so
        `_step_ablation` does not need to see the pre-clip vector; this is
        exact when naive.py clipped with the same `c` (asserted downstream)
        and conservative otherwise (κ reported as 1.0, raw_norm as the
        post-clip norm). The probe's mapping check will catch any divergence.
        """
        if self.step_log is None:
            return
        if self._batch_indices is None:
            raise RuntimeError(
                'StepLog requires per-step batch indices; call set_batch_indices '
                'before step()')
        g_flat = _flatten(grads).cpu()
        raw_norm = float(torch.linalg.vector_norm(g_flat).item())
        # κ was applied *before* reaching the optimizer by naive.py's
        # clip_grad_norm_. We don't have the pre-clip vector; the raw_norm /
        # kappa split is reconstructed from the pre-clip value stashed on the
        # optimizer by naive.py (set_clip_info), falling back to (post, 1.0).
        kappa = float(getattr(self, '_last_kappa', 1.0) or 1.0)
        raw = float(getattr(self, '_last_raw_norm', raw_norm) or raw_norm)
        loss = float(getattr(self, '_last_loss', 0.0) or 0.0)
        # θ_s snapshot was taken at the top of _step_ablation (pre-update);
        # pass it through unchanged. If the caller forgot to pass one, fall
        # back to a post-update snapshot and let the probe's age-0 check
        # catch any discrepancy.
        if self.step_log.store_theta:
            if theta_snapshot_flat is None:
                theta_snapshot_flat = torch.cat([
                    p.detach().reshape(-1).float().cpu()
                    for group in self.param_groups
                    for p in group['params']
                    if p.requires_grad
                ])
        self.step_log.append(
            step=int(n), task=int(task_id),
            idx=self._batch_indices, kappa=kappa, raw_norm=raw, loss=loss,
            G_flat=g_flat, theta_flat=theta_snapshot_flat)

    def set_clip_info(self, raw_norm, kappa, loss=0.0):
        """naive.py calls this after clip_grad_norm_ to record (||g_raw||, κ)
        for the StepLog write at the next step()."""
        self._last_raw_norm = float(raw_norm)
        self._last_kappa = float(kappa)
        self._last_loss = float(loss)

    def _record_metrics(self, n, alpha, m1_hats, mus):
        norm_mu = _global_norm(mus)
        norm_fast = _global_norm(m1_hats)
        denom = norm_fast + alpha * norm_mu
        share = 0.0 if denom == 0.0 else (alpha * norm_mu) / denom
        metrics = {
            'memory/step': int(n),
            'memory/share': float(share),
            'memory/norm_mu': float(norm_mu),
            'memory/M': 0,
            'memory/M_clipped': 0,
        }
        if self.memory is not None:
            self.memory.window_stats()
            metrics['memory/M'] = int(self.memory.last_M)
            metrics['memory/M_clipped'] = int(self.memory.last_M_clipped)
            metrics['memory/stored'] = self.memory.stored_examples()
            if self.memory.last_ess is not None and self.logs_reservoir_ess:
                metrics['memory/ess'] = float(self.memory.last_ess)
        self._memory_metrics = metrics
