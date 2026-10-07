"""
Adapted from: https://pytorch.org/docs/1.6.0/_modules/torch/optim/adam.html
"""
import math
import torch
from torch.optim import Optimizer

from .step_log import StepLog


_SLOW_EMA_REFRESH = 'slow_ema_refresh'
_SUPPORTED_ABLATIONS = frozenset({_SLOW_EMA_REFRESH})


def _flatten(tensors):
    return torch.cat([t.detach().reshape(-1).float() for t in tensors])


def _unflatten(flat, params):
    expected = sum(p.numel() for p in params)
    if flat.numel() != expected:
        raise RuntimeError(
            f'flat vector has {flat.numel()} entries but params need {expected}; '
            'the refresh path and the step path built different param lists'
        )
    pieces = []
    offset = 0
    for param in params:
        n = param.numel()
        pieces.append(flat[offset:offset + n].view_as(param))
        offset += n
    return pieces


def _safe_half_life(beta, eps=1e-12):
    """Standard EMA half-life: ``log(0.5)/log(beta)``.

    For ``m_t = beta * m_{t-1} + (1 - beta) * g_t`` the contribution of the
    gradient from ``k`` steps ago is ``(1 - beta) * beta^k``, so the lag at
    which a past gradient still contributes half as much as the most recent
    one is ``k = log(0.5)/log(beta)``. Reading: "how many past gradients are
    retained at >= half weight".

    Note: this is the standard EMA half-life, off by 1 from the AdEMAMix
    paper's convention (``log(0.5)/log(beta) - 1``). The warmup math inside
    ``linear_hl_warmup_scheduler`` defines its own ``f`` with the ``-1`` so
    that ``f`` and ``f_inv`` are exact inverses; do NOT change ``f`` there.
    This reporter is purely for W&B logging.
    """
    beta = min(max(beta, eps), 1.0 - eps)
    return math.log(0.5) / math.log(beta)


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0-a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5)/math.log(beta+eps)-1

    def f_inv(t):
        return math.pow(0.5, 1/(t+1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm, optionally with a slow second moment.

    Standard AdEMAMix mixes a fast and a slow first-moment EMA in the numerator
    but uses a single second-moment denominator (beta2).  When ``beta4`` is set,
    a slow second-moment EMA is maintained and the denominator mirrors the
    numerator: both fast and slow second moments are linearly combined using the
    same ``alpha``, so the update is fully symmetric:

        numerator = (m_fast / (1 - beta1^t)  +  alpha * m_slow) / (1 + alpha)
        denom     = sqrt((v_fast / (1 - beta2^t)  +  alpha * v_slow / (1 - beta4^t)) / (1 + alpha)) + eps
        update    = numerator / denom

    The ``1 + alpha`` scale keeps the step length comparable to alpha = 0
    (and to 50/50 replay). The paper's additive mix is recovered by omitting
    that factor; here alpha is a mix weight, not a gain on the slow term.

    Both second moments are bias-corrected so each term is an unbiased estimate
    of E[g^2] before being mixed.

    When ``beta4`` is ``None`` (default) the original single-denominator
    behaviour is preserved exactly.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        beta4 (float, optional): slow second-moment decay rate.  When set, a separate
            ``exp_avg_sq_slow`` buffer is maintained and combined with the fast second
            moment (``exp_avg_sq``) in the denominator using the same ``alpha`` as the
            first-moment mixing.  Must satisfy 0 <= beta4 < 1.
            ``None`` disables the slow second moment and restores original AdEMAMix
            behaviour (default: None).
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
        anchor_mu (float, optional): boundary-free primal-anchor weight. When set, replaces
            the plain gradient step with the proximal update

                theta_t = ( P_t/lr * theta_{t-1} + anchor_mu * Q_t * theta_bar_t - d_t )
                          / ( P_t/lr + anchor_mu * Q_t )

            where ``P_t`` is the usual AdEMAMix metric (``denom``), ``Q_t = diag(v_t^s)``
            (the bias-corrected slow second moment — the "historical Fisher diagonal"),
            ``d_t`` is the dual term (fast+slow first-moment mix), and ``theta_bar_t`` is a
            slow EMA of the iterate itself (see ``anchor_tau``). This is a boundary-free,
            *diagonal-only* anchor: no gating/freeze-rearm and no Kronecker-structured
            metric. Requires ``beta4`` to be set (the anchor metric is ``diag(v_t^s)``).
            ``None`` (default) disables the anchor and restores the plain gradient step.
        anchor_tau (float, optional): EMA retention rate for the primal anchor point
            ``theta_bar_t = anchor_tau * theta_bar_{t-1} + (1 - anchor_tau) * theta_{t-1}``.
            Only used when ``anchor_mu`` is set. Initialized to the parameter's value at
            the first optimizer step (anchor to the initial weights ``z^0``).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0,
                 beta3_warmup=None, alpha_warmup=None, beta4=None, eps=1e-8,
                 weight_decay=0, anchor_mu=None, anchor_tau=None,
                 ablation=None, tau=0.0, batch_size=0, total_steps=0,
                 steplog_dtype='fp32', steplog_store_theta=False,
                 steplog_store_grad=True,
                 refresh_batches=0, refresh_sampling='systematic',
                 refresh_importance='none', seed=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if beta4 is not None and not 0.0 <= beta4 < 1.0:
            raise ValueError("Invalid beta4 value: {}".format(beta4))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if anchor_mu is not None:
            if beta4 is None:
                raise ValueError(
                    "anchor_mu requires beta4 to be set (the anchor metric is diag(v_t^s))."
                )
            if not 0.0 <= anchor_mu:
                raise ValueError("Invalid anchor_mu value: {}".format(anchor_mu))
            if anchor_tau is None or not 0.0 <= anchor_tau < 1.0:
                raise ValueError("Invalid anchor_tau value: {}".format(anchor_tau))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta3_warmup=beta3_warmup,
                        alpha_warmup=alpha_warmup, beta4=beta4, weight_decay=weight_decay,
                        anchor_mu=anchor_mu, anchor_tau=anchor_tau)
        super(AdEMAMix, self).__init__(params, defaults)
        self._global_step = 0
        self._current_task_id = None
        self._num_tasks = None
        self._fast_task_mass = {}
        self._slow_task_mass = {}
        self._latest_tracking = {}

        # slow_ema_refresh plumbing. The AdEMAMix native slow EMA (exp_avg_slow,
        # driven by β3) is the stale buffer; the refresh correction C is
        # recomputed per step by naive.py from S' past batches and added to
        # exp_avg_slow in the mix (without mutating the buffer itself, so the
        # correction does not compound). task 0 keeps native AdEMAMix behavior
        # (slow EMA active, no correction); optimizer state is carried across
        # task boundaries.
        self.ablation = ablation or None
        if self.ablation is not None and self.ablation not in _SUPPORTED_ABLATIONS:
            raise ValueError(
                f"AdEMAMix only supports ablations {sorted(_SUPPORTED_ABLATIONS)}; "
                f"got {self.ablation!r}"
            )
        self.ablation_enabled = self.ablation is not None
        self.beta3 = float(betas[2])
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.refresh_batches = int(refresh_batches)
        self.refresh_sampling = str(refresh_sampling)
        self.refresh_importance = str(refresh_importance)
        self._task_id = 0
        self._batch_indices = None
        self._external_correction = None
        self._memory_metrics = None
        self.step_log = None
        if self.ablation == _SLOW_EMA_REFRESH:
            if int(total_steps) <= 0:
                raise ValueError('AdEMAMix slow_ema_refresh needs total_steps > 0')
            if int(batch_size) <= 0:
                raise ValueError('AdEMAMix slow_ema_refresh needs batch_size > 0')
            self.step_log = StepLog(
                capacity=int(total_steps), batch_size=int(batch_size),
                dtype=str(steplog_dtype), seed=int(seed),
                store_theta=bool(steplog_store_theta),
                store_grad=bool(steplog_store_grad))
            print(
                f"=== AdEMAMix ablation:{self.ablation} "
                f"refresh_batches={self.refresh_batches} "
                f"sampling={self.refresh_sampling} importance={self.refresh_importance} "
                f"steplog_store_grad={self.step_log.store_grad} ==="
            )
        self.refresh_uses_cv = bool(
            self.step_log is None or self.step_log.store_grad)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    def set_task_context(self, task_id, num_tasks=None):
        self._current_task_id = int(task_id) if task_id is not None else None
        self._num_tasks = int(num_tasks) if num_tasks is not None else self._num_tasks
        if task_id is not None:
            self._task_id = int(task_id)

    # --- slow_ema_refresh API (duck-typed by naive.py's refresh driver) ---

    @property
    def uses_slow_ema_refresh(self):
        return self.ablation == _SLOW_EMA_REFRESH

    def set_batch_indices(self, indices):
        if indices is None:
            self._batch_indices = None
            return
        self._batch_indices = torch.as_tensor(indices, dtype=torch.long).view(-1).cpu()

    def set_clip_info(self, raw_norm, kappa, loss=0.0):
        self._last_raw_norm = float(raw_norm)
        self._last_kappa = float(kappa)
        self._last_loss = float(loss)

    def needs_refresh_batches(self):
        return (
            self.uses_slow_ema_refresh
            and self._task_id >= 1
            and self.refresh_batches > 0
            and self.step_log is not None
            and self.step_log.n_pushed > 0
        )

    def sample_refresh_steps(self):
        if not self.needs_refresh_batches():
            return []
        n = self._global_step + 1  # next step (the one about to run)
        if self.refresh_importance == 'age':
            steps_pool, q, w = self.step_log.importance_weights(
                n, self.beta3, self.tau)
            if steps_pool.size == 0:
                return []
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
                'step': rec['step'], 'task': rec['task'], 'idx': rec['idx'],
                'G_old': rec['G'], 'weight': weight,
            })
        return out

    def set_refresh_correction(self, C_flat):
        if C_flat is None:
            self._external_correction = None
            return
        self._external_correction = C_flat.detach().cpu().float()

    @torch.no_grad()
    def mu_stale_flat(self):
        """Bias-corrected slow EMA flattened across trainable params, in the
        same (requires_grad, grad is not None) order the refresh plumbing uses.
        exp_avg_slow is stored without bias correction, so this divides by
        (1 − β3^n). Returns zeros before the first step() has allocated state.
        """
        if self.ablation != _SLOW_EMA_REFRESH:
            raise RuntimeError('mu_stale_flat only defined for slow_ema_refresh')
        n = max(1, int(self._global_step))
        pieces = []
        for group in self.param_groups:
            beta3 = float(group['betas'][2])
            corr3 = 1.0 - (beta3 ** n) if 0.0 < beta3 < 1.0 else 1.0
            if corr3 == 0.0:
                corr3 = 1.0
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state.get(p, {})
                m_slow = state.get('exp_avg_slow')
                if m_slow is None:
                    pieces.append(torch.zeros(p.numel(), dtype=torch.float32))
                else:
                    pieces.append(
                        (m_slow.detach().float() / corr3).reshape(-1).cpu())
        return torch.cat(pieces) if pieces else torch.zeros(0)

    def pop_memory_metrics(self):
        metrics = self._memory_metrics
        self._memory_metrics = None
        return metrics

    def _push_step_log(self, grads, params, n, task_id, theta_snapshot_flat=None):
        if self.step_log is None:
            return
        if self._batch_indices is None:
            raise RuntimeError(
                'StepLog requires per-step batch indices; call set_batch_indices '
                'before step()')
        g_flat = _flatten(grads).cpu()
        raw_norm = float(torch.linalg.vector_norm(g_flat).item())
        kappa = float(getattr(self, '_last_kappa', 1.0) or 1.0)
        raw = float(getattr(self, '_last_raw_norm', raw_norm) or raw_norm)
        loss = float(getattr(self, '_last_loss', 0.0) or 0.0)
        if self.step_log.store_theta and theta_snapshot_flat is None:
            theta_snapshot_flat = torch.cat([
                p.detach().reshape(-1).float().cpu() for p in params
            ])
        self.step_log.append(
            step=int(n), task=int(task_id),
            idx=self._batch_indices, kappa=kappa, raw_norm=raw, loss=loss,
            G_flat=g_flat, theta_flat=theta_snapshot_flat)

    @staticmethod
    def _update_branch_task_mass(task_mass, beta, task_id):
        if task_id is None:
            return
        for k in list(task_mass.keys()):
            task_mass[k] *= beta
            if abs(task_mass[k]) < 1e-20:
                del task_mass[k]
        task_mass[task_id] = task_mass.get(task_id, 0.0) + (1.0 - beta)

    def _compute_tracking_snapshot(self, beta1, beta3, alpha, beta4=None):
        step = self._global_step
        fast_denom = 1.0 - beta1 ** step if beta1 != 1.0 else 1.0

        fast_corrected = {}
        if fast_denom > 0:
            fast_corrected = {k: v / fast_denom for k, v in self._fast_task_mass.items()}
        slow_raw = dict(self._slow_task_mass)

        task_ids = set(fast_corrected.keys()) | set(slow_raw.keys())
        if self._num_tasks is not None:
            task_ids |= set(range(self._num_tasks))

        combined_raw = {}
        for tid in task_ids:
            combined_raw[tid] = fast_corrected.get(tid, 0.0) + alpha * slow_raw.get(tid, 0.0)

        combined_total = sum(combined_raw.values())
        if combined_total > 0:
            combined_frac = {k: v / combined_total for k, v in combined_raw.items()}
        else:
            combined_frac = {k: 0.0 for k in combined_raw.keys()}

        current_task = self._current_task_id
        fast_prev = sum(v for k, v in fast_corrected.items() if k != current_task)
        slow_prev = sum(v for k, v in slow_raw.items() if k != current_task)
        combined_prev = sum(v for k, v in combined_frac.items() if k != current_task)

        # Per-task retained-gradient counts. Reading: "this many past gradients
        # from task tid are still alive in the EMA at >= half weight". The
        # half-life is the natural horizon of each EMA branch (fast uses beta1,
        # slow uses the warmed-up beta3); multiplying by the task-mass fraction
        # attributes that horizon across tasks.
        hl_fast = float(_safe_half_life(beta1))
        hl_slow = float(_safe_half_life(beta3))
        fast_task_retained = {k: v * hl_fast for k, v in fast_corrected.items()}
        slow_task_retained = {k: v * hl_slow for k, v in slow_raw.items()}
        fast_prev_retained = sum(v for k, v in fast_task_retained.items() if k != current_task)
        slow_prev_retained = sum(v for k, v in slow_task_retained.items() if k != current_task)

        self._latest_tracking = {
            "step": step,
            "current_task_id": current_task,
            "beta1": float(beta1),
            "beta3": float(beta3),
            "alpha": float(alpha),
            "beta4": float(beta4) if beta4 is not None else None,
            "half_life_beta1": hl_fast,
            "half_life_beta3": hl_slow,
            "half_life_beta4": float(_safe_half_life(beta4)) if beta4 is not None else None,
            "fast_prev_tasks_frac": float(fast_prev),
            "slow_prev_tasks_frac": float(slow_prev),
            "combined_prev_tasks_frac": float(combined_prev),
            "fast_task_frac": fast_corrected,
            "slow_task_frac": slow_raw,
            "combined_task_frac": combined_frac,
            "fast_task_retained": fast_task_retained,
            "slow_task_retained": slow_task_retained,
            "fast_prev_tasks_retained": float(fast_prev_retained),
            "slow_prev_tasks_retained": float(slow_prev_retained),
        }

    def get_tracking_metrics(self, prefix="ademamix"):
        if not self._latest_tracking:
            return {}

        t = self._latest_tracking
        out = {
            f"{prefix}/step": t["step"],
            f"{prefix}/current_task_id": -1 if t["current_task_id"] is None else t["current_task_id"],
            f"{prefix}/beta1": t["beta1"],
            f"{prefix}/beta3": t["beta3"],
            f"{prefix}/alpha": t["alpha"],
            f"{prefix}/half_life_beta1": t["half_life_beta1"],
            f"{prefix}/half_life_beta3": t["half_life_beta3"],
            f"{prefix}/fast_prev_tasks_frac": t["fast_prev_tasks_frac"],
            f"{prefix}/slow_prev_tasks_frac": t["slow_prev_tasks_frac"],
            f"{prefix}/combined_prev_tasks_frac": t["combined_prev_tasks_frac"],
            f"{prefix}/fast_prev_tasks_retained": t["fast_prev_tasks_retained"],
            f"{prefix}/slow_prev_tasks_retained": t["slow_prev_tasks_retained"],
        }
        if t.get("beta4") is not None:
            out[f"{prefix}/beta4"] = t["beta4"]
            out[f"{prefix}/half_life_beta4"] = t["half_life_beta4"]

        n_tasks = self._num_tasks if self._num_tasks is not None else 0
        for tid in range(n_tasks):
            out[f"{prefix}/fast_task{tid+1}_frac"]     = t["fast_task_frac"].get(tid, 0.0)
            out[f"{prefix}/slow_task{tid+1}_frac"]     = t["slow_task_frac"].get(tid, 0.0)
            out[f"{prefix}/combined_task{tid+1}_frac"] = t["combined_task_frac"].get(tid, 0.0)
            out[f"{prefix}/fast_task{tid+1}_retained"] = t["fast_task_retained"].get(tid, 0.0)
            out[f"{prefix}/slow_task{tid+1}_retained"] = t["slow_task_retained"].get(tid, 0.0)
        return out

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        track_beta1 = None
        track_beta3 = None
        track_alpha = None
        self._global_step += 1

        # slow_ema_refresh: unflatten C into per-param pieces keyed by id(p) so
        # the inner loop can look them up without re-ordering risk. C is laid
        # out in [p for p in model.parameters() if p.requires_grad] order by
        # naive.py's refresh driver; mirror that here.
        refresh_C_pieces = None
        if (self.uses_slow_ema_refresh
                and self._external_correction is not None
                and self._task_id >= 1):
            rg_params = [p for group in self.param_groups
                         for p in group['params'] if p.requires_grad]
            if rg_params:
                device = rg_params[0].device
                pieces = _unflatten(
                    self._external_correction.to(device), rg_params)
                refresh_C_pieces = {id(p): piece for p, piece in zip(rg_params, pieces)}

        # θ snapshot BEFORE any param mutation (StepLog theta column).
        theta_snapshot_flat = None
        if self.step_log is not None and self.step_log.store_theta:
            theta_snapshot_flat = torch.cat([
                p.detach().reshape(-1).float().cpu()
                for group in self.param_groups
                for p in group['params'] if p.requires_grad
            ])

        for group in self.param_groups:
            
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            # Tracking uses one global step per optimizer.step() call.
            # A falsy warmup (None or 0) means "no warmup": use the configured
            # final value from the very first step, so the logged alpha/beta3
            # match exactly what was set.
            if alpha_warmup:
                track_alpha = linear_warmup_scheduler(self._global_step, alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
            else:
                track_alpha = alpha_final

            if beta3_warmup:
                track_beta3 = linear_hl_warmup_scheduler(self._global_step, beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
            else:
                track_beta3 = beta3_final
            track_beta1 = beta1

            self._update_branch_task_mass(self._fast_task_mass, beta1, self._current_task_id)
            self._update_branch_task_mass(self._slow_task_mass, track_beta3, self._current_task_id)
        
            beta4 = group.get("beta4", None)
            anchor_mu = group.get("anchor_mu", None)
            anchor_tau = group.get("anchor_tau", None)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdEMAMix does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    if beta1 != 0.0: # save memory in case beta1 is 0.0
                        state['exp_avg_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else: 
                        state['exp_avg_fast'] = None
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values (fast)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Optional slow second moment for per-branch preconditioning
                    if beta4 is not None:
                        state['exp_avg_sq_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Optional primal-anchor point, anchored to the initial
                    # weights z^0 at construction time.
                    if anchor_mu is not None:
                        state['theta_bar'] = p.detach().clone()

                exp_avg_fast, exp_avg_slow, exp_avg_sq = state['exp_avg_fast'], state['exp_avg_slow'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute the effective alpha and beta3 in case warmup is used.
                # Falsy warmup (None or 0) => no warmup, use the configured value.
                if alpha_warmup:
                    alpha = linear_warmup_scheduler(state["step"], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
                else:
                    alpha = alpha_final
                
                if beta3_warmup:
                    beta3 = linear_hl_warmup_scheduler(state["step"], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
                else:
                    beta3 = beta3_final

                # Decay the first and second moment running average coefficients
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction4 = None
                if beta4 is not None:
                    # Mirror the first-moment mixing: combine bias-corrected fast and
                    # slow second moments with the same alpha before taking the sqrt.
                    exp_avg_sq_slow = state['exp_avg_sq_slow']
                    exp_avg_sq_slow.mul_(beta4).addcmul_(grad, grad, value=1 - beta4)
                    bias_correction4 = 1 - beta4 ** state['step']
                    v_combined = (
                        exp_avg_sq / bias_correction2
                        + alpha * exp_avg_sq_slow / bias_correction4
                    ) / (1.0 + alpha)
                    denom = v_combined.sqrt().add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # slow_ema_refresh: fold the refresh correction C into the
                # effective slow EMA for THIS step only; do not mutate
                # exp_avg_slow (otherwise C compounds). task 0 keeps native
                # behavior because refresh_C_pieces is None then.
                if refresh_C_pieces is not None:
                    C_piece = refresh_C_pieces.get(id(p))
                    if C_piece is not None:
                        effective_slow = exp_avg_slow + C_piece
                    else:
                        effective_slow = exp_avg_slow
                else:
                    effective_slow = exp_avg_slow

                d_t = (
                    exp_avg_fast.div(bias_correction1) + alpha * effective_slow
                ) / (1.0 + alpha)

                if anchor_mu is not None:
                    # Boundary-free primal anchor, closed form,
                    # diagonal-only: theta_t = (P_t/lr * theta_{t-1} + mu*Q_t*theta_bar_t - d_t)
                    #                          / (P_t/lr + mu*Q_t), with Q_t = diag(v_t^s).
                    theta_prev = p.detach().clone()
                    Q = exp_avg_sq_slow.div(bias_correction4)  # diag(v_t^s), bias-corrected
                    P_over_lr = denom / lr
                    denom_total = P_over_lr + anchor_mu * Q
                    theta_bar = state['theta_bar']
                    numer_total = P_over_lr * theta_prev + anchor_mu * Q * theta_bar - d_t
                    p.copy_(numer_total / denom_total)
                    if lmbda != 0:
                        # Same decoupled (AdamW-style) weight decay as the non-anchor path.
                        p.mul_(1 - lr * lmbda)
                    theta_bar.mul_(anchor_tau).add_(theta_prev, alpha=1 - anchor_tau)
                else:
                    update = d_t / denom
                    # decay
                    update.add_(p, alpha=lmbda)
                    p.add_(-lr * update)

        if track_beta1 is not None and track_beta3 is not None and track_alpha is not None:
            self._compute_tracking_snapshot(track_beta1, track_beta3, track_alpha,
                                            beta4=group.get("beta4", None))

        # Push to StepLog + clear per-step refresh state. Must use the same
        # (requires_grad, grad is not None) ordering that _external_correction
        # was unflattened against, so sample_refresh_steps() on the next step
        # has a consistent G history.
        if self.step_log is not None:
            params_rg = [p for group in self.param_groups
                         for p in group['params']
                         if p.requires_grad and p.grad is not None]
            grads_rg = [p.grad for p in params_rg]
            self._push_step_log(grads_rg, params_rg, self._global_step,
                                self._task_id,
                                theta_snapshot_flat=theta_snapshot_flat)
        self._external_correction = None
        self._batch_indices = None

        return loss


DoubleAdEMAMix = AdEMAMix


if __name__ == "__main__": # small dummy test

    x = torch.randn((10,7))
    model = torch.nn.Linear(7, 1, bias=False)
    opt = AdEMAMix(params=model.parameters(), lr=1e-2, betas=(0.9, 0.999, 0.9999), alpha=2.0, beta3_warmup=45, alpha_warmup=45, weight_decay=0.1)
    print(model.weight)
    for itr in range(50):
        y = model(x).mean()
        opt.zero_grad()
        y.backward()
        opt.step()

    print(model.weight)