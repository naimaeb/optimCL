"""Per-step gradient and batch-index history for the slow-EMA probe and
``slow_ema_refresh`` arm.

The log stores one record per optimizer step: the clipped gradient $g_s$, the
batch indices $B_s$, the task $t_s$, and bookkeeping ($\\kappa$, raw norm,
loss). The probe reads it to recompute $\\mu^{\\rm fresh}_n=\\sum_s w_s
\\nabla\\ell(B_s;\\theta_n)$ and $\\mu^{\\rm stale}_n$ from the stored
gradients. The refresh arm reads it to form the sampled correction
$C=\\frac{1}{S'}\\sum_k(\\nabla\\ell(B_{s_k};\\theta_n)-G_{s_k})$.

Storage choice: a preallocated contiguous CPU tensor for ``G`` plus plain
numpy arrays for the small per-step scalars. fp32 by default; ``bf16``
halves the memory.
"""

from __future__ import annotations

import numpy as np
import torch

from .replay_memory import systematic_indices


_DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}


class StepLog:
    def __init__(self, capacity, batch_size, dtype='fp32', seed=0,
                 store_theta=False, store_grad=True):
        if capacity <= 0:
            raise ValueError(f'capacity must be positive, got {capacity}')
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        if dtype not in _DTYPES:
            raise ValueError(f'dtype must be fp32 or bf16, got {dtype!r}')
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.dtype = _DTYPES[dtype]
        self.rng = np.random.RandomState(int(seed))
        # Allocated lazily on first append once dim is known.
        self.dim = None
        self.G = None
        self.store_theta = bool(store_theta)
        # store_grad=False drops the per-step G column entirely (saves
        # capacity · dim · {4,2} bytes). Set when the gradient history is
        # too large to hold in CPU RAM (ResNet-scale models), at the cost of
        # the probe's per-batch metrics and the refresh correction's control
        # variate — the refresh arm falls back to
        #   C = mean_k iw_k · g_new_k  −  μ_stale
        # using the optimizer's maintained slow EMA (see
        # naive.py:_recompute_refresh_correction).
        self.store_grad = bool(store_grad)
        # θ_s snapshot: allocated lazily, same (cap, dim) shape as G.
        # Needed wherever θ at step s is used to recompute closed-form
        # quantities or evaluate the Hessian at the right point.
        self.theta = None
        # Small per-step metadata: eagerly allocated.
        self.step = np.zeros(self.capacity, dtype=np.int64)
        self.task = np.zeros(self.capacity, dtype=np.int16)
        self.idx = np.full((self.capacity, self.batch_size), -1, dtype=np.int32)
        self.lengths = np.zeros(self.capacity, dtype=np.int32)
        self.kappa = np.zeros(self.capacity, dtype=np.float32)
        self.raw_norm = np.zeros(self.capacity, dtype=np.float32)
        self.loss = np.zeros(self.capacity, dtype=np.float32)
        self.n_pushed = 0
        # Map global optimizer step -> row index. Appends must be strictly
        # increasing in step (optimizer always calls after a .step()).
        self._step_to_row = {}

    def _ensure(self, dim):
        if self.dim is not None:
            if dim != self.dim:
                raise RuntimeError(f'gradient dim changed: {self.dim} -> {dim}')
            return
        self.dim = int(dim)
        if self.store_grad:
            self.G = torch.empty((self.capacity, self.dim), dtype=self.dtype)
        if self.store_theta:
            # θ is always stored in fp32 so closed-form checks keep precision.
            self.theta = torch.empty((self.capacity, self.dim),
                                     dtype=torch.float32)

    def append(self, step, task, idx, kappa, raw_norm, loss, G_flat,
               theta_flat=None):
        """Push one record. ``idx`` is a 1-D LongTensor of dataset indices,
        length ``<= batch_size``. ``G_flat`` is the clipped flat gradient.

        When ``store_theta`` is on, ``theta_flat`` must also be provided
        (fp32 flat vector of the same length as ``G_flat``).
        """
        if self.n_pushed >= self.capacity:
            raise RuntimeError(
                f'StepLog capacity ({self.capacity}) exhausted at step {step}')
        row = self.n_pushed
        self._ensure(int(G_flat.numel()))
        self.step[row] = int(step)
        self.task[row] = int(task)
        idx_arr = idx.detach().cpu().view(-1).numpy().astype(np.int32)
        L = int(idx_arr.size)
        if L > self.batch_size:
            raise ValueError(f'batch of {L} > StepLog batch_size {self.batch_size}')
        self.idx[row, :L] = idx_arr
        self.idx[row, L:] = -1
        self.lengths[row] = L
        self.kappa[row] = float(kappa)
        self.raw_norm[row] = float(raw_norm)
        self.loss[row] = float(loss)
        if self.store_grad:
            self.G[row].copy_(G_flat.detach().to(dtype=self.dtype, device='cpu'))
        if self.store_theta:
            if theta_flat is None:
                raise ValueError('store_theta=True needs theta_flat passed to append()')
            if int(theta_flat.numel()) != self.dim:
                raise RuntimeError(
                    f'theta_flat has {int(theta_flat.numel())} entries but G has {self.dim}')
            self.theta[row].copy_(
                theta_flat.detach().to(dtype=torch.float32, device='cpu'))
        self._step_to_row[int(step)] = row
        self.n_pushed += 1

    def row_of(self, step):
        """Row holding step ``step``, or raise if absent."""
        row = self._step_to_row.get(int(step))
        if row is None:
            raise KeyError(f'step {step} not in StepLog')
        return row

    def get(self, step):
        row = self.row_of(step)
        out = dict(
            step=int(self.step[row]),
            task=int(self.task[row]),
            idx=self.idx[row, :int(self.lengths[row])].copy(),
            kappa=float(self.kappa[row]),
            raw_norm=float(self.raw_norm[row]),
            loss=float(self.loss[row]),
            G=(self.G[row].clone() if self.store_grad and self.G is not None
               else None),
        )
        if self.store_theta and self.theta is not None:
            out['theta'] = self.theta[row].clone()
        return out

    def weights(self, n, beta3):
        """Bias-corrected geometric weights ``w_s = (1-β3)β3^(n-s)/(1-β3^n)``
        for every stored step ``s`` with ``1 <= s <= n``.

        Returns (steps, w) with the same ordering as insertion order.
        """
        if self.n_pushed == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        steps = self.step[:self.n_pushed].astype(np.int64)
        mask = (steps >= 1) & (steps <= int(n))
        if not mask.any():
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        s = steps[mask].astype(np.float64)
        a = float(n) - s
        beta3 = float(beta3)
        corr = 1.0 - beta3 ** float(n)
        if corr <= 0.0:
            raise ValueError(f'β3^n is 1 (β3={beta3}, n={n}); weights undefined')
        w = (1.0 - beta3) * np.power(beta3, a) / corr
        return steps[mask], w

    def sample(self, n, beta3, S, sampling='systematic', rng=None):
        """Draw ``S`` step values from the distribution ``Pr(s)=w_s``.

        Returns the sampled step values (not row indices). Uses the shared
        ``systematic_indices`` helper from ``replay_memory``; ``iid`` falls
        back to ``rng.choice`` with the same probabilities.
        """
        if S <= 0:
            return np.empty(0, dtype=np.int64)
        steps, w = self.weights(n, beta3)
        if steps.size == 0 or w.sum() == 0.0:
            return np.empty(0, dtype=np.int64)
        rng = rng or self.rng
        w = w / w.sum()
        if sampling == 'systematic':
            picks = systematic_indices(w, int(S), rng)
        elif sampling == 'iid':
            picks = rng.choice(steps.size, size=int(S), replace=True, p=w)
        else:
            raise ValueError(f'unknown sampling {sampling!r}')
        return steps[picks]

    def importance_weights(self, n, beta3, tau):
        """Multiplicative age boost ``q_s ∝ w_s · (1 + age_s/τ)`` used by
        ``--refresh_importance age``. Returns (steps, q, w) with q normalised."""
        steps, w = self.weights(n, beta3)
        if steps.size == 0:
            return steps, np.empty(0, dtype=np.float64), w
        ages = float(n) - steps.astype(np.float64)
        boost = 1.0 + ages / max(float(tau), 1.0)
        q = w * boost
        q = q / q.sum()
        return steps, q, w
