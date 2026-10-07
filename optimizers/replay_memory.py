"""Stream-memory estimators for the ReplaySGD ablations.

One buffer, selected by how support, weighting, and gradients are handled.
The fast momentum and the α mix stay in ReplaySGD. This module only builds ``mu``.

  slow_ema                    none / geometric / stale     slow EMA, owned by ReplaySGD
  fifo_geometric_fresh        fifo / geometric / fresh     index window, gradient recomputed
  reservoir_geometric_stale   reservoir / geometric / stale Vitter reservoir of minibatch grads
  fifo_uniform_stale          fifo / uniform / stale       boxcar window of minibatch grads
  reservoir_uniform_stale     reservoir / uniform / stale
  reservoir_uniform_fresh     reservoir / uniform / fresh  Vitter reservoir of K examples

``mu`` is zero on task 0. FIFO / reservoir entries are still recorded during
task 0 so the window is full when task 1 starts. There is no RMS rescaling.

The FIFO window is ``M = ceil(2 tau)`` and geometric weights use ``beta3``, so
those estimators depend on the horizon r. ``reservoir_uniform_stale`` uses
neither: its reservoir is a uniform sample of the whole stream, so
``E[mu_t]`` is the cumulative mean of all past gradients (the r = inf control).

Example budget K (``capacity_examples``, ER's currency):

* ``fifo_geometric_fresh`` with K > 0 thins the window. Every pushed example
  is kept with probability ``rho = min(1, K / (M B))`` by systematic thinning
  of the stream (one random offset u), so a full window holds K examples and
  the kept set is a uniform subsample of the unthinned window. The draw is
  unchanged: kept examples keep their batch's ``beta3 ** age`` weight, so the
  estimate targets the same kernel-weighted mean. K = 0 (or ``rho = 1``) is the
  unthinned estimator, bit for bit.
* ``reservoir_uniform_fresh`` is ER's buffer (Algorithm R over single
  examples, capacity K) with uniform draws and recomputed gradients. Unlike ER
  it includes the current task, as the other estimators do.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

ABLATION_ESTIMATORS = {
    'slow_ema': ('none', 'geometric', 'stale'),
    # slow_ema_refresh keeps the slow EMA (support='none' so no StreamMemory
    # is allocated) and adds a sampled unbiased correction on top using the
    # StepLog's stored (idx, G) history. 'fresh' marks the correction's
    # gradient recompute at θ_n; naive.py routes the per-step call separately
    # from the StreamMemory-based fresh arms.
    'slow_ema_refresh': ('none', 'geometric', 'fresh'),
    # slow_ema_reset: same as slow_ema except m_slow is zeroed at every task
    # boundary (controlled by naive.py's set_task_context). The bias
    # correction counter is also reset so (1 − β3^k) uses k = steps-in-task.
    'slow_ema_reset': ('none', 'geometric', 'stale'),
    'fifo_geometric_fresh': ('fifo', 'geometric', 'fresh'),
    'reservoir_geometric_stale': ('reservoir', 'geometric', 'stale'),
    'fifo_uniform_stale': ('fifo', 'uniform', 'stale'),
    'reservoir_uniform_stale': ('reservoir', 'uniform', 'stale'),
    'reservoir_uniform_fresh': ('reservoir', 'uniform', 'fresh'),
}

_DTYPES = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}


class IndexedDataset(Dataset):
    """Append the dataset index so a recomputed gradient can gather the example later."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if isinstance(item, tuple):
            return (*item, index)
        return item, index


def geometric_weights(ages, beta):
    """Normalized ``beta ** age`` in a stable log-space.

    GEOMETRIC KERNEL: ``beta3 ** age`` is the weight the slow EMA puts on the
    gradient of a step ``age`` steps ago. Every geometric estimator gets its
    age weighting from here.
    """
    ages = np.asarray(ages, dtype=np.float64)
    if ages.size == 0:
        return ages
    logw = ages * math.log(float(beta))
    logw -= logw.max()
    weights = np.exp(logw)
    weights /= weights.sum()
    return weights


def ess_of(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(1.0 / np.square(weights).sum())


def systematic_indices(weights, n_draw, rng):
    """One uniform start, then equal steps along the cumulative weight.

    Same expectation as sampling with replacement, lower variance.
    """
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if n_draw <= 0 or total <= 0.0 or weights.size == 0:
        raise ValueError('systematic sample needs a positive weight vector')
    cdf = np.cumsum(weights / total)
    cdf[-1] = 1.0
    u0 = float(rng.rand()) / float(n_draw)
    targets = u0 + (np.arange(n_draw, dtype=np.float64) / float(n_draw))
    return np.clip(np.searchsorted(cdf, targets, side='left'), 0, weights.size - 1)


class StreamMemory:
    """FIFO or Vitter reservoir of minibatch gradients, index rows or single examples."""

    def __init__(self, support, weighting, gradients, beta3, tau, buffer_size,
                 batch_size, n_draw, dtype='bf16', seed=0, capacity_examples=0):
        if support not in ('fifo', 'reservoir'):
            raise ValueError(f'memory support {support!r} is not buffered')
        if weighting not in ('geometric', 'uniform'):
            raise ValueError(weighting)
        if gradients not in ('stale', 'fresh'):
            raise ValueError(gradients)
        # beta3 only participates in the geometric kernel; uniform arms ignore it.
        if weighting == 'geometric' and not 0.0 < float(beta3) < 1.0:
            raise ValueError(f'beta3 must lie in (0, 1), got {beta3}')
        self.support = support
        self.weighting = weighting
        self.gradients = gradients
        self.beta3 = float(beta3)
        self.tau = float(tau)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.n_draw = int(n_draw)
        self.dtype = _DTYPES[dtype]
        self.rng = np.random.RandomState(int(seed))
        self.m_max = max(1, math.ceil(2.0 * self.tau))
        if support == 'fifo' and self.m_max < self.batch_size:
            raise ValueError(
                f'FIFO window M={self.m_max} is shorter than the batch ({self.batch_size})'
            )
        self.capacity_examples = int(capacity_examples)
        # A fresh reservoir stores single examples, K of them.
        self.per_example = support == 'reservoir' and gradients == 'fresh'
        if self.per_example and self.capacity_examples <= 0:
            raise ValueError('reservoir_uniform_fresh needs capacity_examples > 0')
        # Fresh FIFO with a budget: keep each example with probability rho.
        self.thin_rho = None
        if support == 'fifo' and gradients == 'fresh' and self.capacity_examples > 0:
            rho = self.capacity_examples / float(self.m_max * self.batch_size)
            if rho < 1.0:
                self.thin_rho = rho
                self.thin_u = float(self.rng.rand())
                self.thin_count = 0
        self.dim = None
        self.n_pushed = 0
        self.filled = 0
        self.pos = 0
        self.store = None
        self.steps = None
        self.indices = None
        self.lengths = None
        self.tasks = None
        self._window_sum = None
        self._cum = None
        self.last_ess = None
        self.last_M = 0
        self.last_M_clipped = 0

    def _capacity(self):
        if self.support == 'fifo':
            return self.m_max
        return self.capacity_examples if self.per_example else self.buffer_size

    def _ensure(self, dim):
        if self.store is not None or self.indices is not None:
            return
        cap = self._capacity()
        self.steps = torch.empty(cap, dtype=torch.long)
        if self.per_example:
            self.indices = torch.empty(cap, dtype=torch.int32)
            self.tasks = torch.empty(cap, dtype=torch.int16)
            return
        if self.gradients == 'fresh':
            self.indices = torch.empty((cap, self.batch_size), dtype=torch.int32)
            self.lengths = torch.empty(cap, dtype=torch.int32)
            self.tasks = torch.empty(cap, dtype=torch.int16)
            return
        self.dim = int(dim)
        self.store = torch.empty((cap, self.dim), dtype=self.dtype)
        self._window_sum = torch.zeros(self.dim, dtype=torch.float32)

    def window_stats(self):
        """``M = min(ceil(2τ), n_elapsed)`` and whether the ESS window was clipped."""
        n_elapsed = int(self.n_pushed)
        if self.support != 'fifo':
            self.last_M = 0
            self.last_M_clipped = 0
            return 0, 0
        M = min(self.m_max, n_elapsed)
        clipped = int(self.m_max > n_elapsed)
        self.last_M = int(M)
        self.last_M_clipped = clipped
        if n_elapsed > 0:
            assert 1 <= M <= n_elapsed
        if n_elapsed >= self.batch_size:
            assert M >= self.batch_size
        return int(M), clipped

    def _ordered_slots(self):
        if self.filled <= 0:
            return np.empty(0, dtype=np.int64)
        if self.support == 'reservoir' or self.filled < self._capacity():
            return np.arange(self.filled, dtype=np.int64)
        start = self.pos  # oldest slot in a full ring
        return (np.arange(self.filled, dtype=np.int64) + start) % self._capacity()

    def _slot_weights(self, n, slots):
        # Age in steps of each stored gradient.
        ages = n - self.steps[slots].numpy().astype(np.float64)
        if np.any(ages < 0):
            raise AssertionError('memory insertion step is ahead of the optimizer step')
        if self.weighting == 'uniform':
            weights = np.ones(slots.size, dtype=np.float64)
            weights /= weights.sum()
            return weights
        # GEOMETRIC KERNEL: stored gradient weighted by beta3 ** age, as in the EMA.
        return geometric_weights(ages, self.beta3)

    def sample_grad(self, n):
        """Mean of ``n_draw`` stored minibatch gradients. ``None`` if empty."""
        self.window_stats()
        if self.filled <= 0 or self.gradients != 'stale':
            self.last_ess = None
            return None
        slots = self._ordered_slots()
        weights = self._slot_weights(n, slots)
        self.last_ess = ess_of(weights)
        if self.weighting == 'geometric':
            # Draw stored gradients in proportion to the geometric kernel.
            pick = self.rng.choice(slots.size, size=self.n_draw, replace=True, p=weights)
        else:
            pick = self.rng.randint(0, slots.size, size=self.n_draw)
        chosen = slots[pick]
        return self.store[chosen].float().mean(dim=0)

    def sample_pairs(self, n):
        """``[n_draw, 2]`` of ``(task_id, dataset_index)``. ``None`` if empty."""
        self.window_stats()
        if self.filled <= 0 or self.gradients != 'fresh':
            self.last_ess = None
            return None
        if self.per_example:
            # UNIFORM KERNEL, no age weighting: draws without replacement, as
            # ER's ReservoirReplayBuffer.sample_batch.
            n_pick = min(self.n_draw, self.filled)
            pick = self.rng.choice(self.filled, size=n_pick, replace=False)
            self.last_ess = float(self.filled)
            task = self.tasks[pick].numpy().astype(np.int64)
            index = self.indices[pick].numpy().astype(np.int64)
            return np.stack([task, index], axis=1)
        slots = self._ordered_slots()
        # lengths = examples kept per step (fewer than the batch when thinned).
        lengths = self.lengths[slots].numpy().astype(np.int64)
        if int(lengths.sum()) <= 0:
            return None
        # Age in steps of each stored batch.
        ages = n - self.steps[slots].numpy().astype(np.float64)
        # GEOMETRIC KERNEL: each batch weighted by beta3 ** age, as the EMA weights it.
        slot_w = geometric_weights(ages, self.beta3)
        # Each kept example inherits its batch's kernel weight; renormalize over
        # examples. Thinning keeps every example with the same probability, so
        # these weights target the unthinned kernel-weighted mean.
        example_w = np.repeat(slot_w, lengths)
        example_w = example_w / example_w.sum()
        self.last_ess = ess_of(example_w)
        # Draw n_draw examples in proportion to the kernel.
        flat = systematic_indices(example_w, self.n_draw, self.rng)
        slot_of = np.repeat(np.arange(slots.size), lengths)
        base = np.cumsum(lengths) - lengths
        local_slot = slot_of[flat]
        offset = flat - base[local_slot]
        ring = slots[local_slot]
        index = self.indices[ring, offset].numpy().astype(np.int64)
        task = self.tasks[ring].numpy().astype(np.int64)
        return np.stack([task, index], axis=1)

    def push_grad(self, flat, step):
        stored = flat.detach().to(dtype=self.dtype, device='cpu')
        self._ensure(stored.numel())
        if stored.numel() != self.dim:
            raise RuntimeError('gradient width changed between steps')
        as_fp = stored.float()
        evicting = self.support == 'fifo' and self.filled == self._capacity()
        if self.support == 'reservoir':
            slot = self._vitter_slot()
        else:
            slot = self.pos
            if evicting:
                self._window_sum -= self.store[slot].float()
            self.pos = (self.pos + 1) % self._capacity()
            self.filled = min(self.filled + 1, self._capacity())
        if slot is not None:
            self.store[slot] = stored
            self.steps[slot] = int(step)
            self._window_sum += as_fp
        self._note_push(as_fp, evicting)
        if (
            self.support == 'fifo'
            and self.weighting == 'uniform'
            and not evicting
            and self._cum is not None
        ):
            exact = self._window_sum / float(self.filled)
            if not torch.allclose(exact, self._cum, rtol=1e-3, atol=1e-5):
                raise AssertionError(
                    'full-window mean disagrees with the cumulative mean'
                )

    def push_indices(self, indices, task_id, step):
        indices = torch.as_tensor(indices, dtype=torch.int32).view(-1).cpu()
        if indices.numel() > self.batch_size:
            raise RuntimeError(
                f'batch of {indices.numel()} exceeds memory width {self.batch_size}'
            )
        self._ensure(0)
        if self.per_example:
            for index in indices.tolist():
                slot = self._vitter_slot()
                if slot is not None:
                    self.indices[slot] = int(index)
                    self.tasks[slot] = int(task_id)
                    self.steps[slot] = int(step)
                self.n_pushed += 1
            return
        if self.thin_rho is not None:
            indices = indices[self._thin_mask(indices.numel())]
        slot = self.pos
        self.indices[slot].zero_()
        if indices.numel():
            self.indices[slot, :indices.numel()] = indices
        self.lengths[slot] = int(indices.numel())
        self.tasks[slot] = int(task_id)
        self.steps[slot] = int(step)
        self.pos = (self.pos + 1) % self._capacity()
        self.filled = min(self.filled + 1, self._capacity())
        self.n_pushed += 1

    def _thin_mask(self, count):
        """Systematic thinning: the c-th example of the stream is kept iff
        ``floor((c+1) rho + u) - floor(c rho + u) == 1``.

        Each example is kept with probability rho (over u), and any M*B
        consecutive examples keep floor or ceil of M*B*rho = K of them.
        """
        c = self.thin_count + np.arange(count, dtype=np.float64)
        keep = (np.floor((c + 1.0) * self.thin_rho + self.thin_u)
                - np.floor(c * self.thin_rho + self.thin_u)) == 1.0
        self.thin_count += int(count)
        return torch.from_numpy(keep)

    def stored_examples(self):
        """Examples (index memories) or gradients currently held."""
        if self.filled <= 0:
            return 0
        if self.gradients == 'fresh' and not self.per_example:
            return int(self.lengths[:self.filled].sum().item())
        return int(self.filled)

    def _vitter_slot(self):
        """Algorithm R, the same draw as ``ReservoirReplayBuffer._add_one``."""
        cap = self._capacity()
        if self.filled < cap:
            slot = self.filled
            self.filled += 1
        else:
            j = int(self.rng.randint(0, self.n_pushed + 1))
            slot = j if j < cap else None
        return slot

    def _note_push(self, as_fp, evicting):
        n_new = self.n_pushed + 1
        if not evicting:
            if self._cum is None:
                self._cum = as_fp.clone()
            else:
                self._cum.mul_((n_new - 1) / n_new).add_(as_fp, alpha=1.0 / n_new)
        self.n_pushed = n_new
