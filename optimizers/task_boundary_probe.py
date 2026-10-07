"""Task-boundary + mid-task staleness probe.

Measures the slow-EMA-weighted staleness contribution of each PAST task s < t
at scheduled probe fractions {0.25, 0.5, 0.75, 1.0} inside each current task
t >= 1. Complements ``optimizers/stale_probe.py`` which captures per-batch
staleness within the near-support of the slow EMA; this probe captures the
cross-task block structure that ``stale_probe`` cannot afford on CIFAR-scale
models (its per-step gradient log would be ~3.5 TB).

Terminology:
  - theta_end[s]   = parameters at the end of task s (bf16 CPU snapshot)
  - theta_mid[s]   = parameters at the 0.5-fraction of task s (bf16 CPU snapshot)
  - H[s]           = held-out batch of B_probe examples from task s's test split
  - N[s]           = number of optimizer steps in task s
  - S[s]           = cumulative step count at the END of task s

The task-s EMA mass at global step n is
    W_s(n) = beta3^(n - S[s]) * (1 - beta3^N[s]) / (1 - beta3^n)

Per probe at task_fraction f of task t >= 1, for each past task s < t:
  g_s_now = grad(H[s]; theta_now)              -- 1 fwd+bwd
  g_s_end = grad(H[s]; theta_end[s])            -- 1 fwd+bwd
  g_s_mid = grad(H[s]; theta_mid[s])            -- 1 fwd+bwd (second reference)
  Delta_s = g_s_now - g_s_ref                   (for ref in {end, mid})
  cos_s   = <g_s_ref, g_s_now> / (||g_s_ref|| ||g_s_now||)
  rel_s   = ||Delta_s|| / ||g_s_ref||
  phi_s   = W_s * <G_old, Delta_s>
and one aggregate row per probe fire with sum_phi and rel_E (end-ref only).

The probe uses the same save/restore p.grad discipline as
optimizers/stale_probe.py (``_held_out_grad`` / ``_batch_grad``) so training's
own gradient buffers and the ReplaySGD optimizer state (m_fast, m_slow) are
untouched. In particular slow_ema's m_slow continues to accumulate across task
boundaries exactly as in uninstrumented runs; the probe never writes to
``optimizer.state[p]``.

Output under ``<output_folder>/probe/``:
  - task_boundary_aggregate.csv       (one row per (task, task_fraction) probe fire)
  - task_boundary_perpasttask.csv     (one row per (task, task_fraction, past_task, ref))
  - task_boundary_manifest.json       (static config)
"""

from __future__ import annotations

import csv
import json
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F


BATCH_COLUMNS = ['n', 't', 'task_fraction', 'past_task', 'ref', 'W_s',
                 'cos_s', 'rel_s', 'delta_norm', 'phi_s', 'g_now_norm',
                 'g_ref_norm']

AGG_COLUMNS = ['n', 't', 'task_fraction', 'W_past_total', 'sum_phi', 'rel_E',
               'E_norm', 'mu_fresh_proxy_norm']


def _default_sum_ce(logits, y):
    return F.cross_entropy(logits, y, reduction='sum')


def _default_mean_ce(logits, y):
    return F.cross_entropy(logits, y, reduction='mean')


def _grad_on_batch(model, x, y, device, loss_mean_fn):
    """Save/restore p.grad, zero, backward on (x, y) at the CURRENT model
    parameters, return flat fp32 CPU tensor of the gradient. Mirrors
    ``optimizers/stale_probe.py::_batch_grad``."""
    saved = [p.grad.detach().clone() if p.grad is not None else None
             for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    logits = model(x, task_id=0, out='logits')
    loss = loss_mean_fn(logits, y)
    loss.backward()
    g = torch.cat([p.grad.detach().reshape(-1).float().cpu()
                   for p in model.parameters() if p.requires_grad])
    for p, prev in zip(model.parameters(), saved):
        p.grad = prev
    return g


def _grad_on_heldouts(model, heldouts, device, loss_sum_fn):
    """``∇ℓ(union of heldouts; θ_current)`` returned as flat fp32 CPU tensor.
    ``heldouts`` is a list of ``(task_id, x, y)`` tuples; sum-reduce and divide
    by total count. Mirrors ``optimizers/stale_probe.py::_held_out_grad``."""
    saved = [p.grad.detach().clone() if p.grad is not None else None
             for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    total_loss = None
    total_n = 0
    for _tid, x, y in heldouts:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, task_id=0, out='logits')
        loss = loss_sum_fn(logits, y)
        total_loss = loss if total_loss is None else (total_loss + loss)
        total_n += int(y.shape[0] if y.dim() > 0 else 1)
    (total_loss / float(max(total_n, 1))).backward()
    g = torch.cat([p.grad.detach().reshape(-1).float().cpu()
                   for p in model.parameters() if p.requires_grad])
    for p, prev in zip(model.parameters(), saved):
        p.grad = prev
    return g


def _flatten_params(model, dtype=torch.bfloat16):
    """Return a flat CPU snapshot of the model's parameters (bf16 by default)."""
    pieces = [p.detach().reshape(-1).to(dtype=dtype, device='cpu')
              for p in model.parameters() if p.requires_grad]
    return torch.cat(pieces)


def _restore_params(model, flat, device):
    """In-place scatter of ``flat`` back into model.parameters(). Dtype is
    converted to each parameter's native dtype on the target device."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            chunk = flat[offset:offset + n].to(dtype=p.dtype,
                                              device=device,
                                              non_blocking=True)
            p.data.copy_(chunk.view_as(p))
            offset += n


class TaskBoundaryProbe:
    """Task-boundary + mid-task slow-EMA staleness probe. See module docstring.

    Lifecycle hooks called from naive.py:
      - ``on_task_start(s, task_dataset_s, device)`` — captures H[s].
      - ``on_task_midpoint(s, model)`` — snapshots theta_mid[s].
      - ``on_task_end(s, model, n_now)`` — snapshots theta_end[s], records N[s], S[s].
      - ``on_probe_step(n, t, task_fraction, model, task_datasets, optimizer,
                        device)`` — runs the probe (requires s < t to exist).

    All grads are computed with the training's p.grad buffer saved and
    restored; the ReplaySGD optimizer state is never touched.
    """

    def __init__(self, out_dir: str, beta3: float, n_tasks: int,
                 probe_batch_size: int = 128,
                 probe_task_fractions=(0.25, 0.5, 0.75, 1.0),
                 seed: int = 0, dtype: str = 'bf16',
                 loss_mean_fn: Optional[Callable] = None,
                 loss_sum_fn: Optional[Callable] = None):
        self.out_dir = os.path.join(out_dir, 'probe')
        os.makedirs(self.out_dir, exist_ok=True)
        self.beta3 = float(beta3)
        self.n_tasks = int(n_tasks)
        self.probe_batch_size = int(probe_batch_size)
        self.probe_task_fractions = tuple(sorted(set(float(x)
                                                     for x in probe_task_fractions)))
        self.seed = int(seed)
        self.snapshot_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
        self.loss_mean_fn = loss_mean_fn or _default_mean_ce
        self.loss_sum_fn = loss_sum_fn or _default_sum_ce
        # per-task state
        self.theta_end = {}          # s -> flat bf16 CPU tensor
        self.theta_mid = {}          # s -> flat bf16 CPU tensor
        self.H = {}                  # s -> (x, y) tensors on CPU (sampled lazily)
        self.N = {}                  # s -> int
        self.S = {-1: 0}             # s -> cumulative step count at END of task s
        # CSV paths
        self.per_path = os.path.join(self.out_dir, 'task_boundary_perpasttask.csv')
        self.agg_path = os.path.join(self.out_dir, 'task_boundary_aggregate.csv')
        self._per_hdr = os.path.isfile(self.per_path)
        self._agg_hdr = os.path.isfile(self.agg_path)
        # manifest
        try:
            with open(os.path.join(self.out_dir, 'task_boundary_manifest.json'),
                      'w') as fh:
                json.dump({
                    'beta3': self.beta3, 'n_tasks': self.n_tasks,
                    'probe_batch_size': self.probe_batch_size,
                    'probe_task_fractions': list(self.probe_task_fractions),
                    'seed': self.seed, 'dtype': dtype,
                }, fh, indent=2)
        except Exception as exc:
            print(f'WARNING: task_boundary_probe manifest write failed ({exc})')

    # ------------------------------------------------------------------ hooks
    def on_task_start(self, s: int, task_dataset_s, device):
        """Sample B_probe examples from task s's TEST split and store as a CPU
        tensor pair in H[s]. task_dataset_s is a torch Dataset (same shape as
        the train split); we use the test split to avoid overlap with any
        training batches."""
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + int(s) * 7 + 13)
        n_available = len(task_dataset_s)
        bs = min(self.probe_batch_size, n_available)
        perm = torch.randperm(n_available, generator=g)[:bs].tolist()
        images, labels = [], []
        for i in perm:
            image, label = task_dataset_s[int(i)][:2]
            images.append(image)
            labels.append(label if torch.is_tensor(label) else int(label))
        x = torch.stack(images)
        y = (torch.stack(labels) if torch.is_tensor(labels[0])
             else torch.tensor(labels, dtype=torch.long))
        self.H[int(s)] = (x, y)

    def on_task_midpoint(self, s: int, model):
        self.theta_mid[int(s)] = _flatten_params(model, dtype=self.snapshot_dtype)

    def on_task_end(self, s: int, model, n_now: int):
        self.theta_end[int(s)] = _flatten_params(model, dtype=self.snapshot_dtype)
        s_i = int(s)
        self.S[s_i] = int(n_now)
        self.N[s_i] = int(n_now) - int(self.S.get(s_i - 1, 0))

    # ------------------------------------------------------------------ probe
    def on_probe_step(self, n: int, t: int, task_fraction: float,
                      model, device,
                      heldout_tasks=None):
        """Run the probe at task_fraction f of task t >= 1. ``heldout_tasks``
        defaults to all past tasks ``[0, 1, ..., t-1]``; may be restricted for
        testing. Writes rows to the two CSVs."""
        t = int(t)
        if t < 1:
            return
        if heldout_tasks is None:
            heldout_tasks = list(range(t))
        heldout_tasks = [int(s) for s in heldout_tasks if int(s) < t
                         and int(s) in self.H]
        if not heldout_tasks:
            return
        # Snapshot current theta so we can restore after loading a reference.
        theta_now_flat = _flatten_params(model, dtype=torch.float32)
        # G_old = grad on the union of past-task held-outs AT theta_now
        heldouts = [(int(s), self.H[int(s)][0], self.H[int(s)][1])
                    for s in heldout_tasks]
        G_old = _grad_on_heldouts(model, heldouts, device, self.loss_sum_fn)
        G_old_norm = float(G_old.norm())
        # Per past task
        per_rows = []
        E_tasklevel = None
        mu_fresh = None
        sum_phi = 0.0
        W_past_total = 0.0
        beta3 = float(self.beta3)
        one_minus_beta3n = 1.0 - beta3 ** n if beta3 < 1.0 else 1.0
        for s in heldout_tasks:
            x = self.H[s][0].to(device, non_blocking=True)
            y = self.H[s][1].to(device, non_blocking=True)
            # g_now at theta_now
            g_now = _grad_on_batch(model, x, y, device, self.loss_mean_fn)
            g_now_norm = float(g_now.norm())
            # kernel weight attributed to task s
            if s in self.N and self.N[s] > 0:
                W_s = (beta3 ** (n - self.S[s])) * (1.0 - beta3 ** self.N[s])
                W_s = W_s / max(one_minus_beta3n, 1e-30)
            else:
                W_s = 0.0
            W_past_total += W_s
            for ref_name, theta_ref in (('end', self.theta_end.get(s)),
                                        ('mid', self.theta_mid.get(s))):
                if theta_ref is None:
                    continue
                _restore_params(model, theta_ref, device)
                g_ref = _grad_on_batch(model, x, y, device, self.loss_mean_fn)
                _restore_params(model, theta_now_flat, device)
                g_ref_norm = float(g_ref.norm())
                delta = g_now - g_ref
                delta_norm = float(delta.norm())
                cos_s = (float((g_ref * g_now).sum())
                         / max(g_ref_norm * g_now_norm, 1e-30))
                rel_s = delta_norm / max(g_ref_norm, 1e-30)
                phi_s = W_s * float((G_old * delta).sum())
                per_rows.append({
                    'n': n, 't': t, 'task_fraction': task_fraction,
                    'past_task': s, 'ref': ref_name,
                    'W_s': W_s, 'cos_s': cos_s, 'rel_s': rel_s,
                    'delta_norm': delta_norm, 'phi_s': phi_s,
                    'g_now_norm': g_now_norm, 'g_ref_norm': g_ref_norm,
                })
                if ref_name == 'end':
                    sum_phi += phi_s
                    E_piece = W_s * delta
                    mu_piece = W_s * g_now
                    E_tasklevel = (E_piece if E_tasklevel is None
                                   else E_tasklevel + E_piece)
                    mu_fresh = (mu_piece if mu_fresh is None
                                else mu_fresh + mu_piece)
        # Aggregate row
        E_norm = float(E_tasklevel.norm()) if E_tasklevel is not None else 0.0
        mu_fresh_norm = float(mu_fresh.norm()) if mu_fresh is not None else 0.0
        rel_E = E_norm / max(mu_fresh_norm, 1e-30)
        agg_row = {
            'n': n, 't': t, 'task_fraction': task_fraction,
            'W_past_total': W_past_total, 'sum_phi': sum_phi,
            'rel_E': rel_E, 'E_norm': E_norm,
            'mu_fresh_proxy_norm': mu_fresh_norm,
        }
        # Write CSV rows.
        self._write_rows(per_rows, agg_row)

    # ------------------------------------------------------------------ IO
    def _write_rows(self, per_rows, agg_row):
        with open(self.per_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=BATCH_COLUMNS)
            if not self._per_hdr:
                w.writeheader()
                self._per_hdr = True
            for r in per_rows:
                w.writerow(r)
        with open(self.agg_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=AGG_COLUMNS)
            if not self._agg_hdr:
                w.writeheader()
                self._agg_hdr = True
            w.writerow(agg_row)

    # ------------------------------------------------------------------ util
    @staticmethod
    def parse_fractions(spec):
        """Parse ``'0.25,0.5,0.75,1.0'`` into a sorted tuple of unique floats
        in (0, 1]."""
        if spec is None:
            return (0.25, 0.5, 0.75, 1.0)
        if isinstance(spec, (list, tuple)):
            parts = spec
        else:
            parts = [x.strip() for x in str(spec).split(',') if x.strip()]
        out = []
        for p in parts:
            try:
                v = float(p)
            except ValueError:
                continue
            if 0.0 < v <= 1.0:
                out.append(v)
        return tuple(sorted(set(out)))

    @staticmethod
    def batch_ids_for_task(L: int, fractions):
        """Return ``{task_steps: task_fraction}`` — a mapping from the
        1-indexed per-task step count at which the probe should fire to the
        task_fraction (0,1] it represents. Fraction 1.0 maps to task_steps=L
        (the last batch of the task); fraction 0.5 maps to task_steps=L/2.
        The naive.py loop uses 1-indexed ``task_steps`` (incremented BEFORE
        the step body), so these keys compare directly."""
        ids = {}
        for f in fractions:
            step = max(1, int(round(f * L)))
            ids[step] = max(ids.get(step, 0.0), f)
        return ids  # dict: task_steps -> task_fraction
