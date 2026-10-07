"""Slow-EMA staleness probe.

At each probe step ``n`` on the ``slow_ema`` arm, this module:

- Reconstructs ``μ^stale_n = v_n / (1 − β3^n)`` from the live slow EMA.
- Computes ``w_s = (1 − β3) β3^(n−s) / (1 − β3^n)`` for every ``s ≤ n``
  logged in the ``StepLog``.
- Recomputes ``g_new_s = clip_c(∇ℓ(B_s; θ_n))`` by rebuilding each stored
  batch (using the per-task-rotated train split, same as ``_recompute_memory_grad``)
  and averaging ``μ^fresh = Σ w_s g_new_s``.
- Projects every batch's drift ``Δ_s = g_new_s − G_s`` onto the held-out
  loss gradient ``G_old = ∇ℓ(H; θ_n)`` to form ``phi_s = w_s ⟨G_old, Δ_s⟩``.
- Separates ``Δ_s = Δ^dir_s + Δ^clip_s`` to tell whether staleness
  bites through the gradient direction or through the clip factor.

Writes two artefacts per probe step:

- ``probe/task_<t>_step_<n>_batches.csv.gz``: one row per past batch.
- A row appended to ``probe/aggregate.csv``: run-level summary at that ``n``.

The probe never mutates model parameters or any optimizer state. Model mode
is saved and restored. Everything inside runs under autograd (``grad`` is
needed) but no in-place writes to parameters.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


BATCH_COLUMNS = [
    'n', 't_n', 's', 't_s', 'pos', 'age', 'w', 'kappa_old', 'kappa_new',
    'raw_norm_old', 'raw_norm_new', 'cos', 'rel', 'phi', 'phi_dir', 'phi_clip',
    'sigma',
]

AGGREGATE_COLUMNS = [
    'n', 't_n', 'n_stored', 'cos_agg', 'cos_dist', 'norm_ratio', 'rel_E',
    'proj_E', 'sum_phi', 'mapping_err', 'held_out_norm', 'seconds',
]


def _flatten_grads(model):
    """One flat fp32 tensor over `requires_grad` parameters, in parameter order."""
    return torch.cat([p.grad.detach().reshape(-1).float()
                      for p in model.parameters() if p.requires_grad])


def _flatten_params(model):
    return {name: p.detach() for name, p in model.named_parameters()}


def _default_sum_ce(logits, y):
    return F.cross_entropy(logits, y, reduction='sum')


def _default_mean_ce(logits, y):
    return F.cross_entropy(logits, y, reduction='mean')


def _held_out_grad(model, heldout_batches, device, loss_sum_fn=None):
    """``∇ℓ(H; θ)`` as a single flat fp32 CPU tensor. Sum-reduce over all held-out
    examples, divide by total count. ``loss_sum_fn(logits, y) -> scalar`` must
    use sum-reduction so the final mean is correct across ragged batches."""
    if loss_sum_fn is None:
        loss_sum_fn = _default_sum_ce
    saved = [p.grad.detach().clone() if p.grad is not None else None
             for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    total_loss = None
    total_n = 0
    for task_id, x, y in heldout_batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, task_id=0, out='logits')
        loss = loss_sum_fn(logits, y)
        total_loss = loss if total_loss is None else (total_loss + loss)
        total_n += int(y.shape[0] if y.dim() > 0 else 1)
    (total_loss / float(max(total_n, 1))).backward()
    g = torch.cat([p.grad.detach().reshape(-1).float().cpu()
                   for p in model.parameters() if p.requires_grad])
    # Restore prior grads.
    for p, prev in zip(model.parameters(), saved):
        p.grad = prev
    return g


def _batch_grad(model, x, y, device, loss_mean_fn=None):
    """One flat gradient of the mean loss on (x,y). ``loss_mean_fn(logits, y)``
    must use mean-reduction (default: cross_entropy)."""
    if loss_mean_fn is None:
        loss_mean_fn = _default_mean_ce
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
    return g, float(loss.detach().item())


class StaleProbe:
    def __init__(self, out_dir, beta3, task_length_L, batch_size, clip_c,
                 chunk_size=64, run_sigma=False, stride=1, window_steps=0,
                 n_tasks=0, seed=0, dtype='fp32',
                 loss_mean_fn=None, loss_sum_fn=None):
        self.out_dir = os.path.join(out_dir, 'probe')
        os.makedirs(self.out_dir, exist_ok=True)
        self.beta3 = float(beta3)
        self.L = int(task_length_L)
        self.batch_size = int(batch_size)
        self.clip_c = float(clip_c)
        self.chunk_size = int(chunk_size)
        self.run_sigma = bool(run_sigma)
        self.stride = max(1, int(stride))
        self.window_steps = int(window_steps)
        # Loss callables for the probe recomputes; defaults match CE on
        # classification. For MSE (e.g. the quadratics stream), callers pass
        # closures over F.mse_loss.
        self.loss_mean_fn = loss_mean_fn
        self.loss_sum_fn = loss_sum_fn
        self.aggregate_path = os.path.join(self.out_dir, 'aggregate.csv')
        self._aggregate_header_written = os.path.isfile(self.aggregate_path)
        # Write manifest once.
        try:
            with open(os.path.join(self.out_dir, 'manifest.json'), 'w') as fh:
                json.dump({
                    'beta3': self.beta3, 'L': self.L, 'seed': int(seed),
                    'chunk_size': self.chunk_size, 'run_sigma': self.run_sigma,
                    'stride': self.stride, 'window_steps': self.window_steps,
                    'n_tasks': int(n_tasks), 'clip_c': self.clip_c,
                    'dtype': dtype,
                }, fh, indent=2)
        except Exception as exc:
            print(f'WARNING: probe manifest write failed ({exc})')

    def _rebuild_batch(self, task_datasets, task_id, idx_arr, device):
        images, labels = [], []
        for i in idx_arr.tolist():
            image, label = task_datasets[int(task_id)][0][int(i)][:2]
            images.append(image)
            labels.append(label if torch.is_tensor(label) else int(label))
        x = torch.stack(images).to(device, non_blocking=True)
        y = (torch.stack(labels).to(device, non_blocking=True)
             if torch.is_tensor(labels[0])
             else torch.tensor(labels, dtype=torch.long, device=device))
        return x, y

    def run(self, *, n, task_id, model, optimizer, task_datasets, heldout_batches,
            device):
        """Execute the probe at optimizer step n. See module docstring for the
        math."""
        import time
        t0 = time.time()
        if task_id < 1:
            return None
        step_log = optimizer.step_log
        if not step_log.store_grad:
            raise RuntimeError(
                'per-batch staleness probe needs StepLog.store_grad=True; '
                'pass --steplog_store_grad 1 (or omit, since 1 is the default)')
        beta3 = self.beta3
        n = int(n)
        # Steps to iterate: all s in [1, n] in StepLog. Optional window.
        steps_all, w_all = step_log.weights(n=n, beta3=beta3)
        if steps_all.size == 0:
            return None
        if self.window_steps > 0:
            mask = (n - steps_all) <= self.window_steps
            steps_all = steps_all[mask]
            w_all = w_all[mask]
            if w_all.sum() <= 0:
                return None

        # (1) μ^stale from the live slow EMA, bias-corrected.
        v_slow_flat = torch.cat([
            optimizer.state[p]['m_slow'].detach().reshape(-1).float().cpu()
            for p in optimizer.param_groups[0]['params']
            if 'm_slow' in optimizer.state.get(p, {})
        ])
        corr3 = 1.0 - beta3 ** n
        mu_stale = v_slow_flat / float(corr3)

        # (3) G_old = ∇ℓ(H; θ_n) on the held-out set.
        was_training = model.training
        model.eval()
        G_old = _held_out_grad(model, heldout_batches, device, loss_sum_fn=self.loss_sum_fn)
        held_out_norm = float(torch.linalg.vector_norm(G_old).item())

        # (4) Iterate stored steps in chunks; compute per-batch g_new and
        # bookkeeping; accumulate acc_fresh, acc_stale, sum_phi, batch rows.
        acc_fresh = torch.zeros_like(mu_stale)
        acc_stale = torch.zeros_like(mu_stale)
        E_est_for_sigma = None
        rows = []
        for chunk_start in range(0, steps_all.size, self.chunk_size):
            chunk_steps = steps_all[chunk_start:chunk_start + self.chunk_size]
            chunk_w = w_all[chunk_start:chunk_start + self.chunk_size]
            for local_i, s in enumerate(chunk_steps):
                s = int(s)
                row_i = step_log.row_of(s)
                t_s = int(step_log.task[row_i])
                idx_arr = step_log.idx[row_i, :int(step_log.lengths[row_i])]
                kappa_old = float(step_log.kappa[row_i])
                raw_old = float(step_log.raw_norm[row_i])
                G_old_s = step_log.G[row_i].float().clone()  # fp32 flat
                w_s = float(chunk_w[local_i])

                x, y = self._rebuild_batch(task_datasets, t_s,
                                           torch.as_tensor(idx_arr, dtype=torch.long),
                                           device)
                V_new, _loss = _batch_grad(model, x, y, device, loss_mean_fn=self.loss_mean_fn)
                v_norm_new = float(torch.linalg.vector_norm(V_new).item())
                kappa_new = (min(1.0, self.clip_c / v_norm_new)
                             if (self.clip_c > 0 and v_norm_new > 0) else 1.0)
                g_new = V_new * kappa_new
                # v_old = g_old / kappa_old; Δ = g_new - g_old; split into dir and clip.
                v_old = G_old_s / max(kappa_old, 1e-30)
                delta = g_new - G_old_s
                delta_dir = (V_new - v_old) * kappa_new
                delta_clip = (kappa_new - kappa_old) * v_old

                g_old_norm = float(torch.linalg.vector_norm(G_old_s).item())
                g_new_norm = float(torch.linalg.vector_norm(g_new).item())
                if g_old_norm > 0 and g_new_norm > 0:
                    cos_s = float((G_old_s @ g_new) / (g_old_norm * g_new_norm))
                else:
                    cos_s = 0.0
                rel_s = (float(torch.linalg.vector_norm(delta).item()) /
                         max(g_new_norm, 1e-30))
                phi = w_s * float(G_old @ delta)
                phi_dir = w_s * float(G_old @ delta_dir)
                phi_clip = w_s * float(G_old @ delta_clip)

                acc_fresh.add_(g_new, alpha=w_s)
                acc_stale.add_(G_old_s, alpha=w_s)
                pos = (float(s - 1) % float(self.L)) / float(self.L) if self.L > 0 else 0.0
                age = n - s
                if (chunk_start + local_i) % self.stride == 0:
                    rows.append(OrderedDict([
                        ('n', n), ('t_n', int(task_id)), ('s', s), ('t_s', t_s),
                        ('pos', pos), ('age', age), ('w', w_s),
                        ('kappa_old', kappa_old), ('kappa_new', kappa_new),
                        ('raw_norm_old', raw_old), ('raw_norm_new', v_norm_new),
                        ('cos', cos_s), ('rel', rel_s),
                        ('phi', phi), ('phi_dir', phi_dir), ('phi_clip', phi_clip),
                        ('sigma', 0.0),
                    ]))

        # (5) Aggregates.
        E = acc_fresh - mu_stale
        E_norm = float(torch.linalg.vector_norm(E).item())
        norm_fresh = float(torch.linalg.vector_norm(acc_fresh).item())
        norm_stale = float(torch.linalg.vector_norm(mu_stale).item())
        if norm_fresh > 0 and norm_stale > 0:
            cos_agg = float((mu_stale @ acc_fresh) / (norm_stale * norm_fresh))
        else:
            cos_agg = 0.0
        norm_ratio = norm_stale / max(norm_fresh, 1e-30)
        rel_E = E_norm / max(norm_fresh, 1e-30)
        proj_E = float(G_old @ E)
        mapping_err = float(torch.linalg.vector_norm(acc_stale - mu_stale).item()) \
            / max(norm_stale, 1e-30)
        sum_phi = sum(r['phi'] for r in rows)  # exact only if stride=1
        # (6) sigma pass.
        if self.run_sigma and E_norm > 0:
            E_cpu = E
            for row in rows:
                s = int(row['s'])
                row_i = step_log.row_of(s)
                t_s = int(row['t_s'])
                idx_arr = step_log.idx[row_i, :int(step_log.lengths[row_i])]
                kappa_old = float(step_log.kappa[row_i])
                G_old_s = step_log.G[row_i].float()
                x, y = self._rebuild_batch(task_datasets, t_s,
                                           torch.as_tensor(idx_arr, dtype=torch.long),
                                           device)
                V_new, _loss = _batch_grad(model, x, y, device, loss_mean_fn=self.loss_mean_fn)
                v_norm_new = float(torch.linalg.vector_norm(V_new).item())
                kappa_new = (min(1.0, self.clip_c / v_norm_new)
                             if (self.clip_c > 0 and v_norm_new > 0) else 1.0)
                g_new = V_new * kappa_new
                delta = g_new - G_old_s
                row['sigma'] = row['w'] * float(delta @ E_cpu) / (E_norm ** 2)

        # Restore model mode.
        if was_training:
            model.train()

        seconds = time.time() - t0
        # Cosine distance ∈ [0, 1]: 0 when μ_stale = μ_fresh direction, 0.5
        # when orthogonal, 1 when anti-parallel. Scale-free, so comparable
        # across widths / angles / SAM settings.
        cos_dist = 0.5 * (1.0 - float(cos_agg))
        aggregate_row = OrderedDict([
            ('n', n), ('t_n', int(task_id)), ('n_stored', int(steps_all.size)),
            ('cos_agg', cos_agg), ('cos_dist', cos_dist),
            ('norm_ratio', norm_ratio),
            ('rel_E', rel_E), ('proj_E', proj_E), ('sum_phi', sum_phi),
            ('mapping_err', mapping_err), ('held_out_norm', held_out_norm),
            ('seconds', seconds),
        ])

        # Write per-batch rows.
        batch_path = os.path.join(
            self.out_dir, f'task_{int(task_id):03d}_step_{n:08d}_batches.csv.gz')
        with gzip.open(batch_path, 'wt', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=BATCH_COLUMNS)
            w.writeheader()
            for row in rows:
                w.writerow(row)

        # Append aggregate row.
        with open(self.aggregate_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=AGGREGATE_COLUMNS)
            if not self._aggregate_header_written:
                w.writeheader()
                self._aggregate_header_written = True
            w.writerow(aggregate_row)
        return aggregate_row, batch_path
