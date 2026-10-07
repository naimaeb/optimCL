"""Sharpness-Aware Minimization (SAM) step helper.

Not a full optimizer: SAM is a two-pass gradient computation that leaves
``p.grad`` holding the perturbed-point gradient ``g_sam``, which the usual
optimizer then consumes with its standard update. This file provides the
single primitive used by ``naive.py``.

At the call site, ``sam_step(...)`` replaces the ``loss.backward()`` +
``clip_grad_norm_`` pair whenever SAM is active for the current task.
Everything downstream (memory push, probe recompute, refresh correction)
stays unchanged — the memory stores whatever
``p.grad`` holds post-``sam_step``, which is the perturbed gradient, and the
probe's own recompute does a plain forward+backward so it continues to use
plain gradients.

Reference: Foret et al. 2021, "Sharpness-Aware Minimization for
Efficiently Improving Generalization" (https://arxiv.org/abs/2010.01412).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch


def _flat_norm(tensors: Iterable[torch.Tensor]) -> float:
    sq = 0.0
    for t in tensors:
        if t is None:
            continue
        sq += float((t.detach() ** 2).sum().item())
    return sq ** 0.5


def sam_step(model: torch.nn.Module,
             compute_loss: Callable[[], torch.Tensor],
             rho: float,
             clip_grad: float = 0.0,
             eps: float = 1e-12) -> dict:
    """Run one SAM-perturbed gradient computation.

    - First pass: ``g = ∇ℓ(θ)`` via ``compute_loss().backward()``.
    - Perturb: ``θ' = θ + ρ · g / (‖g‖ + eps)`` in place.
    - Second pass: ``g_sam = ∇ℓ(θ')`` by zeroing grads, calling
      ``compute_loss().backward()`` again.
    - Restore θ (``θ ← θ' − ε``). ``p.grad`` now holds ``g_sam``.
    - If ``clip_grad > 0``, apply ``clip_grad_norm_`` to ``g_sam`` with
      threshold ``clip_grad``. The returned ``raw_norm`` is the **pre-clip**
      global norm of ``g_sam`` (matches ``torch.nn.utils.clip_grad_norm_``).

    Returns a dict with ``loss``, ``raw_norm`` (||g_sam||), ``g_norm``
    (||g|| of the first pass, for logging), ``kappa`` (clip factor) and
    ``rho`` (as passed). ``loss`` is the SECOND-pass loss (the loss at θ',
    which is the quantity SAM actually minimises).
    """
    if rho <= 0.0:
        raise ValueError(f'sam_step requires rho > 0, got {rho}')
    # First pass: g at θ.
    loss1 = compute_loss()
    loss1.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    grads1 = [p.grad for p in params]
    g_norm = _flat_norm(grads1)
    if g_norm == 0.0:
        # Degenerate: just return the plain gradient. Still clip.
        raw_norm_t = torch.nn.utils.clip_grad_norm_(
            params, clip_grad if clip_grad > 0 else float('inf'))
        raw_norm = float(raw_norm_t.item() if torch.is_tensor(raw_norm_t) else raw_norm_t)
        kappa = min(1.0, clip_grad / raw_norm) if (clip_grad > 0 and raw_norm > 0) else 1.0
        return dict(loss=float(loss1.detach().item()),
                    raw_norm=raw_norm, g_norm=g_norm, kappa=kappa, rho=float(rho))
    scale = float(rho) / (g_norm + eps)
    # Perturb and save the perturbation so it can be undone exactly.
    perturbs = []
    with torch.no_grad():
        for p, g in zip(params, grads1):
            if g is None:
                perturbs.append(None); continue
            e = g.detach().mul(scale)
            perturbs.append(e)
            p.data.add_(e)
    # Second pass: g_sam at θ'.
    model.zero_grad(set_to_none=True)
    loss2 = compute_loss()
    loss2.backward()
    # Restore θ (θ' - ε = θ).
    with torch.no_grad():
        for p, e in zip(params, perturbs):
            if e is not None:
                p.data.sub_(e)
    # Clip g_sam, record pre-clip norm.
    raw_norm_t = torch.nn.utils.clip_grad_norm_(
        params, clip_grad if clip_grad > 0 else float('inf'))
    raw_norm = float(raw_norm_t.item() if torch.is_tensor(raw_norm_t) else raw_norm_t)
    kappa = min(1.0, clip_grad / raw_norm) if (clip_grad > 0 and raw_norm > 0) else 1.0
    return dict(loss=float(loss2.detach().item()),
                raw_norm=raw_norm, g_norm=g_norm, kappa=kappa, rho=float(rho))


def is_active(sam_mode: str, task_id: int) -> bool:
    """Decode ``--sam_mode`` into "run SAM this task?"."""
    if sam_mode in ('none', '', None):
        return False
    if sam_mode == 'all':
        return True
    if sam_mode == 'task0_only':
        return int(task_id) == 0
    raise ValueError(f'unknown sam_mode {sam_mode!r}')
