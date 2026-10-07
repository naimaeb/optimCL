"""Per-example gradient-noise variance estimator.

A proxy for the per-batch gradient-noise variance ``sigma^2`` that AdEMAMix's
slow first-moment EMA is implicitly averaging over. This module estimates it
directly from a minibatch:

    sigma^2 ~= mean_i ||g_i - g_bar||_2^2

where ``g_i`` is the per-example gradient of the loss w.r.t. every trainable
parameter, and ``g_bar = mean_i g_i`` is the ordinary minibatch-averaged
gradient (the same quantity ``loss.backward()`` would produce).

Computed with ``torch.func.vmap(torch.func.grad(...))`` -- a single vmapped
forward+backward pass over the batch dimension -- rather than looping over
batch-size-1 forward/backward passes, since the latter would be prohibitively
slow to call periodically during training.

Intended usage: call this every few epochs (not every step -- see
``--track_grad_noise`` / ``--grad_noise_every_epochs`` in utils/args.py), on a
single minibatch, and log the resulting scalar as ``grad_noise/sigma_sq``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def estimate_gradient_noise(model, x: torch.Tensor, y: torch.Tensor, task_id: int = 0,
                             regression: bool = False) -> float:
    """Estimate ``sigma^2 = mean_i ||g_i - g_bar||^2`` for one minibatch.

    Parameters
    ----------
    model : nn.Module
        Any of this repo's CL models (MultiHeadMLP / ResNet18CL / LinearRegressorCL),
        called as ``model(x, task_id=task_id, out='logits')`` internally.
    x, y : torch.Tensor
        A single minibatch, already moved to the model's device.
    task_id : int
        Head index to evaluate (0 for single-head / domain-incremental setups).
    regression : bool
        If True, use MSE loss (quadratics benchmark); else cross-entropy.

    Returns
    -------
    float
        Scalar gradient-noise-variance estimate for this batch.

    Notes
    -----
    Temporarily switches the model to eval() (so BatchNorm/Dropout do not
    introduce spurious per-example randomness / running-stat mutation into the
    gradient computation), and restores the original mode before returning.
    Does not mutate any model parameters or buffers (uses ``functional_call``
    with detached copies).
    """
    was_training = model.training
    model.eval()
    try:
        params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def compute_loss(params, buffers, x_i, y_i):
            x_i = x_i.unsqueeze(0)
            y_i = y_i.unsqueeze(0)
            y_hat = functional_call(model, (params, buffers), (x_i,), {'task_id': task_id, 'out': 'logits'})
            if regression:
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze(-1)
                return F.mse_loss(y_hat, y_i)
            return F.cross_entropy(y_hat, y_i)

        per_sample_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x, y)
    finally:
        if was_training:
            model.train()

    batch_size = x.shape[0]
    sq_dev_sum = None
    for g in per_sample_grads.values():
        g_flat = g.reshape(batch_size, -1).float()
        g_bar = g_flat.mean(dim=0, keepdim=True)
        dev_sq = ((g_flat - g_bar) ** 2).sum(dim=1)
        sq_dev_sum = dev_sq if sq_dev_sum is None else sq_dev_sum + dev_sq

    if sq_dev_sum is None:
        return 0.0
    return float(sq_dev_sum.mean().item())
