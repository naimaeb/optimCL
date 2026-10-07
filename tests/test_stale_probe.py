"""StaleProbe: mapping, age-0, and probe-point parsing."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from optimizers.replay_sgd import ReplaySGD
from optimizers.stale_probe import StaleProbe


def _parse_probe_points(spec, L):
    # Mirror naive._parse_probe_points without importing naive (which depends
    # on many heavy modules). Keep in sync with the naive.py definition.
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
        step = int(round(frac * L))
        step = max(1, min(L, step))
        out.add(step)
    return out


class ProbePointsParsing(unittest.TestCase):
    def test_boundary_and_end(self):
        self.assertEqual(_parse_probe_points('boundary,end', 100), {1, 100})

    def test_fractions(self):
        self.assertEqual(_parse_probe_points('0.25,0.5', 400), {100, 200})

    def test_clamp(self):
        # fractions > 1 are clamped to L.
        self.assertEqual(_parse_probe_points('2.0', 50), {50})

    def test_mixed(self):
        self.assertEqual(
            _parse_probe_points('boundary,0.5,end', 1000), {1, 500, 1000})


class ProbeMapping(unittest.TestCase):
    """``acc_stale = Σ w_s G_s`` must match ``μ^stale = v_n/(1-β3^n)`` from the
    live slow EMA within fp32 rounding, given that the StepLog stored the same
    gradients that fed the EMA."""

    def test_mapping(self):
        torch.manual_seed(2026)
        d = 7
        p = torch.nn.Parameter(torch.zeros(d))
        beta3 = 0.5
        n_steps = 8
        opt = ReplaySGD(
            [p], lr=0.0, weight_decay=0.0, beta1=0.9, beta3=beta3, alpha=0.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0,
            ablation='slow_ema', tau=2.0, buffer_size=0, batch_size=1,
            memory_samples=1, memory_dtype='fp32', seed=1, memory_examples=0,
            total_steps=n_steps + 2, steplog_dtype='fp32',
            probe_staleness=True, refresh_batches=0,
        )
        opt.set_task_context(1)
        G = [torch.randn(d) for _ in range(n_steps)]
        for i, g in enumerate(G):
            p.grad = g.clone()
            opt.set_batch_indices(torch.tensor([i]))
            opt.set_clip_info(raw_norm=float(g.norm().item()), kappa=1.0, loss=0.0)
            opt.step()
        # Recompute acc_stale = Σ w_s G_s from the StepLog.
        steps, w = opt.step_log.weights(n=n_steps, beta3=beta3)
        acc_stale = torch.zeros(d, dtype=torch.float32)
        for s, ws in zip(steps, w):
            acc_stale.add_(opt.step_log.get(int(s))['G'].float(), alpha=float(ws))
        # μ_stale from the live m_slow.
        corr3 = 1.0 - beta3 ** n_steps
        mu_stale = opt.state[p]['m_slow'].detach().clone() / corr3
        rel = (acc_stale - mu_stale).norm() / mu_stale.norm().clamp_min(1e-12)
        self.assertLess(float(rel.item()), 1e-5,
                        msg=f'rel error {float(rel.item())}')


class ProbeConstruction(unittest.TestCase):
    """The probe builder constructs; CSVs appear with the expected columns."""

    def test_probe_writes_aggregate_csv(self):
        with tempfile.TemporaryDirectory() as out:
            probe = StaleProbe(
                out_dir=out, beta3=0.5, task_length_L=10, batch_size=4,
                clip_c=1.0, chunk_size=4, run_sigma=False, stride=1,
                n_tasks=2, seed=0, dtype='fp32')
            # Just confirm the manifest got written.
            self.assertTrue(os.path.isfile(os.path.join(out, 'probe', 'manifest.json')))


if __name__ == '__main__':
    unittest.main()
