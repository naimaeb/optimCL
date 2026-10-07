"""SAM step helper: two-pass gradient at θ' = θ + ρ·g/‖g‖, θ restored."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers.sam import is_active, sam_step


class _Lin(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.linear = nn.Linear(d, k, bias=False)
    def forward(self, x):
        return self.linear(x)


class SamStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _Lin(5, 3)
        self.x = torch.randn(4, 5)
        self.y = torch.randint(0, 3, (4,))

    def _loss(self):
        return F.cross_entropy(self.model(self.x), self.y)

    def test_theta_restored_after_sam_step(self):
        theta0 = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        info = sam_step(self.model, self._loss, rho=0.05, clip_grad=1.0)
        for n, p in self.model.named_parameters():
            diff = (p.detach() - theta0[n]).abs().max().item()
            self.assertLess(float(diff), 1e-6,
                            msg=f'{n} drifted by {diff}')
        self.assertGreater(info['g_norm'], 0.0)
        self.assertGreater(info['raw_norm'], 0.0)

    def test_grad_differs_from_plain_backward(self):
        # Plain gradient
        self.model.zero_grad(set_to_none=True)
        self._loss().backward()
        g_plain = torch.cat([p.grad.detach().reshape(-1) for p in self.model.parameters()])
        # SAM gradient
        self.model.zero_grad(set_to_none=True)
        sam_step(self.model, self._loss, rho=0.1, clip_grad=0.0)
        g_sam = torch.cat([p.grad.detach().reshape(-1) for p in self.model.parameters()])
        # They should NOT be equal (otherwise SAM is doing nothing).
        self.assertGreater(float((g_sam - g_plain).norm().item()), 1e-6)

    def test_clip_factor_matches_pytorch(self):
        info = sam_step(self.model, self._loss, rho=0.05, clip_grad=0.01)
        # kappa should equal clip / raw_norm (or 1 if raw_norm < clip).
        expected = min(1.0, 0.01 / info['raw_norm']) if info['raw_norm'] > 0 else 1.0
        self.assertAlmostEqual(info['kappa'], expected, places=6)

    def test_rho_zero_rejected(self):
        with self.assertRaises(ValueError):
            sam_step(self.model, self._loss, rho=0.0)


class SamActiveTests(unittest.TestCase):
    def test_mode_dispatch(self):
        self.assertFalse(is_active('none', 0))
        self.assertFalse(is_active('none', 7))
        self.assertTrue(is_active('all', 0))
        self.assertTrue(is_active('all', 7))
        self.assertTrue(is_active('task0_only', 0))
        self.assertFalse(is_active('task0_only', 1))
        with self.assertRaises(ValueError):
            is_active('maybe', 0)

    def test_task0_only_switches_off_from_task_1(self):
        """Under sam_mode='task0_only', the SAM pass fires on task 0 and
        then stops — later tasks use the plain gradient, letting the slow
        EMA pick up normal (unperturbed) gradients."""
        # Task 0: active.
        for t in range(0, 1):
            self.assertTrue(is_active('task0_only', t),
                            msg=f'task0_only must fire during task {t}')
        # Tasks 1..T-1: inactive.
        for t in range(1, 20):
            self.assertFalse(is_active('task0_only', t),
                             msg=f'task0_only must NOT fire during task {t}')


if __name__ == '__main__':
    unittest.main()
