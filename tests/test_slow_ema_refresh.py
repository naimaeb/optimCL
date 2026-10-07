"""slow_ema_refresh arm: control (S'=0 identical to slow_ema), and sanity
smoke on the refresh correction consumption path."""

import unittest

import numpy as np
import torch

from optimizers.replay_sgd import ReplaySGD


def _run(ablation, refresh_batches=0, grads=None, seed=42):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.zeros(5))
    opt = ReplaySGD(
        [p], lr=0.1, weight_decay=0.0, beta1=0.9, beta3=0.5, alpha=1.0,
        nesterov=False, beta3_warmup=0, alpha_warmup=0,
        ablation=ablation, tau=2.0, buffer_size=0, batch_size=1,
        memory_samples=1, memory_dtype='fp32', seed=7, memory_examples=0,
        total_steps=max(len(grads) + 1, 10), steplog_dtype='fp32',
        refresh_batches=refresh_batches,
    )
    opt.set_task_context(1)
    for i, g in enumerate(grads):
        p.grad = g.clone()
        if opt.step_log is not None:
            opt.set_batch_indices(torch.tensor([i]))
            opt.set_clip_info(raw_norm=1.0, kappa=1.0, loss=0.0)
        opt.step()
    return p.detach().clone()


class SlowEmaRefreshControl(unittest.TestCase):
    def test_s_zero_matches_slow_ema_bit_identical(self):
        torch.manual_seed(100)
        grads = [torch.randn(5) for _ in range(6)]
        p_slow = _run('slow_ema', refresh_batches=0, grads=grads)
        p_refresh = _run('slow_ema_refresh', refresh_batches=0, grads=grads)
        self.assertTrue(torch.equal(p_slow, p_refresh),
                        msg=f'slow_ema={p_slow}\nslow_ema_refresh={p_refresh}')

    def test_set_refresh_correction_moves_the_step(self):
        torch.manual_seed(200)
        grads = [torch.randn(5) for _ in range(4)]
        # Baseline slow_ema_refresh with S'=0 (no correction).
        p_baseline = _run('slow_ema_refresh', refresh_batches=0, grads=grads)

        # Now the same arm but with an injected correction = ones(5) on step 4.
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.zeros(5))
        opt = ReplaySGD(
            [p], lr=0.1, weight_decay=0.0, beta1=0.9, beta3=0.5, alpha=1.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0,
            ablation='slow_ema_refresh', tau=2.0, buffer_size=0, batch_size=1,
            memory_samples=1, memory_dtype='fp32', seed=7, memory_examples=0,
            total_steps=10, steplog_dtype='fp32', refresh_batches=1,
        )
        opt.set_task_context(1)
        for i, g in enumerate(grads):
            p.grad = g.clone()
            opt.set_batch_indices(torch.tensor([i]))
            opt.set_clip_info(raw_norm=1.0, kappa=1.0, loss=0.0)
            if i == 3:  # last step: inject a nonzero C to be sure it reaches the update
                opt.set_refresh_correction(torch.ones(5))
            opt.step()
        self.assertFalse(torch.equal(p.detach(), p_baseline),
                         msg='set_refresh_correction did not alter the step')


class SampleRefreshStepsSanity(unittest.TestCase):
    def test_sample_returns_s_prime_entries_from_the_log(self):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(5))
        opt = ReplaySGD(
            [p], lr=0.0, weight_decay=0.0, beta1=0.9, beta3=0.5, alpha=1.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0,
            ablation='slow_ema_refresh', tau=2.0, buffer_size=0, batch_size=1,
            memory_samples=1, memory_dtype='fp32', seed=1, memory_examples=0,
            total_steps=10, steplog_dtype='fp32', refresh_batches=3,
        )
        opt.set_task_context(1)
        for i in range(5):
            p.grad = torch.full((5,), float(i + 1))
            opt.set_batch_indices(torch.tensor([i * 10]))
            opt.set_clip_info(raw_norm=1.0, kappa=1.0, loss=0.0)
            opt.step()
        # Now needs_refresh_batches should be True; sample 3 steps.
        self.assertTrue(opt.needs_refresh_batches())
        draws = opt.sample_refresh_steps()
        self.assertEqual(len(draws), 3)
        for d in draws:
            self.assertIn('step', d)
            self.assertIn('task', d)
            self.assertIn('idx', d)
            self.assertIn('G_old', d)
            self.assertEqual(d['G_old'].shape, (5,))


class SlowEmaResetTests(unittest.TestCase):
    def test_m_slow_zeroed_on_task_boundary(self):
        """slow_ema_reset must zero m_slow when set_task_context flips."""
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(4))
        opt = ReplaySGD(
            [p], lr=0.1, weight_decay=0.0, beta1=0.9, beta3=0.5, alpha=1.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0,
            ablation='slow_ema_reset', tau=2.0, buffer_size=0, batch_size=1,
            memory_samples=1, seed=0, total_steps=10)
        opt.set_task_context(0)
        for _ in range(3):
            p.grad = torch.ones(4)
            opt.step()
        # m_slow should be nonzero at this point.
        m_slow_before = opt.state[p]['m_slow'].clone()
        self.assertGreater(float(m_slow_before.abs().sum().item()), 0.0)
        # Task boundary: should zero m_slow and record the reset step.
        opt.set_task_context(1)
        self.assertEqual(opt._task_boundary_step, 3)
        self.assertEqual(float(opt.state[p]['m_slow'].abs().sum().item()), 0.0)

    def test_reset_matches_slow_ema_within_a_task(self):
        """A single task of slow_ema_reset reproduces slow_ema exactly."""
        torch.manual_seed(0)
        grads = [torch.randn(5) for _ in range(5)]

        def run(ablation):
            torch.manual_seed(42)
            p = torch.nn.Parameter(torch.zeros(5))
            opt = ReplaySGD(
                [p], lr=0.1, weight_decay=0.0, beta1=0.9, beta3=0.5, alpha=1.0,
                nesterov=False, beta3_warmup=0, alpha_warmup=0,
                ablation=ablation, tau=2.0, buffer_size=0, batch_size=1,
                memory_samples=1, seed=7, total_steps=10)
            opt.set_task_context(1)  # start on task 1 so μ is active
            for g in grads:
                p.grad = g.clone()
                opt.step()
            return p.detach().clone()

        self.assertTrue(torch.equal(run('slow_ema'), run('slow_ema_reset')))


if __name__ == '__main__':
    unittest.main()
