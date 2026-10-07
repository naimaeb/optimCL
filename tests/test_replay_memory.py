"""ReplaySGD memory-lattice estimators."""

import unittest

import numpy as np
import torch

from optimizers.replay_memory import (
    StreamMemory,
    ess_of,
    geometric_weights,
    systematic_indices,
)
from optimizers.replay_sgd import ReplaySGD


def _reference_vitter(values, capacity, seed):
    rng = np.random.RandomState(seed)
    stored = [None] * capacity
    steps = [None] * capacity
    filled = 0
    n_seen = 0
    for step, value in enumerate(values, start=1):
        if filled < capacity:
            slot = filled
            filled += 1
        else:
            j = int(rng.randint(0, n_seen + 1))
            slot = j if j < capacity else None
        if slot is not None:
            stored[slot] = value
            steps[slot] = step
        n_seen += 1
    return stored[:filled], steps[:filled]


class MemoryTests(unittest.TestCase):
    def test_systematic_follows_a_point_mass(self):
        idx = systematic_indices(np.array([0.0, 0.0, 5.0]), 7, np.random.RandomState(0))
        self.assertTrue(np.all(idx == 2))

    def test_geometric_weights_prefer_recent_samples(self):
        ages = np.arange(1, 11, dtype=np.float64)
        weights = geometric_weights(ages, 0.5)
        uniform = np.ones_like(weights) / weights.size
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[-1])
        self.assertGreater(ess_of(uniform), ess_of(weights))

    def test_vitter_matches_algorithm_r(self):
        values = [float(i) for i in range(30)]
        memory = StreamMemory(
            'reservoir', 'uniform', 'stale', beta3=0.5, tau=4.0, buffer_size=5,
            batch_size=2, n_draw=2, dtype='fp32', seed=7)
        for step, value in enumerate(values, start=1):
            memory.push_grad(torch.tensor([value]), step)
        got = memory.store[:memory.filled, 0].float().tolist()
        got_steps = memory.steps[:memory.filled].tolist()
        ref, ref_steps = _reference_vitter(values, 5, seed=7)
        self.assertEqual(got, ref)
        self.assertEqual(got_steps, ref_steps)

    def test_uniform_fifo_matches_cumulative_mean(self):
        memory = StreamMemory(
            'fifo', 'uniform', 'stale', beta3=0.9, tau=8.0, buffer_size=4,
            batch_size=2, n_draw=2, dtype='bf16', seed=0)
        total = torch.zeros(3)
        for step in range(1, 6):
            grad = torch.tensor([step, 0.5 * step, -step], dtype=torch.float32)
            memory.push_grad(grad, step)
            total += grad.to(torch.bfloat16).float()
        self.assertTrue(torch.allclose(memory._cum, total / 5, rtol=1e-3, atol=1e-5))

    def test_task0_memory_is_zero_and_task1_baseline_is_not(self):
        param = torch.nn.Parameter(torch.zeros(4))
        opt = ReplaySGD(
            [param], lr=0.1, weight_decay=0.0, beta1=0.0, beta3=0.5, alpha=1.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0, ablation='slow_ema',
            tau=2.0, buffer_size=4, batch_size=2, memory_samples=2, seed=0)
        opt.set_task_context(0)
        param.grad = torch.ones(4)
        opt.step()
        metrics = opt.pop_memory_metrics()
        self.assertEqual(metrics['memory/norm_mu'], 0.0)
        self.assertEqual(metrics['memory/share'], 0.0)
        # β1 = 0, so m1_hat = g. μ = 0 and α = 1, so the step is g / 2.
        self.assertTrue(torch.allclose(param.detach(), torch.full((4,), -0.05)))

        opt.set_task_context(1)
        param.grad = torch.ones(4)
        opt.step()
        metrics = opt.pop_memory_metrics()
        self.assertGreater(metrics['memory/norm_mu'], 0.0)
        self.assertGreater(metrics['memory/share'], 0.0)

    def test_reservoir_logs_ess_and_fifo_uses_recomputed_gradient(self):
        param = torch.nn.Parameter(torch.zeros(3))
        opt = ReplaySGD(
            [param], lr=0.0, weight_decay=0.0, beta1=0.0, beta3=0.5, alpha=1.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0, ablation='reservoir_geometric_stale',
            tau=4.0, buffer_size=6, batch_size=2, memory_samples=2, memory_dtype='fp32',
            seed=1)
        opt.set_task_context(0)
        for _ in range(4):
            param.grad = torch.ones(3)
            opt.step()
        opt.set_task_context(1)
        param.grad = torch.ones(3)
        opt.step()
        metrics = opt.pop_memory_metrics()
        self.assertIn('memory/ess', metrics)
        self.assertGreater(metrics['memory/ess'], 0.0)
        self.assertEqual(metrics['memory/M'], 0)

        param2 = torch.nn.Parameter(torch.zeros(3))
        opt1 = ReplaySGD(
            [param2], lr=0.0, weight_decay=0.0, beta1=0.0, beta3=0.5, alpha=1.0,
            nesterov=False, beta3_warmup=0, alpha_warmup=0, ablation='fifo_geometric_fresh',
            tau=4.0, buffer_size=6, batch_size=2, memory_samples=2, seed=1)
        opt1.set_task_context(0)
        opt1.set_batch_indices(torch.tensor([0, 1]))
        param2.grad = torch.ones(3)
        opt1.step()
        self.assertIsNone(opt1.sample_stored_examples())
        opt1.set_task_context(1)
        self.assertIsNotNone(opt1.sample_stored_examples())
        opt1.set_recomputed_gradient([torch.full((3,), 4.0)])
        opt1.set_batch_indices(torch.tensor([2, 3]))
        param2.grad = torch.ones(3)
        opt1.step()
        metrics = opt1.pop_memory_metrics()
        self.assertGreater(metrics['memory/norm_mu'], 0.0)
        self.assertGreater(metrics['memory/M'], 0)
        # Fresh FIFO reports ESS of its geometric draw over the window.
        self.assertIn('memory/ess', metrics)
        self.assertGreater(metrics['memory/ess'], 0.0)

    def _fifo_fresh(self, capacity_examples, seed=3, tau=8.0, batch=4, n_draw=64):
        return StreamMemory(
            'fifo', 'geometric', 'fresh', beta3=tau / (1.0 + tau), tau=tau,
            buffer_size=0, batch_size=batch, n_draw=n_draw, seed=seed,
            capacity_examples=capacity_examples)

    def test_thinned_fifo_holds_k_examples(self):
        # M = 16 slots of 4 examples = 64 in the window; keep K = 10 of them.
        memory = self._fifo_fresh(10)
        self.assertAlmostEqual(memory.thin_rho, 10 / 64)
        for step in range(1, 60):
            memory.push_indices(torch.arange(4 * step, 4 * step + 4), 0, step)
            if step >= memory.m_max:
                self.assertIn(memory.stored_examples(), (9, 10, 11))
        self.assertEqual(memory.stored_examples(), int(memory.lengths.sum()))

    def test_thinned_fifo_with_k_below_m(self):
        # K < M: most steps keep nothing; the draw still works.
        memory = self._fifo_fresh(3)
        for step in range(1, 40):
            memory.push_indices(torch.arange(4 * step, 4 * step + 4), 0, step)
        self.assertIn(memory.stored_examples(), (2, 3, 4))
        pairs = memory.sample_pairs(40)
        self.assertEqual(pairs.shape, (64, 2))

    def test_unthinned_fifo_is_unchanged(self):
        # K = 0 and K >= M*B disable thinning and draw exactly the same pairs.
        draws = []
        for capacity in (0, 10_000):
            memory = self._fifo_fresh(capacity, seed=11)
            self.assertIsNone(memory.thin_rho)
            for step in range(1, 30):
                memory.push_indices(torch.arange(4 * step, 4 * step + 4), 0, step)
            draws.append(memory.sample_pairs(30))
        self.assertTrue(np.array_equal(draws[0], draws[1]))

    def test_thinning_keeps_the_kernel(self):
        # Mean age of the kernel-weighted draws matches the whole-window arm.
        def mean_age(capacity, seeds):
            ages = []
            for seed in seeds:
                memory = self._fifo_fresh(capacity, seed=seed)
                for step in range(1, 41):
                    memory.push_indices(torch.arange(4 * step, 4 * step + 4), 0, step)
                pairs = memory.sample_pairs(41)
                ages.extend((41 - pairs[:, 1] // 4).tolist())  # index 4s.. was pushed at step s
            return float(np.mean(ages))
        full = mean_age(0, range(50))
        thin = mean_age(16, range(400))
        self.assertAlmostEqual(thin, full, delta=0.35)

    def test_uniform_fresh_reservoir_matches_algorithm_r(self):
        memory = StreamMemory(
            'reservoir', 'uniform', 'fresh', beta3=0.5, tau=4.0, buffer_size=0,
            batch_size=3, n_draw=4, seed=5, capacity_examples=7)
        values = list(range(45))
        for step in range(1, 16):
            memory.push_indices(torch.tensor(values[3 * (step - 1):3 * step]), 1, step)
        ref, _ = _reference_vitter(values, 7, seed=5)
        self.assertEqual(memory.indices[:memory.filled].tolist(), ref)
        pairs = memory.sample_pairs(16)
        self.assertEqual(len(set(pairs[:, 1].tolist())), 4)  # without replacement
        self.assertTrue(set(pairs[:, 1].tolist()) <= set(ref))

    def test_uniform_fresh_reservoir_needs_a_budget(self):
        with self.assertRaises(ValueError):
            StreamMemory('reservoir', 'uniform', 'fresh', beta3=0.5, tau=4.0,
                         buffer_size=10, batch_size=2, n_draw=2)


if __name__ == '__main__':
    unittest.main()
