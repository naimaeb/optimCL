"""StepLog: append/get round-trip, bias-corrected weights, systematic sampling."""

import unittest

import numpy as np
import torch

from optimizers.step_log import StepLog


class StepLogTests(unittest.TestCase):
    def _append(self, log, step, task, idx, G):
        log.append(step=step, task=task,
                   idx=torch.as_tensor(idx, dtype=torch.long),
                   kappa=1.0, raw_norm=float(np.linalg.norm(G)),
                   loss=0.0,
                   G_flat=torch.as_tensor(G, dtype=torch.float32))

    def test_append_get_roundtrip(self):
        log = StepLog(capacity=10, batch_size=4, dtype='fp32', seed=0)
        self._append(log, 1, 0, [10, 11], [1., 0., 0.])
        self._append(log, 2, 0, [12, 13, 14], [0., 1., 0.])
        got = log.get(2)
        self.assertEqual(got['step'], 2)
        self.assertEqual(got['task'], 0)
        self.assertTrue(np.array_equal(got['idx'], np.array([12, 13, 14], dtype=np.int32)))
        self.assertTrue(torch.equal(got['G'], torch.tensor([0., 1., 0.])))
        self.assertEqual(log.n_pushed, 2)

    def test_capacity_overflow(self):
        log = StepLog(capacity=2, batch_size=2, dtype='fp32', seed=0)
        self._append(log, 1, 0, [1], [1., 0.])
        self._append(log, 2, 0, [2], [0., 1.])
        with self.assertRaises(RuntimeError):
            self._append(log, 3, 0, [3], [1., 1.])

    def test_weights_match_reference(self):
        log = StepLog(capacity=20, batch_size=1, dtype='fp32', seed=0)
        for s in range(1, 11):
            self._append(log, s, 0, [s], [float(s), 0., 0.])
        beta3 = 0.9
        steps, w = log.weights(n=10, beta3=beta3)
        self.assertEqual(len(steps), 10)
        # Reference computation
        ref = np.array([(1 - beta3) * beta3 ** (10 - s) / (1 - beta3 ** 10)
                        for s in range(1, 11)])
        self.assertTrue(np.allclose(w, ref, atol=1e-7))
        # Weights must sum to 1 (bias-corrected).
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_systematic_sample_marginal(self):
        log = StepLog(capacity=20, batch_size=1, dtype='fp32', seed=0)
        for s in range(1, 11):
            self._append(log, s, 0, [s], [0.])
        beta3 = 0.9
        # Draw a lot: the marginal should approximate w.
        rng = np.random.RandomState(123)
        total = np.zeros(10, dtype=np.int64)
        trials = 1000
        for _ in range(trials):
            picks = log.sample(n=10, beta3=beta3, S=5, sampling='systematic', rng=rng)
            for s in picks:
                total[int(s) - 1] += 1
        emp = total / total.sum()
        _, w = log.weights(n=10, beta3=beta3)
        self.assertTrue(np.max(np.abs(emp - w)) < 0.05)  # 5pp slack is plenty

    def test_sample_s_le_n(self):
        log = StepLog(capacity=20, batch_size=1, dtype='fp32', seed=0)
        for s in range(1, 11):
            self._append(log, s, 0, [s], [0.])
        picks = log.sample(n=5, beta3=0.9, S=100, sampling='systematic')
        self.assertTrue((picks >= 1).all())
        self.assertTrue((picks <= 5).all())

    def test_bf16_storage_roundtrip(self):
        log = StepLog(capacity=4, batch_size=2, dtype='bf16', seed=0)
        self._append(log, 1, 0, [0], [1.0, 2.0, 3.0])
        got = log.get(1)
        # bf16 has ~3 decimal precision; just check approximately
        self.assertTrue(torch.allclose(got['G'].float(),
                                       torch.tensor([1.0, 2.0, 3.0]), atol=1e-2))

    def test_importance_weights_age_boost(self):
        log = StepLog(capacity=10, batch_size=1, dtype='fp32', seed=0)
        for s in range(1, 6):
            self._append(log, s, 0, [s], [0.])
        steps, q, w = log.importance_weights(n=5, beta3=0.5, tau=1.0)
        # q should be heavier on older steps than w (which decays with age).
        self.assertAlmostEqual(float(q.sum()), 1.0, places=6)
        # Oldest step (s=1, age=4) relative to newest (s=5, age=0) is boosted.
        self.assertGreater(q[0] / w[0], q[-1] / w[-1])


if __name__ == '__main__':
    unittest.main()
