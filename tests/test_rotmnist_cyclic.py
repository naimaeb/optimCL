"""Angle schedule, matched drift, scaled reservoir, and bilinear rotation."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from utils.horizon import beta_from_horizon, task_length
from utils.rotmnist_cyclic import (
    angle_accuracy_metrics,
    angle_schedule,
    apply_replay_per_angle,
    bind_cyclic_schedule,
    validate_angle_schedule,
)
from utils.transforms import BilinearTensorRotation


# Spec table: (T, stream) -> (angles or None, n_distinct, amplitude).
_TABLE = {
    (6, 'cyclic'): ([0, 12, 24, 36, 24, 12], 4, 36),
    (6, 'onepass'): ([0, 12, 24, 36, 48, 60], 6, 60),
    (12, 'cyclic'): (None, 4, 36),
    (12, 'onepass'): (None, 12, 132),
    (24, 'cyclic'): (None, 4, 36),
    (24, 'onepass'): (None, 24, 276),
}

# Rounded exemplars per distinct angle at fixed K.
_FIXED = {
    ('cyclic', 6): (50, 250, 1280),
    ('cyclic', 12): (50, 250, 1280),
    ('cyclic', 24): (50, 250, 1280),
    ('onepass', 6): (33, 167, 853),
    ('onepass', 12): (17, 83, 427),
    ('onepass', 24): (8, 42, 213),
}
_SCALED = {
    ('cyclic', 6): (100, 500),
    ('cyclic', 12): (100, 500),
    ('cyclic', 24): (100, 500),
    ('onepass', 6): (150, 750),
    ('onepass', 12): (300, 1500),
    ('onepass', 24): (600, 3000),
}


class CyclicScheduleTest(unittest.TestCase):
    def test_spec_table(self):
        for (tasks, stream), (expected, n_distinct, amplitude) in _TABLE.items():
            angles = angle_schedule(tasks, stream, 12, 6)
            self.assertEqual(len(set(angles)), n_distinct)
            self.assertEqual(max(angles) - min(angles), amplitude)
            if expected is not None:
                self.assertEqual(angles, expected)
            if stream == 'cyclic':
                self.assertEqual(angles[:6], angles[6:12] if tasks >= 12 else angles[:6])
                cycle = [0, 12, 24, 36, 24, 12]
                self.assertEqual(angles, cycle * (tasks // 6))

    def test_matched_drift_and_onepass_cap(self):
        for tasks in (6, 12, 24):
            for stream in ('cyclic', 'onepass'):
                angles = angle_schedule(tasks, stream, 12, 6)
                self.assertTrue(all(
                    abs(angles[i + 1] - angles[i]) == 12 for i in range(tasks - 1)
                ))
        with self.assertRaises(AssertionError):
            validate_angle_schedule([0, 12, 24, 360], 'onepass', 4, 12, 6)
        with self.assertRaises(AssertionError):
            angle_schedule(7, 'cyclic', 12, 6)
        with self.assertRaises(AssertionError):
            angle_schedule(40, 'onepass', 12, 6)  # 12*39 = 468 >= 360

    def test_broken_drift_is_rejected(self):
        angles = [0, 12, 30, 42]
        with self.assertRaises(AssertionError):
            validate_angle_schedule(angles, 'onepass', 4, 12, 6)

    def test_horizon_betas_match_spec(self):
        L = task_length('rotmnist_cyclic', 128, 1)
        self.assertEqual(L, 469)
        expected = {1: 0.997872340, 6: 0.999644760, 24: 0.999911166}
        for r, beta in expected.items():
            got, _tau = beta_from_horizon(r, L)
            self.assertAlmostEqual(got, beta, places=8)

    def test_scaled_and_fixed_coverage(self):
        for stream in ('cyclic', 'onepass'):
            for tasks in (6, 12, 24):
                angles = angle_schedule(tasks, stream, 12, 6)
                n = len(set(angles))
                self.assertEqual(
                    tuple(int(round(K / n)) for K in (200, 1000, 5120)),
                    _FIXED[(stream, tasks)],
                )
                self.assertEqual(
                    tuple(m * n for m in (25, 125)),
                    _SCALED[(stream, tasks)],
                )

    def test_replay_per_angle_overrides_buffer(self):
        args = SimpleNamespace(
            stream_type='onepass', angle_delta=12, cycle_slots=6, tasks=24,
            replay=1, replay_per_angle=125, replay_buffer_size=1000,
        )
        bind_cyclic_schedule(args)
        apply_replay_per_angle(args)
        self.assertEqual(args.cfg_n_distinct_angles, 24)
        self.assertEqual(args.replay_buffer_size, 3000)
        self.assertEqual(args.cfg_K_schedule, 'scaled')

        fixed = SimpleNamespace(
            replay=1, replay_per_angle=0, replay_buffer_size=5120,
            cfg_n_distinct_angles=4,
        )
        apply_replay_per_angle(fixed)
        self.assertEqual(fixed.replay_buffer_size, 5120)
        self.assertAlmostEqual(fixed.cfg_exemplars_per_angle, 1280.0)
        self.assertEqual(fixed.cfg_K_schedule, 'fixed')

    def test_angle_accuracy_keeps_positions(self):
        # Two visits to 0° disagree; the mean and the spread both show up.
        R = np.array([
            [80.0, 10.0],
            [70.0, 90.0],
        ])
        text, metrics = angle_accuracy_metrics(R, [0, 12])
        self.assertAlmostEqual(metrics['angle_acc/0'], 70.0)
        self.assertAlmostEqual(metrics['pos_final_acc/0'], 70.0)
        self.assertIn('pos 0 angle 0', text)
        R2 = np.array([
            [50.0, 0.0, 0.0],
            [40.0, 80.0, 0.0],
            [30.0, 70.0, 90.0],
        ])
        _, metrics = angle_accuracy_metrics(R2, [0, 12, 0])
        self.assertAlmostEqual(metrics['angle_acc/0'], 60.0)
        self.assertAlmostEqual(metrics['angle_acc_spread/0'], 60.0)

    def test_bilinear_rotation_preserves_shape_and_normalizes(self):
        mean, std = (0.1307,), (0.3081,)
        black = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
        out = BilinearTensorRotation(36, mean, std)(black)
        self.assertEqual(tuple(out.shape), (1, 28, 28))
        self.assertTrue(torch.allclose(out, torch.full_like(out, (0.0 - mean[0]) / std[0])))

        arr = np.zeros((28, 28), dtype=np.uint8)
        arr[:8, :] = 255
        img = Image.fromarray(arr, mode='L')
        a = BilinearTensorRotation(0, mean, std)(img)
        b = BilinearTensorRotation(90, mean, std)(img)
        self.assertEqual(tuple(b.shape), (1, 28, 28))
        self.assertFalse(torch.allclose(a, b))


if __name__ == '__main__':
    unittest.main()
