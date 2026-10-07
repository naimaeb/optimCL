"""sweep_run_name names a sweep run after the parameters its sweep varies."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import run_setup as tools  # noqa: E402

SWEEP_PARAMS = {
    'seed': {'values': [42, 123, 7]},
    'lr': {'value': 3.0e-4},
    'alpha': {'values': [1.0, 4.0]},
    'horizon_ratio': {'values': [1.0, 3.0, 10.0]},
    'beta4_from_horizon': {'values': [0, 1]},
    'angle': {'values': [9]},
}


def _api(params):
    sweep = SimpleNamespace(config={'parameters': params})
    return mock.Mock(return_value=mock.Mock(sweep=mock.Mock(return_value=sweep)))


class TestSweepRunName(unittest.TestCase):

    def setUp(self):
        self.run = SimpleNamespace(sweep_id='abc', entity='e', project='p')
        self.args = SimpleNamespace(seed=42, lr=3e-4, alpha=1.0, horizon_ratio=3.0,
                                    beta4_from_horizon=1, angle=9)

    def test_varied_keys_only_seed_last(self):
        with mock.patch.object(tools.wandb, 'Api', _api(SWEEP_PARAMS)):
            name = tools.sweep_run_name(self.run, self.args)
        self.assertEqual(name, 'alpha=1_r=3_b4h=1_seed=42')

    def test_no_sweep(self):
        self.run.sweep_id = None
        self.assertIsNone(tools.sweep_run_name(self.run, self.args))

    def test_api_failure_keeps_default(self):
        with mock.patch.object(tools.wandb, 'Api', side_effect=RuntimeError('offline')):
            self.assertIsNone(tools.sweep_run_name(self.run, self.args))


if __name__ == '__main__':
    unittest.main()
