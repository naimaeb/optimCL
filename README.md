# Optimizer memory vs. replay in continual learning

Code for the experiments of the submission: the training loop, the optimizers
with slow gradient memory (ReplaySGD, AdEMAMix), the memory-estimator ablations
and probes, and the scripts that build the figures and tables.

## Layout

| Path | Contents |
| --- | --- |
| `naive.py` | Training entry point: trains on a task stream, evaluates on every task after each task, writes the accuracy matrix and CL metrics (ACC, BWT, FWT, forgetting). |
| `optimizers/` | `replay_sgd.py` (SGD with a fast and a slow EMA) and `ademamix.py`, the two `--optim` choices; `replay_memory.py` (memory-estimator ablations, `ABLATION_ESTIMATORS`); `stale_probe.py` and `step_log.py` (slow-memory staleness probe and per-step log); `task_boundary_probe.py`, `sam.py`. |
| `utils/` | Argument parser (`args.py`), task streams and datasets, models, CL metrics, evaluation, replay buffer, horizon-to-β conversion (`horizon.py`), diagnostics. |
| `scripts/analysis/` | Figure and table pipeline (`build_submission.py`); learning-rate selection (`protocol_lr.py`). |
| `scripts/data/download_clear.py` | CLEAR-10 / CLEAR-100 downloader. |
| `tests/` | Unit tests (`unittest`). |

## Setup

Python 3.11.

```bash
pip install -r requirements.txt
```

## Data

MNIST, CIFAR-10 and CIFAR-100 are downloaded by torchvision into `--dataset_path`.
CLEAR-10 is fetched with

```bash
python scripts/data/download_clear.py --dataset clear10 --dest ./data/clear10_raw
```

## A single run

`naive.py` logs to Weights & Biases; set `WANDB_MODE=offline` to run without an
account. Every flag is documented in `utils/args.py`. For example, ReplaySGD with
slow memory on rotated MNIST:

```bash
WANDB_MODE=offline python naive.py \
    --wandb_entity <entity> --wandb_project <project> \
    --wandb_group example --wandb_name example --wandb_job_type train \
    --dataset_path ./data --output_folder ./results/example \
    --dataset_name rotmnist --angle 9 --tasks 20 --epochs_per_task 1 --batch_size 128 \
    --model mlp --width 100 --bias 1 --use_multihead_model 0 \
    --optim replaysgd --lr 0.1 --beta1 0.9 --alpha 1.0 --horizon_ratio 3.0 --warmup_tasks 1.0 \
    --weight_decay 0.0 --clip_grad 1.0 --lr_scheduler none --replay 0 --seed 42
```

## Figures and tables

```bash
python scripts/analysis/build_submission.py
```

writes `results/submission/` (Pareto fronts, ablations, gradient staleness),
with the learning rate of every cell chosen by `scripts/analysis/protocol_lr.py`.
It reads run summaries from W&B (set `ENTITY` / `PROJECT` at the top of the
analysis scripts) and the probe outputs that `naive.py` writes under
`results/<W&B group>/`.

## Tests

```bash
python -m unittest discover -s tests
```
