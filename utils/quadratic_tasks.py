"""Convex multi-task quadratic benchmark for continual learning.

10-task setting where each task defines a convex quadratic regression problem
whose population loss is:

    L_t(w) = (w - w*_t)^T Q_t (w - w*_t) + const

where Q_t is a positive-definite Hessian and w*_t is the task-specific optimum.

Task generation
---------------
All tasks share the same random orthogonal eigenbasis U drawn once from
the Haar measure via QR decomposition.  Per-task eigenvalue vectors λ_t are
drawn log-uniformly from [λ_min, λ_max], giving each task a different
condition number.  The optimal weight vectors w*_t are random unit vectors
scaled by ``w_star_scale``, placed independently to maximise the CL challenge
(large inter-task distances → stronger catastrophic forgetting).

Data generation
---------------
For task t with n samples, each pair (a_i, b_i) satisfies

    a_i = Q_t^{1/2} z_i,   z_i ~ N(0, I_d)
    b_i = a_i^T w*_t

The empirical MSE loss (1/n) Σ ||a_i^T w - b_i||^2 concentrates around the
population quadratic as n grows:

    (1/n) Σ ||a_i^T w - b_i||^2  →  (w - w*_t)^T Q_t (w - w*_t)

The Hessian of the empirical loss is (2/n) A^T A → 2 Q_t, so the Fisher
information equals the Hessian — an ideal benchmark for Fisher-preconditioned
continual learning algorithms where the exact second-order structure is known.

Usage via naive.py
------------------
  python naive.py \\
      --dataset_path ./data \\
      --dataset_name quadratics \\
      --model linreg \\
      --optim ademamix --lr <lr> \\
      --tasks 10 --quad_dim 100 \\
      --batch_size <batch_size> --use_multihead_model 0 --seed <seed> \\
      --output_folder ./results/<name> \\
      --wandb_entity <entity> --wandb_project <project> \\
      --wandb_group <group> --wandb_job_type train --wandb_name <name>
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Task specification (closed-form, no data)
# ---------------------------------------------------------------------------

@dataclass
class QuadraticTaskSpec:
    """Closed-form specification for a single quadratic task.

    Attributes
    ----------
    Q : (d, d) positive-definite Hessian matrix.
    w_star : (d,) optimal parameter vector.
    Q_sqrt : (d, d) symmetric square root of Q, i.e. Q_sqrt @ Q_sqrt.T == Q.
    task_id : int  (0-indexed identifier within a task sequence).
    """
    Q:       torch.Tensor
    w_star:  torch.Tensor
    Q_sqrt:  torch.Tensor
    task_id: int = 0

    def loss(self, w: torch.Tensor) -> torch.Tensor:
        """Evaluate the true population loss (w - w*)^T Q (w - w*)."""
        delta = w.to(self.Q.device) - self.w_star.to(self.Q.device)
        return (delta @ self.Q @ delta)

    def dist_to_opt(self, w: torch.Tensor) -> float:
        """Return ||w - w*||_2."""
        return (w.cpu() - self.w_star.cpu()).norm().item()


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapping finite-sample approximation of one quadratic task
# ---------------------------------------------------------------------------

class QuadraticDataset(Dataset):
    """Finite-sample regression dataset whose MSE loss approximates a quadratic.

    Generates n (feature, label) pairs (a_i, b_i) such that

        a_i = Q_sqrt z_i,   z_i ~ N(0, I_d)
        b_i = a_i^T w*

    The empirical MSE loss with these data converges to the population quadratic
    L(w) = (w - w*)^T Q (w - w*) as n grows.

    Parameters
    ----------
    spec : QuadraticTaskSpec
        Quadratic specification for this task.
    n_samples : int
        Number of (a, b) data points to generate.
    rng : np.random.Generator
        Seeded NumPy RNG for reproducibility.
    """

    def __init__(self, spec: QuadraticTaskSpec, n_samples: int, rng: np.random.Generator):
        d = spec.w_star.shape[0]
        Z = torch.tensor(rng.standard_normal((n_samples, d)), dtype=torch.float32)
        A = Z @ spec.Q_sqrt.T   # (n, d); Cov(a_i) = Q_sqrt Q_sqrt^T = Q
        b = A @ spec.w_star      # (n,)
        self.A = A
        self.b = b

    def __len__(self) -> int:
        return self.A.shape[0]

    def __getitem__(self, idx):
        return self.A[idx], self.b[idx]


# ---------------------------------------------------------------------------
# Task sequence generation
# ---------------------------------------------------------------------------

def generate_quadratic_tasks(
    n_tasks: int = 10,
    dim: int = 100,
    n_train: int = 1000,
    n_test: int = 200,
    eigenvalue_range: tuple[float, float] = (0.1, 10.0),
    w_star_scale: float = 3.0,
    seed: int = 42,
) -> list[tuple[QuadraticTaskSpec, QuadraticDataset, QuadraticDataset]]:
    """Generate a sequence of n_tasks convex quadratic CL tasks.

    All tasks share a common orthogonal eigenbasis U (drawn once from the Haar
    measure via QR).  Per-task eigenvalues are drawn log-uniformly so the tasks
    span a range of condition numbers.  Task optima w*_t are random unit vectors
    independently drawn and scaled by ``w_star_scale``.

    Parameters
    ----------
    n_tasks : int
        Number of tasks (default 10).
    dim : int
        Parameter-space dimension d (feature dimension equals d).
    n_train : int
        Number of training samples per task.
    n_test : int
        Number of test samples per task.
    eigenvalue_range : (float, float)
        (λ_min, λ_max) for log-uniform eigenvalue draws.
        Wider range → larger spread of condition numbers across tasks.
    w_star_scale : float
        L2 norm of each task's optimal weight vector.  Larger values
        increase the inter-task distance and therefore catastrophic forgetting.
    seed : int
        Master seed controlling all task randomness.

    Returns
    -------
    List of (spec, train_dataset, test_dataset) triples, one per task.
    Tasks are indexed 0 … n_tasks-1.
    """
    master_rng = np.random.default_rng(seed)

    # --- Shared orthogonal eigenbasis (Haar measure via QR) ---
    raw_basis = master_rng.standard_normal((dim, dim))
    U, _ = np.linalg.qr(raw_basis)   # U: (d, d) orthogonal

    λ_min, λ_max = eigenvalue_range
    tasks: list[tuple[QuadraticTaskSpec, QuadraticDataset, QuadraticDataset]] = []

    for t in range(n_tasks):
        # Per-task eigenvalues drawn log-uniformly → varied condition numbers
        log_eigvals = master_rng.uniform(np.log(λ_min), np.log(λ_max), size=dim)
        eigvals = np.exp(log_eigvals)

        Q_np      = U @ np.diag(eigvals)       @ U.T   # (d, d) PD
        Q_sqrt_np = U @ np.diag(np.sqrt(eigvals)) @ U.T   # symmetric sqrt

        Q      = torch.tensor(Q_np,      dtype=torch.float32)
        Q_sqrt = torch.tensor(Q_sqrt_np, dtype=torch.float32)

        # Independent random optimum — unit vector scaled to w_star_scale
        w_raw   = master_rng.standard_normal(dim)
        w_star  = torch.tensor(
            w_raw / np.linalg.norm(w_raw) * w_star_scale, dtype=torch.float32
        )

        spec = QuadraticTaskSpec(Q=Q, w_star=w_star, Q_sqrt=Q_sqrt, task_id=t)

        # Task-local sub-RNGs keep train/test data deterministic per task
        # regardless of the number of tasks or RNG state from other tasks.
        rng_train = np.random.default_rng(seed * 10_000 + t * 100 + 1)
        rng_test  = np.random.default_rng(seed * 10_000 + t * 100 + 2)

        train_ds = QuadraticDataset(spec, n_train, rng_train)
        test_ds  = QuadraticDataset(spec, n_test,  rng_test)

        tasks.append((spec, train_ds, test_ds))

    return tasks


def generate_bitflip_tasks(
    n_tasks: int = 10,
    dim: int = 100,
    n_train: int = 1000,
    n_test: int = 200,
    n_bits: int = 4,
    w_star_scale: float = 3.0,
    seed: int = 42,
) -> list[tuple[QuadraticTaskSpec, QuadraticDataset, QuadraticDataset]]:
    """Slowly-changing regression: identity Hessian, sign-flips of w*.

    Task 0 draws a ±1 vector scaled to ``w_star_scale``. Each later task copies
    w* and flips ``n_bits`` coordinates. Q = I so the only shift is those bits.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
    w = signs * (w_star_scale / np.sqrt(dim))
    Q = torch.eye(dim)
    Q_sqrt = torch.eye(dim)
    tasks = []
    for t in range(n_tasks):
        if t > 0:
            k = max(1, min(int(n_bits), dim))
            idx = rng.choice(dim, size=k, replace=False)
            w = w.copy()
            w[idx] *= -1.0
        spec = QuadraticTaskSpec(
            Q=Q, w_star=torch.tensor(w, dtype=torch.float32),
            Q_sqrt=Q_sqrt, task_id=t,
        )
        rng_train = np.random.default_rng(seed * 10_000 + t * 100 + 1)
        rng_test = np.random.default_rng(seed * 10_000 + t * 100 + 2)
        tasks.append((
            spec,
            QuadraticDataset(spec, n_train, rng_train),
            QuadraticDataset(spec, n_test, rng_test),
        ))
    return tasks


# ---------------------------------------------------------------------------
# Loader-interface compatible with the rest of the repo
# ---------------------------------------------------------------------------

def get_quadratic_tasks(args) -> tuple[
    list[tuple[DataLoader, DataLoader]],
    list[QuadraticTaskSpec],
]:
    """Build (train_loader, test_loader) pairs + task specs from CLI args.

    Mirrors the return signature of get_rotated_tasks / get_clear_tasks for the
    loader pairs, but also returns the per-task QuadraticTaskSpec objects so the
    training script can compute distance-to-optimum and population loss metrics.

    Parameters consumed from args
    ------------------------------
    args.tasks              : int   — number of tasks
    args.quad_dim           : int   — parameter-space dimension (default 100)
    args.quad_n_train       : int   — training samples per task (default 1000)
    args.quad_n_test        : int   — test samples per task (default 200)
    args.quad_eigval_min    : float — minimum eigenvalue (default 0.1)
    args.quad_eigval_max    : float — maximum eigenvalue (default 10.0)
    args.quad_w_star_scale  : float — L2 scale of w* vectors (default 3.0)
    args.batch_size         : int
    args.seed               : int

    Returns
    -------
    loaders : list of (train_loader, test_loader)
    specs   : list of QuadraticTaskSpec, one per task
    """
    if getattr(args, 'dataset_name', '') == 'bitflip':
        tasks_raw = generate_bitflip_tasks(
            n_tasks=args.tasks,
            dim=getattr(args, 'quad_dim', 100),
            n_train=getattr(args, 'quad_n_train', 1000),
            n_test=getattr(args, 'quad_n_test', 200),
            n_bits=int(getattr(args, 'bitflip_n_bits', 4) or 4),
            w_star_scale=getattr(args, 'quad_w_star_scale', 3.0),
            seed=args.seed,
        )
    else:
        tasks_raw = generate_quadratic_tasks(
            n_tasks=args.tasks,
            dim=getattr(args, 'quad_dim',          100),
            n_train=getattr(args, 'quad_n_train',   1000),
            n_test=getattr(args, 'quad_n_test',     200),
            eigenvalue_range=(
                getattr(args, 'quad_eigval_min', 0.1),
                getattr(args, 'quad_eigval_max', 10.0),
            ),
            w_star_scale=getattr(args, 'quad_w_star_scale', 3.0),
            seed=args.seed,
        )

    loaders: list[tuple[DataLoader, DataLoader]] = []
    specs:   list[QuadraticTaskSpec]             = []

    for spec, train_ds, test_ds in tasks_raw:
        train_loader = DataLoader(
            train_ds, shuffle=True,  batch_size=args.batch_size, num_workers=0
        )
        test_loader  = DataLoader(
            test_ds,  shuffle=False, batch_size=args.batch_size, num_workers=0
        )
        loaders.append((train_loader, test_loader))
        specs.append(spec)

    return loaders, specs
