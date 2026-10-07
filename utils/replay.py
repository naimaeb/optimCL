"""Replay buffers and the MFAC-matched rehearsal schedule."""
import numpy as np
import torch


def compute_grads_per_task(num_tasks, fisher_capacity, recency_bias=False, recency_decay=2.0):
    """Per-task gradient quota for the M-FAC Fisher refill.

    Two schedules are supported, controlled by ``recency_bias``:

    * ``False`` (default): every seen task gets the same per-task budget
      ``min(fisher_capacity, 1 + fisher_capacity // num_tasks)`` — equal
      samples per task, no recency bias.

    * ``True``: geometric weighting ``w_j = recency_decay ** j`` with
      ``j = 0`` for the oldest seen task and ``j = num_tasks - 1`` for the most
      recent. The buffer capacity is distributed proportionally to the weights
      and clamped so every seen task contributes at least one batch. ``decay=1.0``
      reproduces the uniform schedule; ``decay>1`` puts more gradient samples on
      recent tasks, ``decay<1`` would favour older tasks.

    Returns a ``list[int]`` of length ``num_tasks`` whose entries are each task's
    per-task batch count. The list sums to at most ``fisher_capacity`` (it may be
    smaller when each task needs at least one batch and ``num_tasks > capacity``).
    """
    if num_tasks <= 0:
        return []
    if num_tasks > fisher_capacity:
        # Not enough capacity for even one per seen task — fall back to one each
        # (and let the integration loop stop naturally when the buffer is full).
        return [1] * num_tasks
    if not recency_bias or num_tasks == 1:
        per = min(fisher_capacity, 1 + fisher_capacity // num_tasks)
        return [per] * num_tasks

    weights = [recency_decay ** j for j in range(num_tasks)]
    total_w = sum(weights)
    quota = [max(1, int(round(fisher_capacity * (w / total_w)))) for w in weights]
    # Capacity correction (rounding may overshoot or undershoot).
    while sum(quota) > fisher_capacity:
        # Prefer trimming from oldest tasks that still have room above the floor.
        trimmed = False
        for i in range(num_tasks):
            if quota[i] > 1:
                quota[i] -= 1
                trimmed = True
                break
        if not trimmed:
            break
    while sum(quota) < fisher_capacity:
        quota[-1] += 1
    return quota


class ReservoirReplayBuffer:
    """Vitter reservoir of ``(x, y, task_id)`` for experience-replay baselines.

    Stores examples on CPU. ``sample_batch`` can restrict to past tasks so the
    current task is not rehearsed against itself. Multi-head sampling draws
    every replay example from one randomly chosen stored task so a single
    ``task_id`` can be passed to the model.
    """

    def __init__(self, capacity, seed=0):
        self.capacity = int(capacity)
        self.rng = np.random.RandomState(seed)
        self.n_seen = 0
        self.filled = 0
        self.x = None
        self.y = None
        self.task_id = None

    def __len__(self):
        return self.filled

    def add_batch(self, x, y, task_id):
        x = x.detach().cpu()
        y = y.detach().cpu()
        if torch.is_tensor(task_id) and task_id.ndim > 0:
            tids = task_id.detach().cpu().long()
            for i in range(x.shape[0]):
                self._add_one(x[i], y[i], int(tids[i].item()))
        else:
            tid = int(task_id)
            for i in range(x.shape[0]):
                self._add_one(x[i], y[i], tid)

    def _add_one(self, xi, yi, task_id):
        if self.x is None:
            self.x = torch.empty((self.capacity,) + tuple(xi.shape), dtype=xi.dtype)
            if yi.ndim == 0:
                self.y = torch.empty((self.capacity,), dtype=yi.dtype)
            else:
                self.y = torch.empty((self.capacity,) + tuple(yi.shape), dtype=yi.dtype)
            self.task_id = torch.empty((self.capacity,), dtype=torch.long)

        if self.filled < self.capacity:
            idx = self.filled
            self.filled += 1
        else:
            j = int(self.rng.randint(0, self.n_seen + 1))
            idx = j if j < self.capacity else None
        if idx is not None:
            self.x[idx] = xi
            self.y[idx] = yi
            self.task_id[idx] = task_id
        self.n_seen += 1

    def sample_batch(self, n, single_task=False, max_task_id=None):
        if self.filled == 0 or n <= 0:
            return None
        mask = torch.ones(self.filled, dtype=torch.bool)
        if max_task_id is not None:
            mask &= self.task_id[:self.filled] <= max_task_id
        pool = mask.nonzero(as_tuple=False).view(-1)
        if pool.numel() == 0:
            return None
        if single_task:
            tids = self.task_id[pool].unique()
            tid = int(tids[int(self.rng.randint(0, len(tids)))].item())
            pool = pool[self.task_id[pool] == tid]
        n = min(int(n), int(pool.numel()))
        choice = self.rng.choice(pool.cpu().numpy(), size=n, replace=False)
        idx = torch.as_tensor(choice, dtype=torch.long)
        return self.x[idx], self.y[idx], self.task_id[idx]


def collect_mfac_matching_batches(
    loaders, crt_task_id_finished, fisher_capacity,
    recency_bias=False, recency_decay=2.0,
):
    """Training minibatches sampled like MFAC Fisher refill.

    After finishing task ``crt_task_id_finished`` (0-based), MFAC integrates a
    per-task budget of gradients using **sequential** minibatches from each
    seen task's train ``DataLoader``. This returns those same ``(x, y)`` batches
    for experience replay (stored on CPU), under exactly the same per-task
    schedule that the Fisher fill uses — see ``compute_grads_per_task``.

    Returns ``(batches, grads_per_task)`` where ``batches`` is a list of
    ``(task_id, x_cpu, y_cpu)`` and ``grads_per_task`` is a ``list[int]`` of the
    per-task batch counts actually scheduled (oldest -> newest).
    """
    num_tasks_to_process = crt_task_id_finished + 1
    grads_per_task = compute_grads_per_task(
        num_tasks_to_process, fisher_capacity,
        recency_bias=recency_bias, recency_decay=recency_decay,
    )
    batches = []
    for task_id in range(num_tasks_to_process):
        k = grads_per_task[task_id]
        train_loader, _ = loaders[task_id]
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx == k:
                break
            batches.append((task_id, x.detach().cpu(), y.detach().cpu()))
    return batches, grads_per_task
