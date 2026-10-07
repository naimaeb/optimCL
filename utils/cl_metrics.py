import numpy as np

from utils.horizon import DATASET_DETAILS, task_length

# see page 27 in this paer: https://openreview.net/pdf?id=_IM-AfFhna9


def cl_metric_avg_acc(R):  # ACC
    npR = np.array(R)
    return np.nanmean(npR[-1])


def cl_metric_backward_transfer(R):
    """Mammoth ``backward_transfer`` (https://github.com/ContinualLearner/Mammoth).

    Mean_i ``R[T-1][i] - R[i][i]`` for ``i`` in ``0 .. T-2``.
    """
    npR = np.asarray(R, dtype=float)
    n_tasks = npR.shape[0]
    li = [npR[-1, i] - npR[i, i] for i in range(n_tasks - 1)]
    return float(np.mean(li))


def cl_metric_forward_transfer(results, random_results):
    """Mammoth ``forward_transfer``.

    Mean_i ``R[i-1][i] - random_results[i][0]`` for ``i`` in ``1 .. T-1``.

    ``random_results`` is Mammoth layout: one row per task, each ``[acc_pct, ...]``.
    """
    npR = np.asarray(results, dtype=float)
    n_tasks = npR.shape[0]
    if n_tasks < 2 or random_results is None:
        return float('nan')
    li = []
    for i in range(1, n_tasks):
        base = random_results[i][0] if isinstance(random_results[i], (list, tuple, np.ndarray)) else random_results[i]
        li.append(npR[i - 1, i] - float(base))
    return float(np.mean(li))


def cl_metric_forgetting(R):
    """Mammoth ``forgetting`` (https://aimagelab.github.io/mammoth/_modules/utils/metrics.html#forgetting).

    After training ``T`` tasks, mean over past tasks ``i = 0 .. T-2`` of
    ``max_t R[t][i] - R[T-1][i]``: drop from the best accuracy ever seen on
    task ``i`` to the accuracy on ``i`` after the last task. Shorter rows are
    padded with zeros like the reference. Operates on a copy — does not mutate
    ``R``. Returns NaN when ``T < 2`` (nothing to forget yet).
    """
    results = [list(np.asarray(row, dtype=float)) for row in R]
    n_tasks = len(results)
    if n_tasks < 2:
        return float('nan')
    for i in range(n_tasks - 1):
        results[i] = results[i] + [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results, dtype=float)
    # NaN = not evaluated (e.g. CLEAR streaming far-future). Mammoth pads
    # missing entries with 0; nanmax agrees with that for non-negative acc.
    maxx = np.nanmax(np_res, axis=0)
    li = [maxx[i] - results[-1][i] for i in range(n_tasks - 1)]
    return float(np.mean(li))


def cl_metric_forgetting_after(R, n_seen):
    """Mammoth forgetting after ``n_seen`` completed tasks.

    Uses the leading ``n_seen x n_seen`` prefix of the accuracy matrix so the
    metric can be logged after every task, not only at the end of the run.
    Returns NaN when ``n_seen < 2``.
    """
    if n_seen < 2:
        return float('nan')
    prefix = [list(row[:n_seen]) for row in R[:n_seen]]
    return cl_metric_forgetting(prefix)


class CLEARMetric:
    """Five metrics defined by the CLEAR paper (Lin et al., NeurIPS 2021).

    Reference: https://clear-benchmark.github.io/

    Ported locally so we don't depend on Avalanche. Given an NxN accuracy
    matrix ``matrix[i][j]`` = test accuracy on task j after training through
    task i:

      * ``in_domain``        : avg of diagonal                 (acc on current bucket)
      * ``next_domain``      : avg of superdiagonal            (acc on immediately-next bucket)
      * ``accuracy``         : avg of diagonal + lower triangle (seen tasks only)
      * ``forward_transfer`` : avg of upper triangle           (unseen/future tasks)
      * ``backward_transfer``: avg of lower triangle           (seen past tasks)
    """

    def get_metrics(self, matrix):
        matrix = np.asarray(matrix)
        assert matrix.shape[0] == matrix.shape[1], f"non-square acc matrix {matrix.shape}"
        return {
            "in_domain":         self.in_domain(matrix),
            "next_domain":       self.next_domain(matrix),
            "accuracy":          self.accuracy(matrix),
            "forward_transfer":  self.forward_transfer(matrix),
            "backward_transfer": self.backward_transfer(matrix),
        }

    @staticmethod
    def accuracy(matrix):
        r, _ = matrix.shape
        vals = [matrix[i, j] for i in range(r) for j in range(i + 1)]
        return np.nanmean(vals)

    @staticmethod
    def in_domain(matrix):
        r, _ = matrix.shape
        return np.nanmean([matrix[i, i] for i in range(r)])

    @staticmethod
    def next_domain(matrix):
        r, _ = matrix.shape
        return np.nanmean([matrix[i, i + 1] for i in range(r - 1)])

    @staticmethod
    def forward_transfer(matrix):
        r, _ = matrix.shape
        vals = [matrix[i, j] for i in range(r) for j in range(i + 1, r)]
        return np.nanmean(vals)

    @staticmethod
    def backward_transfer(matrix):
        r, _ = matrix.shape
        vals = [matrix[i, j] for i in range(r) for j in range(i)]
        return np.nanmean(vals)


def cl_r_summaries(R, chance=0.0, kappa=0.2, H_list=(1, 2, 4, 8),
                   higher_is_better=True, R_ref=None, acc_is_percent=True):
    """Chance-normalised CL scalars from an R matrix (LA, ACC, BWT, ρ, FRGT, W(H)).

    ``R[i, j]`` is performance on task ``j`` after training through task ``i``.
    Accuracies in this repo are stored as percent; pass ``acc_is_percent=True``
    and ``chance`` in ``[0, 1]`` (scalar, per-task ``(T,)``, or per-cell).
    Retention / FRGT use only tasks that actually learned
    (peak ≥ max(κ, 0.5 · ref)). Identity: ``ACC = LA + BWT``.
    There is no init row here — use Mammoth FWT vs the random-init baseline.
    For loss matrices set ``higher_is_better=False`` and ``acc_is_percent=False``.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f'R must be 2-D, got {R.shape}')
    # Optional leading init row: drop it so the remainder is T×T.
    if R.shape[0] == R.shape[1] + 1:
        Rm = R[1:]
    else:
        Rm = R
    if acc_is_percent and higher_is_better:
        Rm = Rm / 100.0
        if R_ref is not None:
            R_ref = np.asarray(R_ref, dtype=float) / 100.0

    T = Rm.shape[0]
    if R_ref is None:
        # Best post-learning score on each task (intransigence reference).
        R_ref = np.array([np.nanmax(Rm[j:, j]) for j in range(T)])
    R_ref = np.asarray(R_ref, dtype=float).reshape(-1)[:T]
    chance = np.asarray(chance, dtype=float)

    if higher_is_better:
        if chance.ndim == 0:
            c = np.full(T, float(chance))
        elif chance.shape == (T,):
            c = chance
        elif chance.shape == Rm.shape:
            c = np.array([chance[j, j] for j in range(T)], dtype=float)
            Rn = (Rm - chance) / np.maximum(1.0 - chance, 1e-12)
            ref = (R_ref - c) / np.maximum(1.0 - c, 1e-12)
            peak = np.array([np.nanmax(Rn[j:, j]) for j in range(T)])
            learned = peak >= np.maximum(kappa, 0.5 * ref)
            return _cl_r_summaries_from_normalized(Rn, learned, peak, T, H_list)
        else:
            raise ValueError(f'chance shape {chance.shape} does not match T={T}')
        # \tilde R = (R - c) / (1 - c), broadcast c_j down each column.
        denom = np.maximum(1.0 - c, 1e-12)
        Rn = (Rm - c[None, :]) / denom[None, :]
        ref = (R_ref - c) / denom
    else:
        # Loss: 1 = matched the column's best (lowest) loss, 0 = no improvement.
        ref = np.maximum(R_ref, 1e-12)
        Rn = (ref[None, :] - Rm) / ref[None, :]
        ref = np.ones_like(R_ref)

    peak = np.array([np.nanmax(Rn[j:, j]) for j in range(T)])
    learned = peak >= np.maximum(kappa, 0.5 * ref)
    return _cl_r_summaries_from_normalized(Rn, learned, peak, T, H_list)


def _cl_r_summaries_from_normalized(Rn, learned, peak, T, H_list=(1, 2, 4, 8)):
    """LA / ACC / BWT / ρ / FRGT / windowed W(H) on an already-normalised matrix."""
    # LA: mean diagonal (plasticity). ACC: last-row mean. BWT: last − diagonal.
    la = float(np.nanmean(np.diag(Rn)))
    acc = float(np.nanmean(Rn[T - 1, :]))
    bwt = float(np.nanmean([Rn[T - 1, j] - Rn[j, j] for j in range(T)]))
    # ρ and FRGT skip tasks that never learned, so "did not fit" is not rewarded.
    idx = [j for j in range(T) if learned[j] and peak[j] > 0]
    rho = float(np.nanmean([Rn[T - 1, j] / peak[j] for j in idx])) if idx else float('nan')
    frg = float(np.nanmean([peak[j] - Rn[T - 1, j] for j in idx if j < T - 1])) if idx else float('nan')
    acc_all = [Rn[i, j] for i in range(T) for j in range(i + 1)]
    next_seg = float(np.nanmean([Rn[i, i + 1] for i in range(T - 1)])) if T > 1 else float('nan')
    frg_val = 0.0 if frg != frg else frg
    out = dict(
        LA=la, ACC=acc, BWT=bwt, retention=rho, FRGT=frg,
        ACC_all=float(np.nanmean(acc_all)) if acc_all else float('nan'),
        n_learned=int(learned.sum()), n_cells_dropped=int((~learned).sum()),
        next_seg=next_seg,
        # Flags the case where FRGT collapses to −BWT (as in Class-IL).
        frgt_degenerate=int(abs(frg_val + bwt) < 0.01 * max(abs(la), 1e-9)),
    )
    # W(H): mean accuracy on the last H learned tasks, averaged over rows.
    for H in H_list:
        rows = []
        for i in range(T):
            cells = [Rn[i, j] for j in range(max(0, i - H + 1), i + 1) if learned[j]]
            if cells:
                rows.append(np.nanmean(cells))
        out[f'W_H{H}'] = float(np.nanmean(rows)) if rows else float('nan')
    return out


def cl_chance_level(n_classes, n_tasks, classes_per_task=0, readout='primary'):
    """Random-guess accuracy c_j used to chance-normalise an R matrix.

    Domain-IL: ``1/C``. Task-IL: ``1/k`` (classes in the masked block).
    Class-IL: ``1 / classes-seen-when-task-j-was-learned`` per column.
    """
    C = max(int(n_classes), 1)
    k = int(classes_per_task or 0)
    if readout in ('til', 'taskil') and k:
        return 1.0 / float(k)
    if k:
        return np.array([1.0 / float(min(C, (j + 1) * k)) for j in range(int(n_tasks))], dtype=float)
    return 1.0 / float(C)


def cl_summaries_wandb(prefix, summary):
    """Flatten a summaries dict to ``prefix/LA``, ``prefix/ACC``, … for W&B."""
    return {f'{prefix}/{key}': value for key, value in summary.items()}


def infer_memory_arm(args):
    """Which memory channels are on: A0 none, A1 optimizer-slow, A3 ER, A6 both.

    ReplaySGD/Muon and AdEMAMix both use ``d = m_fast(β1) + α m_slow(β3)``,
    so the slow channel is on only when ``α≠0`` and ``β3≠0``.
    """
    replay = bool(getattr(args, 'replay', 0)) and int(getattr(args, 'replay_buffer_size', 0) or 0) > 0
    try:
        alpha = float(getattr(args, 'alpha', 0.0) or 0.0)
    except (TypeError, ValueError):
        alpha = 0.0
    beta3 = float(getattr(args, 'beta3', 0.0) or 0.0)
    name = getattr(args, 'optim', '')
    if name in ('ademamix', 'replaysgd', 'replaymuon', 'replayadamw'):
        slow = alpha != 0.0 and beta3 != 0.0
    else:
        slow = False
    if not replay and not slow:
        return 'A0'
    if not replay and slow:
        return 'A1'
    if replay and not slow:
        return 'A3'
    return 'A6'


def optimizer_family_name(optim):
    """CLI ``--optim`` → report name (ReplaySGD/Muon are the slowSGD/slowMuon arms)."""
    return {
        'ademamix': 'AdEMAMix',
        'replaysgd': 'slowSGD',
        'replaymuon': 'slowMuon',
        'sgd': 'SGD',
        'muon': 'Muon',
        'adam': 'Adam',
        'adamw': 'AdamW',
    }.get(optim, optim)


def tau3_over_L(args):
    """Slow-EMA horizon in units of steps-per-task: ``(β/(1-β)) / L``.

    ``β`` is ``beta3``. After ``--horizon_ratio`` this equals ``r``. ``L`` is
    the value stored by ``apply_slow_horizon`` (``cfg_L``), otherwise
    ``ceil(N_task/batch)*epochs`` from DATASET_DETAILS.
    """
    beta = float(getattr(args, 'beta3', 0.0) or 0.0)
    if not (0.0 < beta < 1.0):
        return 0.0
    L = getattr(args, 'cfg_L', None)
    if not L:
        name = getattr(args, 'dataset_name', '')
        if name in DATASET_DETAILS:
            L = task_length(
                name,
                int(getattr(args, 'batch_size', 0) or 0),
                int(getattr(args, 'epochs_per_task', 0) or 0),
                args=args,
                n_tasks=getattr(args, 'tasks', None),
            )
    if not L:
        return float('nan')
    return (beta / (1.0 - beta)) / float(L)
