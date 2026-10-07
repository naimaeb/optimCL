"""Class counts per dataset and the Class-IL / Task-IL / Domain-IL protocol."""
import torch


NUM_CLASSES = dict(
    mnist=10, rotmnist=10, rotmnist_gradual=10, rotmnist_cyclic=10, permmnist=10,
    permcontinuous=10, labelpermmnist=10,
    imagenette=10, imagewoof=10,
    cifar10=10, cifar100=100, rotcifar10=10,
    seqcifar10=10, seqcifar100=100, seqtinyimg=200, iblurry=10,
    fivedatasets=50,
    domainnet=345, core50=50, stream51=51, cladc=10,
    wildtime=2, tic=1000, cloc=1000,
    clear10=11, clear100=100,
    clear10_neurips2021=11, clear100_cvpr2022=100,
    quadratics=1, bitflip=1,
)


# Split-class datasets: one Linear(*, C) like Mammoth, trained Class-IL
# (unmasked CE over all C logits). Evaluation reports Class-IL and, by masking
# the logits to task t's class block, Task-IL from the same run (see
# apply_il_protocol). Domain-IL streams keep the same label set and need no mask.
TASK_IL_DATASETS = frozenset({
    'seqcifar10', 'seqcifar100', 'seqtinyimg', 'fivedatasets',
})


def task_il_block_size(args):
    """Classes per task for logit masking, or 0 if this dataset is Domain-IL."""
    name = getattr(args, 'dataset_name', None)
    if name not in TASK_IL_DATASETS:
        return 0
    if name == 'fivedatasets':
        return 10
    return int(getattr(args, 'classes_per_task', 2) or 2)


def mask_task_il_logits(logits, task_id, classes_per_task):
    """Set logits outside task ``task_id``'s class block to ``-inf``.

    Same weight matrix as Mammoth's ``Linear(feat, C)``. Masked softmax puts
    zero CE gradient on the other class rows, so those output weights are
    frozen and forgetting lives in the shared trunk. ``task_id`` may be a
    scalar or a ``(B,)`` tensor for mixed batches.
    """
    k = int(classes_per_task or 0)
    if k <= 0:
        return logits
    if torch.is_tensor(task_id) and task_id.ndim > 0:
        idx = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
        start = task_id.long().unsqueeze(1) * k
        keep = (idx >= start) & (idx < start + k)
        return logits.masked_fill(~keep, float('-inf'))
    start = int(task_id) * k
    end = start + k
    out = logits.clone()
    if start > 0:
        out[..., :start] = float('-inf')
    if end < out.shape[-1]:
        out[..., end:] = float('-inf')
    return out


def apply_il_protocol(args):
    """Single Linear(*, C). Split datasets: Class-IL train, Class-IL + Task-IL eval."""
    args.use_multihead_model = 0
    name = args.dataset_name
    k = task_il_block_size(args)
    if k:
        n_logits = NUM_CLASSES.get(name, k * int(args.tasks))
        print(
            f'IL protocol: {name}: single Linear(*, {n_logits}), unmasked CE at train '
            f'(Class-IL). Eval logs Class-IL (primary) and Task-IL (mask to {k} logits, '
            f'chance={100.0 / k:.1f}%).'
        )
    elif name in ('quadratics', 'bitflip'):
        print(f'IL protocol: shared-parameter regression on {name}.')
    else:
        print(
            f'IL protocol: Domain-IL on {name}: single shared head, '
            'no task id (same label set, input distribution shifts).'
        )
    return args
