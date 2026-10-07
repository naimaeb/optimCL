"""Evaluation: test loop, random-init baseline, per-class accuracy, feature dumps."""
from os.path import join

import numpy as np
import torch
from torch.nn.functional import cross_entropy as CE, mse_loss
from torch.utils.data import DataLoader

from utils.il_protocol import NUM_CLASSES, mask_task_il_logits
from utils.models import MultiHeadMLP


def _is_regression(dataset_name):
    return dataset_name in ('quadratics', 'bitflip')


@torch.no_grad()
def mammoth_random_results_baseline(model, args, task_test_datasets, use_multihead_model,
                                    task_il_cpt=0):
    """Per-task test accuracy at initialization — Mammoth ``random_results[i][0]`` baseline.

    Evaluates the current ``model`` weights (typically right after construction under a
    fixed seed) on each task's test set. Returns Mammoth layout ``[[acc_0], [acc_1], ...]``.
    ``task_il_cpt > 0`` masks the logits to task ``i``'s class block, which is the
    baseline the Task-IL FWT must be compared against.

    For regression streams (quadratics / bitflip), *accuracy* is a pseudo-score
    ``100 * mean(exp(-(y_hat - y)^2))`` so ACC/BWT/FWT/FRGT use the same interface as
    classification benchmarks.
    """
    regression = _is_regression(args.dataset_name)
    model.eval()
    random_results = []
    for task_id in range(len(task_test_datasets)):
        test_ds = task_test_datasets[task_id]
        n_workers = 0 if regression else 4
        loader = DataLoader(
            test_ds, shuffle=False, batch_size=args.batch_size,
            num_workers=n_workers,
            pin_memory=not regression, persistent_workers=n_workers > 0)
        test_correct, _, test_size = test(
            model, loader, task_id=0, regression=regression,
            task_il_cpt=task_il_cpt, mask_task_id=task_id)
        acc_pct = round(test_correct / test_size * 100, 2)
        random_results.append([acc_pct])
        del loader
    return random_results


def save_mammoth_random_baseline(output_folder, random_results,
                                 filename='mammoth_random_baseline.txt'):
    """Write ``filename`` (one ``task_i: acc`` line per task)."""
    if random_results is None:
        return
    with open(join(output_folder, filename), 'w') as w:
        for i, row in enumerate(random_results):
            w.write(f'task{i}: {float(row[0]):.4f}\n')


@torch.no_grad()
def test(model, loader, task_id, regression=False, task_il_cpt=0, mask_task_id=None):
    loss, correct, test_dataset_size = 0, 0, 0
    device = next(model.parameters()).device
    mask_id = task_id if mask_task_id is None else mask_task_id
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_hat = model(x, task_id, out='logits')
        if task_il_cpt:
            y_hat = mask_task_il_logits(y_hat, mask_id, task_il_cpt)
        test_dataset_size += x.shape[0]

        if regression:
            if y_hat.dim() > 1:
                y_hat = y_hat.squeeze(-1)
            loss += mse_loss(y_hat, y, reduction='sum').item()
            correct += torch.exp(-(y_hat - y) ** 2).sum().item()
        else:
            loss += CE(y_hat, y, reduction='sum').item()
            pred = y_hat.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return correct, loss, test_dataset_size


@torch.no_grad()
def compute_accuracy_per_class(model, loader, task_id, dataset_name, path, template_name):
    model.eval()
    num_classes = NUM_CLASSES[dataset_name]

    correct_per_class = [0] * num_classes
    samples_per_class = [0] * num_classes

    confusion_matrix = [[0] * num_classes for i in range(num_classes)]

    for x, y in loader:
        x, y = x.cuda().to(torch.float), y.cuda()

        yhat = model(x, task_id, out='logits')
        pred = yhat.argmax(dim=1, keepdim=True)

        # for i in range(y.shape[0]):
        for y_correct, y_pred in zip(y, pred):
            yc = y_correct.item()
            yp = y_pred.item()

            confusion_matrix[yc][yp] += 1

            for c in range(num_classes):
                if yc == c:  # the correct class is c
                    samples_per_class[c] += 1  # we found one sample in class c, increment
                    if yc == yp:  # equivalent to yc == yp == c: if this sample has class c and is the same as prediction
                        correct_per_class[c] += 1  # increment

    accuracy_per_class = [0] * num_classes
    for c in range(num_classes):
        accuracy_per_class[c] = correct_per_class[c] / samples_per_class[c] * 100.
    if path is not None:
        torch.save(accuracy_per_class, join(path, f'{template_name}-class-acc.pt'))

    with open(join(path, f'{template_name}-detailed-info.txt'), 'w') as w:
        w.write(f'{correct_per_class=}\n')
        w.write(f'{samples_per_class=}\n')
        w.write(f'accuracy: {sum(correct_per_class) / sum(samples_per_class) * 100.:.2f} %\n')
        for c in range(num_classes):
            w.write(f'accuracy for class {c}: {accuracy_per_class[c]:.2f} %\n')
        w.write('Confusion Matrix:\n')
        for i in range(num_classes):
            for j in range(num_classes):
                w.write(f'{confusion_matrix[i][j]: 5d}')
            w.write(f'\n')


@torch.no_grad()
def extract_and_save_features(model, loader, task_id, dataset_name, path, template_name):
    num_classes = NUM_CLASSES[dataset_name]
    class_features = [[] for _ in range(num_classes)] # list with index c stores all features for class c
    printed = False
    for x, y in loader:
        x, y = x.cuda(), y.cuda()

        if dataset_name in ['permmnist', 'rotmnist', 'mnist'] and isinstance(model, MultiHeadMLP):
            x = x.view(x.shape[0], -1)
        # breakpoint()
        features = model(x, task_id, out='features') # shape (B, *others)
        features_linear = features.reshape(features.shape[0], -1)
        if not printed:
            printed = True
            # print(f'{features_linear.shape=}')
        for i in range(features.shape[0]):
            c = y[i].item()
            class_features[c].append(features_linear[i, :].cpu().numpy())
    printed = False

    for c in range(num_classes):
        stacked = np.vstack(class_features[c])
        if not printed:
            printed = True
            # print(f'{stacked.shape=}')
        torch.save(stacked, join(path, f'{template_name}-features-class-{c}.pt'))
        # print(f'Saved features for class {c}')
