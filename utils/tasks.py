"""Per-task (train, test) loaders for the rotated / permuted / split / CLEAR streams."""
import numpy as np
from torch.utils.data import DataLoader

# get_dataset() looks up get_<name>_dataset in globals(), so the dataset
# builders must live in this module namespace.
from utils.datasets import *
from utils.datasets import _WithTaskId
from utils.il_protocol import NUM_CLASSES, TASK_IL_DATASETS
from utils.ladder_datasets import build_ladder_task


def get_dataset(args, **kwargs):
    dataset_func_name = f'get_{args.dataset_name}_dataset'
    if dataset_func_name in globals():
        return globals()[dataset_func_name](args, **kwargs)
    raise RuntimeError(f'Cannot load dataset {args.dataset_name}')


def get_rotated_tasks(args):
    """Build per-task (train_loader, test_loader) pairs for domain/task-IL.

    Shift magnitude is dataset-specific (``--angle``, ``--perm_frac``,
    ``--label_perm_frac``, class split). Each task trains on its own set.
    """
    loaders = []
    n_classes = NUM_CLASSES.get(args.dataset_name, 10)
    classes_per_task = int(getattr(args, 'classes_per_task', 2) or 2)

    if (
        args.dataset_name == 'rotmnist_cyclic'
        and str(getattr(args, 'stream_type', '') or '')
    ):
        from utils.rotmnist_cyclic import bind_cyclic_schedule
        bind_cyclic_schedule(args)

    for task_id in range(args.tasks):
        task_rng = np.random.RandomState(int(args.seed) * 1009 + task_id)
        ladder = build_ladder_task(args, task_id, task_rng)
        if ladder is not None:
            train_data, test_data = ladder
        elif args.dataset_name == 'rotmnist':
            rot_deg = task_id * args.angle
            print(f'Creating task {task_id} rotated with {rot_deg} degrees')
            train_data, test_data = get_rotmnist_dataset(args, rotation_degrees=rot_deg)
        elif args.dataset_name == 'rotmnist_cyclic':
            if getattr(args, 'cyclic_angles', None) is not None:
                rot_deg = int(args.cyclic_angles[task_id])
                print(
                    f'Creating {args.cfg_stream_type} task {task_id} '
                    f'rotated with {rot_deg} degrees'
                )
                train_data, test_data = get_rotmnist_cyclic_dataset(
                    args, rotation_degrees=rot_deg)
            else:
                period = int(getattr(args, 'cycle_period', 0) or 0)
                if period < 2:
                    raise RuntimeError(
                        'rotmnist_cyclic requires --stream_type cyclic|onepass '
                        'or legacy --cycle_period >= 2'
                    )
                rot_deg = (task_id % period) * args.angle
                print(
                    f'Creating cyclic task {task_id} (period={period}, '
                    f'revisit={task_id // period}) rotated with {rot_deg} degrees'
                )
                train_data, test_data = get_rotmnist_dataset(args, rotation_degrees=rot_deg)
        elif args.dataset_name == 'rotcifar10':
            rot_deg = task_id * args.angle
            print(f'Creating task {task_id} rotated with {rot_deg} degrees')
            train_data, test_data = get_rotcifar10_dataset(args, rotation_degrees=rot_deg)
        elif args.dataset_name == 'permmnist':
            frac = float(getattr(args, 'perm_frac', 1.0))
            perm = partial_pixel_perm(28 * 28, frac, task_rng)
            print(f'Creating task {task_id} with perm_frac={frac}')
            train_data, test_data = get_permmnist_dataset(args, perm=perm)
        elif args.dataset_name == 'labelpermmnist':
            frac = float(getattr(args, 'label_perm_frac', 1.0))
            if task_id == 0:
                mapping = np.arange(n_classes)
            else:
                mapping = partial_label_map(n_classes, frac, task_rng)
            print(f'Creating task {task_id} with label_perm_frac={frac}')
            train_data, test_data = get_labelpermmnist_dataset(args, label_map=mapping)
        elif args.dataset_name == 'seqcifar10':
            start = task_id * classes_per_task
            class_ids = list(range(start, start + classes_per_task))
            if class_ids[-1] >= n_classes:
                raise RuntimeError(
                    f'seqcifar10: tasks={args.tasks} x classes_per_task='
                    f'{classes_per_task} exceeds {n_classes} classes'
                )
            print(f'Creating task {task_id} with classes {class_ids}')
            train_data, test_data = get_seqcifar10_dataset(args, class_ids=class_ids)
        elif args.dataset_name == 'mnist':
            train_data, test_data = get_mnist_dataset(args)
        else:
            raise RuntimeError(
                f'get_rotated_tasks does not support dataset_name={args.dataset_name!r}'
            )

        if args.dataset_name in TASK_IL_DATASETS:
            train_data = _WithTaskId(train_data, task_id)

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=args.batch_size,
            num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=args.batch_size,
            num_workers=4, pin_memory=True, persistent_workers=True)
        loaders.append((train_loader, test_loader))

    return loaders


def get_clear_tasks(args):
    """Returns ``[(train_loader, test_loader), ...]`` for each CLEAR bucket.

    Drop-in analogue of :func:`get_rotated_tasks` for the CLEAR benchmark.
    Reads the dataset directly from the on-disk layout documented in
    :mod:`utils.datasets` (downloaded via ``scripts/data/download_clear.py``).
    No dependency on Avalanche.

    CLEAR-10/100 ship with a "seed" bucket 0 plus task-buckets 1..N.
    Following the CLEAR paper we skip bucket 0 and iterate buckets 1..args.tasks.

    DataLoader workers:
      * image mode: ``num_workers=4`` with ``persistent_workers=True`` so
        JPEG decoding is parallelised across epochs.
      * feature mode: ``num_workers=0`` — features live in RAM, so workers
        would just fork-duplicate the tensors with no throughput gain.
    """
    ft = getattr(args, 'clear_feature_type', None)
    is_feature_mode = not ((ft is None) or (isinstance(ft, str) and ft.lower() in ('none', 'null', '')))
    num_workers = 0 if is_feature_mode else 4
    persistent = (not is_feature_mode)

    loaders = []
    for task_id in range(args.tasks):
        bucket_id = task_id + 1  # CLEAR buckets are 1-indexed (skip seed bucket 0)
        print(f'Creating task {task_id} from CLEAR bucket {bucket_id}')
        train_data, test_data = get_clear_dataset(args, bucket_id=bucket_id)

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=args.batch_size,
            num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=args.batch_size,
            num_workers=num_workers, pin_memory=True, persistent_workers=persistent)

        loaders.append((train_loader, test_loader))
    return loaders
