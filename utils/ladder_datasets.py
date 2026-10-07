"""Additional CL benchmark datasets that are not rot/perm/CLEAR.

Each builder returns ``(train_ds, test_ds)`` for one task. Folder-based
streams (DomainNet, CORe50, CLAD-C, Stream-51, Wild-Time, TiC, CLOC) expect
``dataset_path/<domain_or_session>/`` with either class subfolders or
``train/`` + ``test|val/``. Shared ``class_to_idx`` is taken from the union of
all domains so labels stay aligned across tasks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as F

from utils.datasets import (
    _ClassFilter,
    get_cifar10_dataset,
    get_permmnist_dataset,
    partial_pixel_perm,
)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_TINYIMG_MEAN = (0.4802, 0.4480, 0.3975)
_TINYIMG_STD = (0.2770, 0.2691, 0.2821)
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)

# Default spatial size and recommended backbone for folder streams.
FOLDER_SPECS = {
    'domainnet': {'image_size': 64, 'default_tasks': 6},
    'core50': {'image_size': 128, 'default_tasks': 8},
    'stream51': {'image_size': 128, 'default_tasks': 10},
    'cladc': {'image_size': 128, 'default_tasks': 6},
    'wildtime': {'image_size': 32, 'default_tasks': 9},
    'tic': {'image_size': 224, 'default_tasks': 6},
    'cloc': {'image_size': 224, 'default_tasks': 10},
}

_FIVE_NAMES = ('mnist', 'svhn', 'notmnist', 'fashionmnist', 'cifar10')


class _OffsetLabels(Dataset):
    def __init__(self, base, offset):
        self.base = base
        self.offset = int(offset)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, int(y) + self.offset


class _MappedFolder(Dataset):
    """ImageFolder-like, with a shared class_to_idx and optional index subset."""

    def __init__(self, samples, class_to_idx, transform):
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)


class _MixtureDataset(Dataset):
    """Deterministic mixture of member datasets (Gaussian-scheduled perm-MNIST)."""

    def __init__(self, members, weights, length):
        self.members = members
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0)
        s = w.sum()
        self.cum = np.cumsum(w / s if s > 0 else np.ones_like(w) / len(w))
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        u = (idx * 0.6180339887 + 0.1) % 1.0
        k = int(np.searchsorted(self.cum, u, side='right'))
        k = min(k, len(self.members) - 1)
        j = idx % len(self.members[k])
        return self.members[k][j]


class _GradualRotMNIST(Dataset):
    """MNIST slice whose rotation interpolates from start_deg to end_deg."""

    def __init__(self, base, indices, start_deg, end_deg, to_tensor_norm):
        self.base = base
        self.indices = list(indices)
        self.start = float(start_deg)
        self.end = float(end_deg)
        self.to_tensor_norm = to_tensor_norm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, y = self.base[self.indices[idx]]
        n = max(len(self.indices) - 1, 1)
        deg = self.start + (self.end - self.start) * (idx / n)
        img = F.rotate(img, deg)
        if self.to_tensor_norm is not None:
            img = self.to_tensor_norm(img)
        return img, y


def _image_transform(image_size, is_train, mean, std):
    ops = [transforms.Resize(image_size)]
    if is_train:
        ops.extend([
            transforms.RandomCrop(image_size, padding=max(1, image_size // 16)),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        ops.append(transforms.CenterCrop(image_size))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def _class_dirs(root: Path):
    """Return class-folder names, looking inside train/ if present."""
    train = root / 'train'
    scan = train if train.is_dir() else root
    return sorted(
        p.name for p in scan.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    )


def _collect_samples(root: Path, class_to_idx, split=None):
    if split == 'train' and (root / 'train').is_dir():
        root = root / 'train'
    elif split in ('test', 'val'):
        for name in ('test', 'val'):
            if (root / name).is_dir():
                root = root / name
                break
    samples = []
    for cls_name, y in class_to_idx.items():
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.rglob('*')):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.ppm'}:
                samples.append((str(p), y))
    return samples


def discover_domains(args):
    root = Path(args.dataset_path)
    listed = getattr(args, 'domain_list', None) or ''
    if str(listed).strip():
        names = [n.strip() for n in str(listed).split(',') if n.strip()]
        dirs = [root / n for n in names]
    else:
        dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(('.', '_'))
        )
    missing = [p for p in dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            f'{args.dataset_name}: missing domain folders {missing}. '
            f'Expected subdirectories of {root}.'
        )
    if not dirs:
        raise FileNotFoundError(
            f'{args.dataset_name}: no domain folders under {root}.'
        )
    n_tasks = int(args.tasks)
    if n_tasks > len(dirs):
        raise RuntimeError(
            f'{args.dataset_name}: tasks={n_tasks} but only {len(dirs)} domains in {root}'
        )
    return dirs[:n_tasks]


def shared_class_to_idx(domain_dirs):
    names = set()
    for d in domain_dirs:
        names.update(_class_dirs(d))
    if not names:
        raise FileNotFoundError(f'No class folders under {domain_dirs[0].parent}')
    return {n: i for i, n in enumerate(sorted(names))}


def folder_domain_pair(args, task_id, image_size=None):
    """One domain-IL task from a class-folder tree."""
    domains = discover_domains(args)
    class_to_idx = shared_class_to_idx(domains)
    domain = domains[task_id]
    spec = FOLDER_SPECS.get(args.dataset_name, {'image_size': 128})
    size = int(image_size or getattr(args, 'image_size', 0) or spec['image_size'])
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    seed = int(args.seed)

    has_splits = (domain / 'train').is_dir() and (
        (domain / 'test').is_dir() or (domain / 'val').is_dir()
    )
    if has_splits:
        tr_s = _collect_samples(domain, class_to_idx, 'train')
        te_s = _collect_samples(domain, class_to_idx, 'test')
    else:
        all_s = _collect_samples(domain, class_to_idx, None)
        rng = np.random.RandomState(seed + 31 + task_id)
        order = rng.permutation(len(all_s))
        n_tr = max(1, int(0.8 * len(all_s)))
        tr_s = [all_s[i] for i in order[:n_tr]]
        te_s = [all_s[i] for i in order[n_tr:]] or tr_s[:1]

    print(f'Creating task {task_id} from {domain.name} ({len(tr_s)} train / {len(te_s)} test)')
    train_ds = _MappedFolder(tr_s, class_to_idx, _image_transform(size, True, mean, std))
    test_ds = _MappedFolder(te_s, class_to_idx, _image_transform(size, False, mean, std))
    return train_ds, test_ds


def _rgb32_mnist_transform(is_train):
    ops = [transforms.Grayscale(num_output_channels=3), transforms.Resize(32)]
    if is_train:
        ops.append(transforms.RandomCrop(32, padding=4))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def _load_fivedataset_member(name, args, is_train, offset):
    root = args.dataset_path
    tf = _rgb32_mnist_transform(is_train)
    if name == 'mnist':
        ds = datasets.MNIST(root, train=is_train, download=True, transform=tf)
    elif name == 'fashionmnist':
        ds = datasets.FashionMNIST(root, train=is_train, download=True, transform=tf)
    elif name == 'svhn':
        split = 'train' if is_train else 'test'
        ds = datasets.SVHN(root, split=split, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]))
    elif name == 'cifar10':
        ds = get_cifar10_dataset(args, train=is_train)
    elif name == 'notmnist':
        folder = Path(root) / 'notmnist' / ('train' if is_train else 'test')
        alt = Path(root) / 'notmnist'
        src = folder if folder.is_dir() else alt
        if not src.is_dir():
            raise FileNotFoundError(
                '5-Datasets needs notMNIST at '
                f'{Path(root) / "notmnist"} (ImageFolder, 10 letter classes). '
                'See https://github.com/facebookresearch/Adversarial-Continual-Learning'
            )
        ds = datasets.ImageFolder(str(src), transform=tf)
    else:
        raise ValueError(name)
    return _OffsetLabels(ds, offset)


def parse_task_order(args):
    raw = getattr(args, 'task_order', None) or ''
    if str(raw).strip():
        names = [n.strip().lower() for n in str(raw).split(',') if n.strip()]
    else:
        names = list(_FIVE_NAMES)
    for n in names:
        if n not in _FIVE_NAMES:
            raise ValueError(f'Unknown 5-Datasets member {n!r}; expected {_FIVE_NAMES}')
    if int(args.tasks) != len(names):
        raise RuntimeError(
            f'fivedatasets: tasks={args.tasks} but task_order has {len(names)} names'
        )
    return names


def fivedatasets_pair(args, task_id, task_rng):
    names = parse_task_order(args)
    name = names[task_id]
    offset = 10 * task_id
    print(f'Creating 5-Datasets task {task_id}: {name} (labels {offset}-{offset + 9}, Task-IL mask)')
    return (
        _load_fivedataset_member(name, args, True, offset),
        _load_fivedataset_member(name, args, False, offset),
    )


def _cifar100_dataset(args, is_train):
    if is_train:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
    return datasets.CIFAR100(
        root=args.dataset_path, train=is_train, download=True, transform=tf)


def _tinyimagenet_root(args):
    root = Path(args.dataset_path)
    for cand in (root / 'tiny-imagenet-200', root):
        if (cand / 'train').is_dir():
            return cand
    raise FileNotFoundError(
        f'seqtinyimg: expected tiny-imagenet-200 under {root} '
        '(train/<wnid>/images/*.JPEG).'
    )


def split_class_il_pair(args, task_id, n_classes, loader_train, loader_test):
    """Disjoint class block with global labels; Task-IL masking happens in the loss."""
    cpt = int(getattr(args, 'classes_per_task', 2) or 2)
    start = task_id * cpt
    class_ids = list(range(start, start + cpt))
    if class_ids[-1] >= n_classes:
        raise RuntimeError(
            f'{args.dataset_name}: tasks={args.tasks} x classes_per_task={cpt} '
            f'exceeds {n_classes} classes'
        )
    print(f'Creating task {task_id} with classes {class_ids[0]}-{class_ids[-1]}')
    return _ClassFilter(loader_train, class_ids), _ClassFilter(loader_test, class_ids)


def seqcifar100_pair(args, task_id, task_rng):
    return split_class_il_pair(
        args, task_id, 100,
        _cifar100_dataset(args, True),
        _cifar100_dataset(args, False),
    )


def seqtinyimg_pair(args, task_id, task_rng):
    root = _tinyimagenet_root(args)
    wnids = sorted(p.name for p in (root / 'train').iterdir() if p.is_dir())
    cpt = int(getattr(args, 'classes_per_task', 20) or 20)
    start = task_id * cpt
    class_ids = wnids[start:start + cpt]
    if len(class_ids) < cpt:
        raise RuntimeError('seqtinyimg: not enough classes for this task split')
    class_to_idx = {n: i for i, n in enumerate(wnids)}
    size = int(getattr(args, 'image_size', 0) or 64)
    tr_s, te_s = [], []
    for wnid in class_ids:
        y = class_to_idx[wnid]
        img_dir = root / 'train' / wnid / 'images'
        if not img_dir.is_dir():
            img_dir = root / 'train' / wnid
        files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        )
        rng = np.random.RandomState(int(args.seed) + y)
        order = rng.permutation(len(files))
        n_te = max(1, len(files) // 10)
        te_idx = set(order[:n_te].tolist())
        for i, p in enumerate(files):
            (te_s if i in te_idx else tr_s).append((str(p), y))
    print(
        f'Creating seqtinyimg task {task_id} classes {start}-{start + cpt - 1}'
    )
    train_ds = _MappedFolder(
        tr_s, class_to_idx,
        _image_transform(size, True, _TINYIMG_MEAN, _TINYIMG_STD),
    )
    test_ds = _MappedFolder(
        te_s, class_to_idx,
        _image_transform(size, False, _TINYIMG_MEAN, _TINYIMG_STD),
    )
    return train_ds, test_ds


def iblurry_pair(args, task_id, task_rng):
    """CIFAR-10 i-Blurry: N = % disjoint classes (100 = split, 0 = fully shared)."""
    n_classes = 10
    n_tasks = int(args.tasks)
    n_pct = float(getattr(args, 'iblurry_n', 50.0))
    n_disjoint = int(round(n_pct / 100.0 * n_classes))
    rng = np.random.RandomState(int(args.seed) + 7)
    order = rng.permutation(n_classes)
    disjoint = order[:n_disjoint].tolist()
    shared = order[n_disjoint:].tolist()
    per_task = [[] for _ in range(n_tasks)]
    for i, c in enumerate(disjoint):
        per_task[i % n_tasks].append(int(c))
    class_ids = sorted(set(per_task[task_id] + shared))
    print(
        f'Creating i-Blurry task {task_id}: disjoint={per_task[task_id]} '
        f'shared={shared} (N={n_pct}%)'
    )
    tr, te = get_cifar10_dataset(args, train=True), get_cifar10_dataset(args, train=False)
    return _ClassFilter(tr, class_ids), _ClassFilter(te, class_ids)


def permcontinuous_pair(args, task_id, task_rng):
    """Gaussian mixture of permuted-MNIST tasks. σ→0 recovers hard perm-MNIST."""
    n_tasks = int(args.tasks)
    sigma = float(getattr(args, 'transition_width', 1.0))
    frac = float(getattr(args, 'perm_frac', 1.0))
    members_tr, members_te = [], []
    for k in range(n_tasks):
        rng_k = np.random.RandomState(int(args.seed) * 1009 + k)
        perm = partial_pixel_perm(28 * 28, frac, rng_k)
        tr, te = get_permmnist_dataset(args, perm=perm)
        members_tr.append(tr)
        members_te.append(te)
    xs = np.arange(n_tasks, dtype=np.float64)
    if sigma <= 1e-8:
        w = np.zeros(n_tasks)
        w[task_id] = 1.0
    else:
        w = np.exp(-0.5 * ((xs - task_id) / sigma) ** 2)
        w /= w.sum()
    print(f'Creating permcontinuous task {task_id} sigma={sigma} weights={np.round(w, 3)}')
    n_tr = len(members_tr[task_id])
    n_te = len(members_te[task_id])
    return _MixtureDataset(members_tr, w, n_tr), _MixtureDataset(members_te, w, n_te)


def rotmnist_gradual_pair(args, task_id, task_rng):
    """All 10 classes, rotation increasing inside and across tasks."""
    data_dir = args.dataset_path
    base_tr = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
    base_te = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
    to_tn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
    ])
    delta = float(args.angle)
    start = task_id * delta
    end = (task_id + 1) * delta
    n_tr, n_te = len(base_tr), len(base_te)
    # Slice the index axis so consecutive tasks cover consecutive rotation bands
    # of the *same* digit set (wrap by repeating the full set each task).
    print(f'Creating rotmnist_gradual task {task_id}: angles [{start}, {end})')
    return (
        _GradualRotMNIST(base_tr, range(n_tr), start, end, to_tn),
        _GradualRotMNIST(base_te, range(n_te), start, end, to_tn),
    )


def build_ladder_task(args, task_id, task_rng):
    """Dispatch. Returns None if this dataset is handled elsewhere."""
    name = args.dataset_name
    if name == 'fivedatasets':
        return fivedatasets_pair(args, task_id, task_rng)
    if name == 'seqcifar100':
        return seqcifar100_pair(args, task_id, task_rng)
    if name == 'seqtinyimg':
        return seqtinyimg_pair(args, task_id, task_rng)
    if name == 'iblurry':
        return iblurry_pair(args, task_id, task_rng)
    if name == 'permcontinuous':
        return permcontinuous_pair(args, task_id, task_rng)
    if name == 'rotmnist_gradual':
        return rotmnist_gradual_pair(args, task_id, task_rng)
    if name in FOLDER_SPECS:
        return folder_domain_pair(args, task_id)
    return None
