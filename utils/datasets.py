import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

from utils.transforms import BilinearTensorRotation, FixedRotation, Permutation

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)
_CIFAR10_RGB_MEANS = (0.491, 0.482, 0.447)
_CIFAR10_RGB_STDS = (0.247, 0.243, 0.262)

# ImageNet statistics used for raw-image CLEAR (ResNet/ViT-style backbones).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_mnist_dataset(args, **kwargs):
    data_dir = args.dataset_path
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=test_transform)

    return train_dataset, test_dataset

def get_rotmnist_dataset(args, **kwargs):
    assert 'rotation_degrees' in kwargs.keys()
    degrees = kwargs['rotation_degrees']

    data_dir = args.dataset_path

    transform = transforms.Compose([
        FixedRotation(degrees),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)

    return train_dataset, test_dataset

def get_rotmnist_cyclic_dataset(args, **kwargs):
    """Same 60k/10k MNIST images at one fixed angle.

    Rotation is bilinear on the raw tensor and happens before normalization.
    It is a deterministic task transform, not a random augmentation, so a
    reservoir that stores the loader output matches applying that transform
    again on retrieval.
    """
    assert 'rotation_degrees' in kwargs.keys()
    degrees = kwargs['rotation_degrees']
    data_dir = args.dataset_path
    transform = BilinearTensorRotation(degrees, _MNIST_MEAN, _MNIST_STD)
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)
    return train_dataset, test_dataset

def get_permmnist_dataset(args, **kwargs):
    assert 'perm' in kwargs.keys()
    perm = kwargs['perm']
    data_dir = args.dataset_path
    transform = transforms.Compose([Permutation(perm), transforms.ToTensor(), transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)
    return train_dataset, test_dataset

def get_rotcifar10_dataset(args, **kwargs):
    assert 'rotation_degrees' in kwargs.keys()
    degrees = kwargs['rotation_degrees']

    data_dir = args.dataset_path

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        FixedRotation(degrees),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


# =============================================================================
# Controllable task-shift helpers (CL benchmark ladder)
# =============================================================================
# Shift magnitude: --angle / --perm_frac / --label_perm_frac / class split.
# Split CIFAR / TinyImageNet / 5-Datasets are Task-IL (masked CE on a single
# Linear(*, C)). Rot/Perm-MNIST, CLEAR, and folder streams are Domain-IL.


class _RemapLabels(Dataset):
    """Apply a fixed integer label map (concept shift / label permutation)."""

    def __init__(self, base, mapping):
        self.base = base
        self.mapping = np.asarray(mapping, dtype=np.int64)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, int(self.mapping[int(y)])


class _WithTaskId(Dataset):
    """Attach a constant task id so mixed Task-IL batches can mask the right logit block."""

    def __init__(self, base, task_id):
        self.base = base
        self.task_id = int(task_id)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx][:2]
        return x, y, self.task_id


class _ClassFilter(Dataset):
    """Keep only examples whose label is in ``class_ids`` (split CIFAR)."""

    def __init__(self, base, class_ids):
        self.base = base
        allowed = np.asarray(list(class_ids), dtype=np.int64)
        if hasattr(base, 'targets'):
            targets = np.asarray(base.targets)
        elif hasattr(base, 'labels'):
            targets = np.asarray(base.labels)
        else:
            targets = np.array([int(base[i][1]) for i in range(len(base))])
        self.indices = np.nonzero(np.isin(targets, allowed))[0]

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        return self.base[int(self.indices[idx])]


def partial_pixel_perm(n_pixels, frac, rng):
    """Identity permutation with a random shuffle of ``frac`` of the indices."""
    perm = np.arange(n_pixels)
    k = int(round(float(frac) * n_pixels))
    if k <= 1:
        return perm
    chosen = rng.choice(n_pixels, size=k, replace=False)
    perm[chosen] = rng.permutation(chosen)
    return perm


def partial_label_map(n_classes, frac, rng):
    """Identity label map with a random shuffle of ``frac`` of the class ids."""
    mapping = np.arange(n_classes)
    k = int(round(float(frac) * n_classes))
    if k <= 1:
        return mapping
    chosen = rng.choice(n_classes, size=k, replace=False)
    mapping[chosen] = rng.permutation(chosen)
    return mapping


def get_cifar10_dataset(args, **kwargs):
    """Unsplit CIFAR-10. ``kwargs['train']`` selects the torchvision split."""
    data_dir = args.dataset_path
    is_train = bool(kwargs.get('train', True))
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS),
        ])
    return datasets.CIFAR10(root=data_dir, train=is_train, download=True, transform=transform)


def get_seqcifar10_dataset(args, **kwargs):
    """Split CIFAR-10: disjoint class blocks, global labels (Class-IL train)."""
    assert 'class_ids' in kwargs, 'get_seqcifar10_dataset requires class_ids'
    class_ids = kwargs['class_ids']
    train_full = get_cifar10_dataset(args, train=True)
    test_full = get_cifar10_dataset(args, train=False)
    return _ClassFilter(train_full, class_ids), _ClassFilter(test_full, class_ids)


def get_labelpermmnist_dataset(args, **kwargs):
    """MNIST with a per-task label permutation (inputs fixed, targets reassigned)."""
    assert 'label_map' in kwargs, 'get_labelpermmnist_dataset requires label_map'
    train_ds, test_ds = get_mnist_dataset(args)
    mapping = kwargs['label_map']
    return _RemapLabels(train_ds, mapping), _RemapLabels(test_ds, mapping)


# =============================================================================
# CLEAR (Continual LEArning on Real-world imagery) - NeurIPS'21 / CVPR'22
# =============================================================================
# Download
# --------
# Grab the official zip files from the CLEAR wiki
# (https://clear-benchmark.github.io/). A standalone downloader lives at
# ``scripts/data/download_clear.py``; it only needs urllib + zipfile, no Avalanche.
#
#   clear10-train.zip  (~3 GB)  train images (+ MoCo features for some variants)
#   clear10-test.zip   (~200MB) test images
#   clear100-train.zip (~3 GB)
#   clear100-test.zip  (~1.6GB)
#
# Extract both into the same directory. The resulting on-disk tree is:
#
#   <root>/
#     train/
#       class_names.txt                     one class per line (11 for CLEAR-10,
#                                           100 for CLEAR-100; includes a
#                                           "BACKGROUND" class for CLEAR-10).
#       labeled_metadata.json               { "<bucket>": { "<class>": "labeled_metadata/<bucket>/<class>.json", ... }, ... }
#       labeled_metadata/<bucket>/<class>.json
#                                           { "<img_id>": { "IMG_PATH": "labeled_images/<bucket>/<class>/<file>.jpg", ... }, ... }
#       labeled_images/<bucket>/<class>/*.jpg
#       all_metadata.json, all_metadata/... (optional; unlabeled images)
#       features/<feature_type>/features.json
#                                           same shape as labeled_metadata.json but
#                                           values are paths to .pth tensor dicts.
#       features/<feature_type>/<bucket>/<class>.pth
#                                           { "<img_id>": Tensor(D,), ... } where
#                                           D=2048 for all released backbones
#                                           (moco_b0, moco_imagenet, byol_imagenet,
#                                           imagenet).
#     test/
#       (same layout, but no features/ and buckets start at 0)
#
# Buckets
# -------
# CLEAR-10/100 ship with N+1 time-buckets: a "seed" bucket 0 and task buckets
# 1..N (N=10 for CLEAR-10, N=10 for CLEAR-100). The CL protocol in the paper
# trains on buckets 1..N; bucket 0 is only used for optional unsupervised
# pretraining on its unlabeled images. This module therefore exposes buckets
# 1..N by default (see :func:`get_clear_dataset`).
#
# Splits
# ------
# The wiki distinguishes two protocols:
#   * "iid"       : 70/30 per-bucket split of the labeled train/ images.
#   * "streaming" : train on the full labeled bucket, evaluate on future
#                   buckets. The official "streaming" way is to use the
#                   train/ bucket for training and the test/ bucket for eval.
# We implement streaming with the separate train/ and test/ directory trees.
# The "iid" protocol is not implemented here, but the data loaders below make
# it trivial to add by subsampling the `samples` list.


class _ClearImageBucket(torch.utils.data.Dataset):
    """One CLEAR time-bucket of raw labeled images.

    Lazy PIL decoding (like ``ImageFolder``) so DataLoader workers can
    parallelise JPEG decoding. ``samples`` is a list of
    ``(path_relative_to_root, class_idx)`` tuples and ``transform`` is a
    torchvision transform applied per sample.
    """

    def __init__(self, samples, root, transform=None):
        self.samples = samples
        self.root = Path(root)
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        relpath, target = self.samples[idx]
        img = self.loader(str(self.root / relpath))
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class _ClearFeatureBucket(torch.utils.data.Dataset):
    """One CLEAR time-bucket of pre-extracted feature vectors held in RAM."""

    def __init__(self, tensors, targets):
        self.tensors = tensors  # (N, D) float
        self.targets = targets  # (N,) long

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.tensors[idx], int(self.targets[idx].item())


def _clear_image_transforms():
    """Return (train_transform, eval_transform) for 224x224 ResNet/ViT backbones.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return train_transform, eval_transform


def _clear_is_feature_mode(args):
    """``True`` iff pre-extracted features (``moco_b0`` etc.) were requested."""
    ft = getattr(args, 'clear_feature_type', None)
    return not ((ft is None) or (isinstance(ft, str) and ft.lower() in ('none', 'null', '')))


def _rebuild_clear_metadata_index(split_dir: Path) -> dict:
    """Rebuild ``labeled_metadata.json`` by scanning ``labeled_metadata/<bucket>/*.json``.

    The CLEAR zips ship this index, but it is occasionally missing or deleted
    while the per-class JSON files remain. Reconstructing it is cheap and keeps
    training unblocked without a full re-download.
    """
    meta_root = split_dir / 'labeled_metadata'
    if not meta_root.is_dir():
        raise FileNotFoundError(
            f"CLEAR split is missing both labeled_metadata.json and the "
            f"labeled_metadata/ directory under {split_dir}. Re-download with "
            f"`python scripts/data/download_clear.py --dataset clear10 "
            f"--dest {split_dir.parent} --splits {split_dir.name}`."
        )
    meta = {}
    for bucket_dir in sorted(meta_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not bucket_dir.is_dir():
            continue
        class_map = {
            p.stem: f'labeled_metadata/{bucket_dir.name}/{p.name}'
            for p in sorted(bucket_dir.glob('*.json'))
        }
        if class_map:
            meta[bucket_dir.name] = class_map
    if not meta:
        raise FileNotFoundError(
            f"CLEAR {split_dir}/labeled_metadata/ contains no per-class JSON files. "
            f"Re-download the {split_dir.name} split."
        )
    index_path = split_dir / 'labeled_metadata.json'
    with open(index_path, 'w') as f:
        json.dump(meta, f)
    return meta


def _load_clear_metadata_index(root, split) -> dict:
    """Load ``<root>/<split>/labeled_metadata.json``, rebuilding it if absent."""
    root = Path(root)
    split_dir = root / split
    index_path = split_dir / 'labeled_metadata.json'
    if index_path.is_file():
        with open(index_path) as f:
            return json.load(f)
    return _rebuild_clear_metadata_index(split_dir)


def _read_class_names(root):
    """Read CLEAR class names, falling back to the metadata index if needed.

    Official layout stores ``train/class_names.txt``. If that file was deleted
    but ``labeled_metadata.json`` (or the per-class JSON tree) is intact, recover
    the class list from the first available bucket so training can proceed.
    """
    root = Path(root)
    names_path = root / 'train' / 'class_names.txt'
    if names_path.is_file():
        with open(names_path) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names

    # Prefer train metadata; fall back to test if train index is also gone.
    for split in ('train', 'test'):
        try:
            meta = _load_clear_metadata_index(root, split)
        except FileNotFoundError:
            continue
        for bucket_id in sorted(meta.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k)):
            names = list(meta[bucket_id].keys())
            if names:
                # Persist the recovered list so subsequent runs don't re-scan.
                names_path.parent.mkdir(parents=True, exist_ok=True)
                with open(names_path, 'w') as f:
                    f.write('\n'.join(names) + '\n')
                return names

    raise FileNotFoundError(
        f"CLEAR class names missing at {names_path} and could not be recovered "
        f"from labeled metadata under {root}. Re-download with "
        f"`python scripts/data/download_clear.py --dataset clear10 --dest {root}`."
    )


def _collect_clear_image_samples(root, split, bucket_id, class_names):
    """Return ``[(relpath_from_root, class_idx), ...]`` for one bucket/split.

    Reads ``<root>/<split>/labeled_metadata.json`` and follows the per-class
    metadata JSONs; each entry's ``IMG_PATH`` is relative to ``<root>/<split>``.
    If the index JSON is missing it is rebuilt from ``labeled_metadata/``.
    """
    root = Path(root)
    meta = _load_clear_metadata_index(root, split)
    if str(bucket_id) not in meta:
        raise RuntimeError(
            f"CLEAR {split}/ is missing bucket {bucket_id} "
            f"(available: {sorted(int(k) for k in meta.keys())})."
        )
    class_to_relpath = meta[str(bucket_id)]
    samples = []
    for class_idx, cname in enumerate(class_names):
        if cname not in class_to_relpath:
            raise FileNotFoundError(
                f"CLEAR {split}/ bucket {bucket_id} is missing class '{cname}' "
                f"(have: {sorted(class_to_relpath)}). The split looks incomplete — "
                f"re-download with `python scripts/data/download_clear.py "
                f"--dest {root} --splits {split}`."
            )
        with open(root / split / class_to_relpath[cname]) as mf:
            class_md = json.load(mf)
        for v in class_md.values():
            samples.append((os.path.join(split, v['IMG_PATH']), class_idx))
    return samples


def _collect_clear_feature_bucket(root, feature_type, bucket_id, class_names):
    """Load one bucket of pre-extracted features fully into RAM.

    Reads ``<root>/train/features/<feature_type>/features.json`` (same schema
    as ``labeled_metadata.json`` but the values are paths to ``.pth`` tensor
    dicts). Returns ``(tensors: (N, D) float, targets: (N,) long)``.
    """
    root = Path(root)
    feat_dir = root / 'train' / 'features' / feature_type
    if not feat_dir.exists():
        raise FileNotFoundError(
            f"Feature folder {feat_dir} missing. "
            f"Re-download with `scripts/data/download_clear.py --feature_type {feature_type}` "
            f"or pick a variant of CLEAR whose train zip ships features."
        )
    with open(feat_dir / 'features.json') as f:
        meta = json.load(f)
    if str(bucket_id) not in meta:
        raise RuntimeError(
            f"CLEAR features are missing bucket {bucket_id} "
            f"(available: {sorted(int(k) for k in meta.keys())})."
        )
    class_to_relpath = meta[str(bucket_id)]
    tensors_list, targets_list = [], []
    for class_idx, cname in enumerate(class_names):
        t_dict = torch.load(root / 'train' / class_to_relpath[cname], map_location='cpu')
        for _id, t in t_dict.items():
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)
            tensors_list.append(t)
            targets_list.append(class_idx)
    tensors = torch.stack(tensors_list).float()
    targets = torch.as_tensor(targets_list, dtype=torch.long)
    return tensors, targets


def get_clear_dataset(args, **kwargs):
    """Return ``(train_dataset, test_dataset)`` for a single CLEAR bucket.

    Mirrors the ``get_rotmnist_dataset`` / ``get_rotcifar10_dataset`` signature
    used by :func:`utils.tasks.get_rotated_tasks`, so CLEAR can be plugged into
    the existing training loop via a small wrapper that iterates buckets.

    Keyword arguments
    -----------------
    bucket_id : int  (required)
        Bucket index in [1, N]. Bucket 0 (seed) is skipped on purpose.

    Modes
    -----
    * Image mode (``args.clear_feature_type`` is ``None`` / ``'none'``):
        returns ``_ClearImageBucket`` datasets with ImageNet-style augmentation
        for train and a Resize+CenterCrop pipeline for test. The train and
        test datasets come from the separate ``train/`` and ``test/`` on-disk
        directory trees (streaming protocol).

    * Feature mode (``args.clear_feature_type == 'moco_b0'`` etc.):
        returns ``_ClearFeatureBucket`` datasets. Features only exist for the
        training buckets, so the *same* in-RAM tensor is returned for both
        train and test (matches Avalanche's default for feature+streaming).
    """
    assert 'bucket_id' in kwargs, "get_clear_dataset requires bucket_id=<int>"
    bucket_id = int(kwargs['bucket_id'])
    root = args.dataset_path
    class_names = _read_class_names(root)

    if _clear_is_feature_mode(args):
        tensors, targets = _collect_clear_feature_bucket(
            root=root,
            feature_type=args.clear_feature_type,
            bucket_id=bucket_id,
            class_names=class_names,
        )
        train_ds = _ClearFeatureBucket(tensors, targets)
        test_ds  = _ClearFeatureBucket(tensors, targets)
        return train_ds, test_ds

    train_transform, eval_transform = _clear_image_transforms()
    train_samples = _collect_clear_image_samples(root, 'train', bucket_id, class_names)
    test_samples  = _collect_clear_image_samples(root, 'test',  bucket_id, class_names)
    train_ds = _ClearImageBucket(train_samples, root, transform=train_transform)
    test_ds  = _ClearImageBucket(test_samples,  root, transform=eval_transform)
    return train_ds, test_ds
