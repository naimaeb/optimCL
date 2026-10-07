"""Download the CLEAR benchmark directly from the HuggingFace mirror.

No Avalanche dependency. Uses only the Python standard library (urllib +
zipfile). The URLs and file names are the ones published on the official
CLEAR wiki: https://clear-benchmark.github.io/ (linked from
https://linzhiqiu.gitbook.io/the-clear-benchmark/).

Files
-----
Downloaded from the HuggingFace mirror referenced by the CLEAR wiki:

  https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/

    clear10-train.zip    ~3 GB   train images (+ pre-extracted features for
                                 CLEAR-10: moco_b0 / moco_imagenet /
                                 byol_imagenet / imagenet)
    clear10-test.zip   ~200 MB   test images
    clear100-train.zip   ~3 GB   CLEAR-100 train
    clear100-test.zip  ~1.6 GB   CLEAR-100 test

Both zips extract directly into ``<dest>/`` creating sibling ``train/`` and
``test/`` sub-trees (see the on-disk layout comment in ``utils/datasets.py``).

Usage
-----
  # CLEAR-10 (both train and test)
  python scripts/data/download_clear.py --dataset clear10 --dest ./data/clear10_raw

  # CLEAR-100
  python scripts/data/download_clear.py --dataset clear100 --dest ./data/clear100

  # Skip the (large) train zip if you only need eval
  python scripts/data/download_clear.py --dataset clear10 --dest ./data/clear10_raw --splits test

The script is idempotent: it skips any split whose ``labeled_metadata.json``
already exists in the destination tree.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path


_HF_BASE = (
    "https://huggingface.co/datasets/elvishelvis6/"
    "CLEAR-Continual_Learning_Benchmark/resolve/main/"
)

DATASETS = {
    "clear10":  {"train": "clear10-train.zip",  "test": "clear10-test.zip"},
    "clear100": {"train": "clear100-train.zip", "test": "clear100-test.zip"},
}


def _already_present(dest, split):
    """Cheap marker check — avoid re-downloading 3 GB if the split is there."""
    return (Path(dest) / split / "labeled_metadata.json").exists()


def _download_with_progress(url, out_path):
    """``urllib.urlretrieve`` with a simple percentage progress bar."""
    def _hook(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * done / total_size)
            mb_done  = done / (1024 ** 2)
            mb_total = total_size / (1024 ** 2)
            sys.stdout.write(f"\r  {pct:6.2f}%  {mb_done:8.1f} / {mb_total:8.1f} MB")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, out_path, reporthook=_hook)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _extract(zip_path, dest):
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def _download_split(dataset, split, dest, keep_zip):
    if _already_present(dest, split):
        print(f"[{dataset}/{split}] already extracted at {dest}/{split} — skipping.")
        return
    zip_name = DATASETS[dataset][split]
    url = _HF_BASE + zip_name
    zip_path = Path(dest) / zip_name

    print(f"[{dataset}/{split}] downloading {url}")
    _download_with_progress(url, str(zip_path))
    _extract(zip_path, dest)
    if not keep_zip:
        try:
            zip_path.unlink()
        except OSError:
            pass
    print(f"[{dataset}/{split}] done.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="CLEAR variant to download.")
    parser.add_argument("--dest", required=True,
                        help="Destination directory. Zips will be extracted in place "
                             "so that <dest>/train/ and <dest>/test/ exist afterwards.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "test"],
                        help="Which splits to download. Default: both.")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Keep the downloaded .zip files after extraction.")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        _download_split(args.dataset, split, dest, args.keep_zip)

    print()
    print("Done. On-disk layout (see utils/datasets.py for details):")
    print(f"  {dest}/train/  (class_names.txt, labeled_metadata.json, labeled_images/, features/ ...)")
    print(f"  {dest}/test/   (class_names.txt, labeled_metadata.json, labeled_images/ ...)")


if __name__ == "__main__":
    main()
