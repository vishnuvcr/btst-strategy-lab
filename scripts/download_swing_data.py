from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

import kagglehub
import yaml


def _copy_csv_tree(src: Path, destination: Path) -> int:
    """Copy CSV files from a Kaggle dataset tree, preserving relative folders."""
    copied = 0
    for fp in src.rglob("*.csv"):
        rel = fp.relative_to(src)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, target)
        copied += 1
    return copied


def _extract_archives(src: Path, destination: Path) -> int:
    """Extract zip/tar archives found in the Kaggle download directory."""
    extracted = 0
    for archive in src.rglob("*"):
        if not archive.is_file():
            continue
        try:
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(destination)
                extracted += 1
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive) as tf:
                    tf.extractall(destination, filter="data")
                extracted += 1
        except (OSError, zipfile.BadZipFile, tarfile.TarError):
            continue
    return extracted


def download(dataset: str, destination: str) -> str:
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    src = Path(kagglehub.dataset_download(dataset))

    # kagglehub may return a version directory containing either the actual
    # CSV tree or an archive. Support both layouts instead of assuming that
    # immediate children are files/directories containing the OHLCV data.
    copied = _copy_csv_tree(src, path)
    if copied == 0:
        _extract_archives(src, path)
        copied = len(list(path.rglob("*.csv")))

    print(f"Kaggle source: {src}")
    print(f"CSV files available locally: {copied}")
    if copied == 0:
        raise RuntimeError(f"Kaggle download completed but no CSV files were found under {src}")
    return str(src)


def main() -> None:
    with open("config/swing.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["data"]["kaggle"]["nse_all_daily"]
    source = download(dataset, "data/nse_all_daily")
    print(f"Downloaded NSE-wide daily dataset from {dataset}: {source}")


if __name__ == "__main__":
    main()
