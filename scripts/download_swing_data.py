from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import yaml


def download(dataset: str, destination: str) -> str:
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    src = Path(kagglehub.dataset_download(dataset))
    for item in src.iterdir():
        target = path / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return str(src)


def main() -> None:
    with open("config/swing.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["data"]["kaggle"]["nse_all_daily"]
    source = download(dataset, "data/nse_all_daily")
    print(f"Downloaded NSE-wide daily dataset from {dataset}: {source}")


if __name__ == "__main__":
    main()
