from __future__ import annotations

import os
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
    with open("config/btst.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = cfg["data"]["kaggle"]
    download(datasets["nifty50_daily"], "data/daily")
    download(datasets["nifty100_5m"], "data/intraday")
    download(datasets["nifty_indices_minute"], "data/market")
    print("Downloaded configured Kaggle datasets.")


if __name__ == "__main__":
    main()
