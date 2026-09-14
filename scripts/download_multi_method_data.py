from __future__ import annotations

import sys
import yaml

sys.path.insert(0, "scripts")
from download_swing_data import download


if __name__ == "__main__":
    with open("config/multi_method.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    download(cfg, destination="data/nse_all_daily")
