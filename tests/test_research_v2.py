import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from research_v2 import fold_dates


def test_fold_dates_respects_embargo_and_order():
    dates = list(range(2000))
    folds = list(fold_dates(dates, train_days=1000, val_days=100, test_days=100, step_days=100, embargo=2))
    assert folds
    train, val, test = folds[0]
    assert max(train) < min(val)
    assert max(val) + 2 < min(test)
    assert len(train) == 1000
    assert len(val) == 100
    assert len(test) == 100


def test_fold_dates_multiple_non_overlapping_tests():
    dates = list(np.arange(1500))
    folds = list(fold_dates(dates, 700, 100, 100, 100, 1))
    assert len(folds) >= 2
    for a, b in zip(folds, folds[1:]):
        assert max(a[2]) < min(b[2])
