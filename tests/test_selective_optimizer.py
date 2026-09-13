import pandas as pd

from src.selective_optimizer import component_frame, score_candidate, candidate_grid


def test_candidate_grid_is_finite_and_diverse():
    grid = candidate_grid()
    assert len(grid) == 100
    assert len({(p['top_n'], p['use_regime'], p['use_volume_filter'], p['min_vz']) for p in grid}) > 5


def test_mean_reversion_components_are_rank_based():
    d = pd.DataFrame({
        'r_ret5': [0.9, 0.1], 'r_ret20': [0.8, 0.2], 'r_sma20': [0.7, 0.3],
        'r_loc': [0.2, 0.8], 'r_vz': [0.5, 0.5], 'sma50': [-0.1, 0.1], 'vz': [0.0, 1.0]
    })
    x = component_frame(d)
    p = candidate_grid()[0]
    s = score_candidate(x, p)
    assert s.iloc[1] > s.iloc[0]
