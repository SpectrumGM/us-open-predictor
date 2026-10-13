import numpy as np
import pandas as pd

from src.elo import add_elo
from src.features import add_form_features
from src.model import LIN_COLS, fit_lr, walk_forward


def make_matches(n=300, seed=0):
    rng = np.random.default_rng(seed)
    players = [f"P{i} X." for i in range(8)]
    rows = []
    for i in range(n):
        a, b = rng.choice(players, 2, replace=False)
        win = a if rng.random() < 0.5 else b
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "tourney": "T", "surface": ["Hard", "Clay", "Grass"][i % 3],
            "round_order": 1, "best_of": 3, "p1": a, "p2": b, "winner": win,
            "y": int(win == a), "rank1": int(rng.integers(1, 100)),
            "rank2": int(rng.integers(1, 100)), "score": "6-4 6-4",
        })
    return pd.DataFrame(rows)


def flip(m, k):
    """Меняет исход матча k на противоположный."""
    m = m.copy()
    r = m.loc[k]
    m.loc[k, "winner"] = r["p2"] if r["winner"] == r["p1"] else r["p1"]
    m.loc[k, "y"] = 1 - r["y"]
    return m


def test_elo_has_no_leakage():
    m = make_matches()
    k = 150
    a, _ = add_elo(m, 70, 30, 0.2)
    b, _ = add_elo(flip(m, k), 70, 30, 0.2)
    cols = ["elo1", "elo2", "selo1", "selo2"]
    # матчи до k включительно: рейтинг записан ДО матча, не должен измениться
    pd.testing.assert_frame_equal(a.loc[:k, cols], b.loc[:k, cols])
    # sanity: после k что-то обязано поменяться, иначе тест бессмысленный
    assert not np.allclose(a.loc[k + 1:, cols], b.loc[k + 1:, cols])


def test_form_features_have_no_leakage():
    m = make_matches()
    k = 150
    a = add_form_features(m)
    b = add_form_features(flip(m, k))
    cols = ["rest1", "rest2", "form1", "form2", "h2h1", "n1", "n2"]
    pd.testing.assert_frame_equal(a.loc[:k, cols], b.loc[:k, cols])


def test_model_is_antisymmetric():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(500, len(LIN_COLS))), columns=LIN_COLS)
    X["y"] = (X["elo_diff"] + rng.normal(size=500) > 0).astype(int)
    f = fit_lr(X)
    p, q = f(X), f(-X[LIN_COLS].assign(y=0))
    np.testing.assert_allclose(p + q, 1.0, atol=1e-9)


def test_walk_forward_ignores_test_year_labels():
    rng = np.random.default_rng(2)
    n = 4000
    d = pd.DataFrame(rng.normal(size=(n, len(LIN_COLS))), columns=LIN_COLS)
    d["date"] = pd.to_datetime("2006-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 365 * 13, n)), unit="D")
    d["y"] = (d["elo_diff"] + rng.normal(size=n) > 0).astype(int)
    d["p_mkt"] = 0.5
    a = walk_forward(d, 2016, 2018)
    d2 = d.copy()
    yr = d2["date"].dt.year
    d2.loc[yr == 2016, "y"] = 1 - d2.loc[yr == 2016, "y"]   # портим ответы 2016
    b = walk_forward(d2, 2016, 2018)
    # прогнозы на 2016 не должны зависеть от ответов 2016
    np.testing.assert_allclose(a[a.year == 2016]["p"], b[b.year == 2016]["p"])
