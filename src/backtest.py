import itertools
import numpy as np
import pandas as pd
from scipy.special import expit

from .bracket import build_bracket, champion, simulate
from .model import fit_lr, fit_scale, LIN_COLS

STATE_COLS = ["elo", "selo", "rank", "rest", "sets", "form"]


def player_state(r4):
    """Состояние каждого игрока на начало 1/8 финала.
    r4 - матчи раунда 4 одного US Open (из m)."""
    st = {}
    for _, r in r4.iterrows():
        for k in ("1", "2"):
            st[r["p" + k]] = {c: r[c + k] for c in STATE_COLS}
    return st


def pair_features(a, b, st):
    """Признаки для гипотетической пары (a, b): разности состояний.
    h2h_diff = 0, так как для будущих пар истории нет."""
    A, B = st[a], st[b]
    elo_diff = A["elo"] - B["elo"]
    return {
        "elo_diff": elo_diff,
        "selo_diff": A["selo"] - B["selo"],
        "lrank_diff": np.log(A["rank"]) - np.log(B["rank"]),
        "elo_bo5": elo_diff,                      # US Open - best of 5
        "rest_diff": np.log1p(A["rest"]) - np.log1p(B["rest"]),
        "sets_diff": A["sets"] - B["sets"],
        "form_diff": A["form"] - B["form"],
        "h2h_diff": 0.0,
    }


def pair_prob_table(players, st, model_fn=None, elo_only=False):
    """Матрица вероятностей P(a бьёт b) для всех пар.
    Симметризуем: p = (p(a,b) + 1 - p(b,a)) / 2."""
    ordered = list(itertools.permutations(players, 2))
    X = pd.DataFrame([pair_features(a, b, st) for a, b in ordered])
    if elo_only:
        raw = expit(X["elo_diff"].values * np.log(10) / 400)
    else:
        raw = model_fn(X)
    raw = dict(zip(ordered, raw))
    return {(a, b): (raw[(a, b)] + 1 - raw[(b, a)]) / 2 for a, b in ordered}


def champion_probs(levels, table, n_sims=10000, seed=0):
    """P(чемпион) для каждого игрока."""
    prob = lambda a, b: table[(a, b)]
    res = simulate(levels, prob, n_sims=n_sims, seed=seed)
    return res[7]
