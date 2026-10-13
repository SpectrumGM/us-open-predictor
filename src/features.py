import re
import numpy as np
import pandas as pd
from collections import defaultdict, deque


def count_sets(score):
    return sum(bool(re.match(r"\d+-\d+", t)) for t in str(score).split())


def add_form_features(m, window_days=14, form_n=10):
    """Фичи каждого игрока ДО матча: rest, sets_14d, form, n_played, h2h.
    Ожидает m, отсортированный по времени."""
    m = m.copy()
    hist = defaultdict(lambda: deque(maxlen=60))   # (date, sets, win)
    h2h = defaultdict(int)                         # (a, b) -> побед a над b

    cols = {k: np.empty(len(m)) for k in
            ["rest1", "rest2", "sets1", "sets2", "form1", "form2",
             "n1", "n2", "h2h1"]}

    for i, (d, p1, p2, y, sc) in enumerate(
            zip(m["date"], m["p1"], m["p2"], m["y"], m["score"])):
        for j, p in ((1, p1), (2, p2)):
            h = hist[p]
            cols[f"rest{j}"][i] = min((d - h[-1][0]).days, 30) if h else 30
            cols[f"sets{j}"][i] = sum(s for dd, s, _ in h if (d - dd).days <= window_days)
            last = list(h)[-form_n:]
            wins = sum(w for _, _, w in last)
            cols[f"form{j}"][i] = (wins + 2.5) / (len(last) + 5)   # сглаживание к 0.5
            cols[f"n{j}"][i] = len(h)
        w1, w2 = h2h[(p1, p2)], h2h[(p2, p1)]
        cols["h2h1"][i] = (w1 + 1) / (w1 + w2 + 2)                 # сглаженная доля p1

        # только теперь обновляем
        s = count_sets(sc)
        hist[p1].append((d, s, y))
        hist[p2].append((d, s, 1 - y))
        h2h[(p1, p2) if y == 1 else (p2, p1)] += 1

    for k, v in cols.items():
        m[k] = v
    return m
