import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .model import LIN_COLS, fit_lr, fit_scale


def export_model(d, path, cols=LIN_COLS, train_start=2006):
    """Обучает финальную модель на всех данных и пишет коэффициенты в JSON.
    Масштаб калибровки: модель до (last-1), проверка на last, как в walk_forward."""
    year = d["date"].dt.year
    last = int(year.max())
    tr = d[(year >= train_start) & (year <= last)]
    sd = tr[cols].std()
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=2000)
    lr.fit(tr[cols] / sd, tr["y"])
    tr_a = d[(year >= train_start) & (year <= last - 1)]
    va = d[year == last]
    a = fit_scale(fit_lr(tr_a, cols)(va), va["y"])
    obj = {
        "cols": list(cols),
        "sd": {c: float(sd[c]) for c in cols},
        "coef": {c: float(w) for c, w in zip(cols, lr.coef_[0])},
        "scale": a,
        "trained_through": last,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    return obj


def export_players(m, final, path, active_days=365):
    """Текущее состояние игроков: Elo (свежий), Elo по покрытиям, ранг, форма.
    Покрытие/ранг/форма берутся из последней записи игрока (до его последнего матча)."""
    parts = []
    for k in ("1", "2"):
        parts.append(pd.DataFrame({
            "date": m["date"], "surface": m["surface"], "name": m["p" + k],
            "selo": m["selo" + k], "rank": m["rank" + k], "form": m["form" + k],
        }))
    long = pd.concat(parts).sort_values("date")
    last_date = long.groupby("name")["date"].max()
    cur = long.groupby("name")[["rank", "form"]].last()
    by_surf = long.groupby(["name", "surface"])["selo"].last()
    cutoff = long["date"].max() - pd.Timedelta(days=active_days)
    players = {}
    for name, ld in last_date.items():
        if ld < cutoff or name not in final:
            continue
        s = {}
        for surf in ("Hard", "Clay", "Grass"):
            if (name, surf) in by_surf.index and not np.isnan(by_surf[(name, surf)]):
                s[surf] = round(float(by_surf[(name, surf)]), 1)
        rk = cur.loc[name, "rank"]
        fm = cur.loc[name, "form"]
        players[name] = {
            "elo": round(float(final[name]), 1),
            "selo": s,
            "rank": None if np.isnan(rk) else int(rk),
            "form": None if np.isnan(fm) else round(float(fm), 3),
            "last_match": str(ld.date()),
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"as_of": str(long["date"].max().date()), "players": players},
              open(path, "w"), indent=1)
    return players
