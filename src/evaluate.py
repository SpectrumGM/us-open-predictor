import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss


def per_match_ll(y, p):
    """Лог-лосс каждого матча отдельно."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summary(res, cols=("p", "q")):
    """Таблица log-loss / Brier / accuracy по колонкам вероятностей."""
    rows = {}
    for c in cols:
        rows[c] = {
            "log_loss": log_loss(res["y"], res[c]),
            "brier": brier_score_loss(res["y"], res[c]),
            "accuracy": float(((res[c] > 0.5) == (res["y"] == 1)).mean()),
        }
    return pd.DataFrame(rows).T.round(4)


def paired_bootstrap(res, a="p", b="q", n_boot=5000, seed=0):
    """Разница log-loss (a - b) со 95% доверительным интервалом.
    Ресэмплим матчи парами, поэтому шум общий для обеих моделей."""
    diff = per_match_ll(res["y"], res[a]) - per_match_ll(res["y"], res[b])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def calibration_table(res, col="p", bins=10):
    """Средний прогноз vs реальная доля побед по корзинам."""
    t = res[[col, "y"]].copy()
    t["bin"] = pd.qcut(t[col], bins, duplicates="drop")
    g = t.groupby("bin", observed=True).agg(pred=(col, "mean"), real=("y", "mean"), n=("y", "size"))
    return g.round(3)


def by_year(res, a="p", b="q"):
    """Лог-лосс по годам и разрыв с рынком."""
    out = res.groupby("year").apply(
        lambda g: pd.Series({
            "n": len(g),
            "model": log_loss(g["y"], g[a]),
            "market": log_loss(g["y"], g[b]),
        }),
        include_groups=False,
    )
    out["gap"] = out["model"] - out["market"]
    return out.round(4)

