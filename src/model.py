import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

LIN_COLS = ["elo_diff", "selo_diff", "lrank_diff", "elo_bo5",
            "rest_diff", "sets_diff", "form_diff", "h2h_diff"]


def build(m):
    """Строит признаки-разности. Оставляет матчи с коэффициентами и очками."""
    d = m[m["odd1"].notna() & m["odd2"].notna() & m["pts1"].notna() & m["pts2"].notna()].copy()
    d["elo_diff"] = d["elo1"] - d["elo2"]
    d["selo_diff"] = d["selo1"] - d["selo2"]
    d["lrank_diff"] = np.log(d["rank1"]) - np.log(d["rank2"])
    d["elo_bo5"] = d["elo_diff"] * (d["best_of"] == 5)
    d["rest_diff"] = np.log1p(d["rest1"]) - np.log1p(d["rest2"])
    d["sets_diff"] = d["sets1"] - d["sets2"]
    d["form_diff"] = d["form1"] - d["form2"]
    d["h2h_diff"] = d["h2h1"] - 0.5
    d["p_mkt"] = (1 / d["odd1"]) / (1 / d["odd1"] + 1 / d["odd2"])
    return d


def fit_lr(train, cols=LIN_COLS):
    """Логрегрессия без intercept на стандартизованных признаках.
    Без intercept => p(A,B) = 1 - p(B,A). Возвращает функцию df -> вероятность победы p1."""
    sd = train[cols].std()
    lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=2000)
    lr.fit(train[cols] / sd, train["y"])
    return lambda df: lr.predict_proba(df[cols] / sd)[:, 1]


def fit_scale(p, y):
    """Калибровочный масштаб a: p_cal = expit(a * logit(p))."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return float(LogisticRegression(fit_intercept=False, C=1e6).fit(z, y).coef_[0, 0])


def walk_forward(d, first_test=2016, last_test=2026, train_start=2006, cols=LIN_COLS):
    """Для каждого года Y: обучение на train_start..Y-1, масштаб a
    подбирается на Y-1 моделью, обученной до Y-2. Тест = год Y."""
    year = d["date"].dt.year
    out = []
    for Y in range(first_test, last_test + 1):
        te = d[year == Y]
        if te.empty:
            continue
        # масштаб: модель до Y-2, проверка на Y-1
        tr_a = d[(year >= train_start) & (year <= Y - 2)]
        va_a = d[year == Y - 1]
        a = fit_scale(fit_lr(tr_a, cols)(va_a), va_a["y"])
        # основная модель: всё до Y-1
        tr = d[(year >= train_start) & (year <= Y - 1)]
        raw = fit_lr(tr, cols)(te)
        p = expit(a * logit(np.clip(raw, 1e-6, 1 - 1e-6)))
        out.append(te.assign(p_raw=raw, p=p, q=te["p_mkt"], year=Y, scale=a))
    return pd.concat(out)

