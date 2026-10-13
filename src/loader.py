import numpy as np
import pandas as pd

ROUND_ORDER = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Quarterfinals": 5, "Semifinals": 6, "The Final": 7,
}

RENAME = {
    "Tournament": "tourney", "Date": "date", "Series": "series",
    "Court": "court", "Surface": "surface", "Round": "round",
    "Best of": "best_of", "Player_1": "p1", "Player_2": "p2",
    "Winner": "winner", "Rank_1": "rank1", "Rank_2": "rank2",
    "Pts_1": "pts1", "Pts_2": "pts2", "Odd_1": "odd1", "Odd_2": "odd2",
    "Score": "score",
}


def load_matches(path):
    """Читает Kaggle ATP, чистит и сортирует по времени."""
    m = pd.read_csv(path, parse_dates=["Date"]).rename(columns=RENAME)

    # пропуски записаны как -1 (очки/ранг) и -1.0 (коэффициенты)
    for c in ["rank1", "rank2", "pts1", "pts2"]:
        m.loc[m[c] < 0, c] = np.nan
    for c in ["odd1", "odd2"]:
        m.loc[m[c] <= 1, c] = np.nan

    # победитель должен быть одним из двух игроков
    ok = (m["winner"] == m["p1"]) | (m["winner"] == m["p2"])
    if not ok.all():
        raise ValueError(f"winner не совпадает с игроками в {(~ok).sum()} строках")

    m["y"] = (m["winner"] == m["p1"]).astype(int)   # 1, если выиграл Player_1
    m["round_order"] = m["round"].map(ROUND_ORDER).fillna(0).astype(int)

    # стабильная сортировка: дата, затем раунд
    m = m.sort_values(["date", "round_order"], kind="stable").reset_index(drop=True)
    return m
