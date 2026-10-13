import numpy as np
from collections import defaultdict

BASE = 1500.0
SURFACES = ("Hard", "Clay", "Grass")


def expected(r_a, r_b):
    """Вероятность победы A при рейтингах r_a, r_b."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def start_rating(rank, slope):
    """Стартовый Elo новичка по ATP-рангу (известен до матча). Ранг 100 -> 1500."""
    if slope == 0 or not np.isfinite(rank) or rank <= 0:
        return BASE
    return BASE + slope * (np.log(100.0) - np.log(rank))


def add_elo(m, k0=250.0, offset=5.0, shape=0.4, with_surface=True, init_slope=0.0):
    """elo1, elo2 (общий) и selo1, selo2 (по покрытию) ДО матча.
    K = k0 / (n_played + offset) ** shape. Ожидает m, отсортированный по времени."""
    m = m.copy()
    n = len(m)
    rk1 = m["rank1"].to_numpy(float) if "rank1" in m else np.full(n, np.nan)
    rk2 = m["rank2"].to_numpy(float) if "rank2" in m else np.full(n, np.nan)

    rating = {}
    played = defaultdict(int)
    s_rating = {s: {} for s in SURFACES}
    s_played = {s: defaultdict(int) for s in SURFACES}

    elo1, elo2 = np.empty(n), np.empty(n)
    selo1, selo2 = np.empty(n), np.empty(n)

    for i, (p1, p2, y, surf) in enumerate(zip(m["p1"], m["p2"], m["y"], m["surface"])):
        if p1 not in rating:
            rating[p1] = start_rating(rk1[i], init_slope)
        if p2 not in rating:
            rating[p2] = start_rating(rk2[i], init_slope)

        # 1. рейтинги ДО матча
        r1, r2 = rating[p1], rating[p2]
        elo1[i], elo2[i] = r1, r2
        use_s = with_surface and surf in s_rating
        if use_s:
            sr = s_rating[surf]
            if p1 not in sr:
                sr[p1] = start_rating(rk1[i], init_slope)
            if p2 not in sr:
                sr[p2] = start_rating(rk2[i], init_slope)
            sr1, sr2 = sr[p1], sr[p2]
        else:
            sr1, sr2 = r1, r2
        selo1[i], selo2[i] = sr1, sr2

        # 2. только теперь обновляем
        e1 = expected(r1, r2)
        rating[p1] = r1 + k0 / (played[p1] + offset) ** shape * (y - e1)
        rating[p2] = r2 + k0 / (played[p2] + offset) ** shape * ((1 - y) - (1 - e1))
        played[p1] += 1
        played[p2] += 1

        if use_s:
            e1s = expected(sr1, sr2)
            sr[p1] = sr1 + k0 / (s_played[surf][p1] + offset) ** shape * (y - e1s)
            sr[p2] = sr2 + k0 / (s_played[surf][p2] + offset) ** shape * ((1 - y) - (1 - e1s))
            s_played[surf][p1] += 1
            s_played[surf][p2] += 1

    m["elo1"], m["elo2"] = elo1, elo2
    m["selo1"], m["selo2"] = selo1, selo2
    return m, dict(rating)
