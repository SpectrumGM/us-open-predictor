"""Сетка US Open от 1/8 финала и симуляция Монте-Карло.

Дерево строится по реальным матчам: победитель матча раунда r-1 играет в матче раунда r.
Если в данных не хватает ровно одного матча раунда, он достраивается однозначно:
двое победителей предыдущего раунда, которых нет ни в одном записанном матче,
играли друг с другом, а победитель - тот, кто сыграл в следующем раунде.
"""
from collections import Counter
from itertools import product

import numpy as np

EXPECTED = {4: 8, 5: 4, 6: 2, 7: 1}   # матчей в раундах 1/8, 1/4, 1/2, финал


def build_bracket(g):
    """g: матчи одного турнира (колонки round_order, p1, p2, winner).
    Возвращает levels: {4: [...], 5: [...], 6: [...], 7: [...]}.
    Узел: {"a", "b", "win", "feeders", "virtual"}. Бросает ValueError, если год не восстановить."""
    rounds = {r: g[g["round_order"] == r] for r in EXPECTED}
    for r, n in EXPECTED.items():
        if len(rounds[r]) > n:
            raise ValueError(f"раунд {r}: матчей {len(rounds[r])}, должно быть не больше {n}")

    levels = {4: [dict(a=x.p1, b=x.p2, win=x.winner, feeders=None, virtual=False)
                  for x in rounds[4].itertuples()]}
    if len(levels[4]) != EXPECTED[4]:
        raise ValueError(f"в 1/8 финала {len(levels[4])} матчей из {EXPECTED[4]}")

    for r in (5, 6, 7):
        nodes = [dict(a=x.p1, b=x.p2, win=x.winner, virtual=False) for x in rounds[r].itertuples()]
        prev_winners = [n["win"] for n in levels[r - 1]]
        seen = {n["a"] for n in nodes} | {n["b"] for n in nodes}
        unknown = seen - set(prev_winners)
        if unknown:
            raise ValueError(f"раунд {r}: игроки {sorted(unknown)} не выиграли предыдущий раунд")
        missing = [w for w in prev_winners if w not in seen]
        need = EXPECTED[r] - len(nodes)

        if len(missing) != 2 * need:
            raise ValueError(f"раунд {r}: не хватает {need} матчей, но без пары {len(missing)} игроков")
        if need == 1:
            if r == 7:
                raise ValueError("нет финала: чемпиона восстановить нельзя")
            nxt = set(rounds[r + 1]["p1"]) | set(rounds[r + 1]["p2"])
            w = [p for p in missing if p in nxt]
            if len(w) != 1:
                raise ValueError(f"раунд {r}: победителя пропавшего матча не определить")
            nodes.append(dict(a=missing[0], b=missing[1], win=w[0], virtual=True))
        elif need > 1:
            raise ValueError(f"раунд {r}: не хватает {need} матчей")

        by_win = {n["win"]: i for i, n in enumerate(levels[r - 1])}
        for n in nodes:
            assert n["win"] in (n["a"], n["b"])
            n["feeders"] = (by_win[n["a"]], by_win[n["b"]])
        levels[r] = nodes
    return levels


def champion(levels):
    return levels[7][0]["win"]


def possible_pairs(levels):
    """Все пары, которые теоретически могут сыграть. Для 16 игроков их ровно 120."""
    poss = {4: [{n["a"], n["b"]} for n in levels[4]]}
    pairs = {frozenset((n["a"], n["b"])) for n in levels[4]}
    for r in (5, 6, 7):
        cur = []
        for n in levels[r]:
            fa, fb = n["feeders"]
            sa, sb = poss[r - 1][fa], poss[r - 1][fb]
            pairs |= {frozenset(p) for p in product(sa, sb)}
            cur.append(sa | sb)
        poss[r] = cur
    return pairs


def simulate(levels, prob, n_sims=10000, seed=0):
    """prob(a, b) -> вероятность победы a над b. Возвращает {раунд: Counter(игрок -> доля симуляций,
    в которых он выиграл матч этого раунда)}. Победа в раунде 7 - это титул."""
    rng = np.random.default_rng(seed)
    wins = {r: Counter() for r in EXPECTED}
    for _ in range(n_sims):
        prev = None
        for r in (4, 5, 6, 7):
            cur = []
            for node in levels[r]:
                if r == 4:
                    a, b = node["a"], node["b"]
                else:
                    fa, fb = node["feeders"]
                    a, b = prev[fa], prev[fb]
                w = a if rng.random() < prob(a, b) else b
                cur.append(w)
                wins[r][w] += 1
            prev = cur
    return {r: Counter({p: c / n_sims for p, c in cnt.items()}) for r, cnt in wins.items()}


def describe(levels):
    names = {4: "1/8", 5: "1/4", 6: "1/2", 7: "финал"}
    for r in (4, 5, 6, 7):
        for n in levels[r]:
            mark = "  <- достроен" if n["virtual"] else ""
            print(f"{names[r]:6s} {n['a']:22s} vs {n['b']:22s} -> {n['win']}{mark}")

