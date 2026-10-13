import json
import math


def load(model_path, players_path):
    model = json.load(open(model_path))
    pl = json.load(open(players_path))
    return model, pl["players"], pl["as_of"]


def _sig(z):
    return 1 / (1 + math.exp(-z))


def _features(A, B, surface, best_of):
    elo_diff = A["elo"] - B["elo"]
    sa = A["selo"].get(surface, A["elo"])
    sb = B["selo"].get(surface, B["elo"])
    ra, rb = A["rank"] or 300, B["rank"] or 300
    fa = A["form"] if A["form"] is not None else 0.5
    fb = B["form"] if B["form"] is not None else 0.5
    return {
        "elo_diff": elo_diff, "selo_diff": sa - sb,
        "lrank_diff": math.log(ra) - math.log(rb),
        "elo_bo5": elo_diff * (best_of == 5),
        "rest_diff": 0.0, "sets_diff": 0.0,       # нет информации о будущих матчах
        "form_diff": fa - fb, "h2h_diff": 0.0,
    }


def _prob(model, x):
    z = sum(model["coef"][c] * x[c] / model["sd"][c] for c in model["cols"])
    p = min(max(_sig(z), 1e-6), 1 - 1e-6)
    return _sig(model["scale"] * math.log(p / (1 - p)))   # калибровка


def win_prob(model, players, a, b, surface="Hard", best_of=5):
    """P(a побеждает b). Симметризуем: (p(a,b) + 1 - p(b,a)) / 2."""
    A, B = players[a], players[b]
    p_ab = _prob(model, _features(A, B, surface, best_of))
    p_ba = _prob(model, _features(B, A, surface, best_of))
    return (p_ab + 1 - p_ba) / 2
