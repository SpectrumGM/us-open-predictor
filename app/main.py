from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src import predict as P

ROOT = Path(__file__).resolve().parent.parent
model, players, AS_OF = P.load(ROOT / "artifacts/model.json", ROOT / "artifacts/players.json")

app = FastAPI(title="US Open Predictor", version="1.0")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "players": len(players), "as_of": AS_OF,
            "trained_through": model["trained_through"]}


@app.get("/players")
def list_players():
    items = [{"name": n, "elo": p["elo"], "rank": p["rank"]} for n, p in players.items()]
    return sorted(items, key=lambda x: -x["elo"])


@app.get("/predict")
def predict(a: str, b: str,
            surface: Literal["Hard", "Clay", "Grass"] = "Hard",
            best_of: Literal[3, 5] = 5):
    if a == b:
        raise HTTPException(400, "Choose two different players")
    for n in (a, b):
        if n not in players:
            raise HTTPException(404, f"Unknown player: {n}")
    p = P.win_prob(model, players, a, b, surface, best_of)
    return {"a": a, "b": b, "surface": surface, "best_of": best_of,
            "p_a": round(p, 4), "p_b": round(1 - p, 4), "data_as_of": AS_OF}
