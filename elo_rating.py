from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from Ex5 import BLUE, DRAW, ONGOING, RED, SOS
from GameNetwork import GameNetwork
from PUCT_SOS import PUCTPlayer

INITIAL_RATING = 1000
K_BEFORE_200 = 32
K_AFTER_200 = 16
PROVISIONAL_GAMES = 200


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_model_key(model_path: str | Path) -> str:
    resolved = Path(model_path).expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


def model_key_to_path(model_key: str) -> Path:
    path = Path(model_key)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def new_ratings_db() -> Dict:
    return {
        "meta": {
            "initial_rating": INITIAL_RATING,
            "k_before_200_games": K_BEFORE_200,
            "k_after_200_games": K_AFTER_200,
            "provisional_games": PROVISIONAL_GAMES,
            "updated_at": now_utc_iso(),
        },
        "models": {},
    }


def load_ratings_db(path: Path) -> Dict:
    if not path.exists():
        return new_ratings_db()

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "meta" not in data:
        data["meta"] = {}
    if "models" not in data:
        data["models"] = {}

    for model_key, record in data["models"].items():
        if "rating" not in record:
            record["rating"] = INITIAL_RATING
        if "games_played" not in record:
            record["games_played"] = 0
        record["rating"] = int(record["rating"])
        record["games_played"] = int(record["games_played"])

    data["meta"]["updated_at"] = now_utc_iso()
    return data


def save_ratings_db(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["meta"]["updated_at"] = now_utc_iso()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def ensure_model_entry(data: Dict, model_key: str, initial_rating: int = INITIAL_RATING) -> None:
    if model_key not in data["models"]:
        data["models"][model_key] = {
            "rating": int(initial_rating),
            "games_played": 0,
            "created_at": now_utc_iso(),
            "updated_at": now_utc_iso(),
        }
    else:
        data["models"][model_key]["rating"] = int(data["models"][model_key]["rating"])
        data["models"][model_key]["games_played"] = int(data["models"][model_key]["games_played"])


def k_factor(games_played: int) -> int:
    return K_BEFORE_200 if games_played < PROVISIONAL_GAMES else K_AFTER_200


def expected_score(rating_a: int, rating_b: int) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def score_from_status(status: int, color: int) -> float:
    if status == DRAW:
        return 0.5
    if status == color:
        return 1.0
    return 0.0


def apply_elo_update(data: Dict, model_a: str, model_b: str, score_a: float, score_b: float) -> None:
    rec_a = data["models"][model_a]
    rec_b = data["models"][model_b]

    rating_a = int(rec_a["rating"])
    rating_b = int(rec_b["rating"])
    games_a = int(rec_a["games_played"])
    games_b = int(rec_b["games_played"])

    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)

    new_rating_a = int(round(rating_a + k_factor(games_a) * (score_a - expected_a)))
    new_rating_b = int(round(rating_b + k_factor(games_b) * (score_b - expected_b)))

    rec_a["rating"] = new_rating_a
    rec_b["rating"] = new_rating_b
    rec_a["games_played"] = games_a + 1
    rec_b["games_played"] = games_b + 1
    rec_a["updated_at"] = now_utc_iso()
    rec_b["updated_at"] = now_utc_iso()


def build_players(model_keys: Iterable[str], cpuct: float = 1.5) -> Dict[str, PUCTPlayer]:
    players: Dict[str, PUCTPlayer] = {}
    for model_key in model_keys:
        model_path = model_key_to_path(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        network = GameNetwork()
        network.load(str(model_path))
        players[model_key] = PUCTPlayer(network=network, C=cpuct)
    return players


def play_game(red_player: PUCTPlayer, blue_player: PUCTPlayer, iterations: int, rng: np.random.Generator) -> int:
    game = SOS()

    while game.status == ONGOING:
        player = red_player if game.current_player == RED else blue_player
        move = player.choose_move(game, iterations=iterations)
        if move is None:
            legal_moves = list(game.legal_moves())
            if not legal_moves:
                break
            row, col = legal_moves[int(rng.integers(0, len(legal_moves)))]
            letter = "S" if int(rng.integers(0, 2)) == 0 else "O"
            move = (row, col, letter)
        game.make_move(move)

    return int(game.status)


def ranking_lines(data: Dict) -> Tuple[str, ...]:
    ordered = sorted(
        data["models"].items(),
        key=lambda item: (-int(item[1]["rating"]), item[0]),
    )
    return tuple(
        f"{idx}. {model_key}: rating={record['rating']}, games={record['games_played']}"
        for idx, (model_key, record) in enumerate(ordered, start=1)
    )
