from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List

import numpy as np

from Ex5 import BLUE, RED
from elo_rating import (
    INITIAL_RATING,
    apply_elo_update,
    build_players,
    ensure_model_entry,
    load_ratings_db,
    normalize_model_key,
    play_game,
    ranking_lines,
    save_ratings_db,
    score_from_status,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rate one model or all unrated checkpoints in the Elo pool."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--model",
        type=Path,
        help="Path to one checkpoint that should be rated.",
    )
    mode_group.add_argument(
        "--auto-unrated",
        action="store_true",
        help="Rate all unrated checkpoints in --checkpoints-dir.",
    )
    parser.add_argument(
        "--ratings-file",
        type=Path,
        default=Path("checkpoints/elo_ratings.json"),
        help="Ratings database produced by simulate_rate_system.py.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory scanned in --auto-unrated mode.",
    )
    parser.add_argument("--num-games", type=int, default=500, help="Total games to play.")
    parser.add_argument(
        "--puct-iterations",
        type=int,
        default=100,
        help="PUCT iterations per move.",
    )
    parser.add_argument("--cpuct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Progress print interval.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow running this script on a model that already exists in the ratings file.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.puct_iterations <= 0:
        raise ValueError("--puct-iterations must be > 0")
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")
    if not args.ratings_file.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {args.ratings_file}. Run simulate_rate_system.py first."
        )
    if args.model is not None and not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.auto_unrated and not args.checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {args.checkpoints_dir}")


def checkpoint_sort_key(path: Path):
    match = re.match(r"^G(\d+)_", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, path.name)


def discover_unrated_models(data: dict, checkpoints_dir: Path) -> List[Path]:
    candidates = sorted(checkpoints_dir.glob("G*.weights.h5"), key=checkpoint_sort_key)
    unrated = []
    for checkpoint_path in candidates:
        model_key = normalize_model_key(checkpoint_path)
        if model_key not in data["models"]:
            unrated.append(checkpoint_path)
    return unrated


def rate_model(
    data: dict,
    model_path: Path,
    num_games: int,
    puct_iterations: int,
    cpuct: float,
    print_every: int,
    rng: np.random.Generator,
) -> str:
    new_model_key = normalize_model_key(model_path)
    ensure_model_entry(data, new_model_key, initial_rating=INITIAL_RATING)

    opponents = [model_key for model_key in data["models"] if model_key != new_model_key]
    if not opponents:
        raise ValueError("No rated opponents found. Initialize baseline first.")

    players = build_players([new_model_key, *opponents], cpuct=cpuct)

    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(1, num_games + 1):
        opp_key = opponents[int(rng.integers(0, len(opponents)))]
        new_model_is_red = bool(rng.integers(0, 2))
        red_key = new_model_key if new_model_is_red else opp_key
        blue_key = opp_key if new_model_is_red else new_model_key

        status = play_game(
            red_player=players[red_key],
            blue_player=players[blue_key],
            iterations=puct_iterations,
            rng=rng,
        )

        new_color = RED if new_model_is_red else BLUE
        opp_color = BLUE if new_model_is_red else RED
        new_score = score_from_status(status, new_color)
        opp_score = score_from_status(status, opp_color)

        if new_score == 1.0:
            wins += 1
        elif opp_score == 1.0:
            losses += 1
        else:
            draws += 1

        apply_elo_update(data, new_model_key, opp_key, new_score, opp_score)

        if game_idx % print_every == 0 or game_idx == num_games:
            rating = data["models"][new_model_key]["rating"]
            games = data["models"][new_model_key]["games_played"]
            print(
                f"[rate] model={new_model_key} game={game_idx}/{num_games} "
                f"W/L/D(new)={wins}/{losses}/{draws} "
                f"new_rating={rating} new_games={games}"
            )

    return new_model_key


def main():
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)
    data = load_ratings_db(args.ratings_file)

    if args.auto_unrated:
        unrated_models = discover_unrated_models(data, args.checkpoints_dir)
        if not unrated_models:
            print(f"[rate] no unrated checkpoints found in {args.checkpoints_dir}")
            print("[rate] ranking:")
            for line in ranking_lines(data):
                print(line)
            return

        print(f"[rate] found {len(unrated_models)} unrated checkpoints")
        for idx, model_path in enumerate(unrated_models, start=1):
            print(f"[rate] ({idx}/{len(unrated_models)}) rating {model_path}")
            rated_key = rate_model(
                data=data,
                model_path=model_path,
                num_games=args.num_games,
                puct_iterations=args.puct_iterations,
                cpuct=args.cpuct,
                print_every=args.print_every,
                rng=rng,
            )
            save_ratings_db(args.ratings_file, data)
            print(
                f"[rate] completed {rated_key} "
                f"rating={data['models'][rated_key]['rating']} "
                f"games={data['models'][rated_key]['games_played']}"
            )
    else:
        new_model_key = normalize_model_key(args.model)
        already_exists = new_model_key in data["models"]
        if already_exists and not args.allow_existing:
            raise ValueError(
                f"Model already rated: {new_model_key}. "
                "Pass --allow-existing to continue rating it with additional games."
            )

        rated_key = rate_model(
            data=data,
            model_path=args.model,
            num_games=args.num_games,
            puct_iterations=args.puct_iterations,
            cpuct=args.cpuct,
            print_every=args.print_every,
            rng=rng,
        )
        save_ratings_db(args.ratings_file, data)
        print(
            f"[rate] new model: {rated_key}\n"
            f"[rate] final rating={data['models'][rated_key]['rating']} "
            f"games={data['models'][rated_key]['games_played']}"
        )

    save_ratings_db(args.ratings_file, data)

    print(f"[rate] saved ratings to {args.ratings_file}")
    print("[rate] ranking:")
    for line in ranking_lines(data):
        print(line)


if __name__ == "__main__":
    main()
