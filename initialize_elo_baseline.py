from __future__ import annotations

import argparse
from pathlib import Path

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
        description="Initialize baseline Elo ratings by running first vs latest model."
    )
    parser.add_argument(
        "--first-model",
        type=Path,
        default=Path("checkpoints/G0_500.weights.h5"),
        help="Path to the first model checkpoint.",
    )
    parser.add_argument(
        "--latest-model",
        type=Path,
        default=Path("checkpoints/G5_500.weights.h5"),
        help="Path to the latest model checkpoint.",
    )
    parser.add_argument(
        "--ratings-file",
        type=Path,
        default=Path("checkpoints/elo_ratings.json"),
        help="Where Elo ratings are stored.",
    )
    parser.add_argument("--num-games", type=int, default=500, help="Number of games to simulate.")
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
        "--force-reset",
        action="store_true",
        help="Overwrite existing ratings file if it exists.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.puct_iterations <= 0:
        raise ValueError("--puct-iterations must be > 0")
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")
    if not args.first_model.exists():
        raise FileNotFoundError(f"First model not found: {args.first_model}")
    if not args.latest_model.exists():
        raise FileNotFoundError(f"Latest model not found: {args.latest_model}")


def main():
    args = parse_args()
    validate_args(args)

    if args.ratings_file.exists() and not args.force_reset:
        raise FileExistsError(
            f"{args.ratings_file} already exists. "
            "Use --force-reset to recreate baseline ratings."
        )

    rng = np.random.default_rng(args.seed)
    data = load_ratings_db(args.ratings_file)
    data["models"] = {}

    first_key = normalize_model_key(args.first_model)
    latest_key = normalize_model_key(args.latest_model)
    ensure_model_entry(data, first_key, initial_rating=INITIAL_RATING)
    ensure_model_entry(data, latest_key, initial_rating=INITIAL_RATING)

    players = build_players([first_key, latest_key], cpuct=args.cpuct)

    first_wins = 0
    latest_wins = 0
    draws = 0

    for game_idx in range(1, args.num_games + 1):
        first_is_red = bool(rng.integers(0, 2))
        red_key = first_key if first_is_red else latest_key
        blue_key = latest_key if first_is_red else first_key
        status = play_game(
            red_player=players[red_key],
            blue_player=players[blue_key],
            iterations=args.puct_iterations,
            rng=rng,
        )

        first_color = RED if first_is_red else BLUE
        latest_color = BLUE if first_is_red else RED
        first_score = score_from_status(status, first_color)
        latest_score = score_from_status(status, latest_color)

        if first_score == 1.0:
            first_wins += 1
        elif latest_score == 1.0:
            latest_wins += 1
        else:
            draws += 1

        apply_elo_update(data, first_key, latest_key, first_score, latest_score)

        if game_idx % args.print_every == 0 or game_idx == args.num_games:
            first_rating = data["models"][first_key]["rating"]
            latest_rating = data["models"][latest_key]["rating"]
            print(
                f"[baseline] game={game_idx}/{args.num_games} "
                f"W/L/D(first)={first_wins}/{latest_wins}/{draws} "
                f"ratings(first/latest)={first_rating}/{latest_rating}"
            )

    save_ratings_db(args.ratings_file, data)

    print(f"[baseline] saved ratings to {args.ratings_file}")
    print("[baseline] ranking:")
    for line in ranking_lines(data):
        print(line)


if __name__ == "__main__":
    main()
