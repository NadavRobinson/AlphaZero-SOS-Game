"""Generate SOS self-play dataset using vanilla MCTS and save it to disk.

This script only creates data. Use `train_from_dataset.py` to train later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from Ex5 import BOARD_SIZE, ONGOING, SOS
from Ex6_MCTS_SOS import MCTSPlayer


ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 2


def play_self_play_game(mcts_iterations: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Play one full self-play game and collect (state, policy, value) targets."""
    game = SOS()
    mcts = MCTSPlayer()

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    current_players: List[int] = []

    while game.status == ONGOING:
        states.append(game.encode().astype(np.float32))
        current_players.append(game.current_player)

        visit_distribution = mcts.choose_move(
            game,
            iterations=mcts_iterations,
            is_self_play=True,
        )
        action_idx = np.random.choice(len(visit_distribution), p=visit_distribution)
        move = game.decode(action_idx)
        policies.append(np.asarray(visit_distribution, dtype=np.float32))
        game.make_move(move)

    outcome = game.status
    values = [float(outcome * player) for player in current_players]
    return states, policies, values


def generate_dataset(num_games: int, mcts_iterations: int):
    """Generate all self-play positions and targets."""
    all_states: List[np.ndarray] = []
    all_policies: List[np.ndarray] = []
    all_values: List[float] = []

    for i in range(1, num_games + 1):
        states, policies, values = play_self_play_game(mcts_iterations)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)

        if i % 10 == 0 or i == num_games:
            print(f"[data] games={i}/{num_games} positions={len(all_states)}")

    states_arr = np.stack(all_states, axis=0).astype(np.float32)
    policies_arr = np.stack(all_policies, axis=0).astype(np.float32)
    values_arr = np.asarray(all_values, dtype=np.float32).reshape(-1, 1)
    return states_arr, policies_arr, values_arr


def save_dataset(output_path: Path, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        policies=policies,
        values=values,
        board_size=np.asarray(BOARD_SIZE, dtype=np.int32),
        action_size=np.asarray(ACTION_SIZE, dtype=np.int32),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SOS MCTS pretraining dataset")
    parser.add_argument("--num-games", type=int, default=10000, help="Number of self-play games to generate")
    parser.add_argument("--mcts-iterations", type=int, default=1000, help="MCTS iterations per move")
    parser.add_argument("--output", type=Path, default=Path("mcts_dataset.npz"), help="Output .npz file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.mcts_iterations <= 0:
        raise ValueError("--mcts-iterations must be > 0")

    if args.seed is not None:
        np.random.seed(args.seed)

    print("[data] starting generation")
    print(f"[data] num_games={args.num_games}, mcts_iterations={args.mcts_iterations}")
    states, policies, values = generate_dataset(args.num_games, args.mcts_iterations)
    save_dataset(args.output, states, policies, values)
    print(f"[data] saved dataset to {args.output}")
    print(f"[data] states={states.shape}, policies={policies.shape}, values={values.shape}")


if __name__ == "__main__":
    main()
