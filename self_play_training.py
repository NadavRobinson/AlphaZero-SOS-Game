"""Train the SOS policy/value network with PUCT self-play and a replay buffer.

This script combines data generation and learning in one loop:
1) Play self-play games with PUCT (network-guided search).
2) Store (state, policy, value) targets in a fixed-size replay buffer.
3) Sample mini-batches from the replay buffer and train online.

Optional: export the final replay buffer to `.npz` with keys
`states`, `policies`, `values` so it can be reused by other scripts.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from Ex5 import BOARD_SIZE, ONGOING, DRAW, RED, BLUE, SOS
from GameNetwork import GameNetwork
from PUCT_SOS import PUCTPlayer


ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 2
ReplaySample = Tuple[np.ndarray, np.ndarray, float]


def play_self_play_game(
    puct_player: PUCTPlayer,
    puct_iterations: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], int]:
    """Play one full game and return per-position targets."""
    game = SOS()

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    current_players: List[int] = []

    while game.status == ONGOING:
        states.append(game.encode().astype(np.float32))
        current_players.append(game.current_player)

        visit_distribution = puct_player.choose_move(
            game,
            iterations=puct_iterations,
            is_self_play=True,
        )
        action_idx = np.random.choice(len(visit_distribution), p=visit_distribution)
        move = game.decode(action_idx)

        policies.append(np.asarray(visit_distribution, dtype=np.float32))
        game.make_move(move)

    outcome = int(game.status)
    values = [float(outcome * player) for player in current_players]
    return states, policies, values, outcome


def add_game_to_replay(
    replay_buffer: Deque[ReplaySample],
    states: List[np.ndarray],
    policies: List[np.ndarray],
    values: List[float],
):
    for state, policy, value in zip(states, policies, values):
        replay_buffer.append((state, policy, value))


def train_from_replay(
    network: GameNetwork,
    replay_buffer: Deque[ReplaySample],
    batch_size: int,
    train_steps: int,
):
    """Run `train_steps` random mini-batch updates from replay buffer."""
    if train_steps <= 0 or len(replay_buffer) == 0:
        return None

    buffer_list = list(replay_buffer)
    n = len(buffer_list)
    replace = n < batch_size

    avg_policy_loss = 0.0
    avg_value_loss = 0.0

    for _ in range(train_steps):
        idx = np.random.choice(n, size=batch_size, replace=replace)
        batch = [buffer_list[i] for i in idx]

        batch_states = np.stack([x[0] for x in batch], axis=0).astype(np.float32)
        batch_policies = np.stack([x[1] for x in batch], axis=0).astype(np.float32)
        batch_values = np.asarray([x[2] for x in batch], dtype=np.float32).reshape(-1, 1)

        metrics = network.train_step(
            states=batch_states,
            policy_targets=batch_policies,
            value_targets=batch_values,
        )

        avg_policy_loss += float(metrics.get("policy_head_loss", 0.0))
        avg_value_loss += float(metrics.get("value_head_loss", 0.0))

    avg_policy_loss /= train_steps
    avg_value_loss /= train_steps
    return avg_policy_loss, avg_value_loss


def export_replay_buffer(path: Path, replay_buffer: Deque[ReplaySample]):
    if len(replay_buffer) == 0:
        raise ValueError("Replay buffer is empty; nothing to export")

    data = list(replay_buffer)
    states = np.stack([x[0] for x in data], axis=0).astype(np.float32)
    policies = np.stack([x[1] for x in data], axis=0).astype(np.float32)
    values = np.asarray([x[2] for x in data], dtype=np.float32).reshape(-1, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        policies=policies,
        values=values,
        board_size=np.asarray(BOARD_SIZE, dtype=np.int32),
        action_size=np.asarray(ACTION_SIZE, dtype=np.int32),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PUCT self-play training with replay buffer")
    parser.add_argument("--num-games", type=int, default=500, help="Number of self-play games")
    parser.add_argument("--puct-iterations", type=int, default=500, help="PUCT iterations per move")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Max replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Training mini-batch size")
    parser.add_argument(
        "--train-steps-per-game",
        type=int,
        default=4,
        help="How many gradient updates to run after each game",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=1024,
        help="Start training only after replay reaches this many samples",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--init-weights", type=Path, default=None, help="Optional initial weights file")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for periodic checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save checkpoint every N games (0 disables periodic checkpoints)",
    )
    parser.add_argument(
        "--final-weights",
        type=Path,
        default=Path("AlphaZero---SOS-Game/puct_self_play.weights.h5"),
        help="Where to save final model weights",
    )
    parser.add_argument(
        "--export-replay",
        type=Path,
        default=None,
        help="Optional path to save replay buffer as .npz",
    )
    parser.add_argument("--print-every", type=int, default=10, help="Log interval in games")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def validate_args(args):
    if args.num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.puct_iterations <= 0:
        raise ValueError("--puct-iterations must be > 0")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.train_steps_per_game < 0:
        raise ValueError("--train-steps-per-game must be >= 0")
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")


def main():
    args = parse_args()
    validate_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    print("TF:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    network = GameNetwork(board_size=BOARD_SIZE, learning_rate=args.learning_rate)

    if args.init_weights is not None:
        if not args.init_weights.exists():
            raise FileNotFoundError(f"Initial weights not found: {args.init_weights}")
        network.load(str(args.init_weights))
        print(f"[init] loaded weights from {args.init_weights}")

    puct_player = PUCTPlayer(network=network)
    replay_buffer: Deque[ReplaySample] = deque(maxlen=args.buffer_size)

    outcomes = {RED: 0, BLUE: 0, DRAW: 0}

    print("[train] starting self-play training")
    print(
        "[train] "
        f"num_games={args.num_games}, puct_iterations={args.puct_iterations}, "
        f"buffer_size={args.buffer_size}, batch_size={args.batch_size}, "
        f"train_steps_per_game={args.train_steps_per_game}, warmup_samples={args.warmup_samples}"
    )

    for game_idx in range(1, args.num_games + 1):
        states, policies, values, outcome = play_self_play_game(
            puct_player=puct_player,
            puct_iterations=args.puct_iterations,
        )
        add_game_to_replay(replay_buffer, states, policies, values)
        outcomes[outcome] += 1

        did_train = False
        losses = None
        if len(replay_buffer) >= args.warmup_samples:
            losses = train_from_replay(
                network=network,
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                train_steps=args.train_steps_per_game,
            )
            did_train = losses is not None

        should_log = (game_idx % args.print_every == 0) or (game_idx == args.num_games)
        if should_log:
            msg = (
                f"[train] game={game_idx}/{args.num_games} "
                f"buffer={len(replay_buffer)} "
                f"wins(R/B/D)=({outcomes[RED]}/{outcomes[BLUE]}/{outcomes[DRAW]})"
            )
            if did_train and losses is not None:
                policy_loss, value_loss = losses
                msg += f" policy_loss={policy_loss:.4f} value_loss={value_loss:.4f}"
            elif len(replay_buffer) < args.warmup_samples:
                msg += f" warmup={len(replay_buffer)}/{args.warmup_samples}"
            print(msg)

        if args.checkpoint_every > 0 and game_idx % args.checkpoint_every == 0:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = args.checkpoint_dir / f"puct_game_{game_idx}.weights.h5"
            network.save(str(ckpt_path))
            print(f"[train] saved checkpoint to {ckpt_path}")

    network.save(str(args.final_weights))
    print(f"[train] saved final weights to {args.final_weights}")

    if args.export_replay is not None:
        export_replay_buffer(args.export_replay, replay_buffer)
        print(f"[train] exported replay buffer to {args.export_replay}")


if __name__ == "__main__":
    main()
