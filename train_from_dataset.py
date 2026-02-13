"""Load a saved SOS dataset and train GameNetwork on it."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from Ex5 import BOARD_SIZE
from GameNetwork import GameNetwork


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path) as data:
        required = {"states", "policies", "values"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Dataset missing keys: {sorted(missing)}")

        states = data["states"].astype(np.float32)
        policies = data["policies"].astype(np.float32)
        values = data["values"].astype(np.float32)

    if states.ndim != 4:
        raise ValueError(f"states must be rank-4, got shape {states.shape}")
    if policies.ndim != 2:
        raise ValueError(f"policies must be rank-2, got shape {policies.shape}")
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"values must have shape (N, 1), got {values.shape}")

    n = states.shape[0]
    if policies.shape[0] != n or values.shape[0] != n:
        raise ValueError(
            "states, policies, and values must have the same first dimension: "
            f"got {states.shape[0]}, {policies.shape[0]}, {values.shape[0]}"
        )

    return states, policies, values


def train_network(network: GameNetwork, states, policies, values, batch_size: int, epochs: int):
    num_samples = states.shape[0]
    indices = np.arange(num_samples)
    num_batches = math.ceil(num_samples / batch_size)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        states_epoch = states[indices]
        policies_epoch = policies[indices]
        values_epoch = values[indices]

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)

            metrics = network.train_step(
                states=states_epoch[start:end],
                policy_targets=policies_epoch[start:end],
                value_targets=values_epoch[start:end],
            )
            epoch_policy_loss += float(metrics.get("policy_head_loss", 0.0))
            epoch_value_loss += float(metrics.get("value_head_loss", 0.0))

        epoch_policy_loss /= num_batches
        epoch_value_loss /= num_batches
        print(
            f"[train] epoch={epoch}/{epochs} "
            f"policy_loss={epoch_policy_loss:.4f} "
            f"value_loss={epoch_value_loss:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train GameNetwork from saved MCTS dataset")
    parser.add_argument("--dataset", type=Path, default=Path("mcts_dataset.npz"), help="Path to .npz dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrain.weights.h5"), help="Output weights file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    print("TF:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    states, policies, values = load_dataset(args.dataset)
    print(f"[data] loaded {args.dataset}")
    print(f"[data] states={states.shape}, policies={policies.shape}, values={values.shape}")

    if states.shape[2] != BOARD_SIZE or states.shape[3] != BOARD_SIZE:
        print(
            f"[warn] dataset board shape is ({states.shape[2]}, {states.shape[3]}) "
            f"but BOARD_SIZE is {BOARD_SIZE}"
        )

    network = GameNetwork(board_size=BOARD_SIZE, learning_rate=args.learning_rate)
    train_network(network, states, policies, values, args.batch_size, args.epochs)

    network.save(str(args.checkpoint))
    print(f"[train] saved weights to {args.checkpoint}")


if __name__ == "__main__":
    main()
