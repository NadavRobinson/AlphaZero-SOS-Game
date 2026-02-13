"""Load a saved SOS dataset and train GameNetwork on it."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

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


def split_train_val(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    val_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("--val-split must be in [0.0, 1.0)")

    num_samples = states.shape[0]
    if num_samples < 2 or val_split == 0.0:
        return states, policies, values, None, None, None

    val_size = int(num_samples * val_split)
    if val_size == 0:
        val_size = 1
    if val_size >= num_samples:
        val_size = num_samples - 1

    indices = np.random.permutation(num_samples)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        states[train_idx],
        policies[train_idx],
        values[train_idx],
        states[val_idx],
        policies[val_idx],
        values[val_idx],
    )


def train_network(
    network: GameNetwork,
    train_states: np.ndarray,
    train_policies: np.ndarray,
    train_values: np.ndarray,
    val_states: Optional[np.ndarray],
    val_policies: Optional[np.ndarray],
    val_values: Optional[np.ndarray],
    batch_size: int,
    epochs: int,
):
    num_samples = train_states.shape[0]
    indices = np.arange(num_samples)
    num_batches = math.ceil(num_samples / batch_size)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        states_epoch = train_states[indices]
        policies_epoch = train_policies[indices]
        values_epoch = train_values[indices]

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
        if val_states is None or val_policies is None or val_values is None:
            print(
                f"[train] epoch={epoch}/{epochs} "
                f"policy_loss={epoch_policy_loss:.4f} "
                f"value_loss={epoch_value_loss:.4f}"
            )
            continue

        val_metrics = network.model.evaluate(
            x=val_states,
            y={"policy_head": val_policies, "value_head": val_values},
            batch_size=batch_size,
            verbose=0,
            return_dict=True,
        )
        val_policy_loss = float(val_metrics.get("policy_head_loss", 0.0))
        val_value_loss = float(val_metrics.get("value_head_loss", 0.0))
        print(
            f"[train] epoch={epoch}/{epochs} "
            f"policy_loss={epoch_policy_loss:.4f} "
            f"value_loss={epoch_value_loss:.4f} "
            f"val_policy_loss={val_policy_loss:.4f} "
            f"val_value_loss={val_value_loss:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train GameNetwork from saved MCTS dataset")
    parser.add_argument("--dataset", type=Path, default=Path("mcts_dataset.npz"), help="Path to .npz dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data used for validation")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrain.weights.h5"), help="Output weights file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val-split must be in [0.0, 1.0)")

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

    train_states, train_policies, train_values, val_states, val_policies, val_values = split_train_val(
        states, policies, values, args.val_split
    )
    if val_states is None:
        print(f"[data] train_samples={train_states.shape[0]}, validation=disabled")
    else:
        print(f"[data] train_samples={train_states.shape[0]}, val_samples={val_states.shape[0]}")

    strategy = tf.distribute.MirroredStrategy()
    print(f"GPUs in sync: {strategy.num_replicas_in_sync}")
    with strategy.scope():
        network = GameNetwork(board_size=BOARD_SIZE, learning_rate=args.learning_rate)
    train_network(
        network=network,
        train_states=train_states,
        train_policies=train_policies,
        train_values=train_values,
        val_states=val_states,
        val_policies=val_policies,
        val_values=val_values,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    network.save(str(args.checkpoint))
    print(f"[train] saved weights to {args.checkpoint}")


if __name__ == "__main__":
    main()
