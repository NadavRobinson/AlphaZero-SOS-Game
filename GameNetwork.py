"""Keras implementation of the SOS game network with policy and value heads.

The network expects the encoded board tensor produced by `SOS.encode`:
shape (6, BOARD_SIZE, BOARD_SIZE) with channels-first ordering.
Outputs:
    - policy: probability distribution over all actions (2 letters * board cells)
    - value: scalar in [-1, 1] estimating game outcome from the current player's view
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np

try:
    # Use BOARD_SIZE from the game logic to stay aligned with encode/decode.
    from Ex5 import BOARD_SIZE, ONGOING, RED, BLUE, DRAW, SOS
except ImportError:
    BOARD_SIZE = 5  # Fallback default
    ONGOING = -67
    DRAW = 0
    RED, BLUE = -1, 1
    SOS = None  # type: ignore


class GameNetwork:
    """Neural network with separate policy and value heads for SOS."""

    def __init__(self, board_size: int = BOARD_SIZE, learning_rate: float = 1e-3):
        self.board_size = board_size
        self.channels = 6
        self.action_size = board_size * board_size * 2
        self.input_shape: Tuple[int, int, int] = (self.channels, board_size, board_size)
        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate: float) -> tf.keras.Model:
        inputs = layers.Input(shape=self.input_shape, name="board_state")

        x = layers.Conv2D(
            64, kernel_size=3, padding="same", activation="relu",
            data_format="channels_first")(inputs)

        x = layers.Conv2D(
            128, kernel_size=3, padding="same", activation="relu",
            data_format="channels_first")(x)

        # Policy head
        p = layers.Conv2D(
            32, kernel_size=1, activation="relu", padding="same",
            data_format="channels_first", name="policy_conv")(x)
        p = layers.Flatten(name="policy_flat")(p)
        policy_out = layers.Dense(
            self.action_size, activation="softmax", name="policy_head")(p)

        # Value head
        v = layers.Conv2D(
            32, kernel_size=1, activation="relu", padding="same",
            data_format="channels_first", name="value_conv")(x)
        v = layers.Flatten(name="value_flat")(v)
        v = layers.Dense(64, activation="relu", name="value_dense")(v)
        value_out = layers.Dense(1, activation="tanh", name="value_head")(v)

        model = models.Model(inputs=inputs, outputs=[policy_out, value_out], name="GameNetwork")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={
                "policy_head": losses.CategoricalCrossentropy(),
                "value_head": losses.MeanSquaredError(),
            },
            loss_weights={"policy_head": 1.0, "value_head": 1.0},
            metrics={"policy_head": "accuracy", "value_head": "mae"},
        )
        return model

    def predict(self, encoded_state):
        """Return (policy_probs, win_prob) for a batch of encoded states."""
        policy, value = self.model.predict(encoded_state, verbose=0)
        return policy, value

    def train_step(self, states, policy_targets, value_targets):
        """One training step; targets should match predict outputs."""
        return self.model.train_on_batch(
            x=states,
            y={"policy_head": policy_targets, "value_head": value_targets},
            return_dict=True
        )

    def save(self, path: str):
        """Save only weights to `path` (e.g., 'checkpoints/gn.h5')."""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        self.model.save_weights(path)

    def load(self, path: str):
        """Load weights from `path` into the existing model."""
        try:
            self.model.load_weights(path)
            return
        except Exception:
            # Fallback for Keras 3 `.weights.h5` files (new HDF5 layout).
            if self._load_keras3_weights(path):
                return
            raise

    def _load_keras3_weights(self, path: str) -> bool:
        """Load Keras 3 `.weights.h5` files by mapping vars to current layers."""
        try:
            import h5py
        except Exception:
            return False

        with h5py.File(path, "r") as f:
            if "layers" not in f or "layer_names" in f:
                return False

            file_layers = []
            for lname in f["layers"].keys():
                lg = f["layers"][lname]
                if "vars" not in lg:
                    continue
                vars_group = lg["vars"]
                keys = sorted(vars_group.keys(), key=lambda x: int(x))
                weights = [vars_group[k][()] for k in keys]
                shapes = tuple(w.shape for w in weights)
                file_layers.append((shapes, weights))

            shape_to_weights = {}
            for shapes, weights in file_layers:
                shape_to_weights.setdefault(shapes, []).append(weights)

            unmatched = []
            for layer in self.model.layers:
                if not layer.weights:
                    continue
                shapes = tuple(w.shape for w in layer.get_weights())
                candidates = shape_to_weights.get(shapes)
                if not candidates:
                    unmatched.append((layer.name, shapes))
                    continue
                layer.set_weights(candidates.pop(0))

            if unmatched:
                return False

        return True


# Convenience factory for quick use without instantiating the class manually.
def build_default_network(learning_rate: float = 1e-3) -> GameNetwork:
    return GameNetwork(board_size=BOARD_SIZE, learning_rate=learning_rate)
