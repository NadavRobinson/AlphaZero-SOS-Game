"""Pre-training script: generate self-play data with vanilla MCTS (no network)
and train the GameNetwork policy/value heads using the collected targets.

Steps:
1) Use the existing MCTSPlayer (rollout-based) to play 10,000 self-play games.
2) For each root position, target policy = visit-count distribution at root.
3) Target value = final game outcome from the current player's perspective.
"""

from __future__ import annotations

import math
from typing import List, Tuple
import tensorflow as tf
import numpy as np

from Ex5 import BOARD_SIZE, ONGOING, DRAW, RED, BLUE, SOS
from Ex6_MCTS_SOS import MCTSPlayer
from GameNetwork import GameNetwork

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

NUM_GAMES = 3
MCTS_ITERATIONS = 1000          # iterations per move for data generation
BATCH_SIZE = 64
EPOCHS = 100                     # single epoch over the generated dataset
CHECKPOINT_PATH = "test.weights.h5"

ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 2

def play_self_play_game(mcts_iterations: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Play one full game with MCTS on both sides; collect (state, policy, value) targets."""
    game = SOS()
    mcts = MCTSPlayer()

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    current_players: List[int] = []

    while game.status == ONGOING:
        # Capture state before the move (from player-to-move perspective)
        states.append(game.encode().astype(np.float32))
        current_players.append(game.current_player)

        # Run MCTS and grab the root visit distribution
        moves_distr = mcts.choose_move(game, iterations=mcts_iterations, is_self_play=True)
        # To generate diverse dataset, the best move is sampled proportionally to visit counts
        action_idx = np.random.choice(len(moves_distr), p=moves_distr)
        # Decode action index back to (row, col, letter)
        move = game.decode(action_idx)

        policies.append(moves_distr)

        game.make_move(move)

    outcome = game.status  # RED=-1, BLUE=1, DRAW=0
    values = [float(outcome * player) for player in current_players]
    return states, policies, values


def generate_dataset(num_games: int, mcts_iterations: int):
    """Generate self-play data arrays suitable for training."""
    all_states: List[np.ndarray] = []
    all_policies: List[np.ndarray] = []
    all_values: List[float] = []

    for i in range(1, num_games + 1):
        s, p, v = play_self_play_game(mcts_iterations)
        all_states.extend(s)
        all_policies.extend(p)
        all_values.extend(v)

        if i % 100 == 0:
            print(f"[data] Generated {i}/{num_games} games "
                  f"({len(all_states)} positions collected)")

    states_arr = np.stack(all_states, axis=0)
    policies_arr = np.stack(all_policies, axis=0)
    values_arr = np.array(all_values, dtype=np.float32).reshape(-1, 1)
    return states_arr, policies_arr, values_arr


def train_network(network: GameNetwork, states, policies, values,
                  batch_size: int, epochs: int):
    """Simple manual training loop over the collected dataset."""
    num_samples = states.shape[0]
    indices = np.arange(num_samples)
    num_batches = math.ceil(num_samples / batch_size)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        states = states[indices]
        policies = policies[indices]
        values = values[indices]

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, num_samples)
            batch_states = states[start:end]
            batch_policies = policies[start:end]
            batch_values = values[start:end]

            metrics = network.train_step(
                states=batch_states,
                policy_targets=batch_policies,
                value_targets=batch_values)
            
            epoch_policy_loss += metrics.get("policy_head_loss", 0.0)
            epoch_value_loss += metrics.get("value_head_loss", 0.0)

        epoch_policy_loss /= num_batches
        epoch_value_loss /= num_batches
        print(f"[train] Epoch {epoch}/{epochs} "
              f"policy_loss={epoch_policy_loss:.4f} "
              f"value_loss={epoch_value_loss:.4f}")


def main():
    # print("Generating self-play data with MCTS...")
    # #states, policies, values = generate_dataset(NUM_GAMES, MCTS_ITERATIONS)
    # #print(f"Dataset ready: {states.shape[0]} positions.")

    # print("Initializing network...")
    # network = GameNetwork()

    print("TF:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print("Build info:", tf.sysconfig.get_build_info())


    #print("Training network...")
    #train_network(network, states, policies, values, BATCH_SIZE, EPOCHS)

    #print(f"Saving weights to {CHECKPOINT_PATH}...")
    #network.save(CHECKPOINT_PATH)
    #print("Done.")


if __name__ == "__main__":
    main()
