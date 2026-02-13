from Ex5 import ONGOING, DRAW, SOS, RED, BLUE, BOARD_SIZE
import numpy as np
import os
from typing import Optional

from GameNetwork import GameNetwork

NUM_ITERATIONS = 800  # Number of PUCT iterations per move
PRETRAIN_WEIGHTS_PATH = "5x5_pretrain.weights.h5"


def move_to_action_index(move):
    """Map (row, col, letter) to a flat action index expected by the policy head."""
    row, col, letter = move
    letter_idx = 0 if letter == 'S' else 1
    return row * BOARD_SIZE * 2 + col * 2 + letter_idx


class PUCTNode:
    def __init__(self, parent=None, prior: float = 1.0, move=None):
        self.parent = parent
        self.children = {}
        self.P = float(prior)   # prior probability from the parent's policy head
        self.Q = 0.0            # mean value
        self.N = 1

    def leaf(self, board):
         return len(self.children) != len(board.legal_moves()) * 2

class PUCTPlayer:
    def __init__(self, network: Optional[GameNetwork] = None, C: float = 1.5):
        self.root = None
        self.network = network or GameNetwork()
        self.C = C

    # PUCT scoring function: U(s, a) = Q(s, a) + cpuct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    def PUCT(self, node, board):
        return board.current_player * node.Q + self.C * node.P * np.sqrt(node.parent.N) / (1 + node.N)

    def choose_move(self, board, iterations):
        self.root = PUCTNode()

        for _ in range(iterations):
            work_board = board.clone()

            # 1. Selection
            leaf = self.selection(work_board)

            # 2. Expansion & neural evaluation (replaces random rollout)
            if work_board.status == ONGOING:
                value = self.expansion(leaf, work_board)
            else:
                value = work_board.status

            # 3. Backpropagation
            self.backpropagation(leaf, value)

        # Choose the move with the highest visit count
        if not self.root.children:
            return None

        best_child = max(self.root.children, key=lambda c: c.N)
        return self.root.children[best_child]

    def selection(self, board):
        cur = self.root
        while not cur.leaf(board) and board.status == ONGOING:
            best_score = -float('inf')
            best_child = None
            best_move = None
            for child, move in cur.children.items():
                score = self.PUCT(child, board)
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_move = move

            # Descend the tree with the chosen move
            board.make_move(best_move)
            cur = best_child
        return cur

    def expansion(self, node, board):
        # Evaluate the leaf with the neural network and expand all legal moves
        policy, value = self.evaluate(board)

        legal_moves = board.legal_moves()
        if not legal_moves:
            return value

        move_priors = []
        for row, col in legal_moves:
            for letter in ['S', 'O']:
                move = (row, col, letter)
                prior = policy[move_to_action_index(move)]
                move_priors.append((move, prior))

        prior_sum = sum(p for _, p in move_priors)
        if prior_sum <= 0:
            normalized = [(m, 1.0 / len(move_priors)) for m, _ in move_priors]
        else:
            normalized = [(m, p / prior_sum) for m, p in move_priors]

        # Create child nodes with their priors
        for move, prior in normalized:
            child_node = PUCTNode(parent=node, prior=prior)
            node.children[child_node] = move

        return value

    def evaluate(self, board):
        encoded = board.encode().astype(np.float32)
        policy, value = self.network.predict(encoded[np.newaxis, ...])
        policy = policy[0]
        value = float(value[0][0]) * board.current_player
        return policy, value

    def backpropagation(self, node, outcome):
        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += (outcome - cur.Q) / cur.N
            cur = cur.parent

def mainAIvsPlayer():
    game = SOS()
    network = GameNetwork()
    if not os.path.exists(PRETRAIN_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Could not find pretrained weights at '{PRETRAIN_WEIGHTS_PATH}'. "
            "Update PRETRAIN_WEIGHTS_PATH to your checkpoint file."
        )
    network.load(PRETRAIN_WEIGHTS_PATH)
    puct_player = PUCTPlayer(network=network)

    print("Welcome to SOS with PUCT!")
    print("You are RED (R). The AI is BLUE (B).\n")
    print(f"Loaded pretrained weights from: {PRETRAIN_WEIGHTS_PATH}\n")

    while game.status == ONGOING:
        game.print_board()
        current_player = 'RED' if game.current_player == RED else 'BLUE'
        if game.current_player == RED:
            move = input(f"Current player: {current_player}\nenter your move as 'row col letter (S or O)': ")
            if move.lower() == 'end':
                game.end_game()
                break
            try:
                parts = move.split()
                if len(parts) != 3:
                    raise ValueError
                move = (int(parts[0]), int(parts[1]), parts[2].upper())
            except (ValueError, IndexError):
                print("Invalid input format. Please enter: row col letter (e.g. '0 1 S')")
                continue
            move0, move1, move2 = move

            if move0 < 0 or move0 >= BOARD_SIZE or move1 < 0 or move1 >= BOARD_SIZE:
                print("Invalid row or column, try again.")
                continue
            if move2 not in ['S', 'O']:
                print("Invalid letter, use 'S' or 'O'.")
                continue
            try:
                game.make_move(move)
            except ValueError as e:
                print(e)
                continue
            
        else:
            print("AI is thinking...")
            ai_move = puct_player.choose_move(game, iterations=NUM_ITERATIONS)
            game.make_move(ai_move)
            print(f"AI played  {ai_move}.")
        
        print(f"Scores - RED: {game.scores[RED]}, BLUE: {game.scores[BLUE]}")

    print("Final Board:")
    game.print_board()
    if game.status == RED:
        print("\nRED (You) win!")
    elif game.status == BLUE:
        print("\nBLUE (AI) wins!")
    else:
        print("\nIt's a draw!")


def mainAIvsAI():
    game = SOS()
    network1 = GameNetwork()
    network2 = GameNetwork()
    if not os.path.exists(PRETRAIN_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Could not find pretrained weights at '{PRETRAIN_WEIGHTS_PATH}'. "
            "Update PRETRAIN_WEIGHTS_PATH to your checkpoint file."
        )
    network1.load(PRETRAIN_WEIGHTS_PATH)
    network2.load(PRETRAIN_WEIGHTS_PATH)
    puct_player1 = PUCTPlayer(network=network1)
    puct_player2 = PUCTPlayer(network=network2)

    print("Welcome to SOS!")
    print("AI 1 is RED (R). AI 2 is BLUE (B).\n")
    print(f"Loaded pretrained weights from: {PRETRAIN_WEIGHTS_PATH}\n")

    while game.status == ONGOING:
        print(game)
        if game.current_player == RED:
            print("AI 1 is thinking...")
            ai_move = puct_player1.choose_move(game, iterations=NUM_ITERATIONS)
            game.make_move(ai_move)
            print(f"AI 1 played column {ai_move}.")
        else:
            print("AI 2 is thinking...")
            ai_move = puct_player2.choose_move(game, iterations=NUM_ITERATIONS)
            game.make_move(ai_move)
            print(f"AI 2 played column {ai_move}.")

    print(game)
    if game.status == game.RED:
        print("\nRED (AI 1) wins!")
    elif game.status == game.BLUE:
        print("\nBLUE (AI 2) wins!")
    else:
        print("\nIt's a draw!")


if __name__ == "__main__":
    mainAIvsPlayer()
