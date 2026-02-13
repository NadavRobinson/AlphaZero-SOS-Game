from Ex5 import ONGOING, SOS, main, RED, BLUE, BOARD_SIZE
import numpy as np
import random

NUM_ITERATIONS = 10000  # Number of MCTS iterations per move
EPSILON = 0.2  # Rollout: probability of choosing a random move
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 2  # Each cell can have 'S' or 'O'


# --------------------------------- HELPER ------------------------------ #
def move_to_action_index(move):
    """Map (row, col, letter) to a flat action index expected by the policy head."""
    row, col, letter = move
    letter_idx = 0 if letter == 'S' else 1
    return row * BOARD_SIZE * 2 + col * 2 + letter_idx

class MCTSNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.untriedMoves = set()
        self.Q = 0.0
        self.N = 1

    def leaf(self, board):
        return len(self.children) != len(board.legal_moves()) * 2

class MCTSPlayer:

    def __init__(self):
        self.root = None

    C = 1.4  # Exploration parameter for MCTS

    # MCTS methods (selection, expansion, simulation, backpropagation)

    def UCT(self, node, parent_log_N, board):
        return board.current_player * node.Q + self.C * np.sqrt(parent_log_N / (node.N))

    def choose_move(self, board, iterations, is_self_play=False):
        self.root = MCTSNode()  # Reset the root for each move decision
        self.root.untriedMoves = {(pos[0], pos[1], letter) for pos in board.legal_moves() for letter in ['S', 'O']}
        for _ in range(iterations):  # Number of MCTS iterations

            work_board = board.clone()

            # 1. Selection
            leaf = self.selection(work_board)
            
            # 2. Expansion & 3. Simulation
            if work_board.status == ONGOING:
                leaf = self.expansion(leaf, work_board)
                outcome = self.simulation(work_board)
            else:
                # If selection led to a terminal state, that is the outcome
                outcome = work_board.status
            
            # 4. Backpropagation
            self.backpropagation(leaf, outcome)
        
        if is_self_play:
            return self.root_policy_distribution()

        # Choose the move with the highest visit count
        best_move = None
        max_visits = -1
        for child, move in self.root.children.items():
            if child.N > max_visits:
                max_visits = child.N
                best_move = move
        return best_move

    def root_policy_distribution(self) -> np.ndarray:
        """Return visit-count distribution over all actions from the current root."""
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        total = 0.0
        for child, move in self.root.children.items():
            idx = move_to_action_index(move)
            counts[idx] = child.N
            total += child.N
        if total > 0:
            counts /= total
        else:
            counts.fill(1.0 / ACTION_SIZE)
        return counts

    def selection(self, board):
        cur = self.root
        while not cur.leaf(board):
            maxUCT = -float('inf')
            best_child = None
            cur_move = None
            
            # Pre-calculate log factor once for all children of this node
            log_N_parent = np.log(cur.N) 
            
            for child, move in cur.children.items():
                # Standard UCT but using the pre-calculated log
                uct_value = self.UCT(child, log_N_parent, board)
                
                if uct_value > maxUCT:
                    maxUCT = uct_value
                    best_child = child
                    cur_move = move
                    
            board.make_move(cur_move)
            if board.status != ONGOING:
                return best_child
            cur = best_child
        return cur
    
    def expansion(self, node, board):
        move = node.untriedMoves.pop()
        board.make_move(move)
        child_node = MCTSNode(parent=node)
        node.children[child_node] = move
        child_node.untriedMoves = {(pos[0], pos[1], letter) for pos in board.legal_moves() for letter in ['S', 'O']}
        return child_node

    # def simulation(self, board):
    #     while board.status == ONGOING:
    #         possible_cells = list(board.legal_moves())
    #         if not possible_cells:
    #             break
                
    #         # Only check for scoring moves if we are deep enough in the game to likely have them.
            
    #         # Intelligent Random: Still picks a random cell, but tries both letters 
    #         # only for that specific cell to see if it scores.
    #         row, col = random.choice(possible_cells)
    #         # best_move = (row, col, 'S')
            
    #         # Quick check for 'O' score at the same spot
    #         board.make_move((row, col, 'O'))
    #         o_score = board.calculate_new_score((row, col, 'O')) 
    #         board.unmake_move((row, col, 'O'), o_score)
            
    #         if o_score > 0:
    #             best_move = (row, col, 'O')

    #         board.make_move((row, col, 'S'))
    #         s_score = board.calculate_new_score((row, col, 'S'))
    #         board.unmake_move((row, col, 'S'), s_score)
            
    #         board.make_move(best_move)
    #         # else:
    #         #     # Early game: Just pick a random letter and move to save CPU
    #         #     row, col = random.choice(possible_cells)
    #         #     board.make_move((row, col, random.choice(['S', 'O'])))
                
    #     return board.status
    
    def simulation(self, board):
        while board.status == ONGOING:
            possible_moves = list(board.legal_moves())
            row, col = random.choice(possible_moves)
            board.make_move((row, col, random.choice(['S', 'O'])))
        return board.status

    # def simulation(self, board):
    #     # simulation_board = board.clone()

    #     while board.status == ONGOING:
    #         possible_moves = board.legal_moves()
    #         all_moves = [(pos[0], pos[1], letter) for pos in possible_moves for letter in ['S', 'O']]
    #         if not all_moves:
    #             break
    #         if random.random() < EPSILON:
    #             chosen_move = random.choice(all_moves)
    #         else:
    #             best_value = -1
    #             best_moves = []
    #             for move in all_moves:
    #                 additional_score = board.make_move(move)
    #                 if additional_score > best_value:
    #                     best_value = additional_score
    #                     best_moves = [move]
    #                 elif additional_score == best_value:
    #                     best_moves.append(move)
    #                 board.unmake_move(move, additional_score)
    #             chosen_move = random.choice(best_moves) if best_moves else random.choice(all_moves)

    #         board.make_move(chosen_move)
    #     return board.status
    
    def backpropagation(self, node, outcome):
        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += (outcome - cur.Q) / cur.N
            cur = cur.parent

def mainAIvsPlayer():
    game = SOS()
    mcts_player = MCTSPlayer()

    print("Welcome to SOS!")
    print("You are RED (R). The AI is BLUE (B).\n")

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
            ai_move = mcts_player.choose_move(game, iterations=NUM_ITERATIONS)
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

if __name__ == "__main__":
    mainAIvsPlayer()
    # main()
