# Nadav Robinson 212544779
# Yuval Cohen 323071043

import numpy as np

BOARD_SIZE = 5
S = 1
O = -1
EMPTY = 0

BLUE = 1
RED = -1

ONGOING = -67 # Completely arbitrary (OR IS IT???)
DRAW = 0

class SOS:
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY)
        self.current_player = RED
        self.status = ONGOING
        self.scores = {RED: 0, BLUE: 0}
        self.empty_cells = set((row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if self.board[row, col] == EMPTY)

    def legal_moves(self):
        return self.empty_cells

    def calculate_new_score(self, move):
        row, col, letter = move
        letter = S if letter == 'S' else O
        directions = [
            [(0, -1), (0, 1)],    # Horizontal
            [(-1, 0), (1, 0)],    # Vertical
            [(-1, -1), (1, 1)],   # Diagonal \
            [(-1, 1), (1, -1)]    # Diagonal /
        ]
        
        score_increment = 0
        
        for direction in directions:
            if letter == S:
                for d in direction:
                    r, c = row + d[0], col + d[1]
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        if self.board[r, c] == O:
                            # Check the next cell in the same direction
                            r2, c2 = r + d[0], c + d[1]
                            # temp = (r*BOARD_SIZE + c2, r2*BOARD_SIZE + c2)
                            if 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE:
                                if self.board[r2, c2] == S:
                                    score_increment += 1
                                    # self.completed.add(temp)
            elif letter == O:
                # Check both sides for S
                r1, c1 = row + direction[0][0], col + direction[0][1]
                r2, c2 = row + direction[1][0], col + direction[1][1]
                if (0 <= r1 < BOARD_SIZE and 0 <= c1 < BOARD_SIZE and self.board[r1, c1] == S and
                    0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and self.board[r2, c2] == S):
                    score_increment += 1
                        
        return score_increment

    def make_move(self, move):
        legal_moves = self.legal_moves()
        if (move[0], move[1]) not in legal_moves:
            raise ValueError("Illegal move")
        row, col, letter = move
        self.board[row, col] = S if letter == 'S' else O
        self.empty_cells.remove((row, col))
        additional_score = self.calculate_new_score(move)
        self.scores[self.current_player] += additional_score
        self.switch_player()

        # Game is over when the board has no empty cells left after this move.
        if len(self.empty_cells) == 0:
            if self.scores[RED] > self.scores[BLUE]:
                self.status = RED
            elif self.scores[RED] < self.scores[BLUE]:
                self.status = BLUE
            else:
                self.status = DRAW
        return additional_score
            
    def unmake_move(self, move, score_to_deduct=0):
        row, col, _ = move
        self.board[row, col] = EMPTY
        self.empty_cells.add((row, col))
        self.switch_player()
        self.scores[self.current_player] -= score_to_deduct
        self.status = ONGOING

    def switch_player(self):
        self.current_player = BLUE if self.current_player == RED else RED

    def clone(self):
        clone = SOS()
        clone.board = np.copy(self.board)
        clone.current_player = self.current_player
        clone.status = self.status
        clone.scores = self.scores.copy()
        clone.empty_cells = self.empty_cells.copy()
        return clone
    
    def encode(self):
        # Channel 0: S positions
        s_plane = np.where(self.board == S, 1, 0)

        # Channel 1: O positions
        o_plane = np.where(self.board == O, 1, 0)

        # Channel 2: Legal moves
        legal_moves_plane = np.where(self.board == EMPTY, 1, 0)

        # Channel 3: Current player
        current_player_plane = np.full((BOARD_SIZE, BOARD_SIZE), self.current_player)

        # Channel 4: Red score
        red_score_plane = np.full((BOARD_SIZE, BOARD_SIZE), self.scores[RED])

        # Channel 5: Blue score
        blue_score_plane = np.full((BOARD_SIZE, BOARD_SIZE), self.scores[BLUE])

        return np.array([s_plane, o_plane, legal_moves_plane, current_player_plane, red_score_plane, blue_score_plane])
    
    def decode(self, action_index):
        # Suppose we want to palce 'S' in (1,2)
        # If board size is 8, then action_index = 20
        row = action_index // (BOARD_SIZE * 2)
        col = (action_index % (BOARD_SIZE * 2)) // 2
        letter = 'S' if action_index % 2 == 0 else 'O'
        return (row, col, letter)
    
    def print_board(self):
        symbol_map = {S: 'S', O: 'O', EMPTY: '.'}
        for row in range(BOARD_SIZE):
            print(' '.join(symbol_map[self.board[row, col]] for col in range(BOARD_SIZE)))
        print()

    def status(self):
        return self.status
    
    def end_game(self):
        while self.legal_moves():
                legal = self.legal_moves()[0]
                self.make_move((legal[0], legal[1], 'O'))

def main():
    game = SOS()

    legal_moves = game.legal_moves()
    print("Legal moves:", legal_moves)

    while game.status == ONGOING:
        game.print_board()
        current_player = 'RED' if game.current_player == RED else 'BLUE'
        move = input(f"Current player: {current_player}\nenter your move as 'row col letter (S or O)': ")
        if move.lower() == 'end':
            game.end_game()
            break
        try:
            parts = move.split()
            if len(parts) != 3:
                raise ValueError
            move = (int(parts[0]), int(parts[1]), parts[2])
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
        print(f"Scores - RED: {game.scores[RED]}, BLUE: {game.scores[BLUE]}")
        
    print("Final Board:")
    game.print_board()
    print(f"Final Scores - RED: {game.scores[RED]}, BLUE: {game.scores[BLUE]}")
    if game.status == DRAW:
        print("The game is a draw!")
    else:
        winner = 'RED' if game.status == RED else 'BLUE'
        print(f"The winner is {winner}!")

if __name__ == "__main__":
    main()
