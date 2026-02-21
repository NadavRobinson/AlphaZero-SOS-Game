"""
SOS Game GUI - Play against MCTS or PUCT AI
Click on a cell to place S or O
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os

from Ex5 import SOS, ONGOING, DRAW, RED, BLUE, BOARD_SIZE, S, O, EMPTY
from Ex6_MCTS_SOS import MCTSPlayer, NUM_ITERATIONS as MCTS_ITERATIONS
from PUCT_SOS import PUCTPlayer, NUM_ITERATIONS as PUCT_ITERATIONS, PRETRAIN_WEIGHTS_PATH
from GameNetwork import GameNetwork


class SOSGUI:
    CELL_SIZE = 60
    PADDING = 20

    def __init__(self, root):
        self.root = root
        self.root.title("SOS Game")
        self.root.resizable(False, False)

        self.game = None
        self.ai_players = {}
        self.game_mode = tk.StringVar(value="Player vs AI")
        self.ai_mode = tk.StringVar(value="MCTS")
        self.model_var = tk.StringVar()
        self.available_models = self._get_available_models()
        if self.available_models:
            # Prefer 5000_pretrain or any 2nd_training_checkpoints if available
            default_model = self.available_models[0]
            for m in self.available_models:
                if "5000_pretrain" in m:
                    default_model = m
                    break
            self.model_var.set(default_model)

        self.ai_iterations = MCTS_ITERATIONS
        self.ai_thinking = False
        self.ai_vs_ai_started = False
        self.human_color = RED  # Human plays RED (first)

        self._build_ui()
        self._new_game()

    def _get_available_models(self):
        import glob
        # Search for .weights.h5 files in root and subdirectories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pattern = os.path.join(base_dir, "**", "*.weights.h5")
        models = glob.glob(pattern, recursive=True)
        # Convert to relative paths from AlphaZero-SOS-Game for cleaner display
        rel_models = [os.path.relpath(m, base_dir) for m in models]
        return sorted(rel_models)

    def _build_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top controls
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.pack(fill=tk.X, pady=(0, 10))

        # First row of controls
        self.row1 = ttk.Frame(self.control_frame)
        self.row1.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.row1, text="Game Mode:").pack(side=tk.LEFT, padx=(0, 5))
        game_mode_combo = ttk.Combobox(
            self.row1,
            textvariable=self.game_mode,
            values=["Player vs AI", "AI vs AI"],
            state="readonly",
            width=12
        )
        game_mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        game_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._new_game())

        ttk.Label(self.row1, text="AI Mode:").pack(side=tk.LEFT, padx=(0, 5))
        mode_combo = ttk.Combobox(
            self.row1,
            textvariable=self.ai_mode,
            values=["MCTS", "PUCT"],
            state="readonly",
            width=10
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        mode_combo.bind("<<ComboboxSelected>>", self._on_ai_mode_change)

        self.start_button = ttk.Button(self.row1, text="Start AI vs AI", command=self._start_ai_vs_ai)
        self.start_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(self.row1, text="New Game", command=self._new_game).pack(side=tk.LEFT, padx=5)

        # Second row of controls (Model selection, hidden for MCTS)
        self.model_frame = ttk.Frame(self.control_frame)
        # Will be packed/unpacked in _update_ui_visibility
        
        ttk.Label(self.model_frame, text="PUCT Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_combo = ttk.Combobox(
            self.model_frame,
            textvariable=self.model_var,
            values=self.available_models,
            state="readonly",
            width=60
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._new_game())

        # Status labels
        self.status_label = ttk.Label(self.row1, text="", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Score frame
        score_frame = ttk.Frame(main_frame)
        score_frame.pack(fill=tk.X, pady=(0, 10))

        self.red_score_label = ttk.Label(score_frame, text="RED (You): 0", foreground="red", font=("Arial", 12, "bold"))
        self.red_score_label.pack(side=tk.LEFT, padx=20)

        self.blue_score_label = ttk.Label(score_frame, text="BLUE (AI): 0", foreground="blue", font=("Arial", 12, "bold"))
        self.blue_score_label.pack(side=tk.RIGHT, padx=20)

        # Game board canvas
        canvas_size = BOARD_SIZE * self.CELL_SIZE + 2 * self.PADDING
        self.canvas = tk.Canvas(
            main_frame,
            width=canvas_size,
            height=canvas_size,
            bg="white",
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Letter selection frame (initially hidden)
        self.letter_frame = ttk.Frame(main_frame)
        self.letter_frame.pack(fill=tk.X, pady=10)

        self.letter_prompt = ttk.Label(self.letter_frame, text="Choose letter:", font=("Arial", 10))
        self.letter_prompt.pack(side=tk.LEFT, padx=10)

        self.s_button = ttk.Button(self.letter_frame, text="S", width=5, command=lambda: self._place_letter('S'))
        self.s_button.pack(side=tk.LEFT, padx=5)

        self.o_button = ttk.Button(self.letter_frame, text="O", width=5, command=lambda: self._place_letter('O'))
        self.o_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(self.letter_frame, text="Cancel", command=self._cancel_selection)
        self.cancel_button.pack(side=tk.LEFT, padx=10)

        self._hide_letter_selection()
        self._update_ui_visibility()

        # Selected cell
        self.selected_cell = None

    def _on_ai_mode_change(self, event=None):
        self._update_ui_visibility()
        self._new_game()

    def _update_ui_visibility(self):
        if self.ai_mode.get() == "PUCT":
            self.model_frame.pack(fill=tk.X, pady=(0, 5), before=self.row1)
            # Re-pack row1 after model_frame to keep it below if desired, 
            # or just pack it before to have model selection on top.
            # Let's put model selection below row1.
            self.model_frame.pack_forget()
            self.model_frame.pack(fill=tk.X, pady=(0, 5), after=self.row1)
        else:
            self.model_frame.pack_forget()

    def _hide_letter_selection(self):
        for widget in self.letter_frame.winfo_children():
            widget.configure(state="disabled")
        prompt = "Press Start AI vs AI to begin"
        if self.game_mode.get() == "Player vs AI":
            prompt = "Click a cell to place S or O"
        self.letter_prompt.configure(text=prompt)

    def _show_letter_selection(self, row, col):
        self.selected_cell = (row, col)
        self.letter_prompt.configure(text=f"Cell ({row}, {col}) - Choose letter:")
        for widget in self.letter_frame.winfo_children():
            widget.configure(state="normal")

    def _cancel_selection(self):
        self.selected_cell = None
        self._hide_letter_selection()
        self._draw_board()

    def _new_game(self):
        if self.ai_thinking:
            return

        self.game = SOS()
        self.ai_players = {}
        self.ai_vs_ai_started = False
        self.selected_cell = None
        self._hide_letter_selection()
        self._update_ui_visibility()

        # Initialize AI player(s)
        game_mode = self.game_mode.get()
        mode = self.ai_mode.get()
        if mode == "MCTS":
            if game_mode == "Player vs AI":
                self.ai_players = {BLUE: MCTSPlayer()}
            else:
                self.ai_players = {RED: MCTSPlayer(), BLUE: MCTSPlayer()}
            self.ai_iterations = MCTS_ITERATIONS
        else:  # PUCT
            model_path = self.model_var.get()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_model_path = os.path.join(base_dir, model_path)
            
            if not os.path.exists(full_model_path):
                messagebox.showerror(
                    "Error",
                    f"PUCT requires weights at:\n{full_model_path}\n\nSwitching to MCTS mode."
                )
                self.ai_mode.set("MCTS")
                self._update_ui_visibility()
                if game_mode == "Player vs AI":
                    self.ai_players = {BLUE: MCTSPlayer()}
                else:
                    self.ai_players = {RED: MCTSPlayer(), BLUE: MCTSPlayer()}
                self.ai_iterations = MCTS_ITERATIONS
            else:
                if game_mode == "Player vs AI":
                    network = GameNetwork()
                    network.load(full_model_path)
                    self.ai_players = {BLUE: PUCTPlayer(network=network)}
                else:
                    # Mirror PUCT_SOS.mainAIvsAI: separate networks/players for each side.
                    network_red = GameNetwork()
                    network_blue = GameNetwork()
                    network_red.load(full_model_path)
                    network_blue.load(full_model_path)
                    self.ai_players = {
                        RED: PUCTPlayer(network=network_red),
                        BLUE: PUCTPlayer(network=network_blue),
                    }
                self.ai_iterations = PUCT_ITERATIONS

        self._update_display()

    def _is_human_turn(self):
        return self.game_mode.get() == "Player vs AI" and self.game.current_player == self.human_color

    def _is_ai_turn(self):
        if self.game.status != ONGOING:
            return False
        if self.game.current_player not in self.ai_players:
            return False
        if self.game_mode.get() == "AI vs AI" and not self.ai_vs_ai_started:
            return False
        return True

    def _start_ai_vs_ai(self):
        if self.game_mode.get() != "AI vs AI":
            return
        if self.ai_thinking or self.game.status != ONGOING or self.ai_vs_ai_started:
            return

        self.ai_vs_ai_started = True
        self.selected_cell = None
        self._hide_letter_selection()
        self._update_display()
        if self._is_ai_turn():
            self.root.after(50, self._ai_move)

    def _draw_board(self):
        self.canvas.delete("all")

        # Draw grid
        for i in range(BOARD_SIZE + 1):
            # Vertical lines
            x = self.PADDING + i * self.CELL_SIZE
            self.canvas.create_line(x, self.PADDING, x, self.PADDING + BOARD_SIZE * self.CELL_SIZE, fill="black")
            # Horizontal lines
            y = self.PADDING + i * self.CELL_SIZE
            self.canvas.create_line(self.PADDING, y, self.PADDING + BOARD_SIZE * self.CELL_SIZE, y, fill="black")

        # Draw row/column labels
        for i in range(BOARD_SIZE):
            # Row labels
            y = self.PADDING + i * self.CELL_SIZE + self.CELL_SIZE // 2
            self.canvas.create_text(self.PADDING // 2, y, text=str(i), font=("Arial", 10))
            # Column labels
            x = self.PADDING + i * self.CELL_SIZE + self.CELL_SIZE // 2
            self.canvas.create_text(x, self.PADDING // 2, text=str(i), font=("Arial", 10))

        # Draw pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell_value = self.game.board[row, col]
                if cell_value != EMPTY:
                    x = self.PADDING + col * self.CELL_SIZE + self.CELL_SIZE // 2
                    y = self.PADDING + row * self.CELL_SIZE + self.CELL_SIZE // 2
                    letter = "S" if cell_value == S else "O"
                    self.canvas.create_text(x, y, text=letter, font=("Arial", 24, "bold"))

        # Highlight selected cell
        if self.selected_cell:
            row, col = self.selected_cell
            x1 = self.PADDING + col * self.CELL_SIZE + 2
            y1 = self.PADDING + row * self.CELL_SIZE + 2
            x2 = x1 + self.CELL_SIZE - 4
            y2 = y1 + self.CELL_SIZE - 4
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=3)

        # Highlight legal moves
        if self.game.status == ONGOING and self._is_human_turn() and not self.ai_thinking:
            for row, col in self.game.legal_moves():
                if (row, col) != self.selected_cell:
                    x1 = self.PADDING + col * self.CELL_SIZE + 2
                    y1 = self.PADDING + row * self.CELL_SIZE + 2
                    x2 = x1 + self.CELL_SIZE - 4
                    y2 = y1 + self.CELL_SIZE - 4
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="lightgreen", width=1)

    def _update_display(self):
        self._draw_board()

        # Update scores
        if self.game_mode.get() == "Player vs AI":
            self.red_score_label.configure(text=f"RED (You): {self.game.scores[RED]}")
            self.blue_score_label.configure(text=f"BLUE (AI): {self.game.scores[BLUE]}")
        else:
            self.red_score_label.configure(text=f"RED (AI 1): {self.game.scores[RED]}")
            self.blue_score_label.configure(text=f"BLUE (AI 2): {self.game.scores[BLUE]}")

        # Update status
        if self.game.status == ONGOING:
            if self.ai_thinking:
                if self.game_mode.get() == "Player vs AI":
                    status_text = "AI is thinking..."
                else:
                    thinker = "AI 1" if self.game.current_player == RED else "AI 2"
                    status_text = f"{thinker} is thinking..."
                self.status_label.configure(text=status_text, foreground="orange")
            elif self.game_mode.get() == "AI vs AI" and not self.ai_vs_ai_started:
                self.status_label.configure(text="Press Start to begin AI vs AI", foreground="orange")
            elif self._is_human_turn():
                self.status_label.configure(text="Your turn (RED)", foreground="red")
            else:
                if self.game_mode.get() == "Player vs AI":
                    self.status_label.configure(text="AI's turn (BLUE)", foreground="blue")
                elif self.game.current_player == RED:
                    self.status_label.configure(text="AI 1's turn (RED)", foreground="red")
                else:
                    self.status_label.configure(text="AI 2's turn (BLUE)", foreground="blue")
        else:
            if self.game.status == DRAW:
                self.status_label.configure(text="DRAW!", foreground="gray")
            elif self.game_mode.get() == "Player vs AI":
                if self.game.status == RED:
                    self.status_label.configure(text="You WIN!", foreground="green")
                else:
                    self.status_label.configure(text="AI WINS!", foreground="red")
            else:
                if self.game.status == RED:
                    self.status_label.configure(text="AI 1 WINS!", foreground="green")
                else:
                    self.status_label.configure(text="AI 2 WINS!", foreground="green")

        self._update_start_button()

    def _update_start_button(self):
        if self.game_mode.get() != "AI vs AI":
            self.start_button.configure(text="Start AI vs AI", state="disabled")
            return
        if self.ai_vs_ai_started:
            self.start_button.configure(text="AI vs AI Running", state="disabled")
            return
        if self.ai_thinking or self.game.status != ONGOING:
            self.start_button.configure(text="Start AI vs AI", state="disabled")
            return
        self.start_button.configure(text="Start AI vs AI", state="normal")

    def _on_canvas_click(self, event):
        if self.ai_thinking or self.game.status != ONGOING:
            return
        if not self._is_human_turn():
            return

        # Calculate cell from click position
        col = (event.x - self.PADDING) // self.CELL_SIZE
        row = (event.y - self.PADDING) // self.CELL_SIZE

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            if (row, col) in self.game.legal_moves():
                self._show_letter_selection(row, col)
                self._draw_board()

    def _place_letter(self, letter):
        if self.selected_cell is None:
            return

        row, col = self.selected_cell
        move = (row, col, letter)

        try:
            self.game.make_move(move)
        except ValueError as e:
            messagebox.showerror("Invalid Move", str(e))
            self._cancel_selection()
            return

        self.selected_cell = None
        self._hide_letter_selection()
        self._update_display()

        # Check if game is over
        if self.game.status != ONGOING:
            return

        # AI's turn - schedule after UI has updated
        if self._is_ai_turn():
            self.root.after(50, self._ai_move)

    def _ai_move(self):
        if self.ai_thinking or not self._is_ai_turn():
            return

        ai_player = self.ai_players[self.game.current_player]

        self.ai_thinking = True
        self._update_display()

        # Run AI in a separate thread to keep UI responsive
        def ai_thread():
            move = ai_player.choose_move(self.game, iterations=self.ai_iterations)
            self.root.after(0, lambda: self._apply_ai_move(move))

        thread = threading.Thread(target=ai_thread, daemon=True)
        thread.start()

    def _apply_ai_move(self, move):
        if move:
            self.game.make_move(move)

        self.ai_thinking = False
        self._update_display()
        if self._is_ai_turn():
            self.root.after(120, self._ai_move)


def main():
    root = tk.Tk()
    app = SOSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
