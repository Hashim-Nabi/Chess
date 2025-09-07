import tkinter as tk
from tkinter import messagebox
import chess
from PIL import Image, ImageTk
import os
import random
import math
import time
from collections import defaultdict

# Constants and Configuration
TILE_SIZE = 64

PIECE_NAMES = {
    'P': 'w_pawn_1x', 'N': 'w_knight_1x', 'B': 'w_bishop_1x',
    'R': 'w_rook_1x', 'Q': 'w_queen_1x', 'K': 'w_king_1x',
    'p': 'b_pawn_1x', 'n': 'b_knight_1x', 'b': 'b_bishop_1x',
    'r': 'b_rook_1x', 'q': 'b_queen_1x', 'k': 'b_king_1x'
}

# Initialize piece images dictionary
PIECE_IMAGES = {}

IMAGE_DIR = r"C:\Users\LTC\OneDrive\Desktop\chess python\Pieces"

class ChessAI:
    def __init__(self, difficulty="medium"):
        self.difficulty = difficulty
        self.transposition_table = {}
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.nodes_searched = 0
        self.max_depth = self._set_difficulty()
        
    def _set_difficulty(self):
        """Set search parameters based on difficulty level"""
        difficulties = {
            "easy": 2,
            "medium": 4,
            "hard": 6,
            "expert": 8
        }
        return difficulties.get(self.difficulty, 4)
    
    def make_ai_move(self, board):
        """Main interface for AI to make a move"""
        self.nodes_searched = 0
        start_time = time.time()
        
        # Get list of legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        if len(legal_moves) <= 10:
            move = self.alpha_beta_search(board)
        else:
            move = self.iterative_deepening(board)
        
        time_used = time.time() - start_time
        if time_used > 0:
            print(f"AI searched {self.nodes_searched} nodes in {time_used:.2f}s")
        return move
    
    def iterative_deepening(self, board):
        """Iterative deepening search with time management"""
        best_move = None
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        for depth in range(1, self.max_depth + 1):
            try:
                move, evaluation = self.alpha_beta_search(board, depth=depth)
                if move:
                    best_move = move
                    print(f"Depth {depth}: Best move {move} (eval: {evaluation})")
            except Exception as e:
                print(f"Error at depth {depth}: {e}")
                break
        
        return best_move or random.choice(legal_moves)
    
    def alpha_beta_search(self, board, depth=None, alpha=-math.inf, beta=math.inf, maximizing_player=True):
        """Alpha-beta pruning with move ordering and transposition table"""
        if depth is None:
            depth = self.max_depth
        
        self.nodes_searched += 1
        
        # Check transposition table
        board_key = board.fen()
        if board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry['depth'] >= depth:
                return entry['move'], entry['score']
        
        if depth == 0 or board.is_game_over():
            return None, self.evaluate_board(board)
        
        ordered_moves = self.order_moves(board)
        if not ordered_moves:
            return None, self.evaluate_board(board)
        
        best_move = None
        if maximizing_player:
            max_eval = -math.inf
            for move in ordered_moves:
                board.push(move)
                _, evaluation = self.alpha_beta_search(board, depth-1, alpha, beta, False)
                board.pop()
                
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        self._store_killer_move(move, depth, board)
                        break
            
            if best_move:
                self.transposition_table[board_key] = {'move': best_move, 'score': max_eval, 'depth': depth}
            return best_move, max_eval
        else:
            min_eval = math.inf
            for move in ordered_moves:
                board.push(move)
                _, evaluation = self.alpha_beta_search(board, depth-1, alpha, beta, True)
                board.pop()
                
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        self._store_killer_move(move, depth, board)
                        break
            
            if best_move:
                self.transposition_table[board_key] = {'move': best_move, 'score': min_eval, 'depth': depth}
            return best_move, min_eval
    
    def order_moves(self, board):
        """Order moves for optimal alpha-beta pruning"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
            
        scored_moves = []
        for move in legal_moves:
            score = 0
            
            # Prioritize captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * self._piece_value(victim) - self._piece_value(attacker)
            
            # Killer move heuristic
            board_key = board.fen()
            if move in self.killer_moves[board_key]:
                score += 900
                
            # History heuristic
            score += self.history_table.get(move.uci(), 0)
            
            # Promotion bonus
            if move.promotion:
                score += 500
                
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for (score, move) in scored_moves]
    
    def _store_killer_move(self, move, depth, board):
        """Store killer moves for move ordering"""
        board_key = board.fen()
        if move not in self.killer_moves[board_key]:
            self.killer_moves[board_key].insert(0, move)
            if len(self.killer_moves[board_key]) > 2:
                self.killer_moves[board_key].pop()
    
    def evaluate_board(self, board):
        """Comprehensive board evaluation function"""
        if board.is_checkmate():
            return -math.inf if board.turn else math.inf
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # Calculate material balance
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self._piece_value(piece)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        material = white_material - black_material
        
        # Positional evaluation
        positional = self._evaluate_position(board)
        
        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))
        if board.turn == chess.BLACK:
            mobility = -mobility
        
        # King safety and pawn structure
        king_safety = self._evaluate_king_safety(board)
        pawn_structure = self._evaluate_pawn_structure(board)
        
        evaluation = (
            1.0 * material +
            0.5 * positional +
            0.1 * mobility +
            0.3 * king_safety +
            0.2 * pawn_structure
        )
        
        return evaluation if board.turn == chess.WHITE else -evaluation
    
    def _piece_value(self, piece):
        """Return the value of a piece"""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        return values.get(piece.piece_type, 0)
    
    def _evaluate_position(self, board):
        """Evaluate piece positions using piece-square tables"""
        positional_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                file, rank = chess.square_file(square), chess.square_rank(square)
                center_score = (3.5 - abs(file - 3.5)) * (3.5 - abs(rank - 3.5)) * 0.1
                
                # Knights prefer center squares
                if piece.piece_type == chess.KNIGHT:
                    if rank in [2, 3, 4, 5] and file in [2, 3, 4, 5]:
                        center_score *= 2
                
                positional_score += center_score if piece.color == chess.WHITE else -center_score
        
        return positional_score
    
    def _evaluate_king_safety(self, board):
        """Evaluate king safety"""
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square is None or black_king_square is None:
            return 0
        
        # Simple king safety evaluation based on king position
        white_king_file = chess.square_file(white_king_square)
        black_king_file = chess.square_file(black_king_square)
        
        # Penalty for king on open files (simplified)
        white_penalty = 0
        black_penalty = 0
        
        # Count pawns in front of king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                if piece.color == chess.WHITE and file == white_king_file:
                    white_penalty -= 1
                elif piece.color == chess.BLACK and file == black_king_file:
                    black_penalty -= 1
        
        return black_penalty - white_penalty
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure"""
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
        
        white_doubled = self._count_doubled_pawns(white_pawns)
        black_doubled = self._count_doubled_pawns(black_pawns)
        
        white_isolated = self._count_isolated_pawns(white_pawns)
        black_isolated = self._count_isolated_pawns(black_pawns)
        
        white_passed = self._count_passed_pawns(white_pawns, black_pawns)
        black_passed = self._count_passed_pawns(black_pawns, white_pawns)
        
        return (black_doubled - white_doubled + 
                black_isolated - white_isolated + 
                white_passed - black_passed) * 0.5
    
    def _count_doubled_pawns(self, pawns):
        """Count doubled pawns"""
        files = [chess.square_file(p) for p in pawns]
        return len(files) - len(set(files))
    
    def _count_isolated_pawns(self, pawns):
        """Count isolated pawns"""
        files = set(chess.square_file(p) for p in pawns)
        isolated = 0
        for file in files:
            if (file - 1) not in files and (file + 1) not in files:
                isolated += sum(1 for p in pawns if chess.square_file(p) == file)
        return isolated
    
    def _count_passed_pawns(self, pawns, opponent_pawns):
        """Count passed pawns"""
        passed = 0
        opponent_files = set(chess.square_file(p) for p in opponent_pawns)
        for pawn in pawns:
            file = chess.square_file(pawn)
            if file not in opponent_files and (file - 1) not in opponent_files and (file + 1) not in opponent_files:
                passed += 1
        return passed

def load_piece_images():
    """Load all chess piece images"""
    if not os.path.exists(IMAGE_DIR):
        print(f"[WARNING] Image directory not found: {IMAGE_DIR}")
        return False
        
    loaded_count = 0
    for symbol, filename in PIECE_NAMES.items():
        path = os.path.join(IMAGE_DIR, f"{filename}.png")
        if os.path.exists(path):
            try:
                img = Image.open(path)
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                PIECE_IMAGES[symbol] = ImageTk.PhotoImage(img)
                loaded_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to load image {path}: {e}")
        else:
            print(f"[WARNING] Missing image: {path}")
    
    print(f"[INFO] Loaded {loaded_count} piece images")
    return loaded_count > 0

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chess Game - Main Menu")
        self.geometry("300x400")
        self.resizable(False, False)

        # Center buttons vertically
        self.frame = tk.Frame(self)
        self.frame.pack(expand=True)

        title = tk.Label(self.frame, text="Chess Game", font=("Arial", 24, "bold"))
        title.pack(pady=30)

        btn_play = tk.Button(self.frame, text="Play", font=("Arial", 16), width=15, command=self.open_play_menu)
        btn_play.pack(pady=10)

        btn_settings = tk.Button(self.frame, text="Settings", font=("Arial", 16), width=15, command=self.open_settings)
        btn_settings.pack(pady=10)

        btn_exit = tk.Button(self.frame, text="Exit", font=("Arial", 16), width=15, command=self.quit)
        btn_exit.pack(pady=10)

    def open_play_menu(self):
        PlayMenu(self)

    def open_settings(self):
        messagebox.showinfo("Settings", "Settings will be added later.")

class PlayMenu(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Choose Mode")
        self.geometry("300x300")
        self.resizable(False, False)
        self.parent = parent
        self.transient(parent)
        self.grab_set()

        label = tk.Label(self, text="Choose Play Mode", font=("Arial", 18, "bold"))
        label.pack(pady=20)

        difficulty_frame = tk.Frame(self)
        difficulty_frame.pack(pady=10)

        self.difficulty = tk.StringVar(value="medium")
        difficulties = [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard"), ("Expert", "expert")]

        for text, mode in difficulties:
            rb = tk.Radiobutton(
                difficulty_frame, 
                text=text, 
                variable=self.difficulty, 
                value=mode,
                font=("Arial", 12)
            )
            rb.pack(anchor='w', padx=20, pady=2)

        btn_single = tk.Button(self, text="Single Player", font=("Arial", 14), width=20, command=lambda: self.start_game("single"))
        btn_single.pack(pady=10)

        btn_multi = tk.Button(self, text="Multiplayer", font=("Arial", 14), width=20, command=lambda: self.start_game("multi"))
        btn_multi.pack(pady=10)

    def start_game(self, mode):
        ChessGameWindow(self.parent, mode, self.difficulty.get() if mode == "single" else None)
        self.destroy()

class ChessGameWindow(tk.Toplevel):
    def __init__(self, parent, mode, difficulty=None):
        super().__init__(parent)
        self.title("Chess Game - Playing")
        self.mode = mode
        self.geometry(f"{8*TILE_SIZE}x{8*TILE_SIZE + 50}")
        self.resizable(False, False)
        self.transient(parent)

        # Load piece images before creating the board
        if not PIECE_IMAGES:
            images_loaded = load_piece_images()
            if not images_loaded:
                messagebox.showwarning("Images Not Found", 
                                     "Chess piece images not found. The game will use text symbols instead.")

        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        
        # Initialize AI if single player mode
        self.ai = ChessAI(difficulty) if mode == "single" else None

        self.canvas = tk.Canvas(self, width=8*TILE_SIZE, height=8*TILE_SIZE)
        self.canvas.pack()

        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)

        self.resign_btn = tk.Button(button_frame, text="Resign", command=self.resign_game)
        self.resign_btn.pack(side='left', padx=5)

        self.new_game_btn = tk.Button(button_frame, text="New Game", command=self.new_game)
        self.new_game_btn.pack(side='left', padx=5)

        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#F0D9B5", "#B58863"]
        
        # Draw chess board squares
        for row in range(8):
            for col in range(8):
                x1 = col * TILE_SIZE
                y1 = (7 - row) * TILE_SIZE
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                col = chess.square_file(square)
                row = chess.square_rank(square)
                x = col * TILE_SIZE
                y = (7 - row) * TILE_SIZE
                
                # Use images if available, otherwise use text
                if symbol in PIECE_IMAGES:
                    self.canvas.create_image(x, y, anchor="nw", image=PIECE_IMAGES[symbol])
                else:
                    # Fallback to text representation
                    text_color = "white" if piece.color == chess.WHITE else "black"
                    self.canvas.create_text(
                        x + TILE_SIZE//2, y + TILE_SIZE//2,
                        text=symbol, font=("Arial", 24, "bold"),
                        fill=text_color
                    )

        # Highlight selected square
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = chess.square_rank(self.selected_square)
            x1 = col * TILE_SIZE
            y1 = (7 - row) * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="red", width=3)

        # Show legal moves
        for square in self.legal_moves:
            col = chess.square_file(square)
            row = chess.square_rank(square)
            x = col * TILE_SIZE + TILE_SIZE // 2
            y = (7 - row) * TILE_SIZE + TILE_SIZE // 2
            self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="yellow", outline="orange", width=2)

    def on_click(self, event):
        if self.game_over:
            return
            
        col = event.x // TILE_SIZE
        row = 7 - (event.y // TILE_SIZE)
        if not (0 <= col <= 7 and 0 <= row <= 7):
            return
            
        clicked_square = chess.square(col, row)

        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.legal_moves = [move.to_square for move in self.board.legal_moves 
                                  if move.from_square == clicked_square]
        else:
            move = self.handle_move(clicked_square)
            if move:
                self.board.push(move)
                self.selected_square = None
                self.legal_moves = []
                self.draw_board()
                self.check_game_state()
                
                # AI move in single player mode
                if (self.mode == "single" and not self.board.is_game_over() and 
                    self.board.turn == chess.BLACK and not self.game_over):
                    self.after(500, self.make_ai_move)
            else:
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = clicked_square
                    self.legal_moves = [move.to_square for move in self.board.legal_moves 
                                      if move.from_square == clicked_square]
                else:
                    self.selected_square = None
                    self.legal_moves = []

        self.draw_board()

    def make_ai_move(self):
        if self.board.is_game_over() or self.game_over:
            return
            
        move = self.ai.make_ai_move(self.board)
        if move and move in self.board.legal_moves:
            self.board.push(move)
            self.selected_square = None
            self.legal_moves = []
            self.draw_board()
            self.check_game_state()

    def handle_move(self, clicked_square):
        piece = self.board.piece_at(self.selected_square)
        promotion_needed = False

        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(clicked_square)
            if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
                promotion_needed = True

        move = chess.Move(self.selected_square, clicked_square)

        if promotion_needed:
            promotion_piece = self.ask_promotion(piece.color)
            if promotion_piece is None:
                return None
            move = chess.Move(self.selected_square, clicked_square, promotion=promotion_piece)

        return move if move in self.board.legal_moves else None

    def ask_promotion(self, color):
        promotion_window = tk.Toplevel(self)
        promotion_window.title("Choose Promotion Piece")
        promotion_window.geometry("320x100")
        promotion_window.resizable(False, False)
        promotion_window.transient(self)
        promotion_window.grab_set()

        chosen_piece = tk.IntVar()
        chosen_piece.set(0)

        frame = tk.Frame(promotion_window)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        title_label = tk.Label(frame, text="Choose promotion piece:", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        button_frame = tk.Frame(frame)
        button_frame.pack()

        pieces = [
            ('Queen', chess.QUEEN, 'Q' if color == chess.WHITE else 'q'),
            ('Rook', chess.ROOK, 'R' if color == chess.WHITE else 'r'),
            ('Bishop', chess.BISHOP, 'B' if color == chess.WHITE else 'b'),
            ('Knight', chess.KNIGHT, 'N' if color == chess.WHITE else 'n')
        ]

        def select_piece(piece_type):
            chosen_piece.set(piece_type)
            promotion_window.destroy()

        for name, piece_type, symbol in pieces:
            btn_frame = tk.Frame(button_frame)
            btn_frame.pack(side='left', padx=5)

            if symbol in PIECE_IMAGES:
                btn = tk.Button(
                    btn_frame,
                    image=PIECE_IMAGES[symbol],
                    command=lambda p=piece_type: select_piece(p),
                    relief='raised',
                    borderwidth=2,
                    bg='white'
                )
                btn.pack()
                label = tk.Label(btn_frame, text=name, font=('Arial', 8))
                label.pack()
            else:
                btn = tk.Button(
                    btn_frame,
                    text=name,
                    command=lambda p=piece_type: select_piece(p),
                    width=8,
                    height=2
                )
                btn.pack()

        cancel_btn = tk.Button(frame, text="Cancel", command=promotion_window.destroy, bg='lightcoral')
        cancel_btn.pack(pady=(10, 0))

        promotion_window.wait_window()

        selected = chosen_piece.get()
        return selected if selected != 0 else None

    def check_game_state(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            messagebox.showinfo("Game Over", f"{winner} wins by checkmate!")
            self.end_game()
        elif self.board.is_stalemate():
            messagebox.showinfo("Game Over", "Game drawn by stalemate.")
            self.end_game()
        elif self.board.is_insufficient_material():
            messagebox.showinfo("Game Over", "Game drawn by insufficient material.")
            self.end_game()
        elif self.board.is_check():
            current_player = "White" if self.board.turn == chess.WHITE else "Black"
            messagebox.showinfo("Check", f"{current_player} is in check!")

    def resign_game(self):
        if self.game_over:
            return
            
        if messagebox.askyesno("Resign", "Are you sure you want to resign?"):
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            messagebox.showinfo("Game Over", f"{winner} wins by resignation!")
            self.end_game()

    def new_game(self):
        if messagebox.askyesno("New Game", "Start a new game?"):
            self.destroy()
            PlayMenu(self.master)

    def end_game(self):
        self.game_over = True
        self.canvas.unbind("<Button-1>")
        self.resign_btn.config(state='disabled')

if __name__ == "__main__":
    try:
        app = MainMenu()
        app.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start chess game: {e}")