import tkinter as tk
from tkinter import ttk, messagebox, font
import chess
import chess.engine
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import threading
import time
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import json
import os

TILE_SIZE = 80
BOARD_SIZE = 8 * TILE_SIZE
SIDEBAR_WIDTH = 350
WINDOW_HEIGHT = BOARD_SIZE + 120


COLORS = {
    'light_square': '#F5F5DC',
    'dark_square': '#8B7355',
    'highlight': '#FFE135',
    'selected': '#87CEEB',
    'legal_move': '#98FB98',
    'capture_move': '#FFB6C1',
    'check': '#FF6347',
    'last_move': '#DDA0DD',

    'bg_primary': '#1E1E2E',
    'bg_secondary': '#2A2A3E',
    'bg_tertiary': '#3A3A54',
    'card_bg': '#252538',
    'text_primary': '#FFFFFF',
    'text_secondary': '#B4B4CE',
    'text_accent': '#F7C41F',
    'accent_primary': '#6C5CE7',
    'accent_secondary': '#74B9FF',
    'success': '#00B894',
    'warning': '#FDCB6E',
    'danger': '#E84393',
    'info': '#74B9FF'
}


@dataclass
class GameSettings:
    """Game settings configuration"""
    ai_difficulty: str = "medium"
    show_legal_moves: bool = True
    show_coordinates: bool = True
    sound_enabled: bool = True
    auto_promote: bool = False
    highlight_last_move: bool = True
    animation_speed: float = 0.3


class ChessAI:



    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.transposition_table = {}
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.nodes_searched = 0
        self.max_depth = self._get_depth_for_difficulty()
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        self.position_tables = self._initialize_position_tables()

    def _get_depth_for_difficulty(self) -> int:

        depths = {
            "beginner": 2,
            "easy": 3,
            "medium": 4,
            "hard": 5,
            "expert": 6,
            "master": 7
        }
        return depths.get(self.difficulty, 4)

    def _initialize_position_tables(self) -> Dict:

        return {
            chess.PAWN: [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [10, 10, 20, 30, 30, 20, 10, 10],
                [5, 5, 10, 25, 25, 10, 5, 5],
                [0, 0, 0, 20, 20, 0, 0, 0],
                [5, -5, -10, 0, 0, -10, -5, 5],
                [5, 10, 10, -20, -20, 10, 10, 5],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            chess.KNIGHT: [
                [-50, -40, -30, -30, -30, -30, -40, -50],
                [-40, -20, 0, 0, 0, 0, -20, -40],
                [-30, 0, 10, 15, 15, 10, 0, -30],
                [-30, 5, 15, 20, 20, 15, 5, -30],
                [-30, 0, 15, 20, 20, 15, 0, -30],
                [-30, 5, 10, 15, 15, 10, 5, -30],
                [-40, -20, 0, 5, 5, 0, -20, -40],
                [-50, -40, -30, -30, -30, -30, -40, -50]
            ]
        }

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:

        self.nodes_searched = 0
        start_time = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        best_move = None
        best_score = -math.inf if board.turn == chess.WHITE else math.inf


        for depth in range(1, self.max_depth + 1):
            try:
                move, score = self._minimax(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
                if move:
                    best_move = move
                    best_score = score

                elapsed = time.time() - start_time
                if elapsed > 5.0:  # 5 second limit
                    break
            except Exception as e:
                print(f"Error in AI search at depth {depth}: {e}")
                break

        return best_move or random.choice(legal_moves)

    def _minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[
        Optional[chess.Move], float]:
        """Minimax algorithm with alpha-beta pruning"""
        self.nodes_searched += 1

        # Terminal node evaluation
        if depth == 0 or board.is_game_over():
            return None, self._evaluate_position(board)

        best_move = None
        moves = self._order_moves(board)

        if maximizing:
            max_eval = -math.inf
            for move in moves:
                board.push(move)
                _, eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return best_move, max_eval
        else:
            min_eval = math.inf
            for move in moves:
                board.push(move)
                _, eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return best_move, min_eval

    def _order_moves(self, board: chess.Board) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning"""
        moves = list(board.legal_moves)
        scored_moves = []

        for move in moves:
            score = 0

            # Prioritize captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]

            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()

            # Prioritize promotions
            if move.promotion:
                score += 800

            # Prioritize castling
            if board.is_castling(move):
                score += 60

            scored_moves.append((score, move))

        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    def _evaluate_position(self, board: chess.Board) -> float:

        if board.is_checkmate():
            return -math.inf if board.turn == chess.WHITE else math.inf

        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0

        score = 0

        # Material evaluation
        score += self._evaluate_material(board)

        # Positional factors
        score += self._evaluate_mobility(board) * 0.1
        score += self._evaluate_king_safety(board) * 0.5
        score += self._evaluate_pawn_structure(board) * 0.3

        return score

    def _evaluate_material(self, board: chess.Board) -> float:

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        return white_material - black_material

    def _evaluate_mobility(self, board: chess.Board) -> float:

        white_mobility = len(list(board.legal_moves))
        board.turn = not board.turn
        black_mobility = len(list(board.legal_moves))
        board.turn = not board.turn

        return white_mobility - black_mobility

    def _evaluate_king_safety(self, board: chess.Board) -> float:

        score = 0

        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        if white_king is not None and board.is_attacked_by(chess.BLACK, white_king):
            score -= 50

        if black_king is not None and board.is_attacked_by(chess.WHITE, black_king):
            score += 50

        return score

    def _evaluate_pawn_structure(self, board: chess.Board) -> float:

        score = 0

        # Evaluate white pawns
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        # Check for doubled pawns
        for file in range(8):
            white_pawns_on_file = sum(1 for square in white_pawns if chess.square_file(square) == file)
            black_pawns_on_file = sum(1 for square in black_pawns if chess.square_file(square) == file)

            if white_pawns_on_file > 1:
                score -= 10 * (white_pawns_on_file - 1)
            if black_pawns_on_file > 1:
                score += 10 * (black_pawns_on_file - 1)

        return score


class PieceImageLoader:

    def __init__(self, pieces_folder: str = "Pieces"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.pieces_folder = os.path.join(script_dir, pieces_folder)

    def load_piece_images(self, size: int = TILE_SIZE) -> Dict[str, ImageTk.PhotoImage]:

        images = {}
        piece_files = {
            'K': ['w_king_1x.png', 'white_king.png', 'wK.png'],
            'Q': ['w_queen_1x.png', 'white_queen.png', 'wQ.png'],
            'R': ['w_rook_1x.png', 'white_rook.png', 'wR.png'],
            'B': ['w_bishop_1x.png', 'white_bishop.png', 'wB.png'],
            'N': ['w_knight_1x.png', 'white_knight.png', 'wN.png'],
            'P': ['w_pawn_1x.png', 'white_pawn.png', 'wP.png'],
            'k': ['b_king_1x.png', 'black_king.png', 'bK.png'],
            'q': ['b_queen_1x.png', 'black_queen.png', 'bQ.png'],
            'r': ['b_rook_1x.png', 'black_rook.png', 'bR.png'],
            'b': ['b_bishop_1x.png', 'black_bishop.png', 'bB.png'],
            'n': ['b_knight_1x.png', 'black_knight.png', 'bN.png'],
            'p': ['b_pawn_1x.png', 'black_pawn.png', 'bP.png']
        }

        for symbol, possible_names in piece_files.items():
            image_loaded = False
            for filename in possible_names:
                filepath = os.path.join(self.pieces_folder, filename)
                if os.path.exists(filepath):
                    try:
                        img = Image.open(filepath).convert("RGBA")
                        img = img.resize((int(size * 0.8), int(size * 0.8)), Image.Resampling.LANCZOS)

                        shadow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                        shadow_offset = 2
                        shadow.paste(img, (shadow_offset, shadow_offset))
                        shadow = shadow.filter(ImageFilter.GaussianBlur(1))

                        final_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                        final_img.paste(shadow, (0, 0))
                        final_img.paste(img, (0, 0), img)

                        images[symbol] = ImageTk.PhotoImage(final_img)
                        image_loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue

            if not image_loaded:
                images[symbol] = self._create_fallback_image(symbol, size)

        return images

    def _create_fallback_image(self, symbol: str, size: int) -> ImageTk.PhotoImage:

        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        is_white = symbol.isupper()
        fill_color = '#FFFFFF' if is_white else '#000000'
        outline_color = '#000000' if is_white else '#FFFFFF'

        center = size // 2
        radius = size // 3
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill=fill_color,
            outline=outline_color,
            width=2
        )

        try:
            draw.text(
                (center, center),
                symbol.upper(),
                fill=outline_color,
                anchor="mm"
            )
        except:
            pass

        return ImageTk.PhotoImage(img)


class NotificationWidget:


    def __init__(self, parent):
        self.parent = parent
        self.notification_window = None

    def show_notification(self, title: str, message: str, duration: int = 4000,
                          notification_type: str = "info"):

        if self.notification_window:
            self.notification_window.destroy()

        self.notification_window = tk.Toplevel(self.parent)
        self.notification_window.title(title)
        self.notification_window.geometry("400x150")
        self.notification_window.resizable(False, False)

        self.notification_window.overrideredirect(True)

        self.notification_window.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        x = parent_x + (parent_width - 400) // 2
        y = parent_y + (parent_height - 150) // 2
        self.notification_window.geometry(f"400x150+{x}+{y}")

        colors = {
            'info': COLORS['info'],
            'success': COLORS['success'],
            'warning': COLORS['warning'],
            'danger': COLORS['danger']
        }
        bg_color = colors.get(notification_type, COLORS['info'])

        main_frame = tk.Frame(
            self.notification_window,
            bg=bg_color,
            relief='raised',
            bd=0
        )
        main_frame.pack(fill='both', expand=True, padx=2, pady=2)

        content_frame = tk.Frame(main_frame, bg=COLORS['card_bg'])
        content_frame.pack(fill='both', expand=True, padx=3, pady=3)

        title_label = tk.Label(
            content_frame,
            text=title,
            font=('Arial', 14, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card_bg']
        )
        title_label.pack(pady=(15, 5))

        message_label = tk.Label(
            content_frame,
            text=message,
            font=('Arial', 11),
            fg=COLORS['text_secondary'],
            bg=COLORS['card_bg'],
            wraplength=350
        )
        message_label.pack(pady=(0, 10))

        close_btn = tk.Button(
            content_frame,
            text="Close",
            font=('Arial', 9, 'bold'),
            fg='white',
            bg=bg_color,
            activebackground=bg_color,
            relief='flat',
            padx=15,
            pady=3,
            command=self.close_notification
        )
        close_btn.pack(pady=(0, 10))

        self.parent.after(duration, self.close_notification)

        self.notification_window.transient(self.parent)
        self.notification_window.grab_set()
        self.notification_window.focus_set()

    def close_notification(self):

        if self.notification_window:
            self.notification_window.grab_release()
            self.notification_window.destroy()
            self.notification_window = None


class ModernChessGUI:


    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Modern Chess Game")
        self.root.geometry(f"{BOARD_SIZE + SIDEBAR_WIDTH + 20}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLORS['bg_primary'])
        self.root.resizable(True, True)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", lambda event: self.root.attributes("-fullscreen", False))

        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.game_mode = "human"
        self.ai_color = chess.BLACK
        self.game_over = False
        self.move_history = []
        self.last_move = None

        self.settings = GameSettings()

        self.ai = ChessAI(self.settings.ai_difficulty)

        self.notification = NotificationWidget(self.root)

        self.piece_loader = PieceImageLoader()
        self.piece_images = self.piece_loader.load_piece_images()

        self.setup_ui()
        self.draw_board()
        self.update_status("White to move")

    def toggle_fullscreen(self, event=None):

        current_state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current_state)

    def setup_ui(self):

        style = ttk.Style()
        style.theme_use('clam')

        main_container = tk.Frame(self.root, bg=COLORS['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=15, pady=15)

        board_container = tk.Frame(main_container, bg=COLORS['bg_primary'])
        board_container.pack(side='left', padx=(0, 15))

        board_frame = tk.Frame(
            board_container,
            bg=COLORS['bg_secondary'],
            relief='flat',
            bd=0
        )
        board_frame.pack()

        canvas_frame = tk.Frame(board_frame, bg=COLORS['bg_secondary'])
        canvas_frame.pack(padx=8, pady=8)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            bg=COLORS['light_square'],
            highlightthickness=2,
            highlightbackground=COLORS['accent_primary'],
            relief='solid',
            bd=1
        )
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_square_click)
        self.canvas.bind('<Motion>', self.on_mouse_motion)

        right_panel = tk.Frame(main_container, bg=COLORS['bg_primary'])
        right_panel.pack(side='right', fill='y')

        self.create_title_section(right_panel)

        self.create_game_mode_section(right_panel)

        self.create_controls_section(right_panel)

        self.create_status_section(right_panel)

        self.create_history_section(right_panel)

    def create_title_section(self, parent):

        title_frame = tk.Frame(parent, bg=COLORS['bg_primary'])
        title_frame.pack(fill='x', pady=(0, 20))
        title_label = tk.Label(
            title_frame,
            text="♛ Chess ♛",
            font=('Arial', 22, 'bold'),
            fg=COLORS['text_accent'],
            bg=COLORS['bg_primary']
        )
        title_label.pack()
        subtitle_label = tk.Label(
            title_frame,
            text="Enhanced Chess Experience",
            font=('Arial', 10, 'italic'),
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_primary']
        )
        subtitle_label.pack()

    def create_game_mode_section(self, parent):
        """Create the game mode section"""
        mode_frame = self.create_section_frame(parent, "🎮 Game Mode")
        self.mode_var = tk.StringVar(value="human")

        human_frame = tk.Frame(mode_frame, bg=COLORS['card_bg'])
        human_frame.pack(fill='x', padx=5, pady=2)
        tk.Radiobutton(
            human_frame,
            text="👥 Human vs Human",
            variable=self.mode_var,
            value="human",
            command=self.change_game_mode,
            font=('Arial', 11),
            fg=COLORS['text_primary'],
            bg=COLORS['card_bg'],
            selectcolor=COLORS['accent_primary'],
            activebackground=COLORS['card_bg'],
            activeforeground=COLORS['text_primary']
        ).pack(anchor='w', padx=10, pady=5)

        ai_frame = tk.Frame(mode_frame, bg=COLORS['card_bg'])
        ai_frame.pack(fill='x', padx=5, pady=2)
        tk.Radiobutton(
            ai_frame,
            text="🤖 Human vs AI",
            variable=self.mode_var,
            value="ai",
            command=self.change_game_mode,
            font=('Arial', 11),
            fg=COLORS['text_primary'],
            bg=COLORS['card_bg'],
            selectcolor=COLORS['accent_primary'],
            activebackground=COLORS['card_bg'],
            activeforeground=COLORS['text_primary']
        ).pack(anchor='w', padx=10, pady=5)

    def create_controls_section(self, parent):

        controls_frame = self.create_section_frame(parent, "🕹 Game Controls")
        buttons = [
            ("⬆ New Game", self.new_game, COLORS['success']),
            ("→ Undo Move", self.undo_move, COLORS['warning']),
            ("🏳 Resign", self.resign, COLORS['danger'])
        ]

        for text, command, color in buttons:
            self.create_modern_button(controls_frame, text, command, color)

    def create_status_section(self, parent):

        status_frame = self.create_section_frame(parent, "📊 Game Status")
        self.status_label = tk.Label(
            status_frame,
            text="White to move",
            font=('Arial', 12, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card_bg'],
            wraplength=280,
            justify='center'
        )
        self.status_label.pack(padx=15, pady=15)

    def create_history_section(self, parent):

        history_frame = self.create_section_frame(parent, "📜 Move History")
        # Scrollable text widget with modern styling
        text_frame = tk.Frame(history_frame, bg=COLORS['card_bg'])
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.history_text = tk.Text(
            text_frame,
            height=12,
            width=30,
            font=('Consolas', 10),
            fg=COLORS['text_primary'],
            bg=COLORS['bg_primary'],
            insertbackground=COLORS['text_primary'],
            relief='flat',
            bd=0,
            padx=10,
            pady=5
        )
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical')
        scrollbar.config(command=self.history_text.yview)
        self.history_text.config(yscrollcommand=scrollbar.set)
        self.history_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def create_section_frame(self, parent, title):

        container = tk.Frame(parent, bg=COLORS['bg_primary'])
        container.pack(fill='x', pady=(0, 15))

        title_label = tk.Label(
            container,
            text=title,
            font=('Arial', 12, 'bold'),
            fg=COLORS['text_accent'],
            bg=COLORS['bg_primary']
        )
        title_label.pack(anchor='w', pady=(0, 5))

        content_frame = tk.Frame(
            container,
            bg=COLORS['card_bg'],
            relief='flat',
            bd=1
        )
        content_frame.pack(fill='x')
        return content_frame

    def create_modern_button(self, parent, text, command, color):

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Arial', 11, 'bold'),
            fg='white',
            bg=color,
            activebackground=color,
            activeforeground='white',
            relief='flat',
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        btn.pack(fill='x', padx=10, pady=3)

        def on_enter(e):
            btn.config(bg=self.lighten_color(color, 0.2))

        def on_leave(e):
            btn.config(bg=color)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def lighten_color(self, color, factor):

        try:

            color = color.lstrip('#')

            rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

            rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)

            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        except:
            return color

    def draw_board(self):

        self.canvas.delete("all")

        for row in range(8):
            for col in range(8):
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE

                square = chess.square(col, 7 - row)
                is_light = (row + col) % 2 == 0
                color = COLORS['light_square'] if is_light else COLORS['dark_square']

                if square == self.selected_square:
                    color = COLORS['selected']

                if (self.settings.highlight_last_move and self.last_move and
                        (square == self.last_move.from_square or square == self.last_move.to_square)):
                    color = COLORS['last_move']

                if self.board.is_check():
                    king_square = self.board.king(self.board.turn)
                    if square == king_square:
                        color = COLORS['check']

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=color,
                    tags=f"square_{square}"
                )

                if self.settings.show_coordinates:
                    if row == 7:
                        file_letter = chr(ord('a') + col)
                        self.canvas.create_text(
                            x2 - 8, y2 - 8,
                            text=file_letter,
                            font=('Arial', 8, 'bold'),
                            fill=COLORS['text_secondary']
                        )
                    if col == 0:
                        rank_number = str(8 - row)
                        self.canvas.create_text(
                            x1 + 8, y1 + 8,
                            text=rank_number,
                            font=('Arial', 8, 'bold'),
                            fill=COLORS['text_secondary']
                        )

        if self.settings.show_legal_moves and self.selected_square is not None:
            for move in self.legal_moves:
                if move.from_square == self.selected_square:
                    col = chess.square_file(move.to_square)
                    row = 7 - chess.square_rank(move.to_square)
                    x = col * TILE_SIZE + TILE_SIZE // 2
                    y = row * TILE_SIZE + TILE_SIZE // 2

                    if self.board.is_capture(move):

                        self.canvas.create_oval(
                            x - TILE_SIZE // 3, y - TILE_SIZE // 3,
                            x + TILE_SIZE // 3, y + TILE_SIZE // 3,
                            outline=COLORS['capture_move'],
                            width=4,
                            tags="legal_move"
                        )
                    else:

                        self.canvas.create_oval(
                            x - 8, y - 8, x + 8, y + 8,
                            fill=COLORS['legal_move'],
                            outline=COLORS['legal_move'],
                            tags="legal_move"
                        )


        self.draw_pieces()

    def draw_pieces(self):

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.draw_piece(square, piece)

    def draw_piece(self, square, piece):

        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        x = col * TILE_SIZE + TILE_SIZE // 2
        y = row * TILE_SIZE + TILE_SIZE // 2
        piece_symbol = piece.symbol()

        if piece_symbol in self.piece_images:
            self.canvas.create_image(
                x, y,
                image=self.piece_images[piece_symbol],
                tags=f"piece_{square}"
            )

    def on_square_click(self, event):

        if self.game_over:
            return

        col = event.x // TILE_SIZE
        row = event.y // TILE_SIZE

        if 0 <= col < 8 and 0 <= row < 8:
            square = chess.square(col, 7 - row)
            self.handle_square_selection(square)

    def handle_square_selection(self, square):

        piece = self.board.piece_at(square)

        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [move for move in self.board.legal_moves
                                    if move.from_square == square]
                self.draw_board()
        else:

            move = None
            for legal_move in self.legal_moves:
                if legal_move.to_square == square:
                    move = legal_move
                    break

            if move:

                if (move.promotion is None and piece and piece.piece_type == chess.PAWN and
                        ((piece.color == chess.WHITE and chess.square_rank(square) == 7) or
                         (piece.color == chess.BLACK and chess.square_rank(square) == 0))):

                    if self.settings.auto_promote:
                        move = chess.Move(move.from_square, move.to_square, chess.QUEEN)
                    else:
                        promotion_piece = self.ask_promotion()
                        if promotion_piece:
                            move = chess.Move(move.from_square, move.to_square, promotion_piece)
                        else:
                            self.selected_square = None
                            self.legal_moves = []
                            self.draw_board()
                            return

                self.make_move(move)
            else:

                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.legal_moves = [move for move in self.board.legal_moves
                                        if move.from_square == square]
                    self.draw_board()
                else:

                    self.selected_square = None
                    self.legal_moves = []
                    self.draw_board()

    def ask_promotion(self):

        promotion_window = tk.Toplevel(self.root)
        promotion_window.title("Pawn Promotion")
        promotion_window.geometry("300x150")
        promotion_window.configure(bg=COLORS['bg_primary'])
        promotion_window.resizable(False, False)
        promotion_window.transient(self.root)
        promotion_window.grab_set()

        result = [None]
        tk.Label(
            promotion_window,
            text="Choose promotion piece:",
            font=('Arial', 14, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['bg_primary']
        ).pack(pady=20)

        button_frame = tk.Frame(promotion_window, bg=COLORS['bg_primary'])
        button_frame.pack(pady=10)

        pieces = [
            ("♕ Queen", chess.QUEEN),
            ("♖ Rook", chess.ROOK),
            ("♗ Bishop", chess.BISHOP),
            ("♘ Knight", chess.KNIGHT)
        ]

        for text, piece_type in pieces:
            tk.Button(
                button_frame,
                text=text,
                font=('Arial', 12),
                fg='white',
                bg=COLORS['accent_primary'],
                activebackground=COLORS['accent_secondary'],
                relief='flat',
                padx=15,
                pady=5,
                command=lambda p=piece_type: [result.__setitem__(0, p), promotion_window.destroy()]
            ).pack(side='left', padx=2)

        self.root.wait_window(promotion_window)
        return result[0]

    def make_move(self, move):

        try:
            move_notation = self.board.san(move)
        except chess.IllegalMoveError:
            self.notification.show_notification(
                "Illegal Move",
                "That move is not allowed!",
                notification_type="warning"
            )
            return

        self.move_history.append(move_notation)
        self.last_move = move


        self.board.push(move)


        self.selected_square = None
        self.legal_moves = []

        self.draw_board()
        self.update_move_history()
        self.update_status()

        if self.board.is_game_over():
            self.handle_game_over()
        elif self.game_mode == "ai" and self.board.turn == self.ai_color:
            self.root.after(500, self.make_ai_move)

    def make_ai_move(self):

        def ai_move_thread():
            try:
                self.update_status("AI is thinking...")
                ai_move = self.ai.get_best_move(self.board)
                if ai_move:
                    self.root.after(0, lambda: self.make_move(ai_move))
                else:
                    self.root.after(0, self.handle_game_over)
            except Exception as e:
                print(f"AI move error: {e}")
                self.root.after(0, self.handle_game_over)

        threading.Thread(target=ai_move_thread, daemon=True).start()

    def update_status(self, custom_message=None):
        if custom_message:
            self.status_label.config(text=custom_message)
            return

        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            message = f"♚ Checkmate! {winner} wins!"
            self.status_label.config(text=message, fg=COLORS['success'])
        elif self.board.is_stalemate():
            message = "♔ Stalemate! The game is a draw."
            self.status_label.config(text=message, fg=COLORS['warning'])
        elif self.board.is_insufficient_material():
            message = "♙ Draw by insufficient material!"
            self.status_label.config(text=message, fg=COLORS['warning'])
        elif self.board.is_seventyfive_moves():
            message = "♙ Draw by 75-move rule!"
            self.status_label.config(text=message, fg=COLORS['warning'])
        elif self.board.is_fivefold_repetition():
            message = "♙ Draw by fivefold repetition!"
            self.status_label.config(text=message, fg=COLORS['warning'])
        elif self.board.is_check():
            current_player = "White" if self.board.turn == chess.WHITE else "Black"
            message = f"⚠ {current_player} is in check!"
            self.status_label.config(text=message, fg=COLORS['danger'])
        else:
            current_player = "White" if self.board.turn == chess.WHITE else "Black"
            message = f"{current_player} to move"
            self.status_label.config(text=message, fg=COLORS['text_primary'])

    def update_move_history(self):
        self.history_text.delete(1.0, tk.END)
        move_pairs = []
        for i in range(0, len(self.move_history), 2):
            move_num = (i // 2) + 1
            white_move = self.move_history[i]
            black_move = self.move_history[i + 1] if i + 1 < len(self.move_history) else ""
            move_pairs.append(f"{move_num}. {white_move} {black_move}\n")

        for pair in move_pairs:
            self.history_text.insert(tk.END, pair)

        self.history_text.see(tk.END)

    def handle_game_over(self):
        self.game_over = True
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.notification.show_notification(
                "Game Over",
                f"Checkmate! {winner} wins!",
                notification_type="success"
            )
        elif self.board.is_stalemate():
            self.notification.show_notification(
                "Game Over",
                "Stalemate! The game is a draw.",
                notification_type="info"
            )
        elif self.board.is_insufficient_material():
            self.notification.show_notification(
                "Game Over",
                "Draw by insufficient material!",
                notification_type="info"
            )
        elif self.board.is_seventyfive_moves():
            self.notification.show_notification(
                "Game Over",
                "Draw by 75-move rule!",
                notification_type="info"
            )
        elif self.board.is_fivefold_repetition():
            self.notification.show_notification(
                "Game Over",
                "Draw by fivefold repetition!",
                notification_type="info"
            )

    def new_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.move_history = []
        self.last_move = None
        self.draw_board()
        self.update_status("White to move")
        self.history_text.delete(1.0, tk.END)
        self.notification.show_notification(
            "New Game",
            "A new game has started. Good luck!",
            notification_type="success"
        )

    def undo_move(self):
        if len(self.board.move_stack) == 0:
            self.notification.show_notification(
                "Cannot Undo",
                "No moves to undo!",
                notification_type="warning"
            )
            return

        moves_to_undo = 2 if self.game_mode == "ai" and len(self.board.move_stack) >= 2 else 1

        for _ in range(moves_to_undo):
            if len(self.board.move_stack) > 0:
                self.board.pop()
                if self.move_history:
                    self.move_history.pop()

        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
        self.draw_board()
        self.update_status()
        self.update_move_history()

    def resign(self):
        if self.game_over:
            return

        current_player = "White" if self.board.turn == chess.WHITE else "Black"
        winner = "Black" if current_player == "White" else "White"
        self.game_over = True

        self.notification.show_notification(
            "Game Over",
            f"{current_player} resigned. {winner} wins!",
            notification_type="info"
        )

        self.status_label.config(
            text=f"{current_player} resigned. {winner} wins!",
            fg=COLORS['danger']
        )

    def change_game_mode(self):
        self.game_mode = self.mode_var.get()
        if self.game_mode == "ai":
            self.ai_color = chess.BLACK
        self.new_game()

    def on_mouse_motion(self, event):
        pass

    def run(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.notification.show_notification(
            "Welcome to Chess!",
            "Enjoy your game with enhanced AI and modern interface.",
            notification_type="success"
        )

        self.root.mainloop()


if __name__ == "__main__":
    try:
        game = ModernChessGUI()
        game.run()
    except Exception as e:
        print(f"Error starting chess game: {e}")
        messagebox.showerror("Error", f"Failed to start chess game:\n{e}")