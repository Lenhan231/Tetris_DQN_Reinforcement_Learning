"""
STEP 1: TETRIS GAME ENGINE (Core Game Logic)
=============================================
Mục đích: Xây dựng game Tetris cơ bản với đầy đủ mechanics

═════════════════════════════════════════════════════════════════════════════
WHAT IS TETRIS?
═════════════════════════════════════════════════════════════════════════════

Classic tile-matching game:
- Board 20×10 (height × width)
- 7 loại tetromino pieces (I, O, T, S, Z, L, J)
- Mỗi step: piece rơi xuống → khi chạm dừng
- Xóa hàng đầy (full row) → gain points
- Game over khi piece không vừa ở vị trí spawn

═════════════════════════════════════════════════════════════════════════════
KEY MECHANICS
═════════════════════════════════════════════════════════════════════════════

1. PIECES (Tetrominos):
   └─ 7 types: O, T, S, Z, I, L, J (mỗi type có shape khác nhau)
   └─ Mỗi piece có 4 states rotation (xoay 90°)
   └─ O-piece (vàng) không cần rotate

2. COLLISION DETECTION:
   └─ Kiểm tra piece có vượt biên (trái/phải/dưới)
   └─ Kiểm tra piece có chạm khối khác

3. PIECE DROP:
   └─ Piece rơi từ trên xuống
   └─ Dừng khi chạm tường dưới hoặc khối khác
   └─ Lock piece lên board

4. LINE CLEAR:
   └─ Kiểm tra hàng nào đầy (full row)
   └─ Xóa hàng → giảm high → insert empty row ở trên
   └─ Scoring: 1 line = 1 + 1²×10, 2 lines = 1 + 4×10, etc

5. STATE FEATURES (để train RL):
   └─ lines_cleared: số hàng đã xóa
   └─ holes: ô trống bị che phủ bên dưới khối
   └─ bumpiness: tổng chênh lệch chiều cao giữa các cột
   └─ height: tổng chiều cao của tất cả cột

═════════════════════════════════════════════════════════════════════════════

Chạy: python tetris.py (demo game 5 steps)
"""

import random
from collections import deque

class TetrisGame:
    """Tetris game engine: board, pieces, collision, scoring, state tracking"""

    # 7 loại tetromino pieces (mỗi số = piece type/color)
    PIECES = [
        [[1, 1], [1, 1]],                    # O-piece (vàng): 2×2 square
        [[0, 2, 0], [2, 2, 2]],              # T-piece (tím): T shape
        [[0, 3, 3], [3, 3, 0]],              # S-piece (xanh lá): S shape
        [[4, 4, 0], [0, 4, 4]],              # Z-piece (đỏ): Z shape
        [[5, 5, 5, 5]],                      # I-piece (xanh dương): line
        [[0, 0, 6], [6, 6, 6]],              # L-piece (cam): L shape
        [[7, 0, 0], [7, 7, 7]]               # J-piece (xanh): J shape
    ]

    def __init__(self, height=20, width=10):
        """Khởi tạo game

        Args:
            height: Board height (default 20)
            width: Board width (default 10)
        """
        self.height = height
        self.width = width
        self.reset()

    def reset(self):
        """Reset game về trạng thái ban đầu

        Return:
            state_features: (lines_cleared, holes, bumpiness, total_height)
        """
        # Board: 0 = empty, 1-7 = piece type
        self.board = [[0] * self.width for _ in range(self.height)]

        # Game state
        self.score = 0
        self.tetrominoes = 0        # Số khối đã đặt
        self.cleared_lines = 0      # Số hàng đã xóa
        self.game_over = False
        self.current_piece_idx = 0  # Index của piece hiện tại (để track rotation)

        # Piece bag: 7-bag random system
        # (Tránh 2 pieces giống nhau liên tiếp, random hơn)
        self.piece_bag = list(range(len(self.PIECES)))
        random.shuffle(self.piece_bag)

        # Spawn khối đầu tiên
        self._spawn_new_piece()
        return self._get_state_features()

    def _spawn_new_piece(self):
        """Spawn khối tetromino mới ở vị trí spawn (giữa trên cùng)

        Dùng piece bag (7-bag system):
        - Shuffle 7 pieces → pop từ dưới lên
        - Khi bag hết → shuffle lại

        Game over nếu không có chỗ tại spawn position
        """
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.PIECES)))
            random.shuffle(self.piece_bag)

        # Lấy piece từ bag
        self.current_piece_idx = self.piece_bag.pop()
        self.current_piece = [row[:] for row in self.PIECES[self.current_piece_idx]]

        # Vị trí spawn: giữa trên cùng
        self.piece_x = self.width // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0

        # Kiểm tra collision tại spawn → Game Over
        if self._check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

    def _rotate_90(self, piece):
        """Xoay piece 90 độ (clockwise)

        Công thức: transpose + reverse mỗi row
        - Transpose: swap (row, col)
        - Reverse: reverse row order

        Ví dụ:
        [[1, 0],        [[1, 0],
         [1, 1]]  →  transpose →  [0, 1]]
        sau đó reverse → [[0, 1], [1, 0]] xoay 90°

        Args:
            piece: 2D list of piece shape

        Return:
            rotated: xoay 90° clockwise
        """
        num_rows = len(piece)
        num_cols = len(piece[0])
        rotated = []

        for i in range(num_cols):
            new_row = []
            for j in range(num_rows - 1, -1, -1):
                new_row.append(piece[j][i])
            rotated.append(new_row)

        return rotated

    def _check_collision(self, piece, x, y):
        """Kiểm tra collision của piece tại vị trí (x, y)

        Check 3 điều:
        1. Vượt biên trái/phải?
        2. Chạm đáy (y >= height)?
        3. Chạm khối khác trên board?

        Args:
            piece: 2D piece shape
            x, y: top-left position của piece trên board

        Return:
            True nếu collision, False nếu OK
        """
        for py in range(len(piece)):
            for px in range(len(piece[0])):
                if piece[py][px] == 0:  # Skip empty cells
                    continue

                board_x = x + px
                board_y = y + py

                # Check biên trái/phải
                if board_x < 0 or board_x >= self.width:
                    return True

                # Check đáy
                if board_y >= self.height:
                    return True

                # Check chạm khối khác (chỉ check khi y >= 0)
                if board_y >= 0 and self.board[board_y][board_x] != 0:
                    return True

        return False

    def _place_piece(self):
        """Cố định (lock) piece hiện tại lên board

        Duyệt qua tất cả cells của piece:
        - Nếu cell không empty → cập nhật board
        """
        for py in range(len(self.current_piece)):
            for px in range(len(self.current_piece[0])):
                if self.current_piece[py][px] != 0:
                    board_x = self.piece_x + px
                    board_y = self.piece_y + py

                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.board[board_y][board_x] = self.current_piece[py][px]

    def _clear_full_lines(self):
        """Xóa hàng đầy, tính điểm

        Algorithm:
        1. Tìm hàng nào đầy (không có 0)
        2. Xóa ALL full lines cùng lúc (tránh index shift)
        3. Insert N empty rows ở trên một lần (gravity)
        4. Tính điểm theo Tetris standard scoring:

        Return:
            (num_lines_cleared, points)
        """
        full_lines = []

        # Tìm hàng đầy
        for y in range(self.height):
            if 0 not in self.board[y]:
                full_lines.append(y)

        # Delete all full lines cùng lúc (avoid index shift bug)
        for y in sorted(full_lines, reverse=True):
            del self.board[y]

        # Insert N empty rows ở top một lần
        num_lines = len(full_lines)
        for _ in range(num_lines):
            self.board.insert(0, [0] * self.width)

        # Tính điểm (Tetris standard scoring)
        
        points = 0
        if num_lines > 0:
            points = 1 + (num_lines ** 2) * 10  # 1 line=10, 2 lines=40, 3 lines=90, 4 lines=160

        return num_lines, points

    def _get_state_features(self, lines_cleared=0):
        """Trích xuất 4 features từ board state (dùng cho neural network)

        Features:
        1. lines_cleared: Số hàng xóa bởi NƯỚC ĐI vừa rồi/đang simulate,
           KHÔNG phải tổng cộng dồn cả ván (cộng dồn sẽ giống nhau cho mọi
           action trong get_next_states và non-stationary theo thời gian)
        2. holes: Số ô trống bị che phủ (penalty - xấu nếu cao)
        3. bumpiness: Độ gồ ghề bề mặt (penalty - xấu nếu cao)
        4. height: Tổng chiều cao các cột (penalty - xấu nếu cao)

        Args:
            lines_cleared: số hàng xóa bởi nước đi tạo ra board hiện tại

        Return:
            (lines_cleared, holes, bumpiness, total_height)
        """
        lines = lines_cleared
        holes = 0
        heights = []

        for col in range(self.width):
            height = 0

            # Scan từ TRÊN xuống: ô có khối đầu tiên = đỉnh cột.
            # (Scan từ dưới lên sẽ lấy khối THẤP nhất — cột nào chạm đáy
            # luôn ra height=1, làm total_height và bumpiness sai hết)
            for row in range(self.height):
                if self.board[row][col] != 0:
                    height = self.height - row
                    break

            heights.append(height)

            # Scan từ trên xuống để đếm holes
            filled = False
            for row in range(self.height):
                if self.board[row][col] != 0:
                    filled = True
                elif filled:
                    holes += 1

        # Tính bumpiness: tổng chênh lệch chiều cao các cột liền kề
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

        return (lines, holes, bumpiness, sum(heights))

    def step(self, action):
        """Thực hiện 1 action (move + drop)

        Action:
        - action[0] (x_pos): vị trí ngang nơi đặt piece (0-10)
        - action[1] (num_rotations): số lần xoay piece (0-3)

        Workflow:
        1. Xoay piece num_rotations lần
        2. Đặt piece tại x_pos
        3. Drop piece xuống (tìm vị trí cuối cùng)
        4. Lock piece lên board
        5. Xóa hàng đầy
        6. Spawn piece mới
        7. Tính reward (điểm từ lines clear, -2 nếu game over)

        Return:
            (reward, game_over, state_features)
        """
        if self.game_over:
            return 0, True, self._get_state_features()

        x_pos, num_rotations = action

        # 1. Xoay piece (mod 4 vì 4 rotations)
        piece = [row[:] for row in self.current_piece]
        for _ in range(num_rotations % 4):
            piece = self._rotate_90(piece)

        # 2. Đặt vị trí x (clamp vào [0, width - piece_width])
        self.piece_x = max(0, min(x_pos, self.width - len(piece[0])))
        self.current_piece = piece

        # 3. Drop piece: tìm y cuối cùng trước collision
        self.piece_y = 0
        while not self._check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.piece_y += 1
        self.piece_y -= 1  # Lùi 1 bước (lần trước là collision)

        # 4. Lock piece lên board (chỉ nếu piece_y >= 0, tức piece fit trên board)
        if self.piece_y >= 0:
            self._place_piece()
            self.tetrominoes += 1
        else:
            # Piece không fit ở bất kỳ vị trí nào → Game Over
            self.game_over = True

        # 5. Xóa hàng đầy
        lines_cleared, points = self._clear_full_lines()
        self.score += points
        self.cleared_lines += lines_cleared

        # Reward: +1 survival bonus mỗi khối đặt được + thưởng lớn khi xóa hàng
        # (+1 dày đặc dạy agent "sống lâu = tốt"; chỉ thưởng khi xóa hàng thì
        # tín hiệu quá thưa, agent random gần như không bao giờ nhận reward)
        reward = 1 + (lines_cleared ** 2) * self.width

        # 6. Spawn piece mới
        self._spawn_new_piece()

        # 7. Penalty nếu game over
        if self.game_over:
            reward -= 2

        return reward, self.game_over, self._get_state_features(lines_cleared)

    def get_next_states(self):
        """Enumerate tất cả possible next states (dùng cho AI planning)

        Tính tất cả combinations của (rotation, x_position):
        - max_rotations: 1 (O), 2 (S/Z/I), hoặc 4 (T/L/J)
        - max_x: 0 to (width - piece_width)

        Mỗi combination:
        1. Simulate drop (tìm final y)
        2. Simulate lock piece lên board copy
        3. Simulate clear lines
        4. Extract features từ board copy

        Return:
            Dict: {(x_pos, num_rotations): (lines, holes, bumpiness, height)}
        """
        states = {}
        piece = [row[:] for row in self.current_piece]

        # Xác định số lần rotate tối đa (tùy piece type hiện tại)
        piece_idx = self.current_piece_idx
        if piece_idx == 0:  # O-piece: không cần rotate
            max_rotations = 1
        elif piece_idx in [2, 3, 4]:  # S, Z, I: 2 rotations (90° = 180°)
            max_rotations = 2
        else:  # T, L, J: 4 rotations
            max_rotations = 4

        # Vòng 1: Thử mỗi rotation
        for rotation in range(max_rotations):
            max_x = self.width - len(piece[0])

            # Vòng 2: Thử mỗi x position
            for x in range(max_x + 1):
                # Simulate: drop piece (tìm final y)
                y = 0
                while not self._check_collision(piece, x, y):
                    y += 1
                y -= 1

                # Simulate: lock piece lên board copy
                temp_board = [row[:] for row in self.board]
                for py in range(len(piece)):
                    for px in range(len(piece[0])):
                        if piece[py][px] != 0:
                            by = y + py
                            bx = x + px
                            if 0 <= by < self.height and 0 <= bx < self.width:
                                temp_board[by][bx] = piece[py][px]

                # Simulate: clear lines — del HẾT rồi mới insert (giống
                # _clear_full_lines). Insert trong loop làm index dịch xuống,
                # xóa nhầm hàng khi clear 2+ hàng cùng lúc.
                full_lines = [r for r in range(self.height) if 0 not in temp_board[r]]
                for r in sorted(full_lines, reverse=True):
                    del temp_board[r]
                for _ in range(len(full_lines)):
                    temp_board.insert(0, [0] * self.width)

                # Extract features từ board copy (không thay đổi board thật)
                # Truyền số hàng vừa xóa trong simulation làm feature "lines"
                original_board = self.board
                self.board = temp_board
                features = self._get_state_features(len(full_lines))
                self.board = original_board

                # Lưu vào dict
                states[(x, rotation)] = features

            # Xoay piece cho iteration tiếp theo
            piece = self._rotate_90(piece)

        return states

    def print_board(self):
        """In board ra console (debug visualization)

        Ký hiệu:
        - █ = block
        - · = empty
        """
        board_visual = [row[:] for row in self.board]

        # Thêm current piece vào visual (để thấy piece đang rơi)
        for py in range(len(self.current_piece)):
            for px in range(len(self.current_piece[0])):
                if self.current_piece[py][px] != 0:
                    by = self.piece_y + py
                    bx = self.piece_x + px
                    if 0 <= by < self.height and 0 <= bx < self.width:
                        board_visual[by][bx] = self.current_piece[py][px]

        print("\n" + "=" * (self.width * 2 + 2))
        for row in board_visual:
            print("|" + "".join(['█' if cell else '·' for cell in row]) + "|")
        print("=" * (self.width * 2 + 2))
        print(f"Score: {self.score}, Pieces: {self.tetrominoes}, Lines: {self.cleared_lines}")
        features = self._get_state_features()
        print(f"State: Lines={features[0]}, Holes={features[1]}, Bump={features[2]}, Height={features[3]}\n")

    def render(self, window_name="Tetris - DQN", block_size=20):
        """Render board using OpenCV (live visualization)

        Tetromino colors (BGR format for OpenCV):
        - 0: black (empty)
        - 1: yellow (O-piece)
        - 2: purple (T-piece)
        - 3: green (S-piece)
        - 4: red (Z-piece)
        - 5: cyan (I-piece)
        - 6: orange (L-piece)
        - 7: blue (J-piece)

        Args:
            window_name: OpenCV window title
            block_size: Size of each block in pixels
        """
        import numpy as np
        import cv2

        # Color palette (BGR format for OpenCV)
        colors = [
            (0, 0, 0),          # 0: black (empty)
            (0, 255, 255),      # 1: yellow (O-piece)
            (254, 88, 147),     # 2: purple (T-piece)
            (144, 175, 54),     # 3: green (S-piece) - xanh lá
            (0, 0, 255),        # 4: red (Z-piece)
            (255, 255, 0),      # 5: cyan (I-piece) - xanh dương (fixed from yellow)
            (32, 151, 254),     # 6: orange (L-piece)
            (255, 0, 0)         # 7: blue (J-piece)
        ]

        # Create board with current piece
        board_visual = [row[:] for row in self.board]
        for py in range(len(self.current_piece)):
            for px in range(len(self.current_piece[0])):
                if self.current_piece[py][px] != 0:
                    by = self.piece_y + py
                    bx = self.piece_x + px
                    if 0 <= by < self.height and 0 <= bx < self.width:
                        board_visual[by][bx] = self.current_piece[py][px]

        # Create image (each cell = block_size pixels)
        img = np.zeros((self.height * block_size, self.width * block_size, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                cell_value = board_visual[y][x]
                color = colors[cell_value] if cell_value < len(colors) else colors[0]

                y1 = y * block_size
                y2 = (y + 1) * block_size
                x1 = x * block_size
                x2 = (x + 1) * block_size

                # Fill block with color
                img[y1:y2, x1:x2] = color

                # Draw grid
                cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)

        # Add info panel (right side)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (200, 200, 200)
        info_width = 150

        img = np.pad(img, ((0, 0), (0, info_width), (0, 0)), mode='constant', constant_values=10)

        text_x = self.width * block_size + 10
        text_y = 30

        cv2.putText(img, f"Score: {self.score}", (text_x, text_y), font, 0.6, text_color, 1)
        cv2.putText(img, f"Pieces: {self.tetrominoes}", (text_x, text_y + 30), font, 0.6, text_color, 1)
        cv2.putText(img, f"Lines: {self.cleared_lines}", (text_x, text_y + 60), font, 0.6, text_color, 1)

        if self.game_over:
            cv2.putText(img, "GAME OVER", (text_x, text_y + 120), font, 0.8, (0, 0, 255), 2)

        # Show window
        cv2.imshow(window_name, img)
        cv2.waitKey(1)


# Demo: Play cơ bản
if __name__ == "__main__":
    import random

    print("🎮 TETRIS GAME DEMO")
    print("=" * 50)
    print("Playing 5 random steps...\n")

    game = TetrisGame(height=20, width=10)

    for step_num in range(5):
        print(f"\n--- STEP {step_num + 1} ---")

        # Get all possible next states
        next_states = game.get_next_states()
        if next_states:
            # Random action
            action = random.choice(list(next_states.keys()))
        else:
            break

        print(f"Action: x_pos={action[0]}, rotations={action[1]}")
        reward, done, features = game.step(action)
        print(f"Reward: {reward}")

        game.print_board()

        if done:
            print("GAME OVER!")
            break

    print("✅ Demo complete!")
