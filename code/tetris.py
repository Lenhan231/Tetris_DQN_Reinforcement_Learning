import random
from collections import deque
class TetrisGame:
    PIECES = [
        [[1, 1], [1, 1]],                    
        [[0, 2, 0], [2, 2, 2]],             
        [[0, 3, 3], [3, 3, 0]],             
        [[4, 4, 0], [0, 4, 4]],              
        [[5, 5, 5, 5]],                     
        [[0, 0, 6], [6, 6, 6]],              
        [[7, 0, 0], [7, 7, 7]]               
    ]

    def __init__(self, height=20, width=10):
        self.height = height
        self.width = width
        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0        
        self.cleared_lines = 0      
        self.game_over = False
        self.current_piece_idx = 0  
        self.piece_bag = list(range(len(self.PIECES)))
        random.shuffle(self.piece_bag)
        self._spawn_new_piece()
        return self._get_state_features()

    def _spawn_new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.PIECES)))
            random.shuffle(self.piece_bag)

        self.current_piece_idx = self.piece_bag.pop()
        self.current_piece = [row[:] for row in self.PIECES[self.current_piece_idx]]
        self.piece_x = self.width // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0

        if self._check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

    def _rotate_90(self, piece):
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
        for py in range(len(piece)):
            for px in range(len(piece[0])):
                if piece[py][px] == 0:
                    continue
                board_x = x + px
                board_y = y + py
                if board_x < 0 or board_x >= self.width:
                    return True
                if board_y >= self.height:
                    return True
                if board_y >= 0 and self.board[board_y][board_x] != 0:
                    return True
        return False

    def _place_piece(self):
        for py in range(len(self.current_piece)):
            for px in range(len(self.current_piece[0])):
                if self.current_piece[py][px] != 0:
                    board_x = self.piece_x + px
                    board_y = self.piece_y + py
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.board[board_y][board_x] = self.current_piece[py][px]

    def _clear_full_lines(self):
        full_lines = []
        for y in range(self.height):
            if 0 not in self.board[y]:
                full_lines.append(y)
        for y in sorted(full_lines, reverse=True):
            del self.board[y]
        num_lines = len(full_lines)
        for _ in range(num_lines):
            self.board.insert(0, [0] * self.width)

        # tinh diem    
        points = 0
        if num_lines > 0:
            points = 1 + (num_lines ** 2) * 10  

        return num_lines, points

    def _get_state_features(self, lines_cleared=0):
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

    def step(self, action, on_drop_step=None):
        """

        Action:
        - action[0] (x_pos): vị trí ngang nơi đặt piece (0-10)
        - action[1] (num_rotations): số lần xoay piece (0-3)

        on_drop_step: callback gọi ở MỖI hàng piece rơi xuống (dùng để render
        animation soft drop khi test). None = rơi tức thì (train nhanh).

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

        # 3. Drop piece: rơi từng hàng một, dừng ở hàng cuối cùng không collision
        # (piece_y = -1 nếu không fit ngay từ đầu → game over ở bước 4)
        self.piece_y = -1
        while not self._check_collision(self.current_piece, self.piece_x, self.piece_y + 1):
            self.piece_y += 1
            if on_drop_step is not None:
                on_drop_step()  # render 1 frame ở vị trí hiện tại (soft drop)

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
        states = {}
        piece = [row[:] for row in self.current_piece]

        # giam so lan xoay
        piece_idx = self.current_piece_idx
        if piece_idx == 0:  
            max_rotations = 1
        elif piece_idx in [2, 3, 4]: 
            max_rotations = 2
        else: 
            max_rotations = 4

        for rotation in range(max_rotations): #rotation
            max_x = self.width - len(piece[0])
            for x in range(max_x + 1):
                # Simulate: drop piece xuống (tìm final y)  
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



















    # render tag for test.py and train.py if flag render is set not very important skip
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
