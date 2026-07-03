"""
STEP 5: TEST & EVALUATE TRAINED MODEL
======================================
Test trained DQN model on Tetris

Usage:
    python test.py --model_path models/tetris_final.pth              # 10 games + stats
    python test.py --model_path models/tetris_final.pth --num_games 50
    python test.py --model_path models/tetris_final.pth --infinite   # Infinite mode
    python test.py --model_path models/tetris_final.pth --infinite --speed 1.5
"""

import argparse
import torch
import numpy as np
from copy import deepcopy
from tetris import TetrisGame
from network import DeepQNetwork


class DQNTester:
    """Test trained DQN model"""

    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepQNetwork().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded: {model_path} ({self.device})\n")

    def select_best_action(self, env):
        """Select best action using model"""
        with torch.no_grad():
            next_states = env.get_next_states()
            if not next_states:
                return (0, 0), 0.0

            states_list = [torch.FloatTensor(list(f)).to(self.device) for a, f in next_states.items()]
            actions_list = list(next_states.keys())

            q_values = self.model(torch.stack(states_list)).squeeze()
            best_idx = torch.argmax(q_values).item()
            return actions_list[best_idx], q_values[best_idx].item() if q_values.dim() > 0 else q_values.item()

    def render_pygame(self, env, action, reward, q_value, screen=None, clock=None, speed=1.0):
        """Render with Pygame"""
        try:
            import pygame
        except ImportError:
            return None, None

        if screen is None:
            pygame.init()
            block_sz = 25
            screen = pygame.display.set_mode((env.width * block_sz + 180, env.height * block_sz + 40))
            pygame.display.set_caption("Tetris DQN")
            clock = pygame.time.Clock()
        else:
            block_sz = 25

        colors = {0: (0, 0, 0), 1: (255, 255, 0), 2: (147, 88, 254), 3: (54, 175, 144),
                  4: (255, 0, 0), 5: (102, 217, 238), 6: (254, 151, 32), 7: (0, 0, 255)}

        board = [row[:] for row in env.board]
        for py in range(len(env.current_piece)):
            for px in range(len(env.current_piece[0])):
                if env.current_piece[py][px] != 0:
                    by, bx = env.piece_y + py, env.piece_x + px
                    if 0 <= by < env.height and 0 <= bx < env.width:
                        board[by][bx] = env.current_piece[py][px]

        screen.fill((10, 10, 10))
        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(x * block_sz, y * block_sz, block_sz, block_sz)
                pygame.draw.rect(screen, colors.get(board[y][x], (0, 0, 0)), rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)

        font = pygame.font.Font(None, 24)
        info_x = env.width * block_sz + 10
        info = [f"Score: {env.score}", f"Pieces: {env.tetrominoes}", f"Lines: {env.cleared_lines}",
                "", f"Action: x={action[0]}, r={action[1]}", f"Reward: {reward:+.1f}", f"Q: {q_value:.2f}"]
        for i, txt in enumerate(info):
            if txt:
                screen.blit(font.render(txt, True, (200, 200, 200)), (info_x, 10 + i * 25))

        pygame.display.flip()
        clock.tick(int(60 / speed))
        return screen, clock

    def play_game(self, render=False, use_pygame=True, speed=1.0):
        """Play one game"""
        env = TetrisGame(height=20, width=10)
        env.reset()
        screen, clock = None, None
        step = 0

        while not env.game_over:
            action, q_val = self.select_best_action(env)

            # Soft drop: render 1 frame mỗi hàng piece rơi (thay vì hard drop)
            on_drop = None
            if render and use_pygame:
                def on_drop(a=action, q=q_val):
                    nonlocal screen, clock
                    screen, clock = self.render_pygame(env, a, 0.0, q, screen, clock, speed)
            elif render:
                on_drop = lambda: env.render(block_size=20)

            reward, _, _ = env.step(action, on_drop_step=on_drop)

            if render and use_pygame:
                screen, clock = self.render_pygame(env, action, reward, q_val, screen, clock, speed)
            elif render:
                env.render(block_size=20)
            step += 1

        if render and use_pygame and screen:
            import pygame
            import time
            time.sleep(2)
            pygame.quit()

        return env.score, env.tetrominoes, env.cleared_lines, step

    def _play_infinite_game(self, speed=1.0, use_pygame=True):
        """Infinite game loop with restart/rewind"""
        import pygame

        print(f"{'=' * 60}")
        mode = "PYGAME" if use_pygame else "OPENCV"
        print(f"▶ RUNNING INFINITE {mode} MODE")
        print("Controls: SPACE → pause/resume | R → restart | Z → rewind | Q → quit")
        print(f"{'=' * 60}\n")

        game_count = 1
        while True:
            print(f"▶ Game {game_count}...")
            env = TetrisGame(height=20, width=10)
            env.reset()
            initial_env = deepcopy(env)
            history = []
            screen, clock = None, None
            paused = False
            last_action, last_reward, last_q = (0, 0), 0.0, 0.0

            while not env.game_over:
                if use_pygame:
                    screen, clock = self.render_pygame(env, last_action, last_reward, last_q, screen, clock, speed)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            pygame.quit()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = not paused
                                status = "▮▮ PAUSED" if paused else "▶ RUNNING"
                                print(f"{status}")
                            elif event.key == pygame.K_r:
                                env = deepcopy(initial_env)
                                history = []
                                paused = False
                                print("🔄 Reset")
                            elif event.key == pygame.K_z and history:
                                env = history.pop()
                                paused = True
                                print(f"⏮️ Rewind (Step: {len(history)}) - PAUSED")
                else:
                    env.render(block_size=20)

                if not paused:
                    history.append(deepcopy(env))
                    last_action, last_q = self.select_best_action(env)

                    # Soft drop: render từng hàng rơi thay vì nhảy thẳng xuống đáy
                    if use_pygame:
                        def on_drop():
                            nonlocal screen, clock
                            screen, clock = self.render_pygame(
                                env, last_action, last_reward, last_q, screen, clock, speed)
                    else:
                        on_drop = lambda: env.render(block_size=20)

                    last_reward, _, _ = env.step(last_action, on_drop_step=on_drop)

            print(f"Game {game_count} Over! Score: {env.score} | Lines: {env.cleared_lines}")
            if screen:
                pygame.quit()
            input("Press Enter to continue...")
            game_count += 1

    def test_games_infinite(self, speed=1.0):
        """Infinite Pygame mode"""
        self._play_infinite_game(speed=speed, use_pygame=True)

    def test_games(self, num_games=10, use_pygame=True, speed=1.0):
        """Test multiple games and show stats"""
        print(f"{'=' * 60}\n🧪 TESTING {num_games} GAMES\n{'=' * 60}\n")

        scores, pieces_list, lines_list, steps_list = [], [], [], []

        for i in range(num_games):
            score, pieces, lines, steps = self.play_game(render=False, speed=speed)
            scores.append(score)
            pieces_list.append(pieces)
            lines_list.append(lines)
            steps_list.append(steps)
            print(f"Game {i+1:2d}: Score {score:7.0f} | Pieces {pieces:3d} | Lines {lines:3d} | Steps {steps:4d}")

        print(f"\n{'=' * 60}\n📊 STATISTICS\n{'=' * 60}\n")

        def stats(data, name):
            mean, std = np.mean(data), np.std(data)
            median = np.median(data)
            print(f"{name:8s}: μ={mean:7.1f} σ={std:5.1f} | median={median:7.1f} | min={min(data):.0f} max={max(data):.0f}")

        stats(scores, "Score")
        stats(pieces_list, "Pieces")
        stats(lines_list, "Lines")
        stats(steps_list, "Steps")

        avg_lines = np.mean(lines_list)
        print(f"\n{'Performance':─^40}")
        if avg_lines > 100:
            print("🌟 Excellent (avg lines > 100)!")
        elif avg_lines > 50:
            print("👍 Good (avg lines > 50)!")
        else:
            print("⚠️ Can be improved (avg lines ≤ 50)")


def main():
    parser = argparse.ArgumentParser(description="Test trained DQN model")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games")
    parser.add_argument("--infinite", action="store_true", help="Infinite mode (R=restart, Z=rewind, Q=quit)")
    parser.add_argument("--speed", type=float, default=2.0, help="Speed multiplier (default: 2.0)")
    args = parser.parse_args()

    try:
        tester = DQNTester(args.model_path)
        if args.infinite:
            tester.test_games_infinite(speed=args.speed)
        else:
            tester.test_games(num_games=args.num_games, use_pygame=True, speed=args.speed)
    except FileNotFoundError:
        print(f"❌ Model not found: {args.model_path}")


if __name__ == "__main__":
    main()
