"""
STEP 4: DQN TRAINING AGENT
==========================
Mục đích: Huấn luyện Deep Q-Network để chơi Tetris tốt hơn

═════════════════════════════════════════════════════════════════════════════
TRAINING WORKFLOW (từng step):
═════════════════════════════════════════════════════════════════════════════

1. PLAY EPISODE
   └─ Agent chơi 1 game đầy đủ
   └─ Mỗi step: chọn action → nhận reward → lưu experience

2. STORE EXPERIENCE (Replay Buffer)
   └─ Lưu (state, reward, next_state, done) vào memory
   └─ Giúp break correlation giữa samples

3. SAMPLE BATCH
   └─ Lấy random batch từ buffer (vd: 512 samples)
   └─ Tính Q-values dựa trên batch

4. COMPUTE TARGET Q-VALUES (Bellman Equation)
   └─ Nếu game chưa over: Q_target = reward + γ * max_Q(next_state)
   └─ Nếu game over: Q_target = reward
   └─ γ (gamma) = 0.99 = discount factor (future reward ít quan trọng hơn hiện tại)

5. UPDATE NETWORK
   └─ Loss = MSE(Q_predicted - Q_target)
   └─ Backprop và update Q_net weights

6. UPDATE TARGET NETWORK
   └─ Mỗi 100 episodes: copy Q_net → target_net
   └─ Target network dùng để tính Q_target (stable)

═════════════════════════════════════════════════════════════════════════════
KEY CONCEPTS:
═════════════════════════════════════════════════════════════════════════════

EPSILON-GREEDY STRATEGY (Explore vs Exploit):
  Early epochs (ε=1.0):   90% random explore → tìm hiểu toàn bộ
  Late epochs (ε=0.001):  99% best action → khai thác knowledge
  Decay: linear qua 2000 epochs

REPLAY BUFFER:
  - Mỗi step lưu 1 experience: (state, reward, next_state, done)
  - Max 30,000 experiences (cũ nhất xóa)
  - Lấy random batch để train (tránh overfitting sequential data)

TARGET NETWORK:
  - Copy của Q_net dùng để tính Q_target (ổn định)
  - Update mỗi 100 episodes (tránh chasing moving target)

STATE FEATURES (từ step1, TetrisGame._get_state_features):
  - lines_cleared: số hàng vừa xóa (tốt = cao)
  - holes: ô trống bị che phủ (tốt = thấp)
  - bumpiness: độ gồ ghề bề mặt (tốt = thấp)
  - total_height: tổng chiều cao các cột (tốt = thấp)

Q-VALUE:
  Q(state, action) = kỳ vọng reward từ state này nếu chọn action này
  Network học mapping: state → Q-value của mỗi action

═════════════════════════════════════════════════════════════════════════════

Chạy:
  python train.py --num_epochs 3000

  python train.py --num_epochs 3000 --render
  (show live game rendering while training)

  python train.py --num_epochs 3000 --wandb
  (track game + model metrics in Weights & Biases dashboard)
  └─ Game metrics: avg_score/100, avg_rewards/100, best_lines/100, avg_pieces/100, avg_lines/100
  └─ Model metrics: loss, epsilon
"""

import argparse
import os
import torch
import torch.nn as nn
from collections import deque
from random import random, sample, choice
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from tetris import TetrisGame
from network import DeepQNetwork


class DQNAgent:
    """Agent sử dụng Deep Q-Learning để chơi Tetris"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Game environment
        self.env = TetrisGame(height=args.height, width=args.width)

        # Rendering flag
        self.render_enabled = getattr(args, 'render', False)

        # Neural Networks
        self.q_net = DeepQNetwork().to(self.device)  # Network chính (cập nhật mỗi step)
        self.target_net = DeepQNetwork().to(self.device)  # Network phụ (cập nhật mỗi 100 steps)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer & Loss function
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer: lưu (state, reward, next_state, done)
        self.memory = deque(maxlen=args.memory_size)
        self.best_score = 0.0

        # Tracking progress
        self.episode = 0
        self.total_loss = 0.0
        self.loss_count = 0

        # WandB tracking (optional)
        self.use_wandb = args.wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="tetris-dqn",
                config={
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "gamma": args.gamma,
                    "initial_eps": args.initial_eps,
                    "final_eps": args.final_eps,
                    "decay_epochs": args.decay_epochs,
                }
            )

        # Metrics for WandB (per 100 epochs)
        self.epoch_scores = []
        self.epoch_pieces = []
        self.epoch_lines = []
        self.epoch_losses = []
        self.epoch_rewards = []

    def _get_epsilon(self):
        """Epsilon decay: 1.0 → 0.001 qua decay_epochs

        Công thức: ε = final_ε + max(decay_epochs - episode, 0) * (initial_ε - final_ε) / decay_epochs

        Kết quả:
        - Episode 0: ε = 1.0 (100% explore)
        - Episode 1000: ε = 0.5 (50% explore)
        - Episode 2000+: ε = 0.001 (99.9% exploit)
        """
        return self.args.final_eps + max(
            self.args.decay_epochs - self.episode, 0
        ) * (self.args.initial_eps - self.args.final_eps) / self.args.decay_epochs

    def _get_best_action(self, next_states):
        """Dùng Q-network để chọn action tốt nhất

        Input:
            next_states: Dict {(x, rotation): (lines, holes, bumpiness, height)}

        Return:
            action: (x_pos, num_rotations) có Q-value cao nhất
        """
        self.q_net.eval()
        with torch.no_grad():
            if not next_states:
                return (0, 0)

            # Convert state features → tensors
            actions = list(next_states.keys())
            features = [list(next_states[a]) for a in actions]
            states_tensor = torch.FloatTensor(features).to(self.device)

            # Predict Q-values cho tất cả actions
            q_values = self.q_net(states_tensor).squeeze()
            best_idx = torch.argmax(q_values).item()
            return actions[best_idx]

    def select_action(self):
        """Chọn action dùng ε-greedy strategy

        - Xác suất ε: random action (explore)
        - Xác suất 1-ε: best action từ Q-network (exploit)

        Return:
            action: (x_pos, num_rotations)
        """
        eps = self._get_epsilon()
        next_states = self.env.get_next_states()

        if random() < eps:
            # EXPLORE: random action
            return choice(list(next_states.keys())) if next_states else (0, 0)
        else:
            # EXPLOIT: best action từ Q-network
            return self._get_best_action(next_states)

    def train_step(self):
        """Một step training: sample batch → tính loss → update Q_net

        DQN training loop:
        1. Sample random batch từ replay buffer
        2. Tính Q_predicted từ current network
        3. Tính Q_target từ target network + Bellman equation
        4. Loss = MSE(Q_predicted, Q_target)
        5. Backprop + update

        Return:
            loss: giá trị loss của bước này
        """
        if len(self.memory) < self.args.batch_size:
            return 0.0

        # 1. Sample random batch
        batch = sample(self.memory, self.args.batch_size)
        states, rewards, next_states, dones = zip(*batch)

        # 2. Convert to tensors
        s = torch.FloatTensor(np.array([list(x) for x in states])).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(np.array([list(x) for x in next_states])).to(self.device)
        d = torch.BoolTensor(dones).to(self.device)

        # 3. Q_predicted: Q-value từ current network
        q_pred = self.q_net(s)

        # 4. Q_target: Bellman equation
        self.target_net.eval()
        with torch.no_grad():
            q_next = self.target_net(s_next)

        # Bellman: Q_target = reward + γ * max_Q(next_state) * (1 - done)
        # Nếu done=True: Q_target = reward (không có future)
        q_target = r + (1 - d.float().unsqueeze(1)) * self.args.gamma * q_next

        # 5. Loss & backprop
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def play_episode(self):
        """Chơi 1 episode đầy đủ

        Return:
            score, pieces, lines, total_reward
        """
        self.env.reset()
        state = self.env._get_state_features()
        total_reward = 0.0

        while not self.env.game_over:
            action = self.select_action()
            reward, done, next_state = self.env.step(action)
            total_reward += reward

            # Render nếu enabled
            if self.render_enabled:
                self.env.render()

            # Lưu experience vào buffer — LUÔN lưu, kể cả reward âm.
            # (Lọc reward >= 0 sẽ vứt hết transition game-over/done=True,
            # agent không bao giờ học được rằng thua là xấu)
            self.memory.append((state, reward, next_state, done))

            # Update state cho vòng lặp tiếp theo
            state = next_state

        # Train 1 LẦN mỗi episode (giống reference), không phải mỗi khối.
        # Train mỗi khối = ~40-100 updates/ván → policy xoay liên tục,
        # loss tăng dần và score dao động mạnh không ổn định được.
        # Chờ buffer đủ lớn (10% memory_size = 3000 samples) mới bắt đầu.
        if len(self.memory) > self.args.memory_size / 10:
            loss = self.train_step()
            self.total_loss += loss
            self.loss_count += 1

        return self.env.score, self.env.tetrominoes, self.env.cleared_lines, total_reward

    def train(self):
        """Main training loop: chơi N episodes, track progress, save models

        Mỗi 10 episodes: in log (score, loss)
        Mỗi 100 episodes: update target network
        Mỗi save_interval episodes: save model
        """
        print("\n" + "=" * 70)
        print(" TRAINING DQN AGENT")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Episodes: {self.args.num_epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Gamma: {self.args.gamma}")
        print("=" * 70)

        os.makedirs(self.args.save_path, exist_ok=True)

        for ep in range(self.args.num_epochs):
            self.episode = ep

            # Chơi 1 episode
            score, pieces, lines, total_reward = self.play_episode()

            # Collect metrics
            self.epoch_scores.append(score)
            self.epoch_pieces.append(pieces)
            self.epoch_lines.append(lines)
            self.epoch_rewards.append(total_reward)
            self.epoch_losses.append(self.total_loss / self.loss_count if self.loss_count > 0 else 0)

            # Log progress (mỗi 10 episodes)
            if (ep + 1) % 10 == 0:
                avg_loss = (self.total_loss / self.loss_count) if self.loss_count > 0 else 0
                eps = self._get_epsilon()
                print(f"Ep {ep + 1:4d}/{self.args.num_epochs} | "
                      f"Score: {score:6.0f} | "
                      f"Pieces: {pieces:3d} | "
                      f"Lines: {lines:3d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"ε: {eps:.3f}")
                self.total_loss = 0.0
                self.loss_count = 0

            # Log to WandB (mỗi 100 episodes)
            if (ep + 1) % 100 == 0 and self.use_wandb:
                avg_score = np.mean(self.epoch_scores[-100:])
                avg_pieces = np.mean(self.epoch_pieces[-100:])
                avg_lines = np.mean(self.epoch_lines[-100:])
                avg_rewards = np.mean(self.epoch_rewards[-100:])
                avg_loss = np.mean(self.epoch_losses[-100:])

                wandb.log({
                    "epoch": ep + 1,
                    "game/avg_score_100": avg_score,
                    "game/best_lines_100": max(self.epoch_lines[-100:]),
                    "game/avg_pieces_100": avg_pieces,
                    "game/avg_lines_100": avg_lines,
                    "game/avg_rewards_100": avg_rewards,
                    "model/loss": avg_loss,
                    "model/epsilon": self._get_epsilon(),
                })

            # Update target network mỗi 10 episodes. Giờ chỉ train 1 lần/episode
            # nên 10 episodes = 10 gradient steps — target 100 episodes sẽ quá cũ
            # (trước đây train mỗi khối nên 100 episodes mới hợp lý)
            if (ep + 1) % 10 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Save best model mỗi khi đạt score cao mới
            if score > self.best_score:
                print(f"New best score: {score:.0f} (previous: {self.best_score:.0f}) - saving model!")
                self.best_score = score
                best_path = os.path.join(self.args.save_path, "tetris_best.pth")
                torch.save(self.q_net.state_dict(), best_path)

        # Save final model (q_net — target_net là bản copy cũ, có thể lệch tới 100 episodes)
        final_path = os.path.join(self.args.save_path, "tetris_final.pth")
        torch.save(self.q_net.state_dict(), final_path)
        print(f"\n✅ Training complete!")
        print(f"Final model: {final_path}")
        return self.q_net


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train DQN for Tetris")

    # Game settings
    parser.add_argument("--width", type=int, default=10,
                        help="Board width (default: 10)")
    parser.add_argument("--height", type=int, default=20,
                        help="Board height (default: 20)")

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=3000,
                        help="Number of episodes to train (default: 3000; "
                             "cần > decay_epochs=2000 để agent có giai đoạn exploit)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")

    # Q-Learning hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--initial_eps", type=float, default=1.0,
                        help="Initial epsilon for exploration (default: 1.0)")
    parser.add_argument("--final_eps", type=float, default=1e-3,
                        help="Final epsilon for exploitation (default: 0.001)")
    parser.add_argument("--decay_epochs", type=float, default=2000,
                        help="Episodes to decay epsilon (default: 2000)")

    # Memory & Save settings
    parser.add_argument("--memory_size", type=int, default=30000,
                        help="Replay buffer size (default: 30000)")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save model every N episodes (default: 100)")
    parser.add_argument("--save_path", type=str, default="models",
                        help="Path to save models (default: models/)")

    # Visualization & Tracking
    parser.add_argument("--render", action="store_true",
                        help="Enable live rendering during training")
    parser.add_argument("--wandb", action="store_true",
                        help="Track metrics with Weights & Biases (install: pip install wandb)")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    agent = DQNAgent(args)
    agent.train()

    print("\n" + "=" * 70)
    print("✅ STEP 4 COMPLETE - DQN Training Done!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Test model: python test.py --model_path models/tetris_final.pth")
    print("  2. Or continue with step 5 (Play game with trained model)")
    print("=" * 70)
