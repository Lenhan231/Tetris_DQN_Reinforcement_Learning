import argparse
import os
import torch
import torch.nn as nn
from collections import deque
from random import random, sample, choice
import numpy as np
from wandb_config import WandBTracker, WANDB_AVAILABLE
import wandb

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

        # Init 2 Neural Networks
        self.q_net = DeepQNetwork().to(self.device) 
        self.target_net = DeepQNetwork().to(self.device)

        # Load q_net weight to target_net
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer & Loss function
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer: lưu (state, reward, next_state, done)
        self.memory = deque(maxlen=args.memory_size)
        self.best_score = 0.0
        self.last_best_path = None  

        # WandB tracking
        self.use_wandb = getattr(args, 'wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self.wandb_tracker = WandBTracker(args)
        
        # Tracking progress and Metrics storage for logging
        self.episode = 0
        self.epoch_scores = []
        self.epoch_pieces = []
        self.epoch_lines = []
        self.epoch_rewards = []
        self.epoch_losses = []

    # ε = ε_final + (ε_initial - ε_final) * max(1 - episode / decay_epochs, 0)
    def _get_epsilon(self) -> float:
        return self.args.final_eps + max(
            self.args.decay_epochs - self.episode, 0
        ) * (self.args.initial_eps - self.args.final_eps) / self.args.decay_epochs

    def _get_best_action(self, next_states) -> tuple:
        """Return the action with the highest predicted Q-value.

        Args:
            next_states: Mapping from action (x, rotation) to state features.

        Returns:
            Tuple[int, int]: Selected action as (x_position, rotations).
        """
        self.q_net.eval()

        with torch.no_grad():
            if not next_states:
                return (0, 0)

            # Batch all candidate next states for a single forward pass.
            actions = list(next_states.keys())
            features = [list(next_states[a]) for a in actions]
            states_tensor = torch.FloatTensor(features).to(self.device)

            # Select the action with the maximum predicted Q-value.
            q_values = self.q_net(states_tensor).squeeze()
            best_idx = torch.argmax(q_values).item()

            return actions[best_idx]

    def select_action(self) -> tuple:
        if random() < self._get_epsilon():
            return choice(list(self.env.get_next_states().keys())) if self.env.get_next_states() else (0, 0)
        else:
            return self._get_best_action(self.env.get_next_states())

    def train_step(self) -> float:
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
        s = torch.FloatTensor(np.array([list(x) for x in states])).to(self.device) #(batch_size,4)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(np.array([list(x) for x in next_states])).to(self.device)#(batch_size,4)
        d = torch.BoolTensor(dones).to(self.device)

        # 3. Q_predicted: Q-value từ current network
        q_pred = self.q_net(s) # (batch_size, 1)

        # 4. Q_target: Bellman equation dự đoán q-value của next_state bằng target network
        self.target_net.eval()
        with torch.no_grad():
            q_next = self.target_net(s_next)
        
        # Bellman: Q_target = reward + γ * max_Q(next_state) * (1 - done)
        q_target = r + (1 - d.float().unsqueeze(1)) * self.args.gamma * q_next

        # 5. Loss & backprop
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad() # adam
        loss.backward() # loss / dθ
        self.optimizer.step()

        return loss.item()

    def play_episode(self) -> tuple:
        """Play a single episode and collect experience.

        Returns:
            Tuple[score, pieces, lines, total_reward]
        """
        self.env.reset()
        state = self.env._get_state_features()
        total_reward = 0.0

        # End the episode early to limit extremely long games.
        while not self.env.game_over and self.env.cleared_lines < self.args.max_episode_lines:

            if self.render_enabled:
                self.env.render()

            action = self.select_action()
            score, done, next_state = self.env.step(action)
            
            reward = score
            if done:
                reward -= 2
            reward -= self.args.shape_holes * (next_state[1] - state[1])
            reward -= self.args.shape_bump * (next_state[2] - state[2])
            reward -= self.args.shape_height * (next_state[3] - state[3])

            total_reward += reward

            self.memory.append((state, reward, next_state, done))

            state = next_state

        return (
            self.env.score,
            self.env.tetrominoes,
            self.env.cleared_lines,
            total_reward,
        )

    def train(self):
        """Main training loop: play episodes, train, log metrics.
        Flow:
        1. Play episode
        2. Save best model if score > best_score
        3. Train if enough experience
        """
        os.makedirs(self.args.save_path, exist_ok=True)

        for ep in range(self.args.num_epochs):
            self.episode = ep

            # Play one episode
            score, pieces, lines, total_reward = self.play_episode()

            # Save highest
            if score > self.best_score:
                self.best_score = score
                print(f"New best score: {score:.0f} | lines cleared: {lines} | pieces: {pieces} | tetris_best_{score:.0f}.pth")
                scored_path = os.path.join(self.args.save_path, f"tetris_best_{score:.0f}.pth")
                torch.save(self.q_net.state_dict(), scored_path)
                self.target_net.load_state_dict(self.q_net.state_dict())
                if self.last_best_path and self.last_best_path != scored_path \
                        and os.path.exists(self.last_best_path):
                    os.remove(self.last_best_path)
                self.last_best_path = scored_path

            # After 3000 pieces, the agent has enough experience to start training
            if len(self.memory) > min(3000, self.args.memory_size / 10):
                # update target network
                if (ep + 1) % self.args.target_update == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())
                # loss collection for logging
                loss = self.train_step()
                self.epoch_losses.append(loss)
            else:
                self.epoch_losses.append(0.0)

            # metrics collection for logging
            self.epoch_scores.append(score)
            self.epoch_pieces.append(pieces)
            self.epoch_lines.append(lines)
            self.epoch_rewards.append(total_reward)
              
            # Log metrics to WandB every log_interval episodes
            if (ep + 1) % self.args.log_interval == 0 and self.use_wandb:
                window = slice(-self.args.log_interval, None)
                interval = self.args.log_interval

                metrics = {
                    "epoch": ep + 1,
                    f"game/avg_score_{interval}": np.mean(self.epoch_scores[window]),
                    f"game/best_lines_{interval}": max(self.epoch_lines[window]),
                    f"game/avg_pieces_{interval}": np.mean(self.epoch_pieces[window]),
                    f"game/avg_lines_{interval}": np.mean(self.epoch_lines[window]),
                    f"game/avg_rewards_{interval}": np.mean(self.epoch_rewards[window]),
                    "model/loss": np.mean(self.epoch_losses[window]),
                    "model/epsilon": self._get_epsilon(),
                }

                print(
                    f"Ep {ep + 1:4d}/{self.args.num_epochs} | "
                    f"Avg_Score: {metrics[f'game/avg_score_{interval}']:.2f} | "
                    f"Avg_Pieces: {metrics[f'game/avg_pieces_{interval}']:.2f} | "
                    f"Avg_Lines: {metrics[f'game/avg_lines_{interval}']:.2f} | "
                    f"Loss: {metrics['model/loss']:.4f} | "
                    f"ε: {metrics['model/epsilon']:.3f}"
                )

                wandb.log(metrics)

        return self.q_net


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train DQN for Tetris")

    # Game settings
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Q-Learning hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_eps", type=float, default=1.0)
    parser.add_argument("--final_eps", type=float, default=0.001)
    parser.add_argument("--decay_epochs", type=float, default=2000)
    parser.add_argument("--target_update", type=int, default=10)

    # Replay buffer 
    parser.add_argument("--memory_size", type=int, default=3000)
    
    # Reward shaping (phạt theo DELTA feature sau mỗi nước, 0 = tắt)
    parser.add_argument("--shape_holes", type=float, default=-1.0)
    parser.add_argument("--shape_bump", type=float, default=-1.0)
    parser.add_argument("--shape_height", type=float, default=-1.0)

    # Visualization & Tracking
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_episode_pieces", type=int, default=100000)
    parser.add_argument("--max_episode_lines", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    agent = DQNAgent(args)
    print("\n" + "=" * 70)
    print(" TRAINING DQN AGENT")
    print("=" * 70)
    print(f"Device: {agent.device}")
    print(f"Episodes: {agent.args.num_epochs}")
    print(f"Batch size: {agent.args.batch_size}")
    print(f"Learning rate: {agent.args.lr}")
    print(f"Gamma: {agent.args.gamma}")
    print(f"Initial epsilon: {agent.args.initial_eps}")
    print(f"Final epsilon: {agent.args.final_eps}")
    print(f"Epsilon decay episodes: {agent.args.decay_epochs}")
    print(f"Target update interval: {agent.args.target_update}")
    print(f"Replay buffer size: {agent.args.memory_size}")
    print(f"Max episode pieces: {agent.args.max_episode_pieces}")
    print("=" * 70)
    agent.train()

