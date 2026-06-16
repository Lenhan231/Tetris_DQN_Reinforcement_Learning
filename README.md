# 🎮 Tetris Deep Q-Learning: Complete Guide

Learn Deep Reinforcement Learning by building an AI that plays Tetris. This project breaks down complex RL concepts into 5 digestible steps with working code.

## 📍 Documentation Guide

**New here?** Pick your style:
- 🚀 **Quick start** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (5 min, copy-paste commands)
- 📚 **Full learning path** → [LEARNING_PATH.md](LEARNING_PATH.md) (20 min, complete guide)
- 🎨 **Visual learner** → [CONCEPTS_EXPLAINED.md](CONCEPTS_EXPLAINED.md) (30 min, diagrams & intuitions)
- 📖 **Full index** → [docs/INDEX.md](docs/INDEX.md) (navigation for all docs)

---

## What You'll Learn

| Concept | You'll Learn |
|---------|----------|
| **Game Logic** | Build a complete Tetris engine with proper physics |
| **Feature Engineering** | Extract meaningful data from game states |
| **Neural Networks** | Design and train a Deep Q-Network |
| **Reinforcement Learning** | Implement DQN training with experience replay |
| **Model Evaluation** | Test and analyze your trained agent |

## 📦 What's Included

```
tetris_from_scratch/
├── code/
│   ├── tetris.py                      # The game engine
│   ├── network.py                     # The neural network
│   ├── train.py                       # The training loop
│   └── test.py                        # Testing & evaluation
├── models/                             # Trained model checkpoints (created during training)
└── docs/                               # Learning guides
```

## 🚀 Quick Start (5 Minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the game engine demo
python code/tetris.py

# Run the neural network demo
python code/network.py

# Train the agent (takes 2-4 hours for full training)
python code/train.py --num_epochs 300

# Test the trained model
python code/test.py --model_path models/tetris_final.pth
```

## 📚 Learning Path

Start here and follow the steps in order:

### **Step 1: Understanding the Game** (20 min)
**File**: `code/tetris.py`

**What you'll understand**:
- How a Tetris board works (20×10 grid)
- How pieces fall, rotate, and collide
- How lines are cleared and scored
- What data the AI will see

**Run it**: `python code/tetris.py`

**Key concepts**:
- Board representation
- Piece mechanics (7 tetrominos, 4 rotation states)
- Collision detection
- Line clearing logic

---

### **Step 3: Building a Neural Network** (20 min)
**File**: `code/network.py`

**What you'll understand**:
- How neural networks estimate Q-values
- Network architecture (4 inputs → 64 → 32 → 1 output)
- What training means
- The Bellman equation

**Run it**: `python code/network.py`

**Key equation**:
```
Q(state, action) = reward + γ × max_Q(next_state)
```

The network learns to predict Q-values that satisfy this equation.

---

### **Step 4: Training the Agent** (2-4 hours)
**File**: `code/train.py`

**What happens during training**:
1. Agent plays Tetris using epsilon-greedy strategy (explore + exploit)
2. Experiences are stored in a replay buffer (30,000 max)
3. Random batches are sampled and used to train the network
4. Loss decreases and performance improves
5. Model checkpoints saved every 100 episodes

**Run it**:
```bash
# Basic training
python code/train.py --num_epochs 300

# With visualization (slower)
python code/train.py --num_epochs 300 --render

# With performance tracking
python code/train.py --num_epochs 300 --wandb
```

**What to expect**:
- Episodes 1-100: Random play, scores ~5-20 lines
- Episodes 100-300: Learning visible, scores ~20-50 lines
- Episodes 300+: Consistent play, scores ~50-150+ lines

---

### **Step 5: Testing Your Model** (5 min)
**File**: `code/test.py`

**What you'll see**:
- Average score across multiple games
- Standard deviation (consistency)
- Performance rating
- Optional visualization

**Run it**:
```bash
# Quick test (10 games)
python code/test.py --model_path models/tetris_final.pth

# Detailed test (50 games)
python code/test.py --model_path models/tetris_final.pth --num_games 50

# Interactive mode (watch and control)
python code/test.py --model_path models/tetris_final.pth --infinite
```

---

## 🧠 Key Concepts Explained

### Q-Learning

**Core Idea**: Learn the value of each action in each state

**Q-Value**: Expected cumulative future reward
```
Q(state, action) = immediate_reward + γ × expected_future_rewards
```

**Problem**: Can't store all states (too many)  
**Solution**: Use a neural network to estimate Q-values

### Deep Q-Network (DQN)

Instead of a lookup table, use a neural network:
- **Input**: Game state features (lines cleared, holes, bumpiness, height)
- **Output**: Q-value (expected future reward)
- **Training**: Minimize difference between predicted and target Q-values

### Experience Replay

Problem: Sequential game states are highly correlated  
Solution: Store past experiences and train on random batches
- Breaks correlation
- Improves data efficiency
- More stable training

### Epsilon-Greedy Strategy

Balance exploration (trying new things) vs exploitation (using what works):
- **Early training**: Mostly random actions (explore)
- **Late training**: Mostly best actions from network (exploit)
- **Decay**: Gradually shift from exploration to exploitation

---

## 🎛️ Training Hyperparameters

### In `step4_train_dqn.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--num_epochs` | 100 | Training duration (episodes) |
| `--batch_size` | 512 | Experiences per training step |
| `--lr` | 0.001 | Learning rate (how fast network updates) |
| `--gamma` | 0.99 | Discount factor (how much we value future rewards) |
| `--decay_epochs` | 2000 | Episodes to decay exploration |

### Game Rewards:

The agent gets:
- **Positive**: Clearing lines (1 line = 101 points, 4 lines = 401 points)
- **Negative**: Building tall stacks, creating holes, bumpy surface
- **Penalty**: Game over (-10)

---

## 🔍 Understanding the Code

### Game State Features (from `tetris.py`)

The AI sees 4 features:
- **lines_cleared**: Total lines cleared (higher = better)
- **holes**: Unfilled gaps under blocks (lower = better)
- **bumpiness**: Height variation between columns (lower = better)
- **height**: Total stack height (lower = better)

### Network Architecture (`network.py`)

```
Input (4 features)
    ↓
Dense(64) + ReLU
    ↓
Dense(32) + ReLU
    ↓
Output (1 Q-value)
```

**Why this size?**
- 64 neurons: Large enough to learn patterns, small enough to train quickly
- 32 neurons: Bottleneck for compression
- Total: ~4,500 parameters

### Training Loop (`train.py`)

```python
for episode in range(num_epochs):
    # 1. Play one game
    while not done:
        state = game.get_state()
        action = select_action(epsilon)      # Epsilon-greedy
        reward, done = game.step(action)
        buffer.store(state, action, reward)
    
    # 2. Train on random batch
    if len(buffer) > batch_size:
        batch = buffer.sample(batch_size)
        q_target = batch.reward + gamma * max_Q(batch.next_state)
        loss = MSE(Q_network(batch.state) - q_target)
        optimizer.step(loss)
    
    # 3. Decay exploration
    epsilon *= decay_rate
```

---

## 📊 Performance Expectations

| Training Stage | Avg Lines/Game | Notes |
|---------------|----------------|-------|
| Episode 1-50 | 5-15 | Random play |
| Episode 50-200 | 15-40 | Learning starts |
| Episode 200-500 | 40-100 | Consistent improvement |
| Episode 500+ | 50-200+ | Converged (great model) |

---

## ⚙️ Troubleshooting

### Training is slow
```bash
# Disable visualization
python code/step4_train_dqn.py --num_epochs 300
# (remove --render flag)

# Reduce batch size
python code/step4_train_dqn.py --batch_size 256
```

### Model not improving
```bash
# Lower learning rate
python code/step4_train_dqn.py --lr 0.0005

# Train longer
python code/step4_train_dqn.py --num_epochs 500
```

### ModuleNotFoundError
```bash
pip install torch numpy opencv-python pillow
```

### CUDA not found
The code automatically falls back to CPU. No action needed.

---

## 🎓 Next Steps

Once the basic model works, try:

1. **Architecture changes**: Modify hidden layer sizes
2. **Reward shaping**: Adjust reward penalties in Step 1
3. **New features**: Add more state features in `_get_state_features()`
4. **Advanced techniques**:
   - Double DQN (reduces overestimation)
   - Dueling DQN (separate value and advantage)
   - Prioritized Experience Replay (sample important experiences more)

---

## 📖 References

- **RL Book**: [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)
- **DQN Paper**: [Nature 2015](https://www.nature.com/articles/nature14236)
- **PyTorch Docs**: [pytorch.org](https://pytorch.org/)

---

## 📝 License

Educational project. Use freely for learning.

---

**Ready to start? Run Step 1:** 🚀
```bash
python code/step1_tetris_basic.py
```
