# Step 4: Training the Agent

## Overview

You have a game and a neural network. Now: teach the network to play well by training it on real game data.

## The Big Picture

```
REPEAT N times:
  1. Play a game with the current network
  2. Collect experiences: (state, action, reward, next_state)
  3. Train the network on random batches from past experiences
  4. Repeat until network learns
```

## The Training Loop

### Phase 1: Play a Game

The agent plays Tetris. At each step:

```python
# Get current state
state = game.get_state()

# Choose action (explore or exploit)
if random() < epsilon:
    action = random_action()  # Explore: try something random
else:
    action = network.select_best(state)  # Exploit: use network

# Execute action
reward, done, next_state = game.step(action)

# Store experience
buffer.store(state, action, reward, next_state, done)
```

### Phase 2: Train on Random Batch

Random sampling is crucial:

```python
# Sample a random batch (e.g., 512 experiences)
batch = buffer.sample(512)

# For each experience in batch:
for state, action, reward, next_state, done in batch:
    # Calculate target Q-value
    if done:
        target = reward  # No future rewards if game is over
    else:
        next_q = network(next_state)
        target = reward + gamma * next_q
    
    # Train network to predict target
    loss = (network(state) - target)²
    optimizer.step(loss)
```

## Key Concepts

### 1. Epsilon-Greedy Strategy

**Problem**: If network is always wrong, it learns from wrong data (garbage in, garbage out)

**Solution**: Mix exploration and exploitation
- Early training: Mostly random actions (explore different states)
- Late training: Mostly network actions (exploit learned knowledge)

**Decay schedule**:
```
Episode 1:    epsilon = 1.0    (100% random)
Episode 1000: epsilon = 0.3    (70% greedy, 30% random)
Episode 2000: epsilon = 0.001  (99% greedy, 1% random)
```

**Why?**
1. Random play explores many states early on
2. Network gets diverse training data
3. As network improves, trust it more
4. Eventually play greedily

### 2. Experience Replay Buffer

**Problem**: Consecutive game states are highly correlated. If you train on sequential data, the network overfits.

**Example**: Sequence of moves in one game
```
State 1 → Action A → Reward 10 → State 2
State 2 → Action B → Reward 5 → State 3
State 3 → Action A → Reward 8 → State 4
```

Training on these in order → network sees false patterns

**Solution**: Store experiences in a buffer and sample random batches

```python
# Store experiences
for 1000 games:
    store(state, action, reward, next_state)

# Train on random subset
batch = random.sample(buffer, 512)
train(batch)
```

**Benefits**:
- Breaks correlation between samples
- Data is used multiple times (efficient)
- Training is more stable

### 3. Target Network

**Problem**: You're using the network to compute both:
- Predicted Q: `network(state)`
- Target Q: `reward + gamma * network(next_state)`

This creates circular dependency (chasing a moving target)

**Solution**: Keep two copies
- **Q-Network**: Updated every step (learns from data)
- **Target Network**: Updated every 100 episodes (stable targets)

```python
# Every training step
loss = (q_network(state) - target)²  # target uses target_network
optimizer.step(loss)  # update q_network weights

# Every 100 episodes
target_network.load_state_dict(q_network.state_dict())  # copy weights
```

## Running Step 4

```bash
# Quick test (50 episodes)
python code/train.py --num_epochs 50

# Full training (300+ episodes)
python code/train.py --num_epochs 300

# With visualization (slower)
python code/train.py --num_epochs 300 --render

# With performance tracking
python code/train.py --num_epochs 300 --wandb
```

## Understanding Training Progress

### What to Expect

| Episodes | Agent Behavior | Avg Lines |
|----------|----------------|-----------|
| 1-50 | Random gameplay | 5-20 |
| 50-150 | Starting to learn | 15-40 |
| 150-300 | Consistent strategy | 30-80 |
| 300+ | Converged | 50-200+ |

### Metrics to Watch

**Loss**: Should decrease (network getting better)
```
Episode 1: Loss = 2.5
Episode 50: Loss = 1.8
Episode 300: Loss = 0.3
```

**Epsilon**: Should decrease (trust network more)
```
Episode 1: epsilon = 1.0
Episode 1000: epsilon = 0.3
Episode 2000: epsilon = 0.001
```

**Average Reward/Score**: Should increase (agent learning)
```
Episode 1-10: avg score = 10
Episode 50-60: avg score = 25
Episode 300-310: avg score = 80
```

## Common Issues

### Training is Very Slow

**Cause**: Rendering or high batch size
```bash
# Disable rendering
python code/train.py --num_epochs 300
# (remove --render)

# Smaller batch size
python code/train.py --batch_size 256
```

### Loss Not Decreasing

**Cause**: Learning rate too high or data too noisy

```bash
# Lower learning rate
python code/train.py --lr 0.0005

# Train longer (need more data)
python code/train.py --num_epochs 500
```

### Model Performance Plateaus

**Cause**: Network capacity not enough or reward shaping incorrect

```bash
# Modify network size (in code)
# Change: dense1_size=128, dense2_size=64 (instead of 64, 32)

# Or adjust rewards (in code)
# Tweak the reward calculations in step1_tetris_basic.py
```

## What You Should Understand

- ✅ Why we explore early and exploit late
- ✅ Why we sample random batches (experience replay)
- ✅ Why we have two networks (stability)
- ✅ How loss and epsilon change during training
- ✅ What good vs bad training looks like

## Next Steps

After training:
- ✅ Model checkpoints are saved (models/tetris_*.pth)
- ✅ You can test the trained model
- ✅ You can analyze performance

**Next**: Evaluate your trained model on fresh games.
