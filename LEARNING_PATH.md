# Complete Learning Path

Follow this path to learn Deep Reinforcement Learning by building a Tetris-playing AI.

## 🎯 Learning Goals

By the end, you will understand:
- How to build a game environment
- How neural networks estimate value
- How to train agents with reinforcement learning
- How to evaluate model performance

**Time commitment**: 4-6 hours (depending on training duration)

---

## Phase 1: Foundation (1 hour)

### Part 1.1: What is Reinforcement Learning? (15 min)

**Read**: Concepts only, don't worry about math yet

- Agent: Takes actions in an environment
- State: Current situation
- Action: What the agent can do
- Reward: Feedback from environment
- Goal: Maximize cumulative reward

**Key insight**: RL is learning by trial and error

### Part 1.2: What is Q-Learning? (15 min)

**Core idea**: 
```
Q(state, action) = value of taking this action from this state
```

**Intuition**: 
- If Q-value is high, take that action
- If Q-value is low, avoid that action

**Problem**: Too many states to memorize → use neural network

### Part 1.3: What is Deep Q-Learning (DQN)? (15 min)

**Idea**: Use a neural network to estimate Q-values

```
Network Input:  State description
Network Output: Q-value estimate
Training:       Make output match true Q-values
```

**Key algorithm**: Bellman equation
```
Q(s,a) = reward + future_rewards
```

### Part 1.4: The Training Process (15 min)

```
Loop:
  1. Play game → collect experiences
  2. Store in buffer → sample random batch
  3. Train network on batch
  4. Network gets better
  Repeat until convergence
```

---

## Phase 2: Hands-On Learning (3-4 hours)

### Part 2.1: Build the Game (30 min)

**File**: `code/tetris.py`  
**Guide**: `docs/GUIDE_STEP1.md`

**What to do**:
1. Run: `python code/tetris.py`
2. Read the output
3. Open code and understand:
   - How board is represented
   - How pieces work
   - How state is computed
4. Read GUIDE_STEP1.md

**Key learnings**:
- ✅ Board is 20×10 grid
- ✅ 7 pieces with 4 rotations each
- ✅ State = (lines, holes, bumpiness, height)
- ✅ Actions = (x_position, num_rotations)

**Exercise** (optional):
- Modify reward function (increase line clearing bonus)
- Add a new piece type
- Change board size

### Part 2.2: Understand the Network (30 min)

**File**: `code/network.py`  
**Guide**: `docs/GUIDE_STEP3.md`

**What to do**:
1. Run: `python code/network.py`
2. Read the network architecture output
3. Open code and understand:
   - Input layer (4 features)
   - Hidden layers (64 → 32 neurons)
   - Output layer (1 Q-value)
   - ReLU activation
4. Read GUIDE_STEP3.md

**Key learnings**:
- ✅ Network maps state → Q-value
- ✅ ReLU adds non-linearity
- ✅ Bellman equation defines training targets
- ✅ Loss = (predicted - target)²

**Exercise** (optional):
- Visualize network weights
- Try different hidden layer sizes
- Understand gradient descent

### Part 2.3: Train Your Agent (2-3 hours)

**File**: `code/train.py`  
**Guide**: `docs/GUIDE_STEP4.md`

**What to do**:

Start with quick test:
```bash
python code/train.py --num_epochs 50
```

This takes ~30 min and shows:
- Loss decreasing (network learning)
- Epsilon decaying (less exploration)
- Average reward increasing (agent improving)

Read GUIDE_STEP4.md while training.

Then do full training:
```bash
python code/train.py --num_epochs 300
```

This takes 2-3 hours. During this time:
- Read referenced materials
- Understand epsilon-greedy strategy
- Understand experience replay
- Understand target network

**Key learnings**:
- ✅ Epsilon-greedy = explore + exploit
- ✅ Experience replay breaks correlation
- ✅ Target network provides stability
- ✅ Training curve should show improvement

**What to observe**:
- Loss should decrease
- Average reward should increase
- Epsilon should decay to ~0.001
- Checkpoints saved every 100 episodes

### Part 2.4: Evaluate Your Model (30 min)

**File**: `code/test.py`  
**Guide**: `docs/GUIDE_STEP5.md`

**What to do**:

```bash
# Quick evaluation (10 games)
python code/test.py --model_path models/tetris_final.pth

# Detailed evaluation (50 games)
python code/test.py --model_path models/tetris_final.pth --num_games 50

# Interactive mode (watch agent play)
python code/test.py --model_path models/tetris_final.pth --infinite
```

Read GUIDE_STEP5.md.

**Key learnings**:
- ✅ How to interpret metrics (lines, score, std dev)
- ✅ Good vs bad performance
- ✅ How to diagnose problems
- ✅ How to compare models

**What to observe**:
- Average lines cleared
- Consistency (standard deviation)
- Performance across different games

---

## Phase 3: Deeper Understanding (30 min - optional)

### Part 3.1: Theory Deep Dive

**Topics to research** (after basics work):
- Why Bellman equation works
- Convergence guarantees
- Function approximation errors
- Why ReLU works

**Resources**:
- Sutton & Barto: Reinforcement Learning textbook
- OpenAI Spinning Up: RL introduction
- YouTube: DeepMind DQN papers

### Part 3.2: Code Deep Dive

Open each file and understand:

**step1_tetris_basic.py**:
- `_get_next_states()`: How to find valid actions
- `_clear_lines()`: Collision detection
- `_get_state_features()`: Feature engineering

**step3_neural_network.py**:
- Forward pass: How input becomes output
- Backward pass: How gradients flow

**step4_train_dqn.py**:
- Epsilon-greedy selection
- Replay buffer implementation
- Training loop
- Target network update

**step5_test_model.py**:
- Loading trained model
- Inference (no training)
- Metrics calculation

---

## Phase 4: Experiments (1-2 hours - optional)

Once basics work, try:

### Experiment 1: Network Size

```python
# In step3_neural_network.py
# Try: dense1_size=128, dense2_size=64 (larger)
# or:  dense1_size=32, dense2_size=16 (smaller)
```

Then retrain and compare results.

### Experiment 2: Hyperparameters

```bash
# Lower learning rate (more stable)
python code/train.py --lr 0.0005 --num_epochs 500

# Larger batch size (more stable)
python code/train.py --batch_size 1024 --num_epochs 300

# Longer decay (more exploration)
python code/train.py --decay_epochs 5000 --num_epochs 1000
```

### Experiment 3: Reward Shaping

```python
# In step1_tetris_basic.py
# Increase line clearing reward
# Adjust hole penalty
# Change height penalty
```

See how it affects learning.

### Experiment 4: Advanced Techniques

Implement and compare:
- **Double DQN**: Separate networks for stability
- **Dueling DQN**: Separate value and advantage
- **Prioritized Replay**: Sample important experiences more

---

## Checklist: Make Sure You Understand

Check off as you learn:

### Step 1: Game
- [ ] How board is represented
- [ ] How pieces work
- [ ] What actions are valid
- [ ] How state features are computed
- [ ] What game over means

### Step 3: Network
- [ ] Architecture (4 → 64 → 32 → 1)
- [ ] What each layer does
- [ ] Why ReLU
- [ ] Bellman equation
- [ ] How loss is computed
- [ ] Gradient descent updates network

### Step 4: Training
- [ ] Epsilon-greedy strategy
- [ ] Why we explore early
- [ ] Why we exploit late
- [ ] Experience replay (why and how)
- [ ] Target network (why and how)
- [ ] Training loss decreasing
- [ ] Performance improving over time

### Step 5: Evaluation
- [ ] How to interpret metrics
- [ ] Good vs bad performance
- [ ] How to diagnose problems
- [ ] How to compare models

---

## Troubleshooting

### "I don't understand the code"
1. Add print statements to see what's happening
2. Run small examples step-by-step
3. Read the guide multiple times
4. Change code and see what happens

### "Training is too slow"
1. Reduce `--num_epochs` for testing
2. Reduce `--batch_size` (but may be less stable)
3. Disable `--render` flag
4. Use GPU if available

### "Model doesn't improve"
1. Check loss is decreasing
2. Train longer (`--num_epochs 500+`)
3. Try lower learning rate (`--lr 0.0005`)
4. Check data (game is working correctly)

### "Results are inconsistent"
1. Run more test games (`--num_games 50`)
2. Try longer training
3. Increase batch size for stability
4. Extend epsilon decay (`--decay_epochs 5000`)

---

## Summary

You've learned:

1. **Game Design** - How to build an interactive environment
2. **Neural Networks** - How to estimate values
3. **RL Algorithm** - How to train from experience
4. **Training** - How to improve with data
5. **Evaluation** - How to measure success

This foundation applies to many RL problems beyond Tetris!

---

## Next Steps

### Learn More RL
- Double DQN (more stable)
- Actor-Critic methods
- Policy Gradient methods
- Model-based RL

### Apply to Other Games
- Flappy Bird
- Snake
- Pong
- Breakout (classic DQN paper!)

### Dig Deeper
- Read DQN paper (Nature, 2015)
- Study convergence theory
- Learn about exploration-exploitation tradeoff
- Implement from scratch (without framework)

---

**Congratulations on completing the project! 🎉**

You've built a complete deep reinforcement learning system from scratch!
