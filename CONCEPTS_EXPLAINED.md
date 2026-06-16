# Concepts Explained (Simplified)

Visual and intuitive explanations of key concepts without heavy math.

## The Big Picture: What We're Building

```
┌─────────────────────────────────────────────────────────────┐
│                    THE COMPLETE SYSTEM                      │
└─────────────────────────────────────────────────────────────┘

Step 1: Game               Step 3: Brain              Step 4: Training
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  TETRIS ENGINE   │      │  NEURAL NETWORK  │      │  TRAINING LOOP   │
│                  │      │                  │      │                  │
│ • Board (20x10)  │  →   │ 4 features  →    │  →   │ 1. Play games    │
│ • Pieces         │      │ 64 neurons →     │      │ 2. Store data    │
│ • Collision      │      │ 32 neurons →     │      │ 3. Train network │
│ • Line clear     │      │ 1 Q-value        │      │ 4. Improve       │
│                  │      │                  │      │ 5. Repeat        │
└──────────────────┘      └──────────────────┘      └──────────────────┘
       ↓                          ↓                          ↓
"What is the current      "Is this action      "Make the network
 game state?"             good?"               better at predicting"
```

## Key Concept 1: The Q-Value

### What is it?

A number that represents how good a decision is.

```
Q-value = "How many points will I get if I do this?"

Example:
  Current board: somewhat full
  Action: place piece at column 3
  Q-value: 45
  
  Meaning: "If I take this action, I'll get ~45 points on average"
```

### How do we use it?

```
For each possible action:
  1. Calculate Q-value (network prediction)
  2. Choose action with highest Q-value
  3. Execute action
  4. Repeat

Result: Agent makes smart decisions!
```

### Why we need a network

```
WITHOUT neural network:
  States: millions or billions
  Problem: Can't store table of all states
  
WITH neural network:
  Network learns to estimate Q-values
  Input: Current state
  Output: Q-value estimate
  Works for any state!
```

## Key Concept 2: The Training Process

### Why training is necessary

Initially, the network is random and useless:

```
Random Network:
  Input: "board is mostly empty"
  Output: Q = 47
  ❌ Wrong! Should be ~100 (lots of points possible)
```

Training fixes this:

```
After 100 episodes:
  Input: "board is mostly empty"
  Output: Q = 92
  ✓ Better!

After 300 episodes:
  Input: "board is mostly empty"
  Output: Q = 105
  ✓✓ Correct!
```

### How training works (simplified)

```
STEP 1: Play a game
  Network suggests actions
  Agent takes action
  Gets reward (points)
  Store: (state, reward, next_state)

STEP 2: Learn from past
  Take random batch of old experiences
  For each: compute what Q-value SHOULD be
  
  Bellman: Q_target = reward + future_value
  
  STEP 3: Train network
  Loss = (Network output - Q_target)²
  Update network weights to minimize loss
  
STEP 4: Repeat
  Each iteration: network gets better
```

## Key Concept 3: Exploration vs Exploitation

### The Dilemma

```
EXPLOITATION: "Use what I know works"
  Pro: Make good decisions
  Con: Miss new, better strategies
  
EXPLORATION: "Try new things"
  Pro: Discover better strategies
  Con: Make bad decisions

We need BOTH!
```

### The Solution: Epsilon-Greedy

```
Epsilon (ε) = probability of random action

Episode 1:    ε = 1.0   (100% random)
             Explore everything!
             
Episode 500:  ε = 0.3   (70% greedy, 30% random)
             Mostly exploit, some exploration
             
Episode 2000: ε = 0.001 (99.9% greedy, 0.1% random)
             Trust the network!

Graph:
ε
1.0 │ ╲
    │  ╲
0.5 │   ╲___
    │       ╲___
0.0 │___________╲______
    └─────────────────────→ episodes
    1      1000    2000
```

## Key Concept 4: Experience Replay

### The Problem

If we train on game data in order:

```
Move sequence from one game:
  State 1 → Action A → Reward 10 → State 2
  State 2 → Action B → Reward 5 → State 3
  State 3 → Action A → Reward 8 → State 4

These are HIGHLY CORRELATED
Training on them: network sees false patterns
Result: Overfitting
```

### The Solution

Store experiences and sample randomly:

```
Buffer (30,000 experiences):
  Experience 1: (state_a, action_x, reward_5, next_state_j)
  Experience 2: (state_z, action_y, reward_12, next_state_m)
  Experience 3: (state_b, action_z, reward_3, next_state_k)
  ...
  Experience 30000

Sample random batch of 512:
  Pick 512 random experiences
  Train on these
  ✓ No correlation!
  ✓ More stable training!
```

## Key Concept 5: Network Architecture

### Why This Shape? (4 → 64 → 32 → 1)

```
Layer 1 (4 → 64):
  Input has 4 features
  64 neurons = learn 64 patterns
  Why 64? Empirically works well for Tetris
  
Layer 2 (64 → 32):
  Compress from 64 to 32
  "Bottleneck" prevents overfitting
  
Layer 3 (32 → 1):
  Final output: Q-value
  1 number = expected points
```

### What Each Layer Learns

```
Input Layer:
  [lines] [holes] [bumpiness] [height]
      ↓       ↓         ↓           ↓
  "What is the board state?"
  
Hidden Layer 1 (64 neurons):
  Combines features
  Learns patterns like:
  - "High height + holes = bad"
  - "Cleared lines = good"
  - "High bumpiness = bad"
  
Hidden Layer 2 (32 neurons):
  Compresses patterns
  Learns higher-level concepts
  
Output:
  Q-value = single prediction
  "How good is this state?"
```

## Key Concept 6: The Bellman Equation

### The Core Idea

Q-values should satisfy this relationship:

```
Q(state, action) = immediate_reward + future_value
```

### Example

You're in a state where:
- Immediate reward: 10 points
- Next state has max Q-value: 50

The correct Q-value is:
```
Q = 10 + 0.99 × 50 = 10 + 49.5 = 59.5
```

(0.99 is gamma = discount factor = "future is worth 99% of present")

### How Training Uses It

```
1. Predict: Network says Q = 40
2. Target: Should be Q = 59.5 (by Bellman)
3. Error: 59.5 - 40 = 19.5 (big error!)
4. Update: Adjust weights to reduce error
5. Next time: Network says Q = 55 (closer!)
6. Eventually: Network says Q = 59.5 (correct!)
```

## Complete Training Flow (Visual)

```
                           START
                             ↓
                    ┌────────────────┐
                    │  GAME (Step 1)  │
                    └────────────────┘
                             ↓
                    ┌────────────────┐
                    │ Play 1 game    │ ← Using epsilon-greedy
                    │ Store 200 moves│
                    └────────────────┘
                             ↓
                    ┌────────────────────────┐
                    │ EXPERIENCE BUFFER      │
                    │ (30,000 experiences)   │
                    └────────────────────────┘
                             ↓
                    ┌────────────────────────┐
                    │ Sample random batch    │
                    │ (512 experiences)      │
                    └────────────────────────┘
                             ↓
         ┌───────────────────┴───────────────────┐
         ↓                                       ↓
    ┌──────────┐              ┌──────────────────┐
    │ NETWORK  │              │ BELLMAN EQUATION │
    │ (Predict)│  <─────────→ │ (Target Q-value) │
    └──────────┘              └──────────────────┘
         ↓                                       ↓
         └───────────────────┬───────────────────┘
                             ↓
                    ┌────────────────────────┐
                    │ Compute Loss           │
                    │ Loss = (pred - target)²│
                    └────────────────────────┘
                             ↓
                    ┌────────────────────────┐
                    │ Backpropagation       │
                    │ (Update weights)      │
                    └────────────────────────┘
                             ↓
                    ┌────────────────────────┐
    ┌──────────────│ Decay epsilon          │
    │              │ (less exploration)     │
    │              └────────────────────────┘
    │                       ↓
    │              ┌────────────────────────┐
    │              │ Update target network? │
    │              │ (every 100 episodes)   │
    │              └────────────────────────┘
    │                       ↓
    └──────────────────→  REPEAT
```

## Performance Indicators

### During Training

```
LOSS (should DECREASE):
100 │ ╲
    │  ╲
 50 │   ╲____
    │        ╲___
  0 │___________╲____
    └──────────────────→ episodes

EPSILON (should DECREASE):
1.0 │ ╲
    │  ╲
0.5 │   ╲___
    │       ╲___
0.0 │___________╲____
    └──────────────────→ episodes

REWARD (should INCREASE):
100 │          ╱╱
    │       ╱╱╱
 50 │    ╱╱╱
    │  ╱╱
  0 │╱
    └──────────────────→ episodes
```

### After Training

```
GOOD MODEL:
  Average lines: 80 ± 15
  Score: 8000
  Rating: Excellent

OKAY MODEL:
  Average lines: 35 ± 25
  Score: 3500
  Rating: Okay

BAD MODEL:
  Average lines: 8 ± 3
  Score: 800
  Rating: Poor
```

## Debugging Guide

### Problem: Loss Not Decreasing

```
     loss
 100 │ ___________
 50  │ ___________
  0  │ ___________
     └──────────────→ episodes

Likely causes:
  1. Learning rate too high → try 0.0005
  2. Network too small → increase hidden sizes
  3. Not enough data → train longer
  4. Bad features → check game state
```

### Problem: Training Very Slow

```
Time per episode: 10 seconds
Total: 300 episodes × 10s = 50 minutes

Causes:
  1. Rendering enabled → remove --render
  2. Batch size too small → increase
  3. Computation-heavy → use GPU
```

### Problem: Model Plays Badly

```
Average lines: 15 (poor)

Causes:
  1. Training stopped too early → train longer
  2. Learning rate too high → try 0.0005
  3. Wrong rewards → check step1_tetris_basic.py
  4. Network too small → increase hidden sizes
```

## Summary Table

| Concept | Purpose | Key Insight |
|---------|---------|------------|
| **Q-Value** | Estimate action quality | "How good is this decision?" |
| **Network** | Predict Q-values | Avoids lookup table |
| **Training** | Improve predictions | Network learns patterns |
| **Bellman** | Define correct Q-values | Q = reward + future |
| **Epsilon-Greedy** | Explore & exploit | Balance discovery & usage |
| **Replay Buffer** | Break correlation | Random sampling = stability |
| **Target Network** | Stable targets | Don't chase moving target |

---

**Still confused about something?** See the detailed guides:
- `docs/GUIDE_STEP1.md` - Game details
- `docs/GUIDE_STEP3.md` - Network details
- `docs/GUIDE_STEP4.md` - Training details
- `LEARNING_PATH.md` - Complete progression
