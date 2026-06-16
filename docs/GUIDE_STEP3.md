# Step 3: Deep Q-Network (The Brain)

## Overview

You now know how the game works. Next: build a neural network that estimates action quality. This is the "brain" of your AI.

## The Core Problem

In Tetris, there are thousands of possible game states. You can't create a table with an entry for each state.

**Solution**: Use a neural network to estimate Q-values for any state.

## What is a Q-Value?

```
Q(state, action) = Expected future reward if you take this action from this state
```

**Example**:
- Current state: board with some blocks
- Action: place piece at position X with 0 rotations
- Q-value: 25 (means you'll get ~25 points on average from here on)

## Network Architecture

```
INPUT (4 features)
    ↓
    state: (lines, holes, bumpiness, height)
    
HIDDEN LAYER 1 (64 neurons + ReLU)
    ↓
    4 × 64 weights = learns patterns
    ReLU activation = non-linearity
    
HIDDEN LAYER 2 (32 neurons + ReLU)
    ↓
    64 × 32 weights = compress information
    ReLU activation = more non-linearity
    
OUTPUT (1 Q-value)
    ↓
    Linear (no activation) = can output any number
```

## Why This Architecture?

### 64 Neurons in First Layer

- **Too small** (< 32): Network can't learn complex patterns
- **Too large** (> 128): Overfits, trains slowly
- **Just right** (64): Empirically works well for Tetris

### 32 Neurons in Second Layer

- Acts as a "bottleneck" - forces the network to compress information
- Reduces overfitting
- Faster training

### ReLU Activation

```
ReLU(x) = max(0, x)
```

**What does it do?**
- Introduces non-linearity (network can learn curved patterns, not just straight lines)
- Fast to compute
- Helps with training stability

**Without ReLU**:
```
output = w1*input + b1 + w2*input2 + b2 ...
Result: only learns linear relationships
```

**With ReLU**:
```
output = ReLU(w1*input + b1) → w2*ReLU(...) → ...
Result: learns complex curved relationships
```

## How Training Works

### The Bellman Equation

This equation defines what Q-values should be:

```
Q(state, action) = reward + γ × max_Q(next_state)
```

**What it means**:
- The true Q-value is the immediate reward plus future rewards
- The network should learn to output values that satisfy this equation

**Example**:
- You're at a state, take an action, get 10 points
- Next state has a max Q-value of 50
- γ (gamma) = 0.99 (we discount future rewards slightly)
- Target Q = 10 + 0.99 × 50 = 59.5

### Training Process

```
1. Predict Q-value with network
   output = network(state)
   
2. Calculate target Q-value
   target = reward + gamma × network(next_state)
   
3. Calculate loss
   loss = (output - target)²
   
4. Update network
   gradient = ∂loss/∂weights
   weights -= learning_rate × gradient
   
5. Repeat
   After many iterations, network learns correct Q-values
```

## Running Step 3

```bash
python code/network.py
```

This will:
- Create the network
- Show the architecture
- Demonstrate a training step with the Bellman equation
- Show how loss decreases

## Key Concepts

### Loss Function

```
Loss = (Predicted Q - Target Q)²
```

If predicted is 30 and target is 50:
```
Loss = (30 - 50)² = 400
```

The network updates weights to reduce this loss.

### Gradient Descent

The optimizer (Adam) adjusts weights in the direction that reduces loss:

```
new_weight = old_weight - learning_rate × gradient
```

Higher learning rate = faster updates but less stable  
Lower learning rate = slower updates but more stable

## Understanding the Network

### What the Network Learns

After training, the network learns:
- "This state is good, Q-value should be high"
- "This state is bad, Q-value should be low"
- "This action leads to more points than that action"

### What the Network Doesn't Learn

The network doesn't memorize individual games. It learns general patterns about what game states are valuable.

## Next Steps

You now understand:
- ✅ What a Q-value is
- ✅ How neural networks estimate Q-values
- ✅ How the Bellman equation defines correct Q-values
- ✅ How gradient descent updates the network

**Next**: Train the network on real game data.
