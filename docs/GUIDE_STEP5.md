# Step 5: Testing and Evaluation

## Overview

Training is done! Now: evaluate how well your agent plays on fresh games without training.

## Why Testing Matters

Testing shows:
- **Real performance**: How well does it actually play?
- **Generalization**: Does it work on new games or just memorized training data?
- **Stability**: Is performance consistent or does it vary wildly?

## Running Tests

### Quick Test (10 games)

```bash
python code/test.py --model_path models/tetris_final.pth
```

Output:
```
Average Lines Cleared: 45.2 ± 12.3
Average Score: 4520
Average Pieces: 150
Average Steps: 2300

Rating: Good
```

### Detailed Test (50 games)

```bash
python code/test.py --model_path models/tetris_final.pth --num_games 50
```

More games = more reliable statistics (standard deviation will be smaller)

### Interactive Testing

Watch the agent play and control the game:

```bash
python code/test.py --model_path models/tetris_final.pth --infinite
```

Controls:
- **SPACE**: Pause
- **R**: Reset current game
- **Left/Right Arrow**: Rewind moves
- **Q**: Quit

## Understanding the Metrics

### Average Lines Cleared

The main metric - how many lines does the agent clear per game?

**What's good?**
```
< 20 lines:  Poor (training probably didn't work)
20-50 lines: Okay (agent learned something)
50-100 lines: Good (solid strategy)
100+ lines:  Excellent (converged well)
```

### Standard Deviation (±)

Shows consistency. Lower is more consistent.

```
45.2 ± 5.0   = Consistent (mostly plays between 40-50 lines)
45.2 ± 20.0  = Inconsistent (varies 25-65 lines)
```

Lower std dev = better control

### Average Score

Total points across all games. Higher is better.

Rough conversion:
```
Lines × 50 ≈ Score
(1 line ≈ 101 points, varies by combo)
```

### Average Pieces & Steps

How many pieces placed before game over?

```
Pieces = Game length (more = longer game = better)
Steps = Total number of moves taken
```

### Performance Rating

Simple categorization:

```
< 20 lines:     Poor
20-50 lines:    Okay
50-100 lines:   Good
100-150 lines:  Excellent
150+ lines:     Outstanding
```

## What to Look For

### Sign 1: Good Model

```
Average Lines: 80 ± 15
Pieces: 250+
Score: 8000+
Rating: Excellent
```

**Interpretation**: Model plays consistently, has long games, makes good decisions

### Sign 2: Mediocre Model

```
Average Lines: 35 ± 25
Pieces: 120
Score: 3500
Rating: Okay
```

**Interpretation**: Model learned something but training wasn't enough. Needs more epochs.

### Sign 3: Bad Model

```
Average Lines: 8 ± 3
Pieces: 40
Score: 800
Rating: Poor
```

**Interpretation**: Model barely learned. Check:
- Training loss was decreasing
- Training for enough epochs
- Learning rate not too high

## What Model to Test?

During training, models are saved at checkpoints:

```
models/
├── tetris_ep_100.pth    # After 100 episodes
├── tetris_ep_200.pth    # After 200 episodes
├── tetris_ep_300.pth    # After 300 episodes
└── tetris_final.pth     # Final model
```

**Which to test?**
1. Test `tetris_final.pth` first (best trained model)
2. If poor results, test earlier checkpoints to see learning progress
3. Compare different checkpoints to understand when improvement happens

```bash
# Test checkpoint after 100 episodes
python code/test.py --model_path models/tetris_ep_100.pth --num_games 20

# Test checkpoint after 300 episodes
python code/test.py --model_path models/tetris_ep_300.pth --num_games 20

# Test final model
python code/test.py --model_path models/tetris_final.pth --num_games 50
```

## Analyzing Performance

### Case Study 1: Poor Performance

**Test results**:
```
Average Lines: 15 ± 8
Rating: Poor
```

**Diagnosis**:
1. Check training loss: Is it decreasing?
2. Check training duration: Train for more episodes
3. Check network size: Maybe too small to learn
4. Check learning rate: Maybe too high (unstable training)

**Fix**:
```bash
# Train longer
python code/step4_train_dqn.py --num_epochs 500

# Or lower learning rate
python code/step4_train_dqn.py --lr 0.0005 --num_epochs 300
```

### Case Study 2: Inconsistent Performance

**Test results**:
```
Average Lines: 60 ± 35
Rating: Good (but inconsistent)
```

**Interpretation**:
- Some games are great (90+ lines)
- Some games are mediocre (30 lines)
- Agent has good strategy but sometimes makes bad choices

**Cause**: 
- Epsilon decay too fast (stops exploring too soon)
- Network didn't learn state values well

**Fix**:
```bash
# Slower epsilon decay
python code/step4_train_dqn.py --decay_epochs 5000 --num_epochs 1000

# Or train from scratch with different hyperparameters
python code/step4_train_dqn.py --lr 0.0005 --batch_size 256
```

## Visualizing Training Progress

To see how performance improves over training:

```bash
# Test each checkpoint
for f in models/tetris_ep_*.pth; do
    echo "Testing: $f"
    python code/test.py --model_path "$f" --num_games 10
done
```

This shows:
```
Episode 100: Average 15 lines
Episode 200: Average 35 lines
Episode 300: Average 60 lines
Episode 400: Average 75 lines
```

You should see consistent improvement (increasing average lines over time).

## Next Steps

You now understand:
- ✅ How to evaluate model performance
- ✅ What metrics mean
- ✅ How to diagnose problems
- ✅ How to compare different models

## Experiments to Try

Once you have a working model:

1. **Architecture**: Increase hidden layer sizes
   - Change 64→128, 32→64 in step3_neural_network.py
   
2. **Reward Shaping**: Adjust penalties in step1_tetris_basic.py
   - Try higher rewards for lines cleared
   - Try different penalties for holes/height

3. **Hyperparameters**: Train with different settings
   - Lower learning rate for stability
   - Larger batch size for stability
   - Longer decay for more exploration

4. **Advanced Techniques**:
   - Double DQN (reduces overestimation)
   - Dueling DQN (separate value and advantage)
   - Prioritized Experience Replay

## Summary

This is the complete RL pipeline:
1. ✅ **Step 1**: Build environment (Tetris game)
2. ✅ **Step 3**: Build agent (Neural network)
3. ✅ **Step 4**: Train agent (DQN algorithm)
4. ✅ **Step 5**: Evaluate agent (This step)

Congratulations! You've built a complete RL system! 🎉
