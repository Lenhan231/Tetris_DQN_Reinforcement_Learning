# Quick Reference Guide

Fast lookup for running the project.

## Installation

```bash
pip install -r requirements.txt
```

**Required**: Python 3.8+, PyTorch, NumPy, OpenCV-Python

## Running Each Step

### Step 1: Game Demo (2 min)
```bash
python code/tetris.py
```
Shows the Tetris engine with 5 random moves.  
**Learn**: How the game works

### Step 3: Network Demo (2 min)
```bash
python code/network.py
```
Shows the neural network architecture and a training step.  
**Learn**: How Q-values are estimated

### Step 4: Training (30 min - 4 hours)

**Quick test** (30 min):
```bash
python code/train.py --num_epochs 50
```

**Full training** (2-4 hours):
```bash
python code/train.py --num_epochs 300
```

**Common options**:
```bash
# With visualization (slower)
python code/train.py --num_epochs 300 --render

# With performance tracking (Weights & Biases)
python code/train.py --num_epochs 300 --wandb

# Custom hyperparameters
python code/train.py \
    --num_epochs 500 \
    --batch_size 256 \
    --lr 0.0005 \
    --gamma 0.99
```

### Step 5: Testing (5 min)

**Quick test** (10 games):
```bash
python code/test.py --model_path models/tetris_final.pth
```

**Detailed test** (50 games):
```bash
python code/test.py --model_path models/tetris_final.pth --num_games 50
```

**Interactive mode** (watch agent play):
```bash
python code/test.py --model_path models/tetris_final.pth --infinite
```

## File Guide

| File | Purpose |
|------|---------|
| `code/tetris.py` | Game engine |
| `code/network.py` | Neural network |
| `code/train.py` | Training loop |
| `code/test.py` | Evaluation |

## Documentation Guide

| File | Purpose |
|------|---------|
| `README.md` | Overview & concepts |
| `LEARNING_PATH.md` | Complete learning path |
| `QUICK_REFERENCE.md` | This file |
| `docs/GUIDE_STEP1.md` | Step 1 details |
| `docs/GUIDE_STEP3.md` | Step 3 details |
| `docs/GUIDE_STEP4.md` | Step 4 details |
| `docs/GUIDE_STEP5.md` | Step 5 details |

## Learning Order

1. Read `README.md` (overview)
2. Run `code/step1_tetris_basic.py`
3. Read `docs/GUIDE_STEP1.md`
4. Run `code/step3_neural_network.py`
5. Read `docs/GUIDE_STEP3.md`
6. Train: `python code/train.py --num_epochs 300`
7. Read `docs/GUIDE_STEP4.md` (while training)
8. Test: `python code/test.py --model_path models/tetris_final.pth`
9. Read `docs/GUIDE_STEP5.md`
10. Explore `LEARNING_PATH.md` for deeper understanding

## Training Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--num_epochs` | 100 | 50-1000 | More = better but slower |
| `--batch_size` | 512 | 128-1024 | Larger = more stable |
| `--lr` | 0.001 | 0.0001-0.01 | Lower = more stable |
| `--gamma` | 0.99 | 0.9-0.999 | Higher = values future more |
| `--decay_epochs` | 2000 | 500-5000 | Longer = more exploration |

## Performance Indicators

### During Training
- **Loss**: Should decrease over time
- **Epsilon**: Should decay from 1.0 to 0.001
- **Average Reward**: Should increase

### After Training
- **< 20 lines/game**: Poor
- **20-50 lines/game**: Okay
- **50-100 lines/game**: Good
- **100+ lines/game**: Excellent

## Common Commands

```bash
# Full pipeline
python code/tetris.py
python code/network.py
python code/train.py --num_epochs 300
python code/test.py --model_path models/tetris_final.pth --num_games 50

# Compare models at different training stages
python code/test.py --model_path models/tetris_ep_100.pth --num_games 20
python code/test.py --model_path models/tetris_ep_300.pth --num_games 20
python code/test.py --model_path models/tetris_final.pth --num_games 50

# Training with different settings
python code/train.py --num_epochs 500 --lr 0.0005
python code/train.py --num_epochs 300 --batch_size 256
python code/train.py --num_epochs 1000 --decay_epochs 5000
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch numpy opencv-python
```

### "Training is too slow"
```bash
# Remove --render flag
# Reduce --num_epochs for testing
# Reduce --batch_size
python code/train.py --num_epochs 50 --batch_size 256
```

### "Model shows poor performance"
```bash
# Train longer
python code/train.py --num_epochs 500

# Or try lower learning rate
python code/train.py --num_epochs 300 --lr 0.0005
```

### "CUDA not found"
GPU will be used if available. If not, CPU will be used automatically. No action needed.

## Key Concepts (Quick Review)

**Q-Value**: Expected future reward from a state-action pair

**Bellman Equation**: `Q(s,a) = r + γ × max_Q(s')`

**Epsilon-Greedy**: Balance exploration (random) and exploitation (network best)

**Experience Replay**: Store experiences, train on random batches to break correlation

**Target Network**: Separate network for computing Q-targets, updated every 100 episodes

## Model Files

Models are saved in `models/` directory:
- `tetris_ep_100.pth` - Model after 100 episodes
- `tetris_ep_200.pth` - Model after 200 episodes
- `tetris_ep_300.pth` - Model after 300 episodes
- `tetris_final.pth` - Final trained model

Test different models to see learning progress:
```bash
python code/test.py --model_path models/tetris_ep_100.pth --num_games 10
python code/test.py --model_path models/tetris_final.pth --num_games 50
```

## Estimated Time

| Task | Time |
|------|------|
| Setup | 5 min |
| Step 1 Demo | 2 min |
| Step 3 Demo | 2 min |
| Step 4 Training (50 ep) | 30 min |
| Step 4 Training (300 ep) | 2-3 hours |
| Step 5 Testing | 5 min |
| **Total** | **3-4 hours** |

## Next Experiments

Once the basic model works:

1. **Larger network**: Change `dense1_size=128, dense2_size=64`
2. **Lower learning rate**: `--lr 0.0001`
3. **Longer training**: `--num_epochs 1000`
4. **Different rewards**: Modify scoring in `step1_tetris_basic.py`
5. **Advanced techniques**: Implement Double DQN or Dueling DQN

---

**Need details?** See the full guide in `README.md` or step-specific guides in `docs/`.
