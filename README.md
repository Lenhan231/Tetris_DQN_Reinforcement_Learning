# 🎮 Tetris Deep Q-Learning

A Deep Q-Network (DQN) agent trained to play Tetris using reinforcement learning.

Implemented techniques:
- Experience Replay
- Epsilon-Greedy Exploration
- Target Network
- Reward Shaping

---

## 📋 Project Structure

```

tetris_from_scratch/
├── code/
│   ├── tetris.py              # Tetris environment and feature extraction
│   ├── network.py             # DQN model architecture
│   ├── train.py               # Training pipeline
│   ├── wandb_config.py        # WandB configuration
│   └── test.py                # Model evaluation
│
├── models/                    # Saved checkpoints
└── README.md

````

---

# 🚀 Installation

```bash
pip install -r requirements.txt
````

---

# 🏋️ Training

Train the model:

```bash
python code/train.py --num_epochs 3000
```

With rendering:

```bash
python code/train.py --num_epochs 3000 --render
```

With WandB tracking:

```bash
python code/train.py --num_epochs 3000 --wandb
```

You can modify training parameters through command-line arguments.

---

# 🧪 Testing

Evaluate a trained model:

```bash
python code/test.py --model_path models/tetris_best_{score}.pth
```

Options:

* Default: evaluate statistics over 10 games
* `--infinite`: continuously render gameplay

Example:

```bash
python code/test.py \
--model_path models/tetris_best_100000.pth \
--infinite
```

---

# 🎛️ Hyperparameters

| Parameter          | Default | Description          |
| ------------------ | ------: | -------------------- |
| num_epochs         |    3000 | Training episodes    |
| batch_size         |     512 | Training batch size  |
| lr                 |   0.001 | Learning rate        |
| gamma              |    0.99 | Discount factor      |
| initial_eps        |     1.0 | Initial exploration  |
| final_eps          |    0.01 | Final exploration    |
| memory_size        |  100000 | Replay buffer size   |
| max_episode_pieces |    2000 | Episode length limit |

Reward shaping:

| Parameter    | Default | Description       |
| ------------ | ------: | ----------------- |
| shape_holes  |    -1.0 | Hole penalty      |
| shape_bump   |     -1.0 | Bumpiness penalty |
| shape_height |     -1.0 | Height penalty    |

---

# 📊 State Features

The agent uses four handcrafted board features:

| Feature       | Description                       |
| ------------- | --------------------------------- |
| lines_cleared | Number of cleared lines           |
| holes         | Empty spaces below blocks         |
| bumpiness     | Height difference between columns |
| height        | Total stack height                |

---

# 🧠 Model Architecture

```
Input (4 features)
        ↓
Linear(64) + ReLU
        ↓
Linear(64) + ReLU
        ↓
Output Q-value
```

Parameters: ~4.5K

---

# ⚙️ Training Method

The agent uses:

* **Experience Replay**: random batches from stored experiences
* **Target Network**: updated periodically for training stability
* **Epsilon-Greedy**: exploration → exploitation transition
* **Reward Shaping**: encourages better board states

---

# 📦 Output Files

Training generates:

```
models/
├── tetris_best_XXXXX.pth
```

WandB logs:

```
code/wandb/
```

---

# 🛠️ Troubleshooting

## Training is slow

Reduce batch size:

```bash
python code/train.py --batch_size 256
```

Enable CUDA if available.

---

## Model stops improving

Try:

```bash
python code/train.py --num_epochs 5000
```

Adjust reward shaping parameters.

---

