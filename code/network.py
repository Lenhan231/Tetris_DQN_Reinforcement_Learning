"""
STEP 3: DEEP Q-NETWORK (DQN)
============================
Mục đích: Thiết kế neural network để học Q-values (state value estimation)

═════════════════════════════════════════════════════════════════════════════
WHAT IS A Q-NETWORK?
═════════════════════════════════════════════════════════════════════════════

Q-Value Definition:
  Q(state, action) = expected future reward từ state này nếu chọn action này

Problem: Tính toán Q-value cách trực tiếp rất khó
Solution: Dùng neural network để approximate Q-values!

Network mapping:
  Input (4 features) → Hidden layers (learn patterns) → Output (1 Q-value)

Why Q-network?
  ✓ Dapat estimate Q-value cho bất kỳ state nào
  ✓ Scale tốt hơn table-based Q-learning
  ✓ Learn patterns từ experience

═════════════════════════════════════════════════════════════════════════════
ARCHITECTURE VISUALIZATION
═════════════════════════════════════════════════════════════════════════════

INPUT LAYER (4 features):
  [lines_cleared]
  [holes        ]  ← Game state description
  [bumpiness    ]
  [height       ]

HIDDEN LAYER 1 (64 neurons + ReLU):
  INPUT (4) ──→ FC(4×64 = 256 weights) ──→ ReLU activation ──→ (64)

  Why 64? Balance between:
    - Too small: underfitting (can't learn complex patterns)
    - Too large: overfitting (learns noise), slow training
    - 64 = empirically good for Tetris

HIDDEN LAYER 2 (32 neurons + ReLU):
  (64) ──→ FC(64×32 = 2048 weights) ──→ ReLU activation ──→ (32)

  Why 32? Bottleneck layer - gradually compress information

OUTPUT LAYER (1 Q-value):
  (32) ──→ FC(32×1 = 32 weights) ──→ Linear (no activation) ──→ Q-value

  Why linear? Q-values can be negative (or positive)

TOTAL ARCHITECTURE:
  4 ──→ 64 ──→ 32 ──→ 1
  ↓
  Parameters: 4×64 + 64 + 64×32 + 32 + 32×1 + 1 = ~4.5K params

═════════════════════════════════════════════════════════════════════════════
WHY RELU ACTIVATION?
═════════════════════════════════════════════════════════════════════════════

ReLU = max(0, x)

Benefits:
  ✓ Non-linearity: can learn complex patterns (without it = just linear)
  ✓ Fast: simple computation
  ✓ Sparse: keeps many outputs at 0 (efficient)

Visualization:
  Before ReLU (linear):        After ReLU (non-linear):
    │                            │
    │  ╱                         │  ╱
    │╱                          │╱
    └────────                   └──────

  Only learning linear patterns  Learning curved patterns!

═════════════════════════════════════════════════════════════════════════════
TRAINING INTUITION
═════════════════════════════════════════════════════════════════════════════

Goal: Adjust weights so that network outputs correct Q-values

Bellman Equation:
  Q(s,a) = r + γ * max_Q(s', a')

  Network training tries to satisfy this!

Loss Function:
  Loss = (Q_target - Q_predicted)²

  If Q_predicted = 50 and Q_target = 60:
    Loss = (60 - 50)² = 100 → need to update weights!

Gradient Descent:
  1. Forward: compute Q_pred
  2. Backward: compute ∂Loss/∂weights
  3. Update: weights -= learning_rate × gradient

Repeat many times → Network learns!

═════════════════════════════════════════════════════════════════════════════

Chạy: python network.py (demo + examples)
"""

import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    """Deep Q-Network: maps game features → Q-value estimation

    **Dynamic Architecture** (configurable layer sizes)

    Default: 4 → 64 → 32 → 1

    How it works:
    - Input: State features (default: 4 = lines, holes, bumpiness, height)
    - Hidden 1: Configurable neurons learn intermediate patterns
    - Hidden 2: Configurable neurons compress information
    - Output: Q-value prediction (expected future reward)

    Usage examples:
    >>> # Default Tetris network (4 → 64 → 32 → 1)
    >>> model = DeepQNetwork()
    >>> state = torch.FloatTensor([10, 5, 20, 15])
    >>> q_value = model(state)

    >>> # Custom larger network (4 → 256 → 128 → 1)
    >>> model = DeepQNetwork(hidden1_size=256, hidden2_size=128)

    >>> # Different input size
    >>> model = DeepQNetwork(input_size=8, hidden1_size=128)
    """

    def __init__(self, input_size=4, hidden1_size=64, hidden2_size=32, output_size=1):
        """Initialize Dynamic Deep Q-Network

        Args:
            input_size: Number of input features
                       - None (default): auto-detect from step1.NUM_FEATURES (= 4)
                       - int: custom feature count (for testing)
            hidden1_size: Neurons in first hidden layer (default: 64)
            hidden2_size: Neurons in second hidden layer (default: 32)
            output_size: Number of output values (default: 1 for Q-value)

        Example:
            # Default Tetris network (auto from step1)
            model = DeepQNetwork()  # input_size auto = 4

            # Custom layer sizes
            model = DeepQNetwork(hidden1_size=128, hidden2_size=64)

            # Larger network for complex learning
            model = DeepQNetwork(hidden1_size=256, hidden2_size=128)

            # Override input size (testing with different features)
            model = DeepQNetwork(input_size=8, hidden1_size=256)
        """
        super(DeepQNetwork, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # Layer 1: Input → Hidden 1
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()

        # Layer 2: Hidden 1 → Hidden 2 (bottleneck)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()

        # Layer 3: Hidden 2 → Output (Q-value)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        # Initialize weights (important for training stability)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform weight initialization

        Why? Proper initialization helps training converge faster.
        Xavier ensures variance is consistent across layers.

        Alternative: He initialization (for ReLU), Random normal, etc.
        But Xavier is standard for small networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass: state features → Q-value

        Args:
            x: State features
               Shape: (batch_size, input_size) OR (input_size,)
               Example: [lines_cleared, holes, bumpiness, height]

        Return:
            Q-value prediction
            Shape: (batch_size, output_size) OR (output_size,)
            Value: scalar Q-value (can be negative or positive)

        Example:
        >>> model = DeepQNetwork()
        >>> state = torch.FloatTensor([[10, 5, 20, 15]])  # batch of 1
        >>> q = model(state)
        >>> q.shape
        torch.Size([1, 1])
        """
        # Layer 1 + activation
        x = self.fc1(x)      # (batch, input_size) → (batch, hidden1_size)
        x = self.relu1(x)    # Apply ReLU non-linearity

        # Layer 2 + activation
        x = self.fc2(x)      # (batch, hidden1_size) → (batch, hidden2_size)
        x = self.relu2(x)    # Apply ReLU non-linearity

        # Layer 3 (output, no activation)
        q = self.fc3(x)      # (batch, hidden2_size) → (batch, output_size)

        return q

    def count_params(self):
        """Count total trainable parameters

        Return:
            int: number of parameters

        Calculation:
          fc1: 4×64 weights + 64 biases = 320 params
          fc2: 64×32 weights + 32 biases = 2080 params
          fc3: 32×1 weights + 1 bias = 33 params
          Total: ~2500 params
        """
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        """Print network architecture summary"""
        print("\n" + "=" * 70)
        print("🧠 DEEP Q-NETWORK ARCHITECTURE")
        print("=" * 70)
        print(self)
        total = self.count_params()
        print(f"\nTotal trainable parameters: {total:,}")
        print("=" * 70)


# Demo & Examples
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEMO 1: NETWORK ARCHITECTURE")
    print("=" * 70)

    model = DeepQNetwork()
    model.summary()

    print("\n📊 Parameter Breakdown:")
    print("  Layer 1 (fc1):     4 → 64    = (4×64=256 weights) + 64 biases = 320 params")
    print("  Layer 1 (relu):    no params")
    print("  Layer 2 (fc2):     64 → 32   = (64×32=2048 weights) + 32 biases = 2,080 params")
    print("  Layer 2 (relu):    no params")
    print("  Layer 3 (fc3):     32 → 1    = (32×1=32 weights) + 1 bias = 33 params")
    print("  ─────────────────────────────────────────────────")
    print(f"  Total:             {model.count_params():,} params")
    print("\n💡 Why small? Fast training, minimal overfitting risk")

    # Demo 2: Single sample prediction
    print("\n" + "=" * 70)
    print("DEMO 2: SINGLE STATE PREDICTION")
    print("=" * 70)

    state_features = torch.FloatTensor([10, 5, 20, 15])
    print(f"State features: {state_features.tolist()}")
    print(f"  - lines_cleared: 10 (xóa 10 hàng)")
    print(f"  - holes: 5 (có 5 holes)")
    print(f"  - bumpiness: 20 (gồ ghề)")
    print(f"  - height: 15 (tổng chiều cao)")

    model.eval()  # Set to evaluation mode (no dropout, etc.)
    with torch.no_grad():
        q_value = model(state_features).item()

    print(f"\nNetwork prediction: Q-value = {q_value:.4f}")
    print("Interpretation: Expected future reward from this state")

    # Demo 3: Batch processing (multiple states)
    print("\n" + "=" * 70)
    print("DEMO 3: BATCH PROCESSING (Multiple states)")
    print("=" * 70)

    batch_states = torch.FloatTensor([
        [10, 5, 20, 15],   # State A: good (many lines)
        [5, 10, 15, 10],   # State B: medium
        [20, 2, 5, 8],     # State C: very good (no holes)
    ])
    print(f"Input batch shape: {batch_states.shape}")
    print(f"States:\n{batch_states}")

    with torch.no_grad():
        q_values = model(batch_states)

    print(f"\nOutput shape: {q_values.shape}")
    print(f"Q-values:\n{q_values.squeeze().tolist()}")
    print("\nNote: State C should have highest Q-value (no holes)")

    # Demo 4: Training step simulation
    print("\n" + "=" * 70)
    print("DEMO 4: TRAINING STEP (Bellman Update)")
    print("=" * 70)

    model.train()  # Set to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Example: state → action → reward + next_state
    state = torch.FloatTensor([[10, 5, 20, 15]])
    target_q = torch.FloatTensor([[85.0]])  # Bellman: reward + future

    print(f"State: {state.tolist()[0]}")
    print(f"Target Q (from Bellman equation): {target_q.item():.1f}")

    # Forward pass
    q_pred = model(state)
    loss = criterion(q_pred, target_q)
    print(f"\nBefore update:")
    print(f"  Q_predicted: {q_pred.item():.4f}")
    print(f"  Loss: (target - pred)² = ({target_q.item():.1f} - {q_pred.item():.4f})²")
    print(f"       = {loss.item():.4f}")

    # Backward pass & update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check after update
    with torch.no_grad():
        q_after = model(state).item()

    print(f"\nAfter update:")
    print(f"  Q_predicted: {q_after:.4f}")
    print(f"  Loss: {criterion(torch.FloatTensor([[q_after]]), target_q).item():.4f}")
    print(f"  ✓ Q_value moved closer to target {target_q.item():.1f}")

    print("\n📚 Training Loop Summary:")
    print("  1. Sample batch from replay buffer")
    print("  2. Forward: predict Q-values")
    print("  3. Compute loss: MSE(Q_target - Q_predicted)")
    print("  4. Backward: compute gradients")
    print("  5. Update: optimizer.step()")
    print("  6. Repeat thousands of times → network learns Q-values!")

    print("\n✅ Step 3 complete!")
    print("=" * 70)
