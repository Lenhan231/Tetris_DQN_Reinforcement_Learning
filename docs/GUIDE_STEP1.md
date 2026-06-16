# Step 1: The Tetris Game Engine

## Overview

Your first task: build a working Tetris engine. This is the environment where your AI will learn.

## What You Need to Know

### The Board

- **Size**: 20 rows × 10 columns
- **Empty cells**: Represented as `0`
- **Filled cells**: Represented as `1-7` (piece types)
- **Gravity**: Pieces fall automatically

### The 7 Pieces (Tetrominos)

Each piece has 4 rotation states:

```
O-Piece      T-Piece      S-Piece      Z-Piece
[X][X]       [ ][X][ ]    [ ][X][X]    [X][X][ ]
[X][X]       [X][X][X]    [X][X][ ]    [ ][X][X]

I-Piece      L-Piece      J-Piece
[X][X][X][X] [ ][X][ ]    [X][ ][ ]
             [X][X][X]    [X][X][X]
```

### Game Flow

1. **Spawn**: New piece appears at top
2. **Fall**: Piece drops one row each step
3. **Collision**: Piece stops when hitting bottom or another block
4. **Lock**: Piece becomes part of board
5. **Clear**: Check if any rows are complete
6. **Score**: Award points for cleared lines

### State Features (What the AI Sees)

The game represents its state with 4 numbers:

```python
state = (
    lines_cleared,  # Total lines cleared (good = high)
    holes,          # Unfilled gaps under blocks (good = low)
    bumpiness,      # Height variation between columns (good = low)
    height          # Total stack height (good = low)
)
```

**Why these features?**
- **lines_cleared**: Direct reward (agent should maximize this)
- **holes**: Bad structure (wastes space, hard to clear)
- **bumpiness**: Uneven surface (hard to place pieces)
- **height**: Stack growing is bad (runs out of room)

## Running Step 1

```bash
python code/tetris.py
```

This will:
- Create a game
- Simulate 5 random moves
- Print the board and state after each move
- Show how features change

## Understanding the Code

### Creating a Game

```python
from tetris import TetrisGame

game = TetrisGame(height=20, width=10)
state = game.reset()  # Returns (lines, holes, bumpiness, height)
```

### Getting Possible Moves

```python
# Get all valid moves from current state
next_states = game.get_next_states()
# Returns dict: {(x_position, num_rotations): (lines, holes, bumpiness, height)}
```

### Taking an Action

```python
# Choose a move: (x_position, num_rotations)
action = (3, 1)  # Place at column 3 with 1 rotation

# Execute the action
reward, done, state = game.step(action)

# reward: points earned this step
# done: True if game over
# state: new game state (lines, holes, bumpiness, height)
```

## Key Mechanics

### Collision Detection

The game checks:
- Does piece go outside board edges?
- Does piece overlap with existing blocks?

If either is true, the move is invalid.

### Line Clearing

When a piece locks:
1. Check each row from bottom to top
2. If a row is completely filled: clear it
3. Move rows above down
4. Repeat until no more full rows

**Scoring**:
- 1 line: 101 points
- 2 lines: 201 points
- 3 lines: 301 points
- 4 lines: 401 points (Tetris!)

### Game Over

Game ends when a new piece can't spawn (board is too full).

## What to Observe

Run the demo and notice:
- How the board changes
- How state features change with different moves
- What actions clear lines
- What actions create holes

## Next Steps

You now understand:
- ✅ How Tetris works
- ✅ What actions are possible
- ✅ What the state looks like
- ✅ How to interact with the game

**Next**: Study the neural network that will estimate action quality.
