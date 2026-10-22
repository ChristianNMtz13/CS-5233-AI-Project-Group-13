# Simplified AlphaZero Reversi AI

This folder contains an experimental AlphaZero-style Reversi agent built using
PyTorch, a policy-value convolutional neural network, Monte Carlo Tree Search
(MCTS), and self-play training.

The original `AI_logic.py` network and bitboard implementation was kept
unchanged. The additional files connect that network to MCTS and training.

## Files

- `AI_logic.py` - Bitboard functions and policy-value neural network
- `mcts.py` - Monte Carlo Tree Search using PUCT
- `training.py` - Self-play, data augmentation, replay buffer, and training
- `train_self_play.py` - Command-line training program
- `mcts_agent.py` - Tournament-compatible `choose_move()` agent
- `test_alpha_zero.py` - Basic network and MCTS tests
- `reversi_model.pt` - Saved model and optimizer checkpoint

The following files are only used for local comparison:

- `greedy_agent.py` - Professor's original greedy agent
- `local_match.py` - Local match runner
- `reversi.py` - Original Reversi game engine

## Requirements

- Python 3.13
- NumPy
- PyTorch

Install the required packages with:

```powershell
pip install numpy torch
```

## Run the Tests

```powershell
python test_alpha_zero.py
```

## Train From the Beginning

```powershell
python train_self_play.py --games 10 --simulations 50 --epochs 2
```

## Resume Training

```powershell
python train_self_play.py --games 10 --simulations 50 --epochs 2 --resume
```

The checkpoint is saved as `reversi_model.pt`.

## Test Against the Greedy Agent

```powershell
python local_match.py mcts_agent.py greedy_agent.py --games 2 --time-limit 5
```

In this command:

- Agent A is `mcts_agent.py`
- Agent B is `greedy_agent.py`
- The agents switch colors between games

## Current Results

The agent passed the syntax and unit tests without errors.

In a 10-game test using a one-second move limit, the MCTS agent defeated the
greedy agent 10-0. In a two-game test using the full five-second limit, it won
both games with a combined piece lead of 68.

The current checkpoint has completed 12 self-play training games. It is still a
small experimental model and should be trained and evaluated for more games
before being considered a final model.

## How It Works

1. The bitboard code generates legal moves and applies moves efficiently.
2. The neural network predicts a policy over the 64 squares and a board value.
3. MCTS uses those predictions to explore possible future positions.
4. Self-play produces board positions, MCTS policies, and final game results.
5. The network trains on those examples using policy loss and value loss.
6. Board rotations and reflections provide eight training examples per position.

The implementation uses one processing thread. Pass turns are handled inside
MCTS and are not represented as an additional neural-network policy output.