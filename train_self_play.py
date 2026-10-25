"""Run self-play games and train the Reversi network."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import AI_logic as game
from training import (
    ReplayBuffer,
    load_checkpoint,
    play_self_game,
    save_checkpoint,
    train_model
)


def get_arguments() -> argparse.Namespace:
    """Read training options from the command line."""
    parser = argparse.ArgumentParser(
        description="Train the Reversi policy-value network."
    )

    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of self-play games."
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations per move."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training passes after each game."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--checkpoint",
        default="reversi_model.pt"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing checkpoint."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # Prevent PyTorch from creating additional CPU threads.
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    optimizer = torch.optim.Adam(
        game.net.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001
    )

    checkpoint_path = Path(args.checkpoint)
    completed_games = 0

    if args.resume and checkpoint_path.exists():
        completed_games = load_checkpoint(
            checkpoint_path,
            game.net,
            optimizer
        )

        print(
            f"Resumed after {completed_games} games."
        )

    replay = ReplayBuffer()

    for game_number in range(1, args.games + 1):
        (
            examples,
            white_count,
            black_count
        ) = play_self_game(
            game.net,
            simulations=args.simulations
        )

        replay.add_game(examples)

        training_examples = replay.sample(
            min(4096, len(replay))
        )

        losses = train_model(
            game.net,
            optimizer,
            training_examples,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        completed_games += 1

        save_checkpoint(
            checkpoint_path,
            game.net,
            optimizer,
            completed_games
        )

        print(
            f"Game {game_number}/{args.games}: "
            f"white {white_count}, "
            f"black {black_count}, "
            f"examples {len(replay)}, "
            f"loss {losses['loss']:.4f}"
        )


if __name__ == "__main__":
    main()