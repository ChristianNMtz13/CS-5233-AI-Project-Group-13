import copy
import random
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np

import config


# Short loss summary for the printouts.
def describe(losses: dict) -> str:
    return ", ".join(f"{name} {losses[name]:.4f}" for name in ("policy", "value", "score"))


# Sets everything up, then loops: play a batch of games, train on them, save. Runs forever unless GAMES is set.
def main() -> None:
    import torch

    import network
    from training import (
        load_checkpoint,
        load_replay,
        play_in_workers,
        replay_window,
        save_checkpoint,
        save_replay,
        train_model,
        update_average
    )

    for name in ("PARALLEL", "WORKERS", "EPOCHS"):
        if getattr(config, name) < 1:
            raise SystemExit(f"config.py: {name} must be at least 1.")

    if not 1 <= config.REPLAY_SAMPLE <= config.REPLAY_BUFFER_SIZE:
        raise SystemExit(
            "config.py: REPLAY_SAMPLE must be 1 to REPLAY_BUFFER_SIZE, "
            "or training never starts."
        )

    if not 0 < config.FULL_SEARCH_FRACTION <= 1:
        raise SystemExit(
            "config.py: FULL_SEARCH_FRACTION must be above 0 and at most 1, "
            "or no positions are trained on."
        )

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    if network.device.type == "cuda":
        network.single_thread()

        torch.backends.cudnn.benchmark = True

    try:
        model = network.PolicyValueNet(config.BLOCKS, config.CHANNELS, config.HEADS)
    except ValueError as error:
        raise SystemExit(f"config.py: {error}")

    model = model.to(network.device, memory_format=torch.channels_last)

    hidden, rest, undecayed = [], [], []

    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or name.endswith("rope"):
            undecayed.append(parameter)
        elif name.startswith("blocks."):
            hidden.append(parameter)
        else:
            rest.append(parameter)

    # Muon is a new optimizer created fairly recently. 2x more efficient than adamW on specific networks.
    # we only use it for the trunk because thats where it has an advantage over AdamW. AdamW is still used for the heads.
    optimizers = [
        torch.optim.Muon(
            hidden,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            adjust_lr_fn="match_rms_adamw"
        ),
        torch.optim.AdamW(
            [{"params": rest}, {"params": undecayed, "weight_decay": 0.0}],
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    ]

    average = copy.deepcopy(model)

    checkpoint_path = Path(config.CHECKPOINT)
    replay_path = checkpoint_path.with_name(f"{checkpoint_path.stem}-replay.npz")
    completed_games = completed_examples = 0

    replay = deque()

    if checkpoint_path.exists():
        try:
            completed_games, completed_examples = load_checkpoint(checkpoint_path, model, optimizers, average)
        except ValueError as error:
            raise SystemExit(
                f"{error}\nSet BLOCKS, CHANNELS and HEADS in "
                "config.py to match it, "
                "or set CHECKPOINT to a new file to start a new run."
            )

        replay.extend(load_replay(replay_path, completed_games))
        print(
            f"Resumed {checkpoint_path} after {completed_games} games, "
            f"{len(replay)} replay examples."
        )
    else:
        print(f"No {checkpoint_path} found; starting a new run.")

    untrained_average = not checkpoint_path.exists()
    losses = None

    print(
        f"Device: {network.device}, "
        f"network {config.BLOCKS} blocks x {config.CHANNELS} channels, "
        f"{config.HEADS} heads"
    )

    total = config.GAMES or float("inf")
    played = 0

    try:
        while played < total:
            batch = min(config.PARALLEL, total - played)
            print(f"Self-play: {batch} games...")
            started = time.perf_counter()

            results = play_in_workers(
                average,
                config.WORKERS,
                batch,
                config.SEED + completed_games,
                simulations=config.SIMULATIONS,
                full_search_fraction=config.FULL_SEARCH_FRACTION,
                fast_simulations=config.FAST_SIMULATIONS
            )

            completed_examples += sum(len(examples) for examples, _, _ in results)
            replay = deque(
                replay,
                maxlen=replay_window(completed_examples, config.REPLAY_BUFFER_SIZE)
            )

            play_seconds = time.perf_counter() - started
            started = last_report = time.perf_counter()
            filling = len(replay) < config.REPLAY_BUFFER_SIZE

            if filling:
                for examples, _, _ in results:
                    replay.extend(examples)

            for trained, (examples, _, _) in enumerate(results, 1):
                if not filling:
                    replay.extend(examples)

                if len(replay) >= config.REPLAY_SAMPLE:
                    losses = train_model(
                        model,
                        optimizers,
                        random.sample(replay, config.REPLAY_SAMPLE),
                        epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE
                    )

                if trained in (len(results) // 2, len(results)):
                    update_average(average, model, first=untrained_average)
                    untrained_average = untrained_average and losses is None

                if losses is not None and time.perf_counter() - last_report >= 10:
                    last_report = time.perf_counter()
                    print(
                        f"  training {last_report - started:.0f}s: "
                        f"{trained}/{len(results)} games, {describe(losses)}"
                    )

            train_seconds = time.perf_counter() - started
            played += len(results)
            completed_games += len(results)

            save_checkpoint(
                checkpoint_path,
                model,
                optimizers,
                completed_games,
                average,
                completed_examples
            )

            if config.BACKUP_EVERY and completed_games % config.BACKUP_EVERY < len(results):
                backup = checkpoint_path.with_stem(
                    f"{checkpoint_path.stem}-{completed_games}"
                )
                shutil.copyfile(checkpoint_path, backup)
                save_replay(replay_path, replay, completed_games)
                print(f"Backed up to {backup}, replay buffer to {replay_path}")

            scores = ", ".join(
                f"{white}-{black}"
                for _, white, black in results
            )

            if losses is None:
                status = f"filling replay buffer to {config.REPLAY_SAMPLE}"
            else:
                status = describe(losses)

            print(
                f"Games {played}/{config.GAMES or 'inf'} "
                f"(total {completed_games}; white-black {scores}): "
                f"examples {len(replay)} (window {replay.maxlen}), {status} "
                f"[self-play {play_seconds:.0f}s, training {train_seconds:.0f}s]"
            )
    finally:
        if replay:
            print(
                f"Saving the replay buffer to {replay_path} "
                "(Ctrl+C again skips it)..."
            )
            save_replay(replay_path, replay, completed_games)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. ")