# https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
# https://arxiv.org/pdf/1902.10565

import contextlib
import io
import multiprocessing
import os
import random
import time
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

import bitboard
import network
from self_play import TrainingExample, play_as_worker, stack_examples, unstack_examples


# When one worker asks for the network, wait a tiny bit for the others so we can do them all together.
GATHER_SECONDS = 0.02


# Starts the worker processes and answers their network requests. The workers play the games, this side runs the network.
def serve_workers(target, arguments: list[tuple], answer, report) -> list:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    processes = []
    connections = []

    for args in arguments:
        ours, theirs = context.Pipe()
        process = context.Process(target=target, args=(theirs, *args), daemon=True)
        process.start()
        theirs.close()
        processes.append(process)
        connections.append(ours)

    results = [None] * len(connections)
    running = list(connections)
    latest = dict.fromkeys(connections)
    began = last_report = time.perf_counter()

    try:
        while running:
            now = time.perf_counter()

            if now - last_report >= 10:
                last_report = now
                report(now - began, list(latest.values()))

            asking = []

            ready = wait(running, timeout=0.5)

            deadline = time.perf_counter() + GATHER_SECONDS

            while ready and len(ready) < len(running):
                more = wait([connection for connection in running if connection not in ready], timeout=max(0.0, deadline - time.perf_counter()))

                if not more:
                    break

                ready += more

            for connection in ready:
                latest[connection] = connection.recv_bytes()

                if latest[connection]:
                    asking.append(connection)
                else:
                    results[connections.index(connection)] = connection.recv()
                    running.remove(connection)

            if not asking:
                for process in processes:
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(f"A worker process exited with code {process.exitcode}.")

                continue

            replies = answer([latest[connection] for connection in asking])

            for connection, reply in zip(asking, replies):
                connection.send_bytes(reply)
    finally:
        for process in processes:
            process.terminate()
            process.join()

    return results


# Splits the games across the workers as evenly as possible and plays them all.
def play_in_workers(
    model: nn.Module,
    workers: int,
    games: int,
    seed: int,
    **search
) -> list[tuple[list[TrainingExample], int, int]]:
    
    shares = [games // workers + (index < games % workers) for index in range(min(workers, games))]
    evaluated = 0

    def answer(requests: list[bytes]) -> list[np.ndarray]:
        nonlocal evaluated
        replies = network.evaluate_states(model, bitboard.unpack_states(b"".join(requests)))
        evaluated += len(replies)
        return np.split(replies, np.cumsum([len(packed) // bitboard.STATE_BYTES for packed in requests])[:-1])

    def report(seconds: float, latest: list) -> None:
        unfinished = sum(share if packed is None else len(packed) // bitboard.STATE_BYTES for share, packed in zip(shares, latest))
        discs = sum(4 if packed is None else bitboard.packed_discs(packed) if packed else 64 for packed in latest)
        print(
            f"  self-play {seconds:.0f}s: "
            f"~{100 * (discs - 4 * len(latest)) / (60 * len(latest)):.0f}%, "
            f"{games - unfinished}/{games} games done, "
            f"{evaluated / seconds:,.0f} positions/s"
        )

    results = serve_workers(play_as_worker, [(share, seed + index, search) for index, share in enumerate(shares)], answer, report)

    return [(unstack_examples(columns), white, black)
            for chunk in results
            for columns, white, black in chunk]


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(logits, target, reduction="none")


# Like squared error, but doesn't explode over big misses.
def huber(prediction: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    return functional.huber_loss(prediction, target, reduction="none", delta=delta)


# soft policy target
# Flattens the move odds so the net also learns about the second and third best moves, not just the top one.
def soften(policy: torch.Tensor) -> torch.Tensor:
    soft = (policy + 1e-7) ** 0.25
    return soft / soft.sum(-1, keepdim=True)


# How wrong the network was on each thing it guesses. The number in front of each one is how much it matters.
# Numbers are copied from KataGo. I don't think it really matters what they are. 
# policy (which move) and value (who wins) are the main ones, the rest just help it learn.
def head_losses(
    out: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    
    policy, reply = target["policy"].unbind(1)
    has_reply = reply.sum(1)
    logits = out["policy"]
    margin = target["margin"]
    td_margin = target["td_margin"]
    outcome = torch.stack((margin > 0, margin < 0, margin == 0), 1).float()

    # optimistic policy weights, upweight games that went better than expected
    long_surprise = (margin - out["score"].detach()) / torch.sqrt(out["score_stdev"].detach() ** 2 + 0.25)
    long_weight = ((outcome[:, 0] + 0.5 * outcome[:, 2]) ** 2 + torch.sigmoid((long_surprise - 1.5) * 3)).clamp(0, 1)
    
    short_surprise = (td_margin[:, 2] - out["td_score"][:, 2].detach()) / torch.sqrt(out["shortterm_score_error"].detach() + 0.25)
    short_weight = torch.sigmoid((short_surprise - 1.5) * 3)

    # score belief pdf + cdf loss
    belief = out["score_belief"]
    belief_pdf = belief.exp()
    margins = torch.arange(-64.0, 65.0, device=belief.device)
    target_belief = functional.one_hot(margin.long() + 64, len(margins)).float()
    belief_mean = (belief_pdf * margins).sum(1, keepdim=True)
    belief_stdev = torch.sqrt(0.001 + (belief_pdf * (margins - belief_mean) ** 2).sum(1))

    # target for the net's own short-term error (uncertainty-weighted playouts)
    short_error = (out["td_score"][:, 2].detach() - td_margin[:, 2]) ** 2 + 1e-4
    boards = target["boards"].flatten(2)
    future_error = ((torch.tanh(out["futurepos"]) - boards[:, 1:]) ** 2).sum(2)

    losses = {
        "policy": 0.93 * cross_entropy(logits[:, 0], policy),
        "reply": 0.15 * cross_entropy(logits[:, 1], reply),
        "soft_policy": 8 * cross_entropy(logits[:, 2], soften(policy)),
        "soft_reply": 8 * 0.15 * has_reply * cross_entropy(logits[:, 3], soften(reply)),
        "long_optimistic": 0.1 * long_weight * cross_entropy(logits[:, 4], policy),
        "short_optimistic": 0.2 * short_weight * cross_entropy(logits[:, 5], policy),
        "value": 0.6 * 1.2 * cross_entropy(out["value"], outcome),
        "score": 0.0015 * huber(out["score"], margin, 12),
        "score_stdev": 0.001 * huber(out["score_stdev"], belief_stdev, 10),
        "score_belief_pdf": 0.02 * cross_entropy(belief, target_belief),
        "score_belief_cdf": 0.02 * ((belief_pdf.cumsum(1) - target_belief.cumsum(1)) ** 2).sum(1),
        "td_score": 0.0004 * huber(out["td_score"], td_margin, 12).sum(1),
        "shortterm_score_error": 0.00002 * huber(out["shortterm_score_error"], short_error, 100),
        "variance_time": 0.0003 * huber(out["variance_time"], target["variance_time"] + 1e-5, 50),
        "ownership": 1.5 * functional.binary_cross_entropy_with_logits(2 * out["ownership"], (1 + boards[:, 0]) / 2, reduction="none").mean(1),
        "futurepos": 0.25 * (future_error[:, 0] + 0.25 * future_error[:, 1]) / 8
    }

    return {name: loss.mean() for name, loss in losses.items()}


# Rotates or flips the training positions. Basically free extra training data.
def symmetric(columns: dict[str, torch.Tensor], symmetry: int) -> dict[str, torch.Tensor]:
    return {
        name: network.transform(
            column.reshape(len(column), -1, 8, 8), symmetry
        ).reshape(column.shape) if name in ("state", "policy", "boards") else column
        for name, column in columns.items()
    }


# One round of training on the examples we got handed.
def train_model(
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    examples: list[TrainingExample],
    epochs: int,
    batch_size: int
) -> dict[str, torch.Tensor]:
    
    if not examples:
        raise ValueError("Cannot train without self-play examples.")

    device = next(model.parameters()).device

    columns = {
        name: torch.from_numpy(column).to(device, torch.float32)
        for name, column in stack_examples(examples).items()
    }

    model.train()
    totals = {}
    batches = 0

    for _ in range(epochs):
        for batch in torch.randperm(len(examples), device=device).split(batch_size):
            model.zero_grad()
            target = symmetric({name: column[batch] for name, column in columns.items()}, random.randrange(8))

            losses = head_losses(model(target.pop("state")), target)
            loss = sum(losses.values())

            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            for name, value in {"loss": loss, **losses}.items():
                totals[name] = totals.get(name, 0) + value.detach()

            batches += 1

    return {name: total / batches for name, total in totals.items()}


# How slowly the average moves. Each update it moves 1/8 of the way toward the new weights.
SWA_SCALE = 8


# SWA, moving average of the weights
# Smoother than the raw model and usually plays a bit better, so it's the one that plays the self-play games.
def update_average(average: nn.Module, model: nn.Module, first: bool = False) -> None:
    with torch.no_grad():
        for kept, new in zip(average.state_dict().values(), model.state_dict().values()):
            if kept.is_floating_point():
                kept.lerp_(new, 1.0 if first else 1 / SWA_SCALE)
            else:
                kept.copy_(new)


# replay window grows sublinearly with data
# Early on we only keep recent games since the old ones were played badly. The buffer grows as we get more data, just slower than the data does. Caps off at 20mb anyways for discord.
def replay_window(examples: int, minimum: int) -> int:
    if examples <= minimum:
        return minimum

    exponent = 0.65
    grown = (examples ** exponent - minimum ** exponent) / (exponent * minimum ** (exponent - 1))
    return int(0.4 * grown + minimum)


# Writes to a temp file first then swaps it in. If it crashes halfway we still have the old save.
@contextlib.contextmanager
def atomic_write(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with open(temporary, "wb") as file:
        yield file

    os.replace(temporary, path)


# Saves the model, the averaged model, the optimizers and how far we got.
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    completed_games: int,
    average: nn.Module | None = None,
    completed_examples: int = 0
) -> None:
    
    with atomic_write(path) as file:
        torch.save(
            {
                "model_state": model.state_dict(),
                "average_state": (average or model).state_dict(),
                "optimizer_state": [optimizer.state_dict() for optimizer in optimizers],
                "completed_games": completed_games,
                "completed_positions": completed_examples
            },
            file
        )


# Discord's limits uploads of around 20mb. We're at 17mb so im adding this now instead of later. The replay save drops its oldest positions until it fits under this.
REPLAY_FILE_BYTES = 20000000


# Saves the replay buffer so we don't have to refill it after a restart.
def save_replay(
    path: str | Path,
    examples,
    completed_games: int
) -> None:

    examples = list(examples)

    while True:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            completed_games=completed_games,
            one_copy_each=True,
            **stack_examples(examples)
        )
        size = buffer.tell()

        if size < REPLAY_FILE_BYTES:
            break

        examples = examples[-int(len(examples) * REPLAY_FILE_BYTES / size * 0.99):]

    with atomic_write(path) as file:
        file.write(buffer.getbuffer())


def load_replay(path: str | Path, completed_games: int) -> list[TrainingExample]:
    path = Path(path)

    if not path.exists():
        return []

    with np.load(path) as saved:
        saved_games = int(saved["completed_games"])

        if saved_games > completed_games:
            print(
                f"{path} is from game {saved_games}, after {completed_games}; refilling the replay buffer instead.")
            return []

        if saved_games < completed_games:
            print(f"{path} is from game {saved_games}; games since then are missing from it.")

        examples = unstack_examples(saved)

        if "one_copy_each" not in saved.files:
            examples = examples[len(examples) % 8::8]

        return examples


# Loads a saved run. Gives an error if the network size doesn't match config.py.
def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer] | None = None,
    average: nn.Module | None = None
) -> tuple[int, int]:
    
    checkpoint = torch.load(Path(path), map_location=network.device)

    shape = network.architecture(checkpoint["model_state"])

    if shape != network.architecture(model.state_dict()):
        described = ", ".join(f"{key}={value}" for key, value in shape.items())
        raise ValueError(f"{path} holds a network with {described}, which doesn't match the model being loaded.")

    model.load_state_dict(checkpoint["model_state"])

    if average is not None:
        average.load_state_dict(checkpoint.get("average_state", checkpoint["model_state"]))

    if optimizers and "optimizer_state" in checkpoint:
        saved = checkpoint["optimizer_state"]

        try:
            if not isinstance(saved, list) or len(saved) != len(optimizers):
                raise ValueError

            for optimizer, state in zip(optimizers, saved):
                settings = [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in optimizer.param_groups
                ]
                optimizer.load_state_dict(state)

                for group, setting in zip(optimizer.param_groups, settings):
                    group.update(setting)
        except ValueError:
            print("Optimizer state doesn't match; starting it fresh, keeping weights.")

    return (
        int(checkpoint.get("completed_games", 0)),
        int(checkpoint.get("completed_positions", checkpoint.get("completed_examples", 0) // 8))
    )