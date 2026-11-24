# https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from bitboard import legal_moves, play_move, squares


# Squares are 0 to 63, so 64 means pass.
PASS_ACTION = 64

# Most of these values come straight from KataGo. I have idea if these are optimal.
# Theres also no way to test changes unless we train a dozens of models with different values. Waste of time.
VALUE_SCALE = 0.25
CPUCT = 1.0 * VALUE_SCALE
CPUCT_LOG = 0.45 * VALUE_SCALE
CPUCT_BASE = 500.0
STDEV_PRIOR = 0.40 * VALUE_SCALE
STDEV_PRIOR_WEIGHT = 2.0
STDEV_SCALE = 0.85
FPU_REDUCTION = 0.2 * VALUE_SCALE
ROOT_FPU_REDUCTION = 0.1 * VALUE_SCALE

# Training only. Flattens the move odds at the top of the tree so it tries more stuff, mostly early in the game. Good for generalization.
ROOT_TEMP_EARLY = 1.25
ROOT_TEMP = 1.1
TEMP_HALFLIFE = 12.0

# Training only. Random noise so the bot sometimes tries moves it would normally ignore. Good for generalization
ROOT_NOISE_CONCENTRATION = 5
ROOT_NOISE_FRACTION = 0.35

# Used when picking the final move, so we don't pick a move that only looks good because we barely checked it.
LCB_STDEVS = 5.0
LCB_MIN_PROPORTION = 0.15

# Used only for final tournament bot. Positions the network isn't sure about count for less.
UNCERTAINTY_COEFF = 0.25 * VALUE_SCALE
UNCERTAINTY_MAX_WEIGHT = 8.0
POLICY_OPTIMISM = 1.0
ROOT_POLICY_OPTIMISM = 0.2

PolicyValueFunction = Callable[[int, int], tuple]


# One position in the search tree. Remembers how often we've been here and how good it is.
@dataclass(slots=True)
class Node:
    prior: float = 0.0
    visit_count: int = 0
    weight_sum: float = 0.0
    weight_sq_sum: float = 0.0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    terminal_value: float | None = None
    lower: float = -1.0
    upper: float = 1.0

    # Have we looked at this node yet.
    @property
    def expanded(self) -> bool:
        return bool(self.children) or self.terminal_value is not None

    # Average result from here, for whoever's turn it is at this node.
    @property
    def mean_value(self) -> float:
        if self.weight_sum <= 0.0:
            return 0.0

        return self.value_sum / self.weight_sum


# The actual search. Same code for training and playing, play=True just turns on the extra heads.
class MCTS:
    def __init__(
        self,
        policy_value: PolicyValueFunction | None = None,
        solver: bool = False,
        play: bool = False
    ) -> None:
        self.policy_value = policy_value

        self.play = play
        self.terminal_weight = UNCERTAINTY_MAX_WEIGHT if play else 1.0

        self.solver = solver

    # Final score is just the piece difference, squished to between -1 and 1.
    @staticmethod
    def terminal_score(own: int, opponent: int) -> float:
        return (own.bit_count() - opponent.bit_count()) / 64

    # Adds the moves under a node and asks the network what it thinks of the position.
    # The yields are how we ask the network. The search pauses there and picks the answer up later,
    # which is what lets use parallelize the games.
    def expand(
        self,
        node: Node,
        own: int,
        opponent: int,
        temperature: float = 1.0,
        root: bool = False
    ):
        moves = legal_moves(own, opponent)

        if moves == 0:
            opponent_moves = legal_moves(opponent, own)

            if opponent_moves == 0:
                node.terminal_value = self.terminal_score(
                    own,
                    opponent
                )
                return node.terminal_value, self.terminal_weight

            child = Node(prior=1.0)
            node.children[PASS_ACTION] = child

            value, weight = yield from self.expand(child, opponent, own)
            self.backpropagate([child], value, weight)
            return -value, weight

        # uncertainty-weighted playouts
        if self.play:
            logits, value, error = yield own, opponent
            weight = UNCERTAINTY_COEFF / (
                error + UNCERTAINTY_COEFF / UNCERTAINTY_MAX_WEIGHT
            )
        else:
            logits, value = yield own, opponent
            weight = 1.0

        actions = squares(moves)

        for action, prior in zip(
            actions,
            self.priors(logits, actions, temperature, root)
        ):
            node.children[action] = Node(prior=prior)

        return float(value), weight

    # Converts network's output into odds for each legal move.
    def priors(
        self,
        logits: np.ndarray,
        actions: list[int],
        temperature: float,
        root: bool
    ) -> list[float]:
        logits = np.asarray(logits, dtype=np.float64)

        # optimistic policy
        if self.play:
            normal, optimistic = logits
            optimism = ROOT_POLICY_OPTIMISM if root else POLICY_OPTIMISM
            logits = normal + optimism * (optimistic - normal)

        legal_logits = logits[actions] / temperature

        legal_logits -= legal_logits.max()
        probabilities = np.exp(legal_logits)
        total = probabilities.sum()

        if not np.isfinite(total):
            return [1.0 / len(actions)] * len(actions)

        return (probabilities / total).tolist()

    # shaped dirichlet noise, straight from KataGo.
    # Training only. Mixes some randomness into the move odds at the top of the tree so the bot
    # sometimes tries moves it thinks are bad. That's how it finds out some of them are actually good.
    def add_root_noise(self, root: Node) -> None:
        actions = [
            action
            for action in root.children
            if action != PASS_ACTION
        ]

        if not actions:
            return

        priors = np.array([root.children[action].prior for action in actions])
        shape = np.log(np.minimum(0.01, priors) + 1e-20)
        shape = np.maximum(0.0, shape - shape.mean())
        alphas = np.full(len(actions), 1.0 / len(actions))

        if shape.sum() > 0.0:
            alphas = 0.5 * (shape / shape.sum() + alphas)

        noise = np.random.dirichlet(ROOT_NOISE_CONCENTRATION * alphas)

        for action, random_prior in zip(actions, noise):
            child = root.children[action]

            child.prior = (
                (1.0 - ROOT_NOISE_FRACTION) * child.prior
                + ROOT_NOISE_FRACTION * float(random_prior)
            )

    # Picks which move to look at next.
    def select_child(
        self,
        node: Node,
        fpu_reduction: float = FPU_REDUCTION
    ) -> tuple[int, Node]:
        child_weight = 0.0
        visited_prior = 0.0

        for child in node.children.values():
            if child.visit_count:
                child_weight += child.weight_sum
                visited_prior += child.prior

        mean = node.mean_value

        # dynamic variance-scaled cpuct, KataGo.
        if node.weight_sum <= 1:
            stdev = STDEV_PRIOR
        else:
            stdev = math.sqrt(
                max(0.0,
                    ((mean * mean + STDEV_PRIOR ** 2) * STDEV_PRIOR_WEIGHT + node.value_sq_sum) / (STDEV_PRIOR_WEIGHT + node.weight_sum - 1) - mean * mean)
                )

        c_puct = CPUCT + CPUCT_LOG * math.log((child_weight + CPUCT_BASE) / CPUCT_BASE)

        explore_scale = (c_puct * math.sqrt(child_weight + 0.01) * (1.0 + STDEV_SCALE * (stdev / STDEV_PRIOR - 1.0)))

        fpu_value = mean - fpu_reduction * math.sqrt(visited_prior)

        floor = node.lower if self.solver else -math.inf

        def score(item: tuple[int, Node]) -> float:
            _, child = item

            if -child.lower <= floor:
                return -math.inf

            if child.visit_count > 0:
                exploitation = -child.mean_value
            else:
                exploitation = fpu_value

            exploration = (explore_scale * child.prior / (1 + child.weight_sum))

            return exploitation + exploration

        return max(node.children.items(), key=score)

    # Plays a move (or a pass) and gives back the new position.
    def next_position(
        self,
        own: int,
        opponent: int,
        action: int
    ) -> tuple[int, int]:
        if action == PASS_ACTION:
            return opponent, own

        return play_move(own, opponent, 1 << action)

    # Looks through the old tree for the position we're in now.
    def find_subtree(
        self,
        node: Node,
        own: int,
        opponent: int,
        target_own: int,
        target_opponent: int
    ) -> Node | None:
        if own == target_own and opponent == target_opponent:
            return node

        if (own | opponent) & ~(target_own | target_opponent):
            return None

        for action, child in node.children.items():
            found = self.find_subtree(
                child,
                *self.next_position(own, opponent, action),
                target_own,
                target_opponent
            )

            if found is not None:
                return found

        return None

    # Walks back up the path adding the result to every node. Flips the sign each step since the turns alternate.
    @staticmethod
    def backpropagate(
        path: list[Node],
        value: float,
        weight: float = 1.0
    ) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.weight_sum += weight
            node.weight_sq_sum += weight * weight
            node.value_sum += weight * value
            node.value_sq_sum += weight * value * value

            value = -value

    # Solver stuff. We hit the end of the game so we know the exact score here.
    # Pass that up the tree and stop once nothing changes.
    @staticmethod
    def update_bounds(path: list[Node]) -> None:
        leaf = path[-1]
        leaf.lower = leaf.upper = leaf.terminal_value

        for node in reversed(path[:-1]):
            lower = max(-child.upper for child in node.children.values())
            upper = max(-child.lower for child in node.children.values())

            if lower == node.lower and upper == node.upper:
                break

            node.lower = lower
            node.upper = upper

    # Search a single position.
    def search(self, own: int, opponent: int, **options) -> Node:
        root, = run_in_lockstep(
            lambda positions: [self.policy_value(*p) for p in positions],
            [self.search_steps(own, opponent, **options)]
        )
        return root

    # The main search loop. Walk down the tree, ask the network about the new spot, send the result back up. Repeat.
    def search_steps(
        self,
        own: int,
        opponent: int,
        simulations: int = 100,
        deadline: float | None = None,
        add_root_noise: bool = False,
        root: Node | None = None
    ):
        if root is None:
            root = Node()

        temperature = 1.0

        # root policy softmax temperature
        if add_root_noise:
            temperature = temperature_at(own, opponent, ROOT_TEMP_EARLY, ROOT_TEMP)

        if not root.expanded:
            root_value, weight = yield from self.expand(root, own, opponent, temperature, root=True)
            self.backpropagate([root], root_value, weight)
        elif self.play and legal_moves(own, opponent):
            logits, _, _ = yield own, opponent
            priors = self.priors(logits, list(root.children), temperature, True)

            for child, prior in zip(root.children.values(), priors):
                child.prior = prior
        elif temperature != 1.0:
            scaled = [child.prior ** (1.0 / temperature) for child in root.children.values()]
            total = sum(scaled)

            for child, prior in zip(root.children.values(), scaled):
                child.prior = prior / total

        if add_root_noise:
            self.add_root_noise(root)

        # Deepest line this search reached, for the tournament bot's stats.
        self.max_depth = 0

        while root.visit_count <= simulations:
            if (deadline is not None and time.perf_counter() >= deadline):
                break

            if root.terminal_value is not None or root.lower == root.upper:
                break

            node = root
            current_own = own
            current_opponent = opponent
            path = [node]

            while node.children:
                action, node = self.select_child(node, ROOT_FPU_REDUCTION if node is root else FPU_REDUCTION)

                current_own, current_opponent = (self.next_position(current_own, current_opponent, action))

                path.append(node)

            self.max_depth = max(self.max_depth, len(path) - 1)

            if node.terminal_value is not None:
                leaf_value, weight = node.terminal_value, self.terminal_weight

            else:
                leaf_value, weight = yield from self.expand(node, current_own, current_opponent)

            if self.solver and node.terminal_value is not None:
                self.update_bounds(path)

            self.backpropagate(path, leaf_value, weight)

        return root


# This is what makes training fast. Runs a bunch of searches at once. Each one pauses when it needs
# the network, then we ask the network about all of them in one big batch. GPUs are way faster doing a lot at once.
def run_in_lockstep(
    evaluate: Callable[[list], list],
    generators: list
) -> list:
    results = [None] * len(generators)
    running = list(enumerate(generators))
    replies = [None] * len(running)

    while running:
        requests = []
        still_running = []

        for (index, generator), reply in zip(running, replies):
            try:
                requests.append(generator.send(reply))
                still_running.append((index, generator))
            except StopIteration as done:
                results[index] = done.value

        running = still_running

        if requests:
            replies = evaluate(requests)

    return results


# Starts at early and slides toward late as the game goes on.
def temperature_at(own: int, opponent: int, early: float, late: float) -> float:
    moves_played = (own | opponent).bit_count() - 4
    return late + (early - late) * 0.5 ** (moves_played / TEMP_HALFLIFE)


# A pessimistic guess of how good a move is. Moves we've barely checked get knocked down more.
def lower_bound(child: Node) -> tuple[float, float]:
    weight, weight_sq = child.weight_sum, child.weight_sq_sum

    if child.visit_count <= 0 or weight <= 0.0 or weight_sq <= 0.0:
        return -2 * LCB_STDEVS, 2 * LCB_STDEVS

    mean = child.mean_value
    prior = weight / (weight * weight / weight_sq) ** 3
    sq_mean = max(child.value_sq_sum / weight, mean * mean + 1e-8)
    sq_mean = (sq_mean * weight + (sq_mean + VALUE_SCALE ** 2) * prior) / (weight + prior)
    weight += prior
    weight_sq += prior * prior
    radius = LCB_STDEVS * math.sqrt((sq_mean - mean * mean) * weight_sq / (weight * weight))
    return -mean - radius, radius


# Mostly just how much each move got searched.
# If a less searched move looks much better than the most searched one, give it a boost.
def play_values(children: dict[int, Node]) -> dict[int, float]:
    values = {action: child.weight_sum for action, child in children.items()}
    top = max(values.values())
    bounds = {action: lower_bound(child) for action, child in children.items()}
    eligible = [a for a, v in values.items() if 0 < v and LCB_MIN_PROPORTION * top <= v]

    if not eligible:
        return values

    best = max(eligible, key=lambda a: bounds[a][0])
    raised = values[best]

    for action, (bound, radius) in bounds.items():
        excess = bounds[best][0] - bound

        if action != best and excess >= 0:
            factor = (radius + excess) / (radius + 0.2 * excess)
            raised = max(raised, factor * factor * values[action])

    values[best] = raised
    return values


# Picks the move we actually play. Normally the most searched move, but if the solver proved a move is safe
# and the other one doesn't look any better than that, take the safe one.
def best_action(children: dict[int, Node]) -> int:
    values = play_values(children)

    def by_value(action: int) -> tuple[float, float]:
        return values[action], children[action].prior

    safe = max(children, key=lambda a: (-children[a].upper, *by_value(a)))
    floor = -children[safe].upper

    open_moves = [a for a in children if -children[a].lower > floor]

    if not open_moves:
        return safe

    best = max(open_moves, key=by_value)
    child = children[best]
    value = min(max(-child.mean_value, -child.upper), -child.lower)

    return best if value > floor else safe


# What the network learns to copy in training. Moves the search spent more time on get more weight.
def visit_policy(root: Node) -> np.ndarray:
    policy = np.zeros(64, dtype=np.float32)
    actions = [action for action in root.children if action != PASS_ACTION]

    if not actions:
        return policy

    weights = np.array([root.children[action].visit_count for action in actions], dtype=np.float64)

    if weights.sum() <= 0.0:
        weights = np.array([root.children[action].prior for action in actions])

    policy[actions] = weights / weights.sum()
    return policy


# temperature 0 means just play the best move. Above 0 adds some randomness, only used in training so games don't all look the same.
def choose_action(root: Node, temperature: float = 0.0) -> int:
    if not temperature:
        return best_action(root.children)

    values = play_values(root.children)
    actions = list(values)
    weights = np.array([values[action] for action in actions])

    if weights.max() <= 0.0:
        weights = np.array([root.children[action].prior for action in actions])

    weights[weights < min(1.0, weights.max() / 64)] = 0.0
    weights = (weights / weights.max()) ** (1.0 / temperature)
    return actions[np.random.choice(len(actions), p=weights / weights.sum())]