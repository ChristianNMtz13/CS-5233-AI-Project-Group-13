# https://github.com/lightvector/KataGo/blob/master/docs/NetworkArchitectures.md
# https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
# https://arxiv.org/pdf/1902.10565

import contextlib
import math
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from bitboard import INPUT_PLANES, pack_states, split_replies, unpack_states


# Use the graphics card if there is one.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How many attention layers live inside each block.
INNER_LAYERS = 3

# policy, opponent reply, soft policy, soft reply, 2 optimistic policies
POLICY_PLANES = 6

# The final score guess is blended together from this many separate guesses.
SCORE_BELIEFS = 4

SILU_GAIN = math.sqrt(2)


# Keeps torch to one cpu thread so it doesn't fight with everything else that's running.
def single_thread() -> None:
    torch.set_num_threads(1)

    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


# fixed variance initialization
# Starts the weights at sensible random sizes so the numbers don't blow up or fade out going through the layers.
def init_weights(
    weight: torch.Tensor,
    scale: float = 1.0,
    gain: float = SILU_GAIN,
    fan_in: int | None = None
) -> None:
    fan_in = fan_in or weight[0].numel()
    std = gain * scale / math.sqrt(fan_in) / 0.8796256610342398
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


# Keeps an output positive, but still gives it a little push to learn even when it's way too low.
def softplus_floor(x: torch.Tensor, square: bool = False) -> torch.Tensor:
    floored = 0.05 * x + 0.95 * functional.softplus(x)
    value = functional.softplus(0.5 * x) ** 2 if square else functional.softplus(x)
    return floored + (value - floored).detach()


# The x, y spot of each of the 64 squares.
SQUARE_XY = torch.stack((torch.arange(64) % 8, torch.arange(64) // 8), -1).float()[:, None, None]


# How the attention knows where each square is. Spins the numbers around by an angle based on the square's position.
def rotate(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    xy: torch.Tensor = SQUARE_XY
) -> torch.Tensor:
    angles = (xy * frequencies).sum(-1)
    cos, sin = angles.cos(), angles.sin()
    x0, x1 = x.unflatten(-1, (-1, 2)).unbind(-1)
    turned = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1)
    return turned.flatten(-2).type_as(x)


# fixed scale instead of batch norm inside the trunk
# Just scales things down a bit. The deeper the block, the more it gets scaled down.
class FixedNorm(nn.Module):
    def __init__(self, channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.gamma * self.scale) + self.beta


# transformer attention, RMSNorm + learned 2D RoPE
# this is the bread and butter how "smart" the model is.
class Attention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.RMSNorm(channels, eps=1e-6)
        self.q, self.k, self.v, self.out = (nn.Linear(channels, channels, bias=False) for _ in range(4))

        shape = (heads, channels // heads // 2, 2)
        signs = torch.randint(0, 2, shape) * 2 - 1
        self.rope = nn.Parameter(torch.empty(shape).uniform_(math.log(1 / 20), 0).exp() * signs)
        self.register_buffer("xy", SQUARE_XY, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, squares, channels = x.shape
        x = self.norm(x)
        q, k, v = (projection(x).view(n, squares, self.heads, -1) for projection in (self.q, self.k, self.v))
        q, k = rotate(torch.stack((q, k)), self.rope, self.xy)
        out = functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(out.transpose(1, 2).reshape(n, squares, channels))


# SwiGLU feed forward
class FeedForward(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(channels, eps=1e-6)
        self.up, self.gate = (
            nn.Linear(channels, 3 * channels, bias=False) for _ in range(2)
        )
        self.down = nn.Linear(3 * channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.down(functional.silu(self.up(x)) * self.gate(x))


# nested bottleneck block
# Shrinks the channels in half, runs a few attention layers, then grows it back. Cheaper than doing it all at full size.
# Straight from KataGo.
class Block(nn.Module):
    def __init__(self, index: int, channels: int, heads: int) -> None:
        super().__init__()
        mid = channels // 2

        self.norm_in = FixedNorm(channels, 1 / math.sqrt(index + 1))
        self.project_in = nn.Linear(channels, mid, bias=False)
        self.layers = nn.ModuleList(
            layer
            for _ in range(INNER_LAYERS)
            for layer in (Attention(mid, heads), FeedForward(mid))
        )
        self.norm_out = FixedNorm(mid, 1 / math.sqrt(INNER_LAYERS + 1))
        self.project_out = nn.Linear(mid, channels, bias=False)
        init_weights(self.project_in.weight)
        init_weights(self.project_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project_in(functional.silu(self.norm_in(x)))

        for layer in self.layers:
            h = h + layer(h)

        return self.project_out(functional.silu(self.norm_out(h)))


# Decides which moves look good. Gives a score for every square.
class PolicyHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        width = channels // 4
        self.spatial = nn.Conv2d(channels, width, 1)
        self.pooled = nn.Conv2d(channels, width, 1)
        self.mix = nn.Linear(2 * width, width, bias=False)
        self.out = nn.Conv2d(width, POLICY_PLANES, 1, bias=False)

        init_weights(self.spatial.weight, 0.8)
        init_weights(self.pooled.weight)
        init_weights(self.mix.weight, 0.6)
        init_weights(self.out.weight, 0.3, gain=1)
        nn.init.zeros_(self.spatial.bias)
        nn.init.zeros_(self.pooled.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global pooling (mean + max)
        pooled = functional.silu(self.pooled(x))
        pooled = torch.cat([pooled.mean((2, 3)), pooled.amax((2, 3))], dim=1)
        h = self.spatial(x) + self.mix(pooled)[:, :, None, None]
        return self.out(functional.silu(h)).flatten(2)


# Guesses who's winning and by how much, plus a bunch of side guesses that help it learn faster.
class ValueHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        width = channels // 4
        hidden = 3 * channels // 4
        belief_width = channels // 2
        self.conv = nn.Conv2d(channels, width, 1)
        self.hidden = nn.Linear(width, hidden)
        self.value = nn.Linear(hidden, 3)
        self.misc = nn.Linear(hidden, 7)
        self.ownership = nn.Conv2d(width, 1, 1, bias=False)
        self.futurepos = nn.Conv2d(channels, 2, 1, bias=False)
        self.belief_pooled = nn.Linear(width, belief_width)
        self.belief_offset = nn.Linear(1, belief_width, bias=False)
        self.belief_parity = nn.Linear(1, belief_width, bias=False)
        self.belief_out = nn.Linear(belief_width, SCORE_BELIEFS)
        self.belief_mix = nn.Linear(width, SCORE_BELIEFS)
        self.register_buffer("margins", torch.arange(-64.0, 65.0)[:, None], persistent=False)

        init_weights(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        for layer, gain, scale, bias_scale in (
            (self.hidden, SILU_GAIN, 1.0, 0.2),
            (self.value, 1.0, 1.0, 0.2),
            (self.misc, 1.0, 1.0, 0.2),
            (self.belief_pooled, SILU_GAIN, 1.0, 1.0),
            (self.belief_out, 1.0, 0.5, 0.1),
            (self.belief_mix, 1.0, 1.0, 0.2)
        ):
            init_weights(layer.weight, scale, gain)
            init_weights(layer.bias, bias_scale, gain, layer.in_features)

        init_weights(self.belief_offset.weight, fan_in=belief_width)
        init_weights(self.belief_parity.weight, fan_in=belief_width)
        init_weights(self.ownership.weight, 0.2, gain=1)
        init_weights(self.futurepos.weight, 0.2, gain=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = functional.silu(self.conv(x))
        pooled = features.mean((2, 3))
        hidden = functional.silu(self.hidden(pooled))
        misc = self.misc(hidden)

        # score "belief" distribution
        belief = functional.silu(
            self.belief_pooled(pooled)[:, None]
            + self.belief_offset(0.05 * self.margins)
            + self.belief_parity(0.5 - self.margins % 2)
        )
        
        belief = (
            functional.log_softmax(self.belief_out(belief), 1)
            + functional.log_softmax(self.belief_mix(pooled), 1)[:, None]
        ).logsumexp(2)

        return {
            "value": self.value(hidden),
            "score": 20 * misc[:, 0],
            "score_stdev": 20 * softplus_floor(misc[:, 1]),
            "variance_time": 40 * softplus_floor(misc[:, 2]),
            # short-term score targets
            "td_score": 20 * misc[:, 3:6],
            # net's own error guess, used for uncertainty-weighted playouts
            "shortterm_score_error": 150 * softplus_floor(misc[:, 6], square=True),
            # ownership target
            "ownership": self.ownership(features).flatten(1),
            "futurepos": self.futurepos(x).flatten(2),
            "score_belief": belief
        }


# The whole network. Board goes in, move odds and score guesses come out.
class PolicyValueNet(nn.Module):
    def __init__(
        self,
        blocks: int = 2,
        channels: int = 32,
        heads: int = 2
    ) -> None:
        super().__init__()

        if blocks < 1 or heads < 1 or channels < 1 or channels % (4 * heads):
            raise ValueError(
                "need blocks >= 1, heads >= 1 and channels a positive multiple of "
                "4 x heads (blocks halve the channels and split them into "
                f"heads of an even size, for RoPE), got blocks={blocks}, "
                f"channels={channels}, heads={heads}"
            )

        self.stem = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        init_weights(self.stem.weight, 0.8)
        self.position = nn.Parameter(torch.zeros(64, channels))
        self.blocks = nn.ModuleList(
            Block(index, channels, heads) for index in range(blocks)
        )

        # the one batch norm, right before the heads
        self.norm = nn.BatchNorm2d(channels, eps=1e-4, momentum=0.01)
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = len(x)

        x = self.stem(x).flatten(2).transpose(1, 2) + self.position

        for block in self.blocks:
            x = x + block(x)

        x = x.transpose(1, 2).reshape(n, -1, 8, 8)
        x = functional.silu(self.norm(x))
        return {"policy": self.policy_head(x), **self.value_head(x)}


# Rotates or mirrors the board. Reversi plays the same no matter which way you turn the board, so there's 8 ways to look at any position.
def transform(boards: torch.Tensor, symmetry: int, inverse: bool = False) -> torch.Tensor:
    turns, mirror = symmetry % 4, symmetry >= 4

    if inverse:
        boards = boards.flip(-1) if mirror else boards
        return torch.rot90(boards, -turns, (-2, -1))

    boards = torch.rot90(boards, turns, (-2, -1))
    return boards.flip(-1) if mirror else boards


# Figures out how big a saved network is just from its weights.
def architecture(state: dict) -> dict:
    return {
        "blocks": sum(key.endswith("project_in.weight") for key in state),
        "channels": state["stem.weight"].shape[0],
        "heads": state["blocks.0.layers.0.rope"].shape[0]
    }


# Builds the right size network and loads the weights into it.
def from_state_dict(state: dict) -> PolicyValueNet:
    model = PolicyValueNet(**architecture(state))
    model.load_state_dict(state)
    return model


# Ask the network about a list of positions.
def evaluate_batch(
    model: nn.Module,
    positions: list[tuple[int, int]],
    play: bool = False,
    symmetry: int | None = None
) -> list[tuple]:
    return split_replies(evaluate_states(model, unpack_states(pack_states(positions)), play, symmetry), play)


def evaluate_states(
    model: nn.Module,
    states: np.ndarray,
    play: bool = False,
    symmetry: int | None = None
) -> np.ndarray:
    model_device = next(model.parameters(), torch.empty(0)).device

    if symmetry is None:
        symmetry = random.randrange(8)

    if model.training:
        model.eval()

    with torch.inference_mode():
        outputs = model(transform(torch.from_numpy(states).to(model_device), symmetry).float())

    policy = transform(outputs["policy"].unflatten(-1, (8, 8)), symmetry, inverse=True).flatten(-2)

    values = (outputs["score"] / 64).clamp(-1, 1)
    columns = [policy[:, 0], values[:, None]]

    if play:
        columns[1:1] = [policy[:, 5]]
        columns.append(outputs["shortterm_score_error"].sqrt()[:, None] / 64)

    return torch.cat(columns, 1).float().cpu().numpy()
