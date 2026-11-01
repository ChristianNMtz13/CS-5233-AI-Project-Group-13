"""Policy-value network shared by the MCTS agent and training."""

import numpy as np
import torch
from torch import nn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class PolicyValueNet(nn.Module):
    """Four 3x3 convolutions, then a 64-square policy and a value in [-1, 1].

    Layer names and shapes match reversi_model.pt.
    """

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.policy = nn.Linear(channels * 64, 64)
        self.value = nn.Linear(channels * 64, 1)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = torch.relu(conv(x))

        x = x.flatten(1)
        return self.policy(x), torch.tanh(self.value(x))


net = PolicyValueNet().to(device)


def state_array(
    own: int,
    opponent: int
) -> np.ndarray:
    """Create the two input layers expected by the network."""
    planes = np.zeros(
        (2, 8, 8),
        dtype=np.float32
    )

    for action in range(64):
        row, column = divmod(action, 8)

        planes[0, row, column] = (
            own >> action
        ) & 1

        planes[1, row, column] = (
            opponent >> action
        ) & 1

    return planes


def to_tensor(
    own: int,
    opponent: int
) -> torch.Tensor:
    """Turn one position into a batch of one for the network."""
    return torch.from_numpy(
        state_array(own, opponent)
    ).unsqueeze(0).to(device)
