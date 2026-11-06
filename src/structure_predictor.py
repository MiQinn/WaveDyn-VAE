from torch import Tensor, nn

from src.predictor import Block


class StructurePredictor(nn.Module):

    def __init__(
        self,
        latent_size: int = 256,
        hidden_sizes: list[int] = None,
        dropout: float = 0.1,
        out_dim: int = 4,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []

        layers.append(
            Block(
                latent_size,
                hidden_sizes[0],
                dropout=dropout,
            )
        )

        for i in range(1, len(hidden_sizes)):
            layers.append(
                Block(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    dropout=dropout,
                )
            )

        layers.append(
            nn.Linear(
                hidden_sizes[-1],
                out_dim,
            )
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.mlp(x)
