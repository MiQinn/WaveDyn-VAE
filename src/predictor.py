from torch import Tensor, nn


class Block(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = out_features

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Mish(),
            nn.Linear(in_features, hidden_features),

            nn.BatchNorm1d(hidden_features),
            nn.Mish(),
            nn.Dropout(dropout),

            nn.Linear(hidden_features, out_features),
        )

        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor):
        return self.block(x) + self.skip(x)


class Predictor(nn.Module):

    def __init__(
        self,
        latent_size: int,
        hidden_sizes: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []

        # first block
        layers.append(
            Block(
                latent_size,
                hidden_sizes[0],
                dropout=dropout,
            )
        )

        # middle blocks
        for i in range(1, len(hidden_sizes)):
            layers.append(
                Block(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    dropout=dropout,
                )
            )

        # output
        layers.append(
            nn.Linear(hidden_sizes[-1], 1)
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.mlp(x)
