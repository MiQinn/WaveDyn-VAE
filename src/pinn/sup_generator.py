from abc import ABC
import torch
from torch import nn, Tensor

from src.vae_top_k import VAE
from src.predictor import Predictor


class Generator(nn.Module, ABC):
    latent_size: int
    reverse_size: int

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError


class SupVAEGenerator(Generator):
    def __init__(self, model: VAE):
        super().__init__()

        self.model = model

        self.latent_size = model.latent_dim
        self.reverse_size = model.max_len * model.vocab_size

    def forward(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class PropGenerator(Generator):

    def __init__(
        self,
        vae: VAE,
        model: Predictor,
        structure_predictor=None,
        use_structure: bool = False,
    ):
        super().__init__()

        self.vae = vae
        self.model = model

        self.structure_predictor = structure_predictor
        self.use_structure = use_structure

        self.latent_size = vae.latent_dim
        self.reverse_size = 1

    def forward(self, z):

        if self.use_structure:

            structure_feat = self.structure_predictor(z)

            x = torch.cat(
                [z, structure_feat],
                dim=-1,
            )

        else:

            x = z

        return self.model(x)
