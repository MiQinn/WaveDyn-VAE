import torch
from torch import Tensor
import numpy as np
import random

from cd2root import cd2root
cd2root()

from src.utils.scores_basic import *
from src.vae_top_k_loader import load_vae
from src.pinn.pde import load_sup_wavepde
from src.pinn import PropGenerator
from src.predictor import Predictor
from src.structure_predictor import StructurePredictor

MODES = [
    "random",
    "random_1d",
    "chemspace",
    "wave_sup",
]


class Traversal:
    """
    Perform one-step latent traversal
    """

    def __init__(
        self,
        method: str,
        prop: str,
        vae_name: str = "osimertinib_sca_screened",
        step_size: float = 0.1,
        relative: bool = True,
        minimize: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):

        self.method = method
        self.prop = prop
        self.vae_name = vae_name
        self.step_size = step_size
        self.relative = relative
        self.minimize = minimize
        self.device = device

        assert self.method in MODES, f"mode must be one of {MODES}"

        # Load VAE
        self.dm, self.vae = load_vae(
            file_path=f"data/processed/osimertinib/{self.vae_name}.smi",
            model_path=f"checkpoints/vae/{self.vae_name}/checkpoint.pt",
            device=device,
        )

        # Random traversal
        if method == "random":
            self.u_z = torch.randn(self.vae.latent_dim, device=device)
            return
        elif method == "random_1d":
            self.u_z = torch.zeros(self.vae.latent_dim, device=device)
            idx = random.randint(0, self.vae.latent_dim - 1)
            self.u_z[idx] = (1 if random.random() < 0.5 else -1)
            return
        # Chemspace
        elif method == "chemspace":
            boundary = np.load(
                f"src/chemspace/boundaries_{self.vae_name}/"
                f"boundary_{self.prop}.npy"
            )
            self.u_z = torch.tensor(boundary, device=device).squeeze()
            return

        # Load property predictor
        predictor_ckpt = torch.load(
            f"checkpoints/prop_predictor/osimertinib/random/"
            f"{prop}/checkpoint.pt",
            map_location=device,
        )

        state_dict = predictor_ckpt["model"]
        state_dict = {
            k.replace("cls.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("cls.")
        }

        predictor_input_dim = None

        for k, v in state_dict.items():
            if (
                    "weight" in k
                    and len(v.shape) == 2
            ):
                predictor_input_dim = v.shape[1]
                break

        print(
            f"[Traversal] Predictor input dim: "
            f"{predictor_input_dim}"
        )

        self.predictor = Predictor(
            latent_size=predictor_input_dim,
            hidden_sizes=[512, 256, 128],
            dropout=0.1,
        ).to(device)

        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()

        for p in self.predictor.parameters():
            p.requires_grad = False

        # Structure predictor
        structure_predictor = None
        use_structure = False

        if prop in ["logp", "fsp3"]:
            structure_ckpt = torch.load(
                "checkpoints/structure_predictor/osimertinib/"
                "checkpoint.pt",
                map_location=device,
            )

            structure_state = structure_ckpt["model"]
            structure_state = {
                k.replace("cls.", "", 1): v
                for k, v in structure_state.items()
                if k.startswith("cls.")
            }

            structure_predictor = StructurePredictor(
                latent_size=self.vae.latent_dim,
                hidden_sizes=[512, 256, 128],
                dropout=0.1,
                out_dim=4,
            ).to(device)

            structure_predictor.load_state_dict(structure_state)
            structure_predictor.eval()

            for p in structure_predictor.parameters():
                p.requires_grad = False

            use_structure = True

            print("[Traversal] Structure predictor loaded.")

        # Generator
        self.generator = PropGenerator(
            vae=self.vae,
            model=self.predictor,
            structure_predictor=structure_predictor,
            use_structure=use_structure,
        ).to(device)

        # WavePDE
        if self.method == "wave_sup":
            ckpt = torch.load(
                f"checkpoints/wavepde_prop/osimertinib/random/{prop}/checkpoint.pt",
                map_location=device,
            )

            k_model = ckpt["c"].shape[0]

            self.pde = load_sup_wavepde(
                checkpoint=f"checkpoints/wavepde_prop/osimertinib/random/{prop}/checkpoint.pt",
                generator=self.generator,
                k=k_model,
                time_steps=20,
                n_in=self.vae.latent_dim,
                device=device
            )

            self.pde.eval()

            for p in self.pde.parameters():
                p.requires_grad = False

            self.idx = 0
            self.k = self.pde.k
            self.half_range = self.pde.half_range


    def step(self, z: Tensor, t: int = 0) -> Tensor:
        """
        Perform 1 step of traversal in latent space, return u_z
        When t=0, return 0 tensor
        """
        if t == 0:
            return torch.zeros_like(z)

        # Random
        if self.method in ["random", "random_1d"]:
            u_z = normalize(self.u_z, self.step_size, self.relative)
            return u_z
        # Chemspace
        elif self.method == "chemspace":
            u_z = normalize(self.u_z, self.step_size, self.relative)
            if self.minimize:
                u_z = -u_z
            return u_z
        # WavePDE
        elif self.method == "wave_sup":
            _, u_z = self.pde.inference(self.idx, z, t % self.half_range)
            u_z = normalize(u_z, self.step_size, self.relative)
            return u_z
        else:
            raise ValueError(f"Unknown method {self.method}")
