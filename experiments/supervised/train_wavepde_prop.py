import os
import sys
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, PROJECT_ROOT)

import torch
import lightning as L
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import CSVLogger    # WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser

from torch.utils.data import DataLoader, TensorDataset, random_split

from pathlib import Path

from cd2root import cd2root

cd2root()

from src.predictor import Predictor
from src.vae_top_k_loader import load_vae
from src.vae_top_k import VAE
from src.pinn import PropGenerator
from src.pinn.pde import SupWavePDEModel
from src.utils.scores_basic import *
from src.structure_predictor import StructurePredictor


class DataModule(LightningDataModule):
    def __init__(self, n: int = 50_000, batch_size: int = 128):
        super().__init__()

        self.save_hyperparameters()

        self.n = n
        self.batch_size = batch_size
        self.latent_dim = None

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        assert self.latent_dim is not None

        self.dataset = TensorDataset(torch.randn(self.n, self.latent_dim))

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [int(self.n * 0.9), self.n - int(self.n * 0.9)],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-o", "--output", type=str, default="checkpoints")
    parser.add_argument("-p", "--prop", type=str, default="logp")

    parser.add_argument("--bucket", action="store_true")

    parser.add_lightning_class_args(SupWavePDEModel, "model")
    parser.add_lightning_class_args(DataModule, "data")

    args = parser.parse_args()

    del args.model.generator

    return args


def main():
    args = parse_args()

    L.seed_everything(args.seed)

    mode = "bucket" if args.bucket else "random"

    output_path = (Path("checkpoints") / "wavepde_prop" / mode / args.prop / "test")
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, vae = load_vae(
        file_path="data/processed/training_set/training_set_sca_screened.smi",
        model_path="checkpoints/vae/training_set_sca_screened/checkpoint.pt",
        device=device,
    )

    datamodel = DataModule(**args.data)
    datamodel.latent_dim = vae.latent_dim

    predictor_ckpt = torch.load(
        f"checkpoints/prop_predictor/{mode}/{args.prop}/checkpoint.pt",
        map_location=device,
    )

    state_dict = predictor_ckpt["model"]

    print("\n========================")
    print("STATE_DICT KEYS")
    print("========================")

    for k, v in state_dict.items():
        if "weight" in k:
            print(k, v.shape)

    print("========================\n")

    # 去掉 "cls." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("cls."):
            new_state_dict[k.replace("cls.", "", 1)] = v

    # 自动恢复 predictor 输入维度
    first_linear_key = None

    for k, v in new_state_dict.items():
        if "weight" in k and len(v.shape) == 2:
            first_linear_key = k
            break

    predictor_input_dim = new_state_dict[first_linear_key].shape[1]

    print("\n========================")
    print("PREDICTOR INPUT DIM")
    print("========================")
    print(predictor_input_dim)
    print("========================\n")

    args.model.n_in = vae.latent_dim

    predictor = Predictor(
        latent_size=predictor_input_dim,
        hidden_sizes=[512, 256, 128],
        dropout=0.1,
    ).to(device)

    predictor.load_state_dict(new_state_dict)

    print("\n========================")
    print("PREDICTOR EXPECT INPUT")
    print("========================")

    for k, v in predictor.state_dict().items():
        if "mlp.0.block.2.weight" in k:
            print(k, v.shape)

    print("========================\n")

    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    structure_predictor = None

    if args.prop in ["logp", "fsp3"]:

        structure_ckpt = torch.load(
            "checkpoints/structure_predictor/checkpoint.pt",
            map_location=device,
        )

        structure_state_dict = structure_ckpt["model"]

        structure_state_dict = {
            k.replace("cls.", "", 1): v
            for k, v in structure_state_dict.items()
            if k.startswith("cls.")
        }

        structure_predictor = StructurePredictor(
            latent_size=vae.latent_dim,
            hidden_sizes=[512, 256, 128],
            dropout=0.1,
            out_dim=4,
        ).to(device)

        structure_predictor.load_state_dict(structure_state_dict)

        structure_predictor.eval()

        for p in structure_predictor.parameters():
            p.requires_grad = False

        print("\n========================")
        print("STRUCTURE PREDICTOR LOADED")
        print("========================")
        print("checkpoint: checkpoints/structure_predictor/checkpoint.pt")
        print("========================\n")

    use_structure = args.prop in ["logp", "fsp3"]

    generator = PropGenerator(
        vae=vae, model=predictor, structure_predictor=structure_predictor, use_structure=use_structure
    ).to(device)

    model = SupWavePDEModel(
        generator=generator,
        **args.model,
    ).to(device)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=CSVLogger("logs", name=f"wavepde_{mode}_{args.prop}/test"),
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                dirpath=output_path,
            ),
        ],
    )

    print("Training..")

    trainer.fit(
        model,
        datamodel,
    )

    best_model = SupWavePDEModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        generator=generator,
        **args.model,
    ).to(device)

    torch.save(best_model.pde.state_dict(), output_path / "checkpoint.pt")

    print("\n========================")
    print("WAVEPDE SAVED")
    print("========================")
    print(output_path / "checkpoint.pt")
    print("========================\n")


if __name__ == "__main__":
    main()
