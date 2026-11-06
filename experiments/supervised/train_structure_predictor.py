import os
import sys
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger    # WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint, EarlyStopping)

from pathlib import Path
from scipy.stats import linregress

from cd2root import cd2root
cd2root()

from src.predictor import Predictor
from src.structure_predictor import StructurePredictor

class Model(LightningModule):

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_sizes: list = [512, 256, 128],
        dropout: float = 0.1,
        optimizer: str = "adam",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.cls = StructurePredictor(latent_size=latent_dim, hidden_sizes=hidden_sizes, dropout=dropout, out_dim=4)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

        self.val_y_hat = []
        self.val_y = []

    def forward(self, x: Tensor):
        return self.cls(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.val_y_hat.append(y_hat.detach())
        self.val_y.append(y.detach())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_y_hat.clear()
        self.val_y.clear()

    def on_validation_epoch_end(self):

        y_hat = torch.cat(self.val_y_hat).cpu().numpy()
        y = torch.cat(self.val_y).cpu().numpy()

        rs = []
        for i in range(4):
            try:
                r = linregress(y[:, i], y_hat[:, i]).rvalue
            except:
                r = 0.0
            rs.append(r)

        mean_r = sum(rs) / 4.0

        self.log("mean_r", mean_r, prog_bar=True)

        if self.global_rank == 0:
            print("\n========================")
            print("VALIDATION METRICS")
            print("========================")
            print(f"n_star         : {rs[0]:.4f}")
            print(f"n_ring         : {rs[1]:.4f}")
            print(f"n_atoms        : {rs[2]:.4f}")
            print(f"aromatic_ratio : {rs[3]:.4f}")
            print(f"mean_r         : {mean_r:.4f}")
            print("========================\n")

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.AdamW(
                self.cls.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.cls.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class PropDataset(LightningDataModule):

    def __init__(
        self,
        n: int = 15656,
        batch_size: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.y_mean = None
        self.y_std = None

    def setup(self, stage=None):

        base = f"prop_predictor_random_{self.n}_seed{self.seed}_scaffold"
        pt_path = Path("data/interim/props/random") / f"{base}.pt"
        csv_path = Path("data/interim/props/random") / f"{base}.csv"

        print("\n========================")
        print("LOADING DATASET")
        print("========================")
        print(pt_path)
        print(csv_path)

        data = torch.load(pt_path)
        x = data["x"].float()
        df = pd.read_csv(csv_path).set_index("smiles").loc[data["smiles"]]
        y = torch.tensor(
            df[
                ["n_star", "n_ring", "n_atoms", "aromatic_ratio"]
            ].values, dtype=torch.float,
        )

        print("\n========================")
        print("STRUCTURE STATS")
        print("========================")
        print(df[["n_star", "n_ring", "n_atoms", "aromatic_ratio"]].describe())
        print("========================\n")

        self.y_mean = y.mean(dim=0)
        self.y_std = y.std(dim=0) + 1e-8

        y = (y - self.y_mean) / self.y_std
        dataset = TensorDataset(x, y)
        n_total = len(dataset)
        train_size = int(n_total * 0.9)
        val_size = n_total - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        print("\n========================")
        print("DATASET INFO")
        print("========================")
        print(n_total, train_size, val_size)
        print("========================\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--no_early_stop", action="store_true")

    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(PropDataset, "data")

    return parser.parse_args()


def main():

    args = parse_args()
    L.seed_everything(args.data.seed)

    dm = PropDataset(**args.data)
    dm.setup()

    sample_x, _ = dm.train_dataset[0]
    args.model.latent_dim = sample_x.shape[-1]
    model = Model(**args.model)

    save_dir = Path("checkpoints/structure_predictor/test")
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="mean_r",
            mode="max",
            save_top_k=1,
            save_last=True,
            dirpath=save_dir,
            filename="best",
        ),
    ]

    if not args.no_early_stop:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=20)
        )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=CSVLogger("logs", name="structure_predictor/test"),
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)

    best_path = trainer.checkpoint_callback.best_model_path

    print("\n========================")
    print("BEST CHECKPOINT")
    print("========================")
    print(best_path)
    print("========================\n")

    ckpt = torch.load(best_path, map_location="cpu")
    torch.save(
        {"model": ckpt["state_dict"], "y_mean": dm.y_mean, "y_std": dm.y_std},
        save_dir / "checkpoint.pt",
    )

    print("\n========================")
    print("SAVED")
    print("========================")
    print(save_dir / "checkpoint.pt")
    print("========================\n")


if __name__ == "__main__":
    main()
