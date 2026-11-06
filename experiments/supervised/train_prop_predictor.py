import os
import sys
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
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

        self.cls = Predictor(latent_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

        self.val_y_hat = []
        self.val_y = []

    def forward(self, x: Tensor):
        return self.cls(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        self.val_y_hat.append(y_hat.detach())
        self.val_y.append(y.detach())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_y_hat.clear()
        self.val_y.clear()

    def on_validation_epoch_end(self):
        val_y_hat = torch.cat(self.val_y_hat).flatten().cpu().numpy()
        val_y = torch.cat(self.val_y).flatten().cpu().numpy()

        try:
            r = linregress(val_y, val_y_hat).rvalue
        except:
            r = 0.0

        self.log("lin_reg_r", float(r), prog_bar=True)

        if self.global_rank == 0:
            print("\n========================")
            print("VALIDATION METRICS")
            print("========================")
            print(f"lin_reg_r = {r:.4f}")
            print("========================\n")

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.AdamW(
                self.cls.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
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
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class PropDataset(LightningDataModule):

    def __init__(
        self,
        prop: str = "logp",
        n: int = 18341,
        batch_size: int = 256,
        seed: int = 42,
        num_workers: int = 4,
        bucket: bool = False,
    ):
        super().__init__()
        self.prop = prop
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.bucket = bucket

    def setup(self, stage=None):
        #  根据 bucket 开关选择路径
        if self.bucket:
            base_name = f"prop_predictor_bucket_{self.n}_seed{self.seed}_scaffold"
            base_dir = Path("data/interim/props/bucket")
        else:
            base_name = f"prop_predictor_random_{self.n}_seed{self.seed}_scaffold"
            base_dir = Path("data/interim/props/random")

        pt_path = base_dir / f"{base_name}.pt"
        csv_path = base_dir / f"{base_name}.csv"

        print("\n========================")
        print("LOADING DATASET")
        print("========================")
        print("MODE:", "BUCKET" if self.bucket else "RANDOM")
        print("PT:", pt_path)
        print("CSV:", csv_path)

        data = torch.load(pt_path)
        x = data["x"].float()
        s = data["s"].float()
        smiles = data["smiles"]

        df = pd.read_csv(csv_path).set_index("smiles").loc[smiles]
        y = torch.tensor(df[self.prop].values, dtype=torch.float)

        print("\n========================")
        print("FULL DATA logP distribution")
        print("========================")
        print(df[self.prop].describe())
        print("========================\n")

        if self.prop in ["logp", "fsp3"]:
            print("Using latent + structure as input")
            x = torch.cat([x, s], dim=-1)
        else:
            print("Using latent ONLY as input")

        print("\n========================")
        print("PROPERTY STATS")
        print("========================")
        print(f"{self.prop} min  = {y.min().item():.4f}")
        print(f"{self.prop} max  = {y.max().item():.4f}")
        print(f"{self.prop} mean = {y.mean().item():.4f}")
        print(f"{self.prop} std  = {y.std().item():.4f}")
        print("========================\n")

        self.y_mean = y.mean().item()
        self.y_std = y.std().item()
        y = (y - self.y_mean) / (self.y_std + 1e-8)

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
        print(f"Total samples: {n_total}")
        print(f"Train samples: {train_size}")
        print(f"Val samples:   {val_size}")
        print("========================\n")

        # 保存 sample 用于推断维度
        self.sample = dataset[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=50,)
    # 新增：是否关闭 early stopping
    parser.add_argument("--no_early_stop", action="store_true", help="Disable EarlyStopping")
    parser.add_argument("--bucket", action="store_true")

    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(PropDataset, "data")

    return parser.parse_args()


def main():
    args = parse_args()

    L.seed_everything(args.data.seed)
    args.data.bucket = args.bucket
    dm = PropDataset(**args.data)
    dm.setup()

    sample_x, _ = dm.sample
    args.model.latent_dim = sample_x.shape[-1]

    model = Model(**args.model)

    mode = "bucket" if dm.bucket else "random"
    save_dir = Path("checkpoints/prop_predictor") /mode / dm.prop / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    # callbacks 动态控制
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="lin_reg_r",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            dirpath=save_dir,
            filename="best",
        ),
    ]

    if not args.no_early_stop:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=20))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=CSVLogger("logs", name=f"{mode}_{dm.prop}_lr{args.model.learning_rate}/test"),
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
        {"model": ckpt["state_dict"], "y_mean": dm.y_mean, "y_std": dm.y_std, "prop": dm.prop},
        save_dir / "checkpoint.pt",
    )

    print("\n========================")
    print("FINAL MODEL SAVED")
    print("========================")
    print(save_dir / "checkpoint.pt")
    print("========================\n")


if __name__ == "__main__":
    main()
