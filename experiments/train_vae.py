import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger  # , CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser
from rdkit import Chem

from pathlib import Path
from rdkit import Chem  # === Check: For validity check
import warnings

warnings.filterwarnings("ignore")

from cd2root import cd2root

cd2root()

# from src.vae.utils import *
from src.vae_top_k import VAE
from src.vae.datamodule import MolDataModule
from src.vae.datamodule import tokenize_smiles


def parse_args():
    parser = LightningArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("-o", "--output", type=str, default="checkpoints/vae/training_set_sca_screened")
    parser.add_argument("--debug", action="store_true",
                        help="Disable Wandb for debugging.")  # "Disable WandB logging and fallback to CSVLogger"

    # === Check: add check mode
    parser.add_argument("--check", action="store_true",
                        help="Enable training-time checks for tokenizer, vocab, reconstruction, and validity.")

    parser.add_lightning_class_args(VAE, "model")
    parser.add_lightning_class_args(MolDataModule, "data")

    args = parser.parse_args()

    del args.model.max_len, args.model.vocab_size

    return args


def main():
    args = parse_args()

    L.seed_everything(args.seed)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    dm = MolDataModule(**args.data)
    dm.prepare_data()
    dm.setup()

    print("max len:", dm.max_len)
    print("vocab size:", dm.vocab_size)

    # === Check ①: tokenizer sample output
    if args.check:
        print("\n===== Check ① Tokenization Sample =====")
        for i in range(3):
            smiles = dm.dataset.smiles[i]
            tokens = tokenize_smiles(smiles)
            print(f"SMILES: {smiles}")
            print(f"TOKENS: {tokens}")

    # === Check ②: vocab content
    if args.check:
        print("\n===== Check ② Vocab Info =====")
        print(f"Vocab size: {len(dm.dataset.vocab)}")
        special_tokens = [t for t in dm.dataset.vocab if ':' in t and t.startswith('[')]
        print(f"Sample special tokens: {special_tokens[:10]}")

    vae = VAE(
        max_len=dm.max_len,
        vocab_size=dm.vocab_size,
        **args.model,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)

    # [ADDED] 训练前快速数值检查（可选）
    if args.check:
        vae.eval()
        with torch.no_grad():
            z = torch.randn(4, vae.latent_dim, device=device)
            logits = vae.decode(z)
            print("\n===== Pretrain decode check =====")
            print("decode logits finite:", torch.isfinite(logits).all().item())
            print("logits stats: min/max/mean/std =",
                  logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
            probs = torch.softmax(logits, dim=-1)
            print("probs finite:", torch.isfinite(probs).all().item(),
                  "probs sum first row:", probs.sum(-1)[0, :5])

    # 选择 logger
    loggers = []
    if not args.debug:
        loggers.append(WandbLogger(project="scadffold_vae"))

    trainer = L.Trainer(
        # gpus=1,
        # accelerator="gpu",
        # devices=1,
        max_epochs=args.epochs,
        # logger=L.loggers.CSVLogger("logs"),
        logger=loggers if loggers else None,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath=output_path,
            ),
        ],
        gradient_clip_val=1.0,  # [ADDED] 防止梯度爆炸
        # precision=32, # [NOTE] Lightning 默认 32 位
        # enable_checkpointing=False,
    )
    print("Training..")
    trainer.fit(vae, datamodule=dm)

    # === Check ③: show best loss
    if args.check:
        print("\n===== Check ③ Best Checkpoint Info =====")
        print(f"Best checkpoint: {trainer.checkpoint_callbacks[0].best_model_path}")
        print(f"Best val_loss: {trainer.callback_metrics.get('val_loss')}")

        # === Load best checkpoint (fix bug: vocab_len -> vocab_size)
        vae = VAE.load_from_checkpoint(
            trainer.checkpoint_callbacks[0].best_model_path,
            max_len=dm.max_len,
            vocab_size=dm.vocab_size,
        )

        # === Check ④+⑤: reconstruction + validity
        if args.check:
            print("\n===== Check ④+⑤ Reconstruction & Validity =====")
            vae.eval()
            with torch.no_grad():
                z = torch.randn(8, vae.latent_dim).to(vae.device)
                token_ids_batch = vae.decode_to_tokens(z, dm, top_k=20)
                smiles_list = dm.decode(token_ids_batch)
                vocab = dm.dataset.vocab
                valid_count = 0
                for i, (token_ids, smiles) in enumerate(zip(token_ids_batch, smiles_list)):
                    try:
                        # 修改点，替换原来错误的调用
                        # tokens = list(vocab.decode(token_ids))  # ❌ 旧代码，vocab没decode方法
                        tokens = [dm.dataset.id_to_token[i.item()] for i in token_ids if i.item() != 0]  # ✅ 正确
                        is_valid = (isinstance(smiles, str) and smiles.strip() and Chem.MolFromSmiles(smiles) is not None)
                    except Exception:
                        tokens = []
                        is_valid = False

                    print(f"[{i}] Tokens: {tokens} -> SMILES: {smiles}    Valid: {is_valid}")

                valid_count = sum(
                    1
                    for smiles in smiles_list
                    if isinstance(smiles, str)
                    and smiles.strip()
                    and Chem.MolFromSmiles(smiles) is not None
                )
                print(f"\nValid SMILES: {valid_count}/{len(smiles_list)}")

    print("Saving..")
    torch.save(vae.state_dict(), output_path / "checkpoint.pt")
    # [ADDED] 也保存 Lightning 完整 checkpoint（便于复现优化器状态）
    trainer.save_checkpoint(str(output_path / "lightning_checkpoint.ckpt"))


if __name__ == "__main__":
    main()
