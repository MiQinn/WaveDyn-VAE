# src/vae_top_k_loader.py

import torch
from pathlib import Path
from src.vae.datamodule import MolDataModule
from src.vae_top_k import VAE


def load_vae(
    file_path: str,
    model_path: str,
    latent_dim: int = 256,
    embedding_dim: int = 128,
    device: torch.device = torch.device("cpu"),
):
    model_path = Path(model_path)
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"

    dm = MolDataModule(file=file_path)
    dm.prepare_data()
    dm.setup()

    vae = VAE(
        vocab_size=dm.vocab_size,
        max_len=dm.max_len,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        p=0.1
    ).to(device)

    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return dm, vae


# 外部统一调用接口
def load_model(**kwargs):
    return load_vae(**kwargs)


# 可选测试代码
if __name__ == "__main__":
    from cd2root import cd2root
    cd2root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm, vae = load_model(
        file_path="data/processed/osimertinib/osimertinib_sca_screened.smi",
        model_path="checkpoints/vae/osimertinib_sca_screened/checkpoint.pt",
        latent_dim=256,
        embedding_dim=128,
        device=device,
    )
    z = torch.randn(1, 256).to(device)
    smiles_list = vae.decode_to_smiles(z, dm, top_k=5)
    print("Sampled SMILES:", smiles_list)
