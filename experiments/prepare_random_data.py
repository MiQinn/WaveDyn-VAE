import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pandarallel import pandarallel

import torch
from tqdm import tqdm, trange
from tap import Tap
from pathlib import Path
from rdkit import Chem
from accelerate.utils import set_seed

from cd2root import cd2root
cd2root()

from src.vae_top_k_loader import load_vae
from src.vae_top_k import VAE
from src.vae.datamodule import MolDataModule
from src.utils.scores_basic import *


class Args(Tap):
    smiles_file: str = "data/processed/training_set/training_set_sca_screened.smi"
    vae_path: str = "checkpoints/vae/training_set_sca_screened/checkpoint.pt"
    seed: int = 42
    n: int = 500_000
    batch_size: int = 64
    top_k: int = 20


PROPS = ["logp", "fsp3"]
STRUCT_KEYS = ["n_star", "n_ring", "n_atoms", "aromatic_ratio"]


def make_dataset(args, dm, vae, device, output_dir):

    set_seed(args.seed)

    vae = vae.to(device)
    vae.eval()

    z0 = torch.randn(args.n, vae.latent_dim, device=device)

    valid_smiles = []
    valid_x = []
    valid_struct = []
    seen = set()  # 去重

    print("Generating molecules...")

    for i in tqdm(range(0, args.n, args.batch_size)):
        z_batch = z0[i: i + args.batch_size]

        with torch.no_grad():
            smiles_batch = vae.decode_to_smiles(z_batch, dm=dm, top_k=args.top_k)

        for j, smi in enumerate(smiles_batch):

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # canonical SMILES
            canon = Chem.MolToSmiles(mol)

            if canon in seen:
                continue

            # structure features
            n_star = canon.count("*")
            n_ring = mol.GetRingInfo().NumRings()

            real_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
            n_atoms = len(real_atoms)

            aromatic_ratio = (sum(a.GetIsAromatic() for a in real_atoms) / (n_atoms + 1e-8))

            seen.add(canon)

            valid_smiles.append(canon)
            valid_x.append(z_batch[j].detach().cpu())
            valid_struct.append([n_star, n_ring, n_atoms, aromatic_ratio])

    x0 = torch.stack(valid_x)

    # normalize structure features
    s_raw = torch.tensor(valid_struct, dtype=torch.float)
    s_mean = s_raw.mean(dim=0, keepdim=True)
    s_std = s_raw.std(dim=0, keepdim=True) + 1e-8
    s0 = (s_raw - s_mean) / s_std

    print(f"Valid scaffolds: {len(valid_smiles)} / {args.n}")

    torch.save(
        {
            "x": x0,
            "s": s0,
            "s_mean": s_mean,
            "s_std": s_std,
            "smiles": valid_smiles,
        },
        output_dir / f"raw_random_seed{args.seed}.pt"
    )

    return x0, s0, s_raw, s_mean, s_std, valid_smiles

def soft_clip_logp(x, low=-4, high=4):
    return np.tanh((x - (low + high)/2) / (high - low)) * (high - low)/2 + (low + high)/2

def compute_properties(x, s, s_raw, smiles):

    rows = []
    keep_idx = []

    print("Computing scaffold properties...")

    for i, smi in enumerate(tqdm(smiles)):

        row = {
            "smiles": smi,
            "n_star": float(s_raw[i][0]),
            "n_ring": float(s_raw[i][1]),
            "n_atoms": float(s_raw[i][2]),
            "aromatic_ratio": float(s_raw[i][3]),
        }

        ok = True

        for prop in PROPS:
            try:
                val = PROP_FN_SCAFFOLD[prop](smi)
                if val is None or np.isnan(val):
                    ok = False
                    break
                if prop == "logp":
                    val = soft_clip_logp(val)
                row[prop] = val
            except:
                ok = False
                break

        if ok:
            rows.append(row)
            keep_idx.append(i)

    print(f"After filtering: {len(keep_idx)}")

    x = x[keep_idx]
    s = s[keep_idx]
    s_raw = s_raw[keep_idx]
    smiles = [smiles[i] for i in keep_idx]

    df = pd.DataFrame(rows)

    return x, s, s_raw, smiles, df

def main():
    args = Args().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("data/interim/props/random/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    dm, vae = load_vae(
        file_path=args.smiles_file,
        model_path=args.vae_path
    )

    x, s, s_raw, s_mean, s_std, smiles = make_dataset(args, dm, vae, device, output_dir)

    x, s, s_raw, smiles, df = compute_properties(x, s, s_raw, smiles)

    name = f"prop_predictor_random_{len(df)}_seed{args.seed}_scaffold"

    torch.save(
        {
            "x": x,
            "s": s,
            "s_mean": s_mean,
            "s_std": s_std,
            "smiles": smiles,
        },
        output_dir / f"{name}.pt"
    )

    df.to_csv(output_dir / f"{name}.csv", index=False)

    print("\nSaved dataset:")
    print(" name:", name)
    print(" size:", len(df))


if __name__ == "__main__":
    main()
