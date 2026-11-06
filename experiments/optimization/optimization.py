import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from accelerate.utils import set_seed

from absl import logging
import os
from collections import defaultdict
from pandarallel import pandarallel
from tap import Tap
from tqdm import trange
from typing import Literal
import pickle
import random
import json

from rdkit import Chem
from pathlib import Path

from cd2root import cd2root
cd2root()

from experiments.utils.traversal_step import Traversal
from src.utils.scores_basic import MINIMIZE_PROPS, PROP_FN_SCAFFOLD


class Args(Tap):
    prop: str = "logp"  # property to optimize
    n: int | None = None  # number of molecules to generate
    steps: int = 1000  # number of optimization steps
    method: Literal[
        "random",
        "random_1d",
        "chemspace",
        "wave_sup",
        "wave_unsup",
    ] = "wave_sup"  # optimization method
    step_size: float = 0.1  # step sizes
    relative: bool = False  # relative step size
    data_name: str = "training_set_sca_screened"  # data name

    def process_args(self):
        self.model_name = self.prop + "_" + self.method
        self.model_name += f"_{self.step_size}"
        self.model_name += "_relative" if self.relative else "_absolute"


def valid_scaffold_pair(original, new):
    if original == new:
        return False

    mol = Chem.MolFromSmiles(new)

    if mol is None:
        return False

    if original.count("*") != new.count("*"):
        return False

    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if len(ring) >= 8:
            return False
    return True


def main():

    logging.set_verbosity(logging.INFO)
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True, verbose=2)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    set_seed(42)

    args = Args().parse_args()

    traversal = Traversal(
        method=args.method,
        prop=args.prop,
        data_name=args.data_name,
        step_size=args.step_size,
        relative=args.relative,
        minimize=args.prop in MINIMIZE_PROPS,
        device=device,
    )

    with open(
            "data/processed/training_set/training_set_sca_metadata.json",
            "r"
    ) as f:
        metadata = json.load(f)

    # 使用训练集筛选后保留的真实骨架
    original_scaffolds = [
        item["scaffold"]
        for item in metadata
        if item["screening"]["kept"]
    ]

    print(
        f"Found {len(original_scaffolds)} "
        f"kept scaffolds in metadata."
    )

    # 如果指定了 n，则只优化前 n 个
    if args.n is not None:
        original_scaffolds = original_scaffolds[:args.n]

    print(
        f"Using {len(original_scaffolds)} "
        f"scaffolds for optimization."
    )

    x = traversal.dm.encode(original_scaffolds).to(device)

    with torch.no_grad():
        z, *_ = traversal.vae.encode(x)

    del x, metadata

    print(f"prop: {args.prop}")

    results = []

    for t in trange(args.steps):
        u_z = traversal.step(z, t)
        z = z + u_z

        with torch.no_grad():
            smiles = (traversal.vae.decode_to_smiles(z, traversal.dm, top_k=20))

        for i, smi in enumerate(smiles):
            results.append(
                {"idx": i, "step": t,
                 "original_scaffold": original_scaffolds[i], "new_scaffold": smi}
            )

    df = pd.DataFrame(results)

    print("Filtering scaffolds...")

    df = df[
        df.apply(
            lambda row:
            valid_scaffold_pair(row["original_scaffold"], row["new_scaffold"]),
            axis=1,
        )
    ]

    if len(df) == 0:
        print("No valid scaffolds generated.")
        return

    # 保留首次出现
    df = (df.sort_values("step").drop_duplicates(["original_scaffold", "new_scaffold"]))

    print("Calculating properties...")

    df[args.prop] = df["new_scaffold"].apply(PROP_FN_SCAFFOLD[args.prop])

    output_path = (Path("data/interim/optimization") / args.prop /"test")
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = (output_path / args.model_name)

    csv_path = (prefix.with_suffix(".csv"))
    smi_path = (output_path / f"{args.model_name}.smi")
    json_path = (output_path / f"{args.model_name}_mapping.json")

    # csv
    df.to_csv(csv_path, index=False)
    # smi
    (df["new_scaffold"].drop_duplicates().to_csv( smi_path, index=False, header=False))
    # json
    mapping = defaultdict(list)

    for _, row in df.iterrows():
        mapping[row["original_scaffold"]].append(
            {"smiles": row["new_scaffold"], "step": int(row["step"])}
        )

    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"CSV saved to {csv_path}")
    print(f"SMI saved to {smi_path}")
    print(f"Mapping saved to {json_path}")


if __name__ == "__main__":
    main()
