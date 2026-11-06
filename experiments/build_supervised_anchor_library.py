import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================
METADATA_FILE = "data/processed/training_set/training_set_sca_metadata.json"
OPT_DIR = "data/interim/optimization"

TASKS = ["logp", "fsp3", "n_star", "n_ring", "n_atoms"]

OUTPUT_DIR = "result/sup/anchor_library/test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUP_SMI = os.path.join(OUTPUT_DIR, "supervised_scaffolds.smi")
SUP_CSV = os.path.join(OUTPUT_DIR, "supervised_scaffolds.csv")
SUP_JSON = os.path.join(OUTPUT_DIR, "supervised_scaffolds.json")

ANCHOR_LIBRARY = os.path.join(OUTPUT_DIR, "anchor_library.json")
ANCHOR_LIBRARY_COMBO = os.path.join(OUTPUT_DIR, "anchor_library_combo.json")
SCAFFOLD_ANCHOR = os.path.join(OUTPUT_DIR, "scaffold_anchor_library.json")


# ============================================================
# Utilities
# ============================================================
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def find_star_atoms(mol):
    return [a for a in mol.GetAtoms() if a.GetSymbol() == "*"]


def find_star_atoms_with_index(mol):
    return sorted([a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"])


def atom_signature_from_rdkit(atom):
    return {
        "element": atom.GetSymbol(),
        "aromatic": atom.GetIsAromatic(),
        "hybrid": str(atom.GetHybridization()),
        "in_ring": atom.IsInRing(),
        "degree": atom.GetDegree()
    }


def infer_scaffold_star_info(star):
    neighbors = star.GetNeighbors()

    soft_aromatic = any(n.GetIsAromatic() for n in neighbors)
    soft_in_ring = any(n.IsInRing() for n in neighbors)
    real_neighbors = [n for n in neighbors if n.GetSymbol() != "*"]

    hard_neighbors = [atom_signature_from_rdkit(n) for n in real_neighbors]

    return {
        "soft_aromatic": soft_aromatic,
        "soft_in_ring": soft_in_ring,
        "hard_neighbors": hard_neighbors
    }


def infer_sidechain_dummy_info(mol):

    infos = []

    for star_idx in find_star_atoms_with_index(mol):

        star = mol.GetAtomWithIdx(star_idx)

        real_neighbors = [
            n for n in star.GetNeighbors()
            if n.GetSymbol() != "*"
        ]

        if len(real_neighbors) != 1:
            continue

        nb = real_neighbors[0]

        infos.append({
            "element": nb.GetSymbol(),
            "aromatic": nb.GetIsAromatic(),
            "hybrid": str(nb.GetHybridization()),
            "in_ring": nb.IsInRing(),
            "degree": nb.GetDegree()
        })

    return infos


def infer_dummy_pair_distances(mol):
    stars = find_star_atoms_with_index(mol)

    if len(stars) <= 1:
        return {}

    dmat = rdmolops.GetDistanceMatrix(mol)
    dist = {}

    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            dist[f"{i},{j}"] = int(dmat[stars[i], stars[j]])

    return dist


def build_distance_hist(distance_dict):
    hist = defaultdict(int)

    for d in distance_dict.values():
        hist[d] += 1

    return dict(hist)


# ============================================================
# 1. Sidechain library (RDKit reconstruction)
# ============================================================
def build_sidechain_library():

    print("\nBuilding sidechain anchor library...")

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    anchor_library = {}
    anchor_library_combo = {}
    seen = defaultdict(set)
    total_sidechains = 0
    skipped = 0

    for entry in tqdm(metadata):
        original = entry.get("original")
        sidechains = entry.get("sidechains", [])

        for sc in sidechains:
            smi = sc.get("smiles")
            if smi is None:
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1
                continue

            dummy_info = infer_sidechain_dummy_info(mol)
            if len(dummy_info) == 0:
                skipped += 1
                continue

            dummy_pair_distances = infer_dummy_pair_distances(mol)
            distance_hist = build_distance_hist(dummy_pair_distances)

            combo_sig = [{
                "element": d["element"],
                "aromatic": d["aromatic"],
                "hybrid": d["hybrid"],
                "in_ring": d["in_ring"],
                "degree": d["degree"]}
                for d in dummy_info]

            key = json.dumps(
                {
                    "n_attach": len(dummy_info),
                    "attachment_signatures": sorted(
                        combo_sig,
                        key=lambda x: (
                            x["element"],
                            x["degree"],
                            x["aromatic"],
                            x["in_ring"]))
                },
                sort_keys=True)

            record = {
                "sidechain": smi,
                "dummy_atom_spec": dummy_info,
                "dummy_pair_distances": dummy_pair_distances,
                "distance_hist": distance_hist,
                "source_mol": original
            }

            total_sidechains += 1

            if key not in anchor_library:
                anchor_library[key] = []

            if smi not in seen[key]:
                anchor_library[key].append(record)
                seen[key].add(smi)

            if key not in anchor_library_combo:
                anchor_library_combo[key] = []

            anchor_library_combo[key].append(record)

    with open(ANCHOR_LIBRARY, "w") as f:
        json.dump(anchor_library, f, indent=2, ensure_ascii=False)

    with open(ANCHOR_LIBRARY_COMBO, "w") as f:
        json.dump(anchor_library_combo, f, indent=2, ensure_ascii=False)

    print(f"Total sidechains: {total_sidechains}")
    print(f"Unique signatures: {len(anchor_library)}")
    print(f"Skipped sidechains: {skipped}")


# ============================================================
# 2. scaffold set (unchanged)
# ============================================================
def build_supervised_scaffolds():

    print("\nBuilding supervised scaffold library...")

    scaffold_set = set()
    meta_json = defaultdict(lambda: {"tasks": [], "sources": []})
    all_csv = []

    for task in TASKS:
        task_dir = os.path.join(OPT_DIR, task)
        files = os.listdir(task_dir)

        smi_file = None
        csv_file = None
        json_file = None

        for f in files:
            if f.endswith(".smi"):
                smi_file = os.path.join(task_dir, f)
            elif f.endswith(".csv") and "_wave_sup_" in f:
                csv_file = os.path.join(task_dir, f)
            elif f.endswith(".json") and "mapping" in f:
                json_file = os.path.join(task_dir, f)

        if smi_file:
            with open(smi_file, "r") as f:
                for line in f:
                    smi = line.strip()
                    can = canonicalize_smiles(smi)
                    if can:
                        scaffold_set.add(can)

        if csv_file:
            df = pd.read_csv(csv_file)
            df["task"] = task
            all_csv.append(df)

        if json_file:
            with open(json_file, "r") as f:
                mapping = json.load(f)

            for original, items in mapping.items():
                for item in items:
                    smi = item["smiles"]
                    can = canonicalize_smiles(smi)

                    if can:
                        meta_json[can]["tasks"].append(task)
                        meta_json[can]["sources"].append({
                            "task": task,
                            "original": original,
                            "step": item.get("step")
                        })

    with open(SUP_SMI, "w") as f:
        for smi in sorted(scaffold_set):
            f.write(smi + "\n")

    if all_csv:
        pd.concat(all_csv, ignore_index=True).to_csv(SUP_CSV, index=False)

    with open(SUP_JSON, "w") as f:
        json.dump(dict(meta_json), f, indent=2, ensure_ascii=False)

    print(f"Unique scaffolds: {len(scaffold_set)}")
    return scaffold_set


# ============================================================
# 3. scaffold anchor (CORRECT)
# ============================================================
def build_scaffold_anchor_library(scaffold_set):

    print("\nBuilding scaffold anchor library...")

    scaffold_anchor = {}

    for smi in tqdm(scaffold_set):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        stars = find_star_atoms(mol)
        scaffold_anchor[smi] = [
            infer_scaffold_star_info(star)
            for star in stars
        ]

    with open(SCAFFOLD_ANCHOR, "w") as f:
        json.dump(scaffold_anchor, f, indent=2, ensure_ascii=False)

    print(f"Entries: {len(scaffold_anchor)}")


# ============================================================
# MAIN
# ============================================================
def main():
    build_sidechain_library()
    scaffold_set = build_supervised_scaffolds()
    build_scaffold_anchor_library(scaffold_set)
    print("\nDone!")


if __name__ == "__main__":
    main()
