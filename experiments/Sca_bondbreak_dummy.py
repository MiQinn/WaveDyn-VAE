import numpy as np
import scaffoldgraph as sg
import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdmolops import GetSymmSSSR
import matplotlib.pyplot as plt
import random
import argparse
import csv
import json
import re

# =============== ONLY normalize dummy labels ===============
def normalize_dummy(smiles):
    return re.sub(r'\[\d+\*\]', '*', smiles)

# =============== scaffold filter ===============
def filter_sca(sca):
    mol = Chem.MolFromSmiles(sca)
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    benzene = Chem.MolFromSmiles("c1ccccc1")

    if ri.NumRings() == 1 and mol.HasSubstructMatch(benzene):
        return None
    elif mol.GetNumHeavyAtoms() > 20:
        return None
    elif Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) > 3:
        return None

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            return None

    return True

# =============== consistency check ===============
def mol_if_equal_sca(mol_smi, sca_smi):
    mol = Chem.MolFromSmiles(mol_smi)
    sca = Chem.MolFromSmiles(sca_smi)

    if mol is None or sca is None:
        return False

    match = mol.GetSubstructMatch(sca)
    if not match:
        return False

    mask = np.zeros(mol.GetNumAtoms())
    for i in match:
        mask[i] = 1

    return not (mask.all() or (mask == 0).all())

# =============== atom info ===============
def get_atom_info(atom, bond_type):
    return {
        "element": atom.GetSymbol(),
        "aromatic": atom.GetIsAromatic(),
        "hybridization": str(atom.GetHybridization()),
        "in_ring": atom.IsInRing(),
        "degree": atom.GetDegree(),
        "neighbors": [n.GetSymbol() for n in atom.GetNeighbors()],
        "bond_type": str(bond_type)
    }

# =============== core extraction ===============
def extract_with_bond_cut(mol_smi, sca_smi):
    mol = Chem.MolFromSmiles(mol_smi)
    sca = Chem.MolFromSmiles(sca_smi)

    if mol is None or sca is None:
        return None

    for atom in mol.GetAtoms():
        atom.SetIntProp("_OrigIdx", atom.GetIdx())

    match = mol.GetSubstructMatch(sca)
    if not match:
        return None

    scaffold_atoms = set(match)

    cut_bonds = []
    attachment_info = []

    for mol_idx in match:
        atom = mol.GetAtomWithIdx(mol_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in scaffold_atoms:
                bond = mol.GetBondBetweenAtoms(mol_idx, nbr_idx)
                if bond is not None:
                    cut_bonds.append((bond.GetIdx(), mol_idx, nbr_idx))
                    attachment_info.append({
                        "side_atom_idx": nbr_idx,
                        "info": get_atom_info(nbr, bond.GetBondType())
                    })

    if len(cut_bonds) == 0:
        return None

    bond_indices = [b[0] for b in cut_bonds]
    frag_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    clean_frags = []
    for frag in frags:
        try:
            Chem.Kekulize(frag, clearAromaticFlags=True)
        except:
            pass
        try:
            Chem.SanitizeMol(frag)
            clean_frags.append(frag)
        except:
            clean_frags.append(frag)

    frags = clean_frags

    scaffold_frag = None
    sidechain_frags = []

    for frag in frags:
        orig_atoms = set()
        for a in frag.GetAtoms():
            if a.GetAtomicNum() != 0 and a.HasProp("_OrigIdx"):
                orig_atoms.add(a.GetIntProp("_OrigIdx"))

        if len(orig_atoms & scaffold_atoms) > 0:
            scaffold_frag = frag
        else:
            sidechain_frags.append(frag)

    if scaffold_frag is None:
        return None

    scaffold_smi = normalize_dummy(Chem.MolToSmiles(scaffold_frag))

    sidechains = []

    for frag in sidechain_frags:
        smi = normalize_dummy(Chem.MolToSmiles(frag))

        attach = None
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:
                nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 0]
                if nbrs:
                    orig_idx = nbrs[0].GetIntProp("_OrigIdx")
                    for info in attachment_info:
                        if info["side_atom_idx"] == orig_idx:
                            attach = info["info"]
                            break
                break

        sidechains.append({
            "smiles": smi,
            "attachment_info": attach
        })

    return {
        "scaffold": scaffold_smi,
        "sidechains": sidechains
    }

# =============== scaffold screening ===============
ALLOWED_ELEMENTS = {"C","N","O","F","Cl","S","P","Br","I","B","H","*"}
MAX_ATOMS = 20
MAX_RING_SIZE = 8

def screen_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, ["Invalid SMILES"]

    if not all(a.GetSymbol() in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
        return False, ["Disallowed elements"]

    if mol.GetNumHeavyAtoms() > MAX_ATOMS:
        return False, [f"HeavyAtoms>{MAX_ATOMS}"]

    rings = list(GetSymmSSSR(mol))
    if rings and max(len(r) for r in rings) > MAX_RING_SIZE:
        return False, [f"RingSize>{MAX_RING_SIZE}"]

    return True, []

# =============== main ===============
def main(args):
    network = sg.ScaffoldNetwork.from_smiles_file(args.data)
    molecules = list(network.get_molecule_nodes())

    with open(args.data) as f:
        original_smiles_list = [line.strip() for line in f]

    # ===== 作图 =====
    counts = network.get_hierarchy_sizes()
    x, y = zip(*sorted(counts.items()))
    plt.figure(figsize=(8,6))
    plt.bar(x,y)
    plt.savefig(args.save_png)
    plt.close()

    records = []
    fail_log = []

    # ===== 提取阶段 =====
    for idx, mol_id in enumerate(molecules):
        predecessors = list(nx.bfs_tree(network, mol_id, reverse=True))

        original = original_smiles_list[idx]
        smile = network.nodes[predecessors[0]]['smiles']

        sca_list = [p for p in predecessors[1:] if filter_sca(p)]

        if not sca_list:
            fail_log.append((original,"no_valid_scaffold"))
            continue

        sca = random.choice(sca_list)

        if not mol_if_equal_sca(smile, sca):
            fail_log.append((original,"sca_not_match"))
            continue

        result = extract_with_bond_cut(smile, sca)
        if result is None:
            fail_log.append((original,"extract_failed"))
            continue

        records.append({
            "original": original,
            "canonical": smile,
            "scaffold": result["scaffold"],
            "sidechains": result["sidechains"]
        })

    # ===== screening =====
    kept, failed = [], []
    screen_dict = {}

    for rec in records:
        smi = rec["scaffold"]
        ok, reasons = screen_scaffold(smi)
        if ok:
            kept.append(smi)
            screen_dict[smi] = {"kept": True, "reason": ""}
        else:
            failed.append((smi,";".join(reasons)))
            screen_dict[smi] = {"kept": False, "reason": ";".join(reasons)}

    # ===== 写文件 =====
    with open(args.save_pairs,"w",newline='') as f:
        w = csv.writer(f)
        w.writerow(['original_smiles','scaffold','sidechains','sca_kept'])
        for r in records:
            sc = screen_dict[r["scaffold"]]
            r["screening"] = sc
            w.writerow([r["original"],r["scaffold"],json.dumps(r["sidechains"]),sc["kept"]])

    with open(args.save_metadata,"w") as f:
        json.dump(records,f,indent=2)

    with open(args.save_sca_screened,"w") as f:
        for s in kept:
            f.write(s+"\n")

    with open(args.save_sca_failed_csv,"w",newline='') as f:
        w = csv.writer(f)
        w.writerow(["SMILES","Failed_Reasons"])
        for s,r in failed:
            w.writerow([s,r])

    with open(args.save_failed,"w") as f:
        for s,r in fail_log:
            f.write(f"{s}\t{r}\n")

    print("\n===== Pipeline Summary =====")
    print(f"Total molecules: {len(original_smiles_list)}")
    print(f"Success: {len(records)}")
    print(f"Failed: {len(fail_log)}")
    print(f"Scaffold kept: {len(kept)}")
    print(f"Scaffold removed: {len(failed)}")

# =============== args ===============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/training_set/training_set.smi')
    parser.add_argument('--save_pairs', default='./data/training_set/test/training_set_sca_pairs.csv')
    parser.add_argument('--save_metadata', default='./data/training_set/test/training_set_sca_metadata.json')
    parser.add_argument('--save_failed', default='./data/training_set/test/training_set_failed.smi')
    parser.add_argument('--save_png', default='./data/training_set/test/training_set_sca_hierarchy_distribution.png')
    parser.add_argument('--save_sca_screened', default='./data/training_set/test/training_set_sca_screened.smi')
    parser.add_argument('--save_sca_failed_csv', default='./data/training_set/test/training_set_sca_screen_failed.csv')
    args = parser.parse_args()
    main(args)
