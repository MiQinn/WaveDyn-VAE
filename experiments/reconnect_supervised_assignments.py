import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm
from rdkit import Chem


ASSIGNMENTS = "result/intermediate_622/test/intermediate_assignments_soft.json"
OUT_JSON = "result/final/test/final_reconnected.json"
OUT_SMI = "result/final/test/final_reconnected.smi"


# =========================
# utils
# =========================
def load(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("bad smiles")
    return mol


def stars(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]


def is_connected(mol):
    return len(Chem.GetMolFrags(mol)) == 1


# =========================
# reconstruction
# =========================
def reconnect(assignment):

    scaffold = load(assignment["scaffold_smiles"])
    sc_stars = stars(scaffold)

    combined = Chem.RWMol(scaffold)

    offsets = []

    # attach sidechains
    for conn in assignment["connections"]:
        side = load(conn["sidechain_smiles"])

        offset = combined.GetNumAtoms()
        offsets.append(offset)

        combined = Chem.RWMol(Chem.CombineMols(combined, side))

    sc_map = {i: sc_stars[i] for i in range(len(sc_stars))}

    # connect bonds
    for idx, conn in enumerate(assignment["connections"]):

        perm = conn.get("perm", [0])

        side = load(conn["sidechain_smiles"])
        side_stars = stars(side)

        offset = offsets[idx]

        for i, sc_pos in enumerate(conn["star_ids"]):

            sc_atom = sc_map[sc_pos]
            sd_atom = offset + side_stars[perm[i]]

            sc_nb = [
                n.GetIdx()
                for n in combined.GetAtomWithIdx(sc_atom).GetNeighbors()
                if n.GetSymbol() != "*"
            ][0]

            sd_nb = [
                n.GetIdx()
                for n in combined.GetAtomWithIdx(sd_atom).GetNeighbors()
                if n.GetSymbol() != "*"
            ][0]

            combined.AddBond(sc_nb, sd_nb, Chem.rdchem.BondType.SINGLE)

    # remove stars
    star_atoms = [a.GetIdx() for a in combined.GetAtoms() if a.GetSymbol() == "*"]
    for i in sorted(star_atoms, reverse=True):
        combined.RemoveAtom(i)

    mol = combined.GetMol()

    Chem.SanitizeMol(mol)

    # 过滤 disconnected molecules
    if not is_connected(mol):
        return None

    return Chem.MolToSmiles(mol)


# =========================
# main
# =========================
def main():

    data = json.load(open(ASSIGNMENTS))

    results = []
    ok, fail = 0, 0

    for i, a in enumerate(tqdm(data)):

        try:
            smi = reconnect(a)

            if smi is None:
                fail += 1
                continue

            results.append({
                "idx": i,
                "scaffold": a["scaffold_smiles"],
                "connections": [
                    {
                        "star_ids": c["star_ids"],
                        "sidechain_smiles": c["sidechain_smiles"]
                    }
                    for c in a["connections"]
                ],
                "final_smiles": smi
            })

            ok += 1

        except Exception:
            fail += 1

    json.dump(results, open(OUT_JSON, "w"), indent=2)

    with open(OUT_SMI, "w") as f:
        for r in results:
            f.write(r["final_smiles"] + "\n")

    print("OK:", ok, "FAIL:", fail)


if __name__ == "__main__":
    main()
