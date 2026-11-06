import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import itertools
from collections import defaultdict, Counter
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdmolops
from multiprocessing import Pool, cpu_count
# ============================================================
# GLOBAL
# ============================================================
SIDE_BY_N = None
MAX_OUT = 200   # ✔ 加回 max_out（你要求的）


def init_worker(side_by_n):
    global SIDE_BY_N
    SIDE_BY_N = side_by_n


# ============================================================
# IO
# ============================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json_streaming(path):
    """
    JSON list streaming writer
    """
    f = open(path, "w")
    f.write("[\n")
    first = True

    def write(obj):
        nonlocal first
        if not first:
            f.write(",\n")
        json.dump(obj, f)
        first = False

    def close():
        f.write("\n]\n")
        f.close()

    return write, close


# ============================================================
# scaffold parsing
# ============================================================
def get_scaffold_star_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    stars = sorted(stars)

    dmat = rdmolops.GetDistanceMatrix(mol)

    dist = {}
    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            dist[(i, j)] = int(dmat[stars[i], stars[j]])

    return stars, dist


# ============================================================
# grouping
# ============================================================
def regroup_sidechains(combo_lib):
    by_n = defaultdict(list)
    for lst in combo_lib.values():
        for c in lst:
            by_n[len(c["dummy_atom_spec"])].append(c)
    return by_n


# ============================================================
# env constraint
# ============================================================
def env_ok(sc_info, dummy):
    return (
        sc_info.get("soft_aromatic", False) == dummy.get("aromatic", False)
        and sc_info.get("soft_in_ring", False) == dummy.get("in_ring", False)
    )


# ============================================================
# distance penalty (UNCHANGED)
# ============================================================
def distance_penalty(actual, expected, n):
    if expected is None:
        return 0
    if n <= 2:
        tol = 1
    else:
        tol = 2
    return 0 if abs(actual - expected) <= tol else 1


# ============================================================
# histogram
# ============================================================
def build_scaffold_hist(scaffold_dist):
    hist = Counter()
    for (_, _), d in scaffold_dist.items():
        hist[d] += 1
    return hist


def histogram_ok(scaffold_hist, combo_hist):
    for k, v in combo_hist.items():
        if scaffold_hist.get(k, 0) < v:
            return False
    return True


# ============================================================
# CSP solver (UNCHANGED LOGIC)
# ============================================================
def solve_csp_soft(group, combo, scaffold_infos, scaffold_dist):

    dummy = combo["dummy_atom_spec"]
    n = len(group)

    candidates = []
    for i in range(n):
        ci = []
        sc = scaffold_infos[group[i]]
        for j in range(n):
            if env_ok(sc, dummy[j]):
                ci.append(j)
        if not ci:
            return None
        candidates.append(ci)

    order = sorted(range(n), key=lambda x: len(candidates[x]))

    used = [False] * n
    perm = [-1] * n

    best_perm = None
    best_score = float("inf")

    def dfs(depth, score):
        nonlocal best_perm, best_score

        if score >= best_score:
            return

        if depth == n:
            if score < best_score:
                best_score = score
                best_perm = perm.copy()
            return

        i = order[depth]

        for di in candidates[i]:
            if used[di]:
                continue

            used[di] = True
            perm[i] = di
            new_score = score

            for j in range(depth):
                dj = perm[order[j]]

                a, b = sorted((group[i], group[order[j]]))
                actual = scaffold_dist.get((a, b))

                expected = combo["dummy_pair_distances"].get(f"{di},{dj}")
                if expected is None:
                    expected = combo["dummy_pair_distances"].get(f"{dj},{di}")

                new_score += distance_penalty(actual, expected, n)

            if new_score < best_score:
                dfs(depth + 1, new_score)

            used[di] = False
            perm[i] = -1

    dfs(0, 0)
    return best_perm


# ============================================================
# ENUMERATION
# ============================================================
def enumerate_assignments(scaffold_smiles, scaffold_infos):
    side_by_n = SIDE_BY_N

    stars, dist = get_scaffold_star_info(scaffold_smiles)
    if not stars:
        return []

    scaffold_hist = build_scaffold_hist(dist)

    results = []
    visited = set()

    def dfs(remaining, cur):

        if len(results) >= MAX_OUT:
            return

        if not remaining:
            key = tuple(sorted(
                (c["sidechain_smiles"], tuple(c["perm"]))
                for c in cur
            ))
            if key not in visited:
                visited.add(key)
                results.append({
                    "scaffold_smiles": scaffold_smiles,
                    "connections": cur.copy()
                })
            return

        anchor = remaining[0]
        rest = remaining[1:]

        for n, combos in side_by_n.items():
            if n > len(remaining):
                continue

            for others in itertools.combinations(rest, n - 1):
                group = (anchor,) + others

                for combo in combos:

                    combo_hist = combo.get("distance_hist", None)
                    if combo_hist is not None:
                        if not histogram_ok(scaffold_hist, combo_hist):
                            continue

                    perm = solve_csp_soft(group, combo, scaffold_infos, dist)

                    if perm is None:
                        continue

                    new_remaining = tuple(x for x in remaining if x not in group)

                    cur.append({
                        "star_ids": list(group),
                        "sidechain_smiles": combo["sidechain"],
                        "dummy_atom_spec": combo["dummy_atom_spec"],
                        "perm": perm,
                        "source_mol": combo.get("source_mol")
                    })

                    dfs(new_remaining, cur)
                    cur.pop()

    dfs(tuple(range(len(stars))), [])
    return results


# ============================================================
# WORKER
# ============================================================
def worker(args):
    smi, infos = args

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    if len([a for a in mol.GetAtoms() if a.GetSymbol() == "*"]) > 10:
        return []

    return enumerate_assignments(smi, infos)


# ============================================================
# MAIN (FIXED MEMORY + PARALLEL + STREAM OUTPUT)
# ============================================================
def main():

    scaffold_lib = load_json("result/sup/anchor_library/scaffold_anchor_library.json")
    combo_lib = load_json("result/sup/anchor_library/anchor_library_combo.json")

    side_by_n = regroup_sidechains(combo_lib)
    items = list(scaffold_lib.items())
    n_proc = max(1, cpu_count() - 1)

    write_json, close_json = save_json_streaming(
        "result/intermediate_622/test/intermediate_assignments_soft.json"
    )

    total = 0

    with Pool(processes=n_proc, initializer=init_worker, initargs=(side_by_n,)) as pool:

        for i, res in enumerate(pool.imap_unordered(worker, items, chunksize=5), 1):
            for r in res:
                write_json(r)   # ✔ 直接写文件，不进内存
                total += 1
            if i % 50 == 0:
                print(f"{i}/{len(items)} processed | total={total}")

    close_json()
    print("DONE:", total)


if __name__ == "__main__":
    main()
