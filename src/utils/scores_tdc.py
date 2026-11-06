from tdc import Oracle
from rdkit import Chem
from .scores_basic import mol2logp  # 引用 basic 中的函数

smiles2sa = Oracle(name="SA")
smiles2qed = Oracle(name="QED")
smiles2plogp = Oracle(name="LogP")  # 实际为 pLogP
smiles2gsk3b = Oracle(name="GSK3B")
smiles2jnk3b = Oracle(name="JNK3")
smiles2drd2 = Oracle(name="DRD2")

def smiles2uplogp(smiles: str | list[str]) -> float | list[float]:
    if isinstance(smiles, str):
        return _smiles2uplogp(smiles)
    return [_smiles2uplogp(s) for s in smiles]

def _smiles2uplogp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -100
    logp = mol2logp(mol)
    sa = smiles2sa(smiles)
    plogp = logp - sa
    plogp1 = plogp
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) > 6:
            plogp1 -= 1
    return plogp1
