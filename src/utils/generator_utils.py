
"""Simple fragment-based mutation utilities using RDKit/BRICS.

Functions:
 - brics_decompose(smiles)
 - safe_mutations(smiles, n=50)
"""
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

def brics_decompose(smi):
    try:
        return list(BRICS.BRICSDecompose(smi))
    except Exception:
        return []

def safe_mutations(smi, n=50):
    out = set()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    for _ in range(n):
        try:
            rw = Chem.RWMol(m)
            idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 6]
            if not idxs:
                continue
            i = idxs[0]
            rw.AddAtom(Chem.Atom(6))
            rw.AddBond(i, rw.GetNumAtoms()-1, Chem.BondType.SINGLE)
            nm = rw.GetMol()
            Chem.SanitizeMol(nm)
            out.add(Chem.MolToSmiles(nm, True))
        except Exception:
            continue
    return list(out)
