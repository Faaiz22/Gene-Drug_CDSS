
"""Generate 3D conformers with RDKit ETKDG and embed multiple conformers safely."""
from rdkit import Chem
from rdkit.Chem import AllChem

def embed_3d(smiles, num_confs=1, max_attempts=5):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.maxAttempts = max_attempts
    try:
        cids = AllChem.EmbedMultipleConfs(m, num_confs, params)
        for cid in cids:
            AllChem.UFFOptimizeMolecule(m, confId=cid)
        return m
    except Exception:
        return None
