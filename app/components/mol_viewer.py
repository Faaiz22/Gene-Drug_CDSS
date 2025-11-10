
from rdkit import Chem
from rdkit.Chem import Draw
import io

def mol_to_png_bytes(smiles, size=(300,200)):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    img = Draw.MolToImage(m, size=size)
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    bio.seek(0)
    return bio.getvalue()
