
"""Feature constructors for drugs and proteins.

- Drug: expects RDKit-calculated 3D descriptors (external)
- Protein: amino-acid composition (AAC) + simple physchem features
"""
import numpy as np
from collections import Counter

AA = list('ACDEFGHIKLMNPQRSTVWY')

def aa_composition(seq: str, max_len:int=1000):
    seq = (seq or "")[:max_len]
    counts = Counter(seq)
    vec = np.array([counts.get(a,0)/max(1,len(seq)) for a in AA], dtype=float)
    return vec

def protein_physchem_features(seq: str):
    # simple: length, aromatic fraction (placeholder), charge proxy
    seq = (seq or "")
    length = len(seq)
    aromatic = sum(1 for c in seq if c in 'FWYH') / max(1,length)
    charge = (seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')) / max(1,length)
    return np.array([length, aromatic, charge], dtype=float)

def build_protein_embedding(seq: str):
    aac = aa_composition(seq)
    phys = protein_physchem_features(seq)
    return np.concatenate([aac, phys])
