
"""Compute simple 3D descriptors used in the notebook (placeholders).

Descriptors:
 - radius of gyration (approx)
 - inertia tensor-based shape descriptors (placeholders)
"""
import numpy as np
from rdkit.Chem import rdMolTransforms

def radius_of_gyration(mol, confId=0):
    # approximate using atomic coordinates
    conf = mol.GetConformer(confId)
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coords = np.array([[c.x, c.y, c.z] for c in coords])
    centroid = coords.mean(axis=0)
    rog = np.sqrt(((coords - centroid)**2).sum(axis=1).mean())
    return float(rog)

def asphericity_placeholder(mol, confId=0):
    # placeholder numeric derived from coordinates variance
    conf = mol.GetConformer(confId)
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    vals = np.var(coords, axis=0)
    return float((vals.max() - vals.min()) / (vals.sum() + 1e-9))
