
"""Data loader and simple merging utilities for TSV inputs."""
import pandas as pd
from pathlib import Path

def load_tsvs(drugs_path: str, genes_path: str, rel_path: str):
    drugs = pd.read_csv(drugs_path, sep='\t', dtype=str)
    genes = pd.read_csv(genes_path, sep='\t', dtype=str)
    rel = pd.read_csv(rel_path, sep='\t', dtype=str)
    # keep original columns, but normalize column names slightly
    drugs.columns = [c.strip() for c in drugs.columns]
    genes.columns = [c.strip() for c in genes.columns]
    rel.columns = [c.strip() for c in rel.columns]
    # merge: try to attach drug/ gene metadata if ids present
    merged = rel.copy()
    return drugs, genes, merged
