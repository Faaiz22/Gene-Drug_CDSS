
"""Produce and save UMAP/PCA visualizations for fusion embeddings."""
from src.utils.visualization import plot_umap_2d
import numpy as np, os

def save_umap(fusion_emb, outpath='artifacts/association_clusters_umap.png'):
    coords = plot_umap_2d(fusion_emb, outpath=outpath)
    return coords
