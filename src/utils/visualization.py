
"""Visualization helpers to create and save UMAP/PCA plots and cluster figures."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

def plot_umap_2d(fusion_embeddings, labels=None, outpath=None, sample=5000):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(fusion_embeddings)
    except Exception:
        coords = PCA(n_components=2).fit_transform(fusion_embeddings)
    x,y = coords[:,0], coords[:,1]
    plt.figure(figsize=(6,5))
    if labels is None:
        plt.scatter(x[:sample], y[:sample], s=4, alpha=0.6)
    else:
        sns.scatterplot(x=x[:sample], y=y[:sample], hue=labels[:sample], s=10, palette='viridis', legend=None)
    plt.title('Fusion embedding 2D projection')
    if outpath:
        plt.savefig(outpath, dpi=300)
    plt.close()
    return coords
