import scanpy as sc
from typing import List


def plot_cell_activity_score_along_path(adata, path_differentiation: List[str], genes: List[str]):
    obs_mask = adata.obs["clusters"].isin(path_differentiation)
    sc.pl.umap(adata, layer="activity_score", mask_obs=obs_mask, color=genes, cmap="viridis", vmin=0, vmax=1)