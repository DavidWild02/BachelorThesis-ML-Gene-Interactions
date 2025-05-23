import scanpy as sc
from typing import List
import pandas as pd


def plot_cell_activity_score_along_path(adata, genes: List[str] | pd.Index):
    path_differentiation = adata.uns["path_differentiation"]
    obs_mask = adata.obs["clusters"].isin(path_differentiation)
    sc.pl.umap(adata, layer="activity_score", mask_obs=obs_mask, color=genes, cmap="viridis", vmin=0, vmax=1)