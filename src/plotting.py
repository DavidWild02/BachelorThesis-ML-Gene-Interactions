import scanpy as sc
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_cell_activity_score_along_path(adata, genes: List[str] | pd.Index):
    path_differentiation = adata.uns["path_differentiation"]
    obs_mask = adata.obs["clusters"].isin(path_differentiation)
    sc.pl.umap(adata, layer="activity_score", mask_obs=obs_mask, color=genes, cmap="viridis", vmin=0, vmax=1)


def display_gene_expression_distribution(adata, gene, clusters: List[str] | None = None, ax=None): 
    if ax is None:
        # create ax if not passed by caller
        fig, ax = plt.subplots()
    
    df_gene = adata.to_df()[gene]

    if clusters is None:
        sns.kdeplot(df_gene, ax=ax)
    else:
        cluster_indices = adata.obs["cluster"].values
        cluster_labels = adata.obs["cluster"].cat.categories
        colors = adata.uns["cluster_colors"]
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in clusters:
                continue
            sns.kdeplot(df_gene[cluster_indices == cluster_label], color=colors[i], label=f"Cluster {cluster_label}", ax=ax)
        ax.legend()
    ax.set_title(f"Gene expression distribution {gene}")
    ax.set_xlabel("Expression")
    ax.set_ylabel("Density")


