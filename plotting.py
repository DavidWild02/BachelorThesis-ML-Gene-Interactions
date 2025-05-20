from sklearn.metrics import PredictionErrorDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def show_score_distribution(df_gen_scores, column_name):
    df_gen_scores_lower_quantile = df_gen_scores[column_name].quantile(0.005)
    df_gen_scores_without_outliers = df_gen_scores[df_gen_scores[column_name] >= df_gen_scores_lower_quantile]

    plt.title(f"KDE plot of {column_name} across target genes")
    sns.kdeplot(df_gen_scores_without_outliers, fill=True)
    plt.show()



def display_gene_residual_plots(y: pd.DataFrame, y_pred: pd.DataFrame, selected_genes):
    y_array = y.loc[:, selected_genes].values
    y_pred_array = y_pred.loc[:, selected_genes].values

    n_genes = len(selected_genes)
    _, axes = plt.subplots(nrows=n_genes, ncols=2, figsize=(12, 6*n_genes))

    for i, gene_name in enumerate(selected_genes):
        ax = axes[i]
        ax[0].set_title(gene_name)
        ax[1].set_title(gene_name)
        PredictionErrorDisplay.from_predictions(y_array[:, i], y_pred_array[:, i], ax=ax[0], kind="actual_vs_predicted")
        PredictionErrorDisplay.from_predictions(y_array[:, i], y_pred_array[:, i], ax=ax[1], kind="residual_vs_predicted")

    plt.show()

def display_gene_expression_distribution(adata, gene, by_cluster=True):
    plt.title(f"Gene expression distribution {gene}")
    
    df_genes = adata.to_df()

    if not by_cluster:
        sns.kdeplot(df_genes.loc[:, gene])
    else:
        cluster_indices = adata.obs["clusters"].values
        colors = adata.uns["clusters_colors"]
        num_clusters = len(colors)
        for i in range(num_clusters):
            sns.kdeplot(df_genes.loc[cluster_indices == str(i), gene], color=colors[i], label=f"Cluster {i}")
        plt.legend()
    plt.show()
    
    
        
def calculate_cluster_statistics_for_gene(adata, gene: str):
    cluster_indices = adata.obs["clusters"].values
    clusters = adata.to_df().groupby(cluster_indices, as_index=True)
    cluster_statistics = clusters[gene].agg(["mean", "var", "count"])
    cluster_statistics.columns = ["mean", "variance", "count"]
    return cluster_statistics
    
