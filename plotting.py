from sklearn.metrics import PredictionErrorDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

def display_gene_expression_distribution(data: pd.DataFrame, selected_genes):
    n_genes = len(selected_genes)
    _, axes = plt.subplots(nrows=n_genes, figsize=(8, 6*n_genes)) 

    for i, gene_name in enumerate(selected_genes):   
        axes[i].set_title(f"Gene expression distribution {gene_name}")
        sns.kdeplot(data.loc[:, gene_name], ax=axes[i])
