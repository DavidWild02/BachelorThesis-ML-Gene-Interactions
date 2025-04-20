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



def display_gene_residual_plots(y, y_pred, gene_indices, df_grn):
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values

    mask = df_grn.columns.isin(gene_indices)
    y = y[:, mask]
    y_pred = y_pred[:, mask]

    n_genes = len(gene_indices)
    fig, axes = plt.subplots(nrows=n_genes, ncols=2, figsize=(12, 6*n_genes))

    for i, gene_name in enumerate(gene_indices):
        ax = axes[i]
        ax[0].set_title(gene_name)
        ax[1].set_title(gene_name)
        PredictionErrorDisplay.from_predictions(y[:, i], y_pred[:, i], ax=ax[0], kind="actual_vs_predicted")
        PredictionErrorDisplay.from_predictions(y[:, i], y_pred[:, i], ax=ax[1], kind="residual_vs_predicted")

    plt.show()
