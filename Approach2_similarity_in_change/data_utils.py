import pandas as pd

def extract_samples_of_cell_cluster(df_data: pd.DataFrame, clusters: pd.Series, cluster_id: str) -> pd.DataFrame:
    mask_cell_cluster = clusters.values == cluster_id
    samples_cell_cluster = df_data.values[mask_cell_cluster, :]
    df_stem_cells = pd.DataFrame(samples_cell_cluster,
                                        index=df_data.index[mask_cell_cluster],
                                        columns=df_data.columns)
    return df_stem_cells
    
def extract_random_samples():
    pass # TODO
