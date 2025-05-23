import pandas as pd

def extract_samples_of_cell_cluster(adata, cluster_id: str) -> pd.DataFrame:
    obs_mask_cell_cluster = adata.obs["clusters"].values == cluster_id
    samples_cell_cluster = adata.X[obs_mask_cell_cluster, :]
    df_stem_cells = pd.DataFrame(samples_cell_cluster,
                                        index=adata.obs_names[obs_mask_cell_cluster],
                                        columns=adata.var_names)
    return df_stem_cells