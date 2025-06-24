import numpy as np
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import networkx as nx
import torch

from typing import List, Literal

from data_utils import extract_samples_of_cell_cluster

eta = 0.1e-10 # Used for avoiding divisions through zero


def mmc(x, y):
    """
    Calculates the maximum mean change.
    """
    return (y - x)/(np.maximum(x, y) + eta)

def calculate_relation_mean_change_matrix(cluster_a, cluster_b):
    mean_cluster_a = np.mean(cluster_a, axis=0)
    mean_cluster_b = np.mean(cluster_b, axis=0)
    mean_diff = mean_cluster_a - mean_cluster_b
    mean_diff[mean_diff == 0] = eta
    relation_mean_diff = np.abs(mean_diff) / np.abs(mean_diff).T
    correlation_direction = np.sign(mean_diff * mean_diff.T)
    return correlation_direction * relation_mean_diff

def score_similarity_relative_change(x, y):
    correlation_direction = np.sign(x * y)
    change_absolute_value = mmc(np.abs(x), np.abs(y))
    similarity_magnitude = 1 - np.abs(change_absolute_value)
    return correlation_direction * similarity_magnitude

def calculate_mean_change_similarity_matrix(cluster_a, cluster_b):
    mean_cluster_a = np.mean(cluster_a, axis=0)
    mean_cluster_b = np.mean(cluster_b, axis=0)
    relative_changes = mmc(mean_cluster_a, mean_cluster_b)
    n = relative_changes.shape[0]
    x = np.tile(relative_changes, (n, 1))
    similarity_matrix = score_similarity_relative_change(x, x.T)
    return similarity_matrix


def mean_kernel_matrix(x, y, sigma = 1., device=None):
    """
        Uses the radial basis function kernel
    """
    n, d = x.shape
    m, _ = y.shape

    denominator = 2 * sigma ** 2

    # cannot use vectorization here, because it needs to much space (97 GB)
    summed_kernels = torch.zeros((d, d), device=device)
    for i in tqdm(range(n), desc="Calculating mean kernel matrix..."):
        x_i = x[i, :, None]
        for j in range(m):
            y_j = y[j, None, :]
            distance_matrix = (x_i - y_j) ** 2
            kernel_matrix = torch.exp(-distance_matrix / denominator)
            summed_kernels += kernel_matrix
    return summed_kernels / (n * m)


def calculate_mmdd_similarity_matrix(cluster_a, cluster_b, sigma = 1.):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(cluster_a, pd.DataFrame):
        cluster_a = cluster_a.values
    if isinstance(cluster_b, pd.DataFrame):
        cluster_b = cluster_b.values

    if isinstance(cluster_a, np.ndarray):
        cluster_a = torch.from_numpy(cluster_a).float()
    if isinstance(cluster_b, np.ndarray):
        cluster_b = torch.from_numpy(cluster_b).float()
    cluster_a = cluster_a.to(device)
    cluster_b = cluster_b.to(device)

    k_aa = mean_kernel_matrix(cluster_a, cluster_a, sigma=sigma, device=device)
    k_bb = mean_kernel_matrix(cluster_b, cluster_b, sigma=sigma, device=device)
    k_ab = mean_kernel_matrix(cluster_a, cluster_b, sigma=sigma, device=device)
    k_ba = k_ab.T

    k_diag = torch.diag(k_bb) - 2 * torch.diag(k_ba) + torch.diag(k_aa)

    squared_mmdd_matrix = (
        k_diag[:, None] + k_diag[None, :]
        - 2. * k_bb - 2. * k_aa
        + 2. * k_ab + 2. * k_ba
    )

    UPPER_BOUND_KERNEL = 1 # For RBF kernel the upper bound is 1
    UPPER_BOUND_MMDD = 8 * UPPER_BOUND_KERNEL # used as normalization factor, so that result is between -1 and +1
    mmdd_matrix = torch.sqrt(squared_mmdd_matrix / UPPER_BOUND_MMDD )

    diff_means = torch.mean(cluster_b, dim=0) - torch.mean(cluster_a, dim=0)
    correlation_directions = torch.sign(diff_means[:, None] * diff_means[None, :])
    correlation_directions[correlation_directions == 0] = 1. # Needed because squared_mmdd_matrix is 0, if both distrubtions are the same. For 0, torch.sign returns 0.

    similarity_matrix = correlation_directions * (1. - mmdd_matrix)

    return similarity_matrix.cpu().numpy()

def activation_score_of_cells(target_expression_levels, grn):
    _num_obs, num_target_genes = target_expression_levels.shape
    _num_driver_genes, num_target_genes = grn.shape

    # normalize the rows by the target gene expression level
    # (num_obs, num_driver_genes, num_target_genes)
    normalized_weights = grn[np.newaxis, :, :] / target_expression_levels[:, np.newaxis, :]
    normalized_weights[normalized_weights == 0] = 1e-10 # to avoid divisions by zero
    normalized_weights = np.abs(normalized_weights) # only consider the absolute value
    # sort the rows
    sorted_weights =np.sort(normalized_weights, axis=2)
    area_under_curve = np.sum(np.cumsum(sorted_weights, axis=2) / num_target_genes, axis=2)
    total_weight = np.sum(sorted_weights, axis=2)
    return area_under_curve / total_weight

def activation_score_of_group(mean_target_expression_levels, grn):
    _num_driver_genes, num_target_genes = grn.shape
    assert (mean_target_expression_levels.size == num_target_genes)

    # normalize the rows by the target gene expression level
    # (num_driver_genes, num_target_genes)
    normalized_weights = grn[:, :] / mean_target_expression_levels[np.newaxis, :]
    normalized_weights[normalized_weights == 0] = 1e-10 # to avoid divisions by zero
    normalized_weights = np.abs(normalized_weights) # only consider the absolute value
    # sort the rows
    sorted_weights = np.sort(normalized_weights, axis=1)
    area_under_curve = np.sum(np.cumsum(sorted_weights, axis=1) / num_target_genes, axis=1)
    total_weight = np.sum(sorted_weights, axis=1)
    return area_under_curve / total_weight


def create_transition_grn_graph(df_transitions: pd.DataFrame, df_data: pd.DataFrame, clusters: pd.Series, 
                                calculate_similarity_func, max_samples_per_cluster: int | None = 100, **kwargs):
    num_cells, _num_genes = df_data.shape
    assert (clusters.size == num_cells)

    graph = nx.DiGraph()

    for node in df_transitions.index:
        node_cluster = extract_samples_of_cell_cluster(df_data, clusters, node)
        graph.add_node(node, attr=node_cluster)

        for neighbor in df_transitions.columns:
            if df_transitions.loc[node, neighbor] == 0:
                continue

            neighbor_cluster = extract_samples_of_cell_cluster(df_data, clusters, neighbor)

            if max_samples_per_cluster:
                pass # TODO: randomly pick max num samples from cluster
                # Needed if similarity function takes very long

            similarity_matrix = calculate_similarity_func(node_cluster, neighbor_cluster, **kwargs)
            graph.add_edge(node, neighbor, attr=similarity_matrix)            

    return graph


def calculate_cell_activity_scores_along_path(adata, graph: nx.DiGraph, path_differentiation: List[str] | pd.Index, 
                                              strategy_target_expr_levels: Literal["from_group_a", "mean_shift"] = "from_group_a"):
    cell_activity_scores = pd.DataFrame(np.zeros(adata.X.shape), index=adata.obs_names, columns=adata.var_names)

    iter_path = iter(path_differentiation)
    current_node = next(iter_path)
    for next_node in iter_path:
        grn = graph.edges[current_node, next_node]["attr"]

        group_a = graph.nodes[current_node]["attr"]
        group_b = graph.nodes[next_node]["attr"]

        if strategy_target_expr_levels == "mean_shift":
            mean_shift =  group_b.mean() - group_a.mean()
            target_expression_levels = group_a + mean_shift 
        else:
            target_expression_levels = group_a

        activation_scores = activation_score_of_cells(target_expression_levels.values, grn)
        cell_activity_scores.loc[group_a.index, group_a.columns] = activation_scores

        current_node = next_node

    adata.layers["activity_score"] = cell_activity_scores
    adata.uns["path_differentiation"] = path_differentiation




