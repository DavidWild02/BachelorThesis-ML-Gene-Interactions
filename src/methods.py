import numpy as np
from tqdm import tqdm
import torch

def mmc(x, y):
    """
    Calculates the maximum mean change.
    """
    eta = 0.1e-10 # Used for avoiding divisions through zero
    return (y - x)/(np.maximum(x, y) + eta)

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


def activation_score(target_expression_levels, grn):
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
