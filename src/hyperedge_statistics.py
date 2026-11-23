import math
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans


def _edge_rng(seed: int, edge_id: int) -> np.random.Generator:
    rng_seed = (int(seed) + int(edge_id) * 1000003) % (2**32)
    return np.random.default_rng(rng_seed)


def _sample_nodes(
    nodes: np.ndarray,
    edge_id: int,
    seed: int,
    max_nodes_per_edge: int,
) -> np.ndarray:
    if len(nodes) <= max_nodes_per_edge:
        return nodes
    rng = _edge_rng(seed, edge_id)
    return rng.choice(nodes, size=max_nodes_per_edge, replace=False)


def _pairwise_cosine_stats(
    z_sample: torch.Tensor,
    edge_id: int,
    seed: int,
    max_pairs_per_edge: int,
    eps: float,
) -> tuple:
    sample_size = z_sample.shape[0]
    if sample_size < 2:
        return 0.0, 0.0, 0

    z_norm = z_sample / (z_sample.norm(dim=1, keepdim=True) + eps)
    pair_i, pair_j = torch.triu_indices(sample_size, sample_size, offset=1)
    num_pairs = pair_i.numel()

    if num_pairs > max_pairs_per_edge:
        rng = _edge_rng(seed + 17, edge_id)
        sampled_pairs = rng.choice(num_pairs, size=max_pairs_per_edge, replace=False)
        sampled_pairs = torch.as_tensor(sampled_pairs, dtype=torch.long)
        pair_i = pair_i[sampled_pairs]
        pair_j = pair_j[sampled_pairs]

    sims = (z_norm[pair_i] * z_norm[pair_j]).sum(dim=1)
    return (
        float(sims.mean().item()),
        float(sims.var(unbiased=False).item()),
        int(pair_i.numel()),
    )


def _scatter(values: torch.Tensor) -> float:
    if values.shape[0] == 0:
        return 0.0
    center = values.mean(dim=0, keepdim=True)
    return float(((values - center) ** 2).sum(dim=1).mean().item())


def _l2_normalize(values: torch.Tensor, eps: float) -> torch.Tensor:
    return values / (values.norm(dim=1, keepdim=True) + eps)


def _subcluster_stats(
    z_sample: torch.Tensor,
    edge_size: int,
    edge_id: int,
    seed: int,
    subcluster_k: int,
    eps: float,
) -> tuple:
    if edge_size < 4 or z_sample.shape[0] < 4 or subcluster_k < 2:
        return 0.0, 1.0

    n_clusters = min(subcluster_k, z_sample.shape[0])
    z_np = z_sample.numpy()

    if np.unique(z_np, axis=0).shape[0] < n_clusters:
        return 0.0, 1.0

    kmeans = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=int(seed) + int(edge_id)
    )
    labels = kmeans.fit_predict(z_np)
    centers = torch.as_tensor(kmeans.cluster_centers_, dtype=z_sample.dtype)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    cluster_sizes = torch.bincount(labels_t, minlength=n_clusters).float()
    if torch.any(cluster_sizes == 0):
        return 0.0, 1.0

    if n_clusters == 2:
        between = float(((centers[0] - centers[1]) ** 2).sum().item())
    else:
        center_weights = cluster_sizes.unsqueeze(1)
        global_center = (centers * center_weights).sum(dim=0) / center_weights.sum()
        between = float(
            (((centers - global_center) ** 2).sum(dim=1) * cluster_sizes).sum().item()
        )
        between /= float(cluster_sizes.sum().item())

    assigned_centers = centers[labels_t]
    within = float(((z_sample - assigned_centers) ** 2).sum(dim=1).mean().item())
    major_ratio = float(cluster_sizes.max().item() / cluster_sizes.sum().item())
    return between / (within + eps), major_ratio


def compute_hyperedge_stats(
    Z: torch.Tensor,
    X: Optional[torch.Tensor],
    hyperedge_members: List[List[int]],
    num_nodes: int,
    seed: int = 0,
    max_nodes_per_edge: int = 256,
    max_pairs_per_edge: int = 4096,
    subcluster_k: int = 2,
    eps: float = 1e-8,
    dataset: str = "",
) -> List[Dict[str, float]]:
    if max_nodes_per_edge < 1:
        raise ValueError("max_nodes_per_edge must be >= 1")
    if max_pairs_per_edge < 1:
        raise ValueError("max_pairs_per_edge must be >= 1")

    Z_cpu = Z.detach().float().cpu()
    if Z_cpu.shape[0] != num_nodes:
        raise ValueError(f"Z has {Z_cpu.shape[0]} nodes, expected {num_nodes}")

    X_cpu = None
    if X is not None:
        X_cpu = X.detach().float().cpu()
        if X_cpu.shape[0] != num_nodes:
            raise ValueError(f"X has {X_cpu.shape[0]} nodes, expected {num_nodes}")

    node_degrees = np.zeros(num_nodes, dtype=np.float64)
    for members in hyperedge_members:
        for node_id in members:
            node_degrees[int(node_id)] += 1.0

    rows = []
    for edge_id, members in enumerate(hyperedge_members):
        nodes = np.asarray(members, dtype=np.int64)
        edge_size = int(nodes.shape[0])
        log_size = math.log1p(edge_size)

        is_sampled = int(edge_size > max_nodes_per_edge)
        if edge_size == 0:
            degree_values = np.asarray([0.0], dtype=np.float64)
            z_sample = Z_cpu.new_zeros((0, Z_cpu.shape[1]))
            sampled_nodes = nodes
            embedding_norm_mean = 0.0
            embedding_norm_std = 0.0
        else:
            degree_values = node_degrees[nodes]
            sampled_nodes = _sample_nodes(nodes, edge_id, seed, max_nodes_per_edge)
            z_sample = Z_cpu[torch.as_tensor(sampled_nodes, dtype=torch.long)]
            z_full = Z_cpu[torch.as_tensor(nodes, dtype=torch.long)]
            embedding_norms = z_full.norm(dim=1)
            embedding_norm_mean = float(embedding_norms.mean().item())
            embedding_norm_std = float(embedding_norms.std(unbiased=False).item())

        num_nodes_used = int(z_sample.shape[0])
        mean_sim, var_sim, num_pairs_used = _pairwise_cosine_stats(
            z_sample, edge_id, seed, max_pairs_per_edge, eps
        )
        embedding_scatter = _scatter(z_sample)
        subcluster_score, major_ratio = _subcluster_stats(
            z_sample, edge_size, edge_id, seed, subcluster_k, eps
        )
        z_sample_unit = (
            _l2_normalize(z_sample, eps) if z_sample.shape[0] > 0 else z_sample
        )
        scatter_norm = _scatter(z_sample_unit)
        subcluster_score_norm, _ = _subcluster_stats(
            z_sample_unit, edge_size, edge_id, seed, subcluster_k, eps
        )

        if X_cpu is None or edge_size == 0:
            feature_var = 0.0
        else:
            x_edge = X_cpu[torch.as_tensor(nodes, dtype=torch.long)]
            feature_var = _scatter(x_edge)

        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "edge_id": edge_id,
                "size": edge_size,
                "log_size": log_size,
                "is_sampled": is_sampled,
                "num_nodes_used": num_nodes_used,
                "num_pairs_used": num_pairs_used,
                "mean_sim": mean_sim,
                "var_sim": var_sim,
                "scatter": embedding_scatter,
                "scatter_norm": scatter_norm,
                "subcluster_score": subcluster_score,
                "subcluster_score_norm": subcluster_score_norm,
                "major_ratio": major_ratio,
                "degree_mean": float(np.mean(degree_values)),
                "degree_std": float(np.std(degree_values)),
                "degree_max": float(np.max(degree_values)),
                "feature_var": feature_var,
                "embedding_norm_mean": embedding_norm_mean,
                "embedding_norm_std": embedding_norm_std,
            }
        )

    return rows
