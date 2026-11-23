import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.hyperedge_statistics import (
    _l2_normalize,
    _pairwise_cosine_stats,
    _sample_nodes,
    _scatter,
    compute_hyperedge_stats,
)


MOE_FEATURE_COLUMNS = [
    "log_size",
    "mean_sim",
    "var_sim",
    "scatter_norm",
    "subcluster_score_norm",
    "major_ratio",
    "degree_mean",
    "degree_std",
    "degree_max",
    "feature_var",
]

MOE_DIAGNOSTIC_FEATURES = [
    "log_size",
    "mean_sim",
    "scatter_norm",
    "subcluster_score_norm",
]

MOE_STATIC_FEATURE_COLUMNS = [
    "log_size",
    "degree_mean",
    "degree_std",
    "degree_max",
    "feature_var",
]

MOE_DYNAMIC_FEATURE_COLUMNS = [
    "mean_sim",
    "var_sim",
    "scatter_norm",
]

MOE_SUBCLUSTER_FEATURE_COLUMNS = [
    "subcluster_score_norm",
    "major_ratio",
]


def _column_indices(columns: List[str]) -> List[int]:
    return [MOE_FEATURE_COLUMNS.index(column) for column in columns]


def compute_static_moe_feature_block(
    X: Optional[torch.Tensor],
    hyperedge_members: List[List[int]],
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
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
    for members in hyperedge_members:
        nodes = np.asarray(members, dtype=np.int64)
        edge_size = int(nodes.shape[0])
        log_size = math.log1p(edge_size)

        if edge_size == 0:
            degree_values = np.asarray([0.0], dtype=np.float64)
            feature_var = 0.0
        else:
            degree_values = node_degrees[nodes]
            if X_cpu is None:
                feature_var = 0.0
            else:
                x_edge = X_cpu[torch.as_tensor(nodes, dtype=torch.long)]
                feature_var = _scatter(x_edge)

        rows.append(
            [
                log_size,
                float(np.mean(degree_values)),
                float(np.std(degree_values)),
                float(np.max(degree_values)),
                feature_var,
            ]
        )

    return torch.as_tensor(rows, dtype=dtype, device=device).detach()


def compute_dynamic_moe_feature_block(
    Z: torch.Tensor,
    hyperedge_members: List[List[int]],
    num_nodes: int,
    seed: int = 0,
    max_nodes_per_edge: int = 256,
    max_pairs_per_edge: int = 4096,
    eps: float = 1e-8,
) -> torch.Tensor:
    device = Z.device
    dtype = Z.dtype if Z.is_floating_point() else torch.float32
    Z_cpu = Z.detach().float().cpu()
    if Z_cpu.shape[0] != num_nodes:
        raise ValueError(f"Z has {Z_cpu.shape[0]} nodes, expected {num_nodes}")

    rows = []
    for edge_id, members in enumerate(hyperedge_members):
        nodes = np.asarray(members, dtype=np.int64)
        if nodes.shape[0] == 0:
            z_sample = Z_cpu.new_zeros((0, Z_cpu.shape[1]))
        else:
            sampled_nodes = _sample_nodes(nodes, edge_id, seed, max_nodes_per_edge)
            z_sample = Z_cpu[torch.as_tensor(sampled_nodes, dtype=torch.long)]

        mean_sim, var_sim, _ = _pairwise_cosine_stats(
            z_sample, edge_id, seed, max_pairs_per_edge, eps
        )
        z_sample_unit = (
            _l2_normalize(z_sample, eps) if z_sample.shape[0] > 0 else z_sample
        )
        rows.append([mean_sim, var_sim, _scatter(z_sample_unit)])

    return torch.as_tensor(rows, dtype=dtype, device=device).detach()


def compute_subcluster_moe_feature_block(
    Z: torch.Tensor,
    hyperedge_members: List[List[int]],
    num_nodes: int,
    seed: int = 0,
    max_nodes_per_edge: int = 256,
    max_pairs_per_edge: int = 4096,
    subcluster_k: int = 2,
    eps: float = 1e-8,
    dataset: str = "",
) -> torch.Tensor:
    rows = compute_hyperedge_stats(
        Z=Z.detach(),
        X=None,
        hyperedge_members=hyperedge_members,
        num_nodes=num_nodes,
        seed=seed,
        max_nodes_per_edge=max_nodes_per_edge,
        max_pairs_per_edge=max_pairs_per_edge,
        subcluster_k=subcluster_k,
        eps=eps,
        dataset=dataset,
    )
    values = [[row["subcluster_score_norm"], row["major_ratio"]] for row in rows]
    return torch.as_tensor(values, dtype=Z.dtype, device=Z.device).detach()


class HyperedgeMoEFeatureCache:
    def __init__(
        self,
        X: Optional[torch.Tensor],
        hyperedge_members: List[List[int]],
        num_nodes: int,
        seed: int,
        max_nodes_per_edge: int = 256,
        max_pairs_per_edge: int = 4096,
        subcluster_k: int = 2,
        eps: float = 1e-8,
        dataset: str = "",
    ):
        device = X.device if X is not None else torch.device("cpu")
        dtype = X.dtype if X is not None and X.is_floating_point() else torch.float32
        self.hyperedge_members = hyperedge_members
        self.num_nodes = num_nodes
        self.seed = seed
        self.max_nodes_per_edge = max_nodes_per_edge
        self.max_pairs_per_edge = max_pairs_per_edge
        self.subcluster_k = subcluster_k
        self.eps = eps
        self.dataset = dataset
        self.raw_features = torch.zeros(
            len(hyperedge_members), len(MOE_FEATURE_COLUMNS), dtype=dtype, device=device
        )
        self.dynamic_initialized = False
        self.subcluster_initialized = False

        static_block = compute_static_moe_feature_block(
            X=X,
            hyperedge_members=hyperedge_members,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
        )
        self.raw_features[:, _column_indices(MOE_STATIC_FEATURE_COLUMNS)] = static_block

    def maybe_update_dynamic(
        self,
        Z: torch.Tensor,
        epoch: int,
        stage: str,
        update_interval: int,
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        update_interval = max(1, int(update_interval))
        should_update = (
            force or not self.dynamic_initialized or epoch % update_interval == 0
        )
        if not should_update:
            return

        start_time = time.time()
        dynamic_block = compute_dynamic_moe_feature_block(
            Z=Z,
            hyperedge_members=self.hyperedge_members,
            num_nodes=self.num_nodes,
            seed=self.seed,
            max_nodes_per_edge=self.max_nodes_per_edge,
            max_pairs_per_edge=self.max_pairs_per_edge,
            eps=self.eps,
        )
        self.raw_features = self.raw_features.to(device=Z.device, dtype=Z.dtype)
        self.raw_features[:, _column_indices(MOE_DYNAMIC_FEATURE_COLUMNS)] = (
            dynamic_block
        )
        self.dynamic_initialized = True
        if verbose:
            print(
                f"MoE feature update: epoch={epoch}, stage={stage}, "
                f"update_time={time.time() - start_time:.3f}s"
            )

    def update_subcluster_once(self, Z: torch.Tensor) -> None:
        if self.subcluster_initialized:
            return
        subcluster_block = compute_subcluster_moe_feature_block(
            Z=Z,
            hyperedge_members=self.hyperedge_members,
            num_nodes=self.num_nodes,
            seed=self.seed,
            max_nodes_per_edge=self.max_nodes_per_edge,
            max_pairs_per_edge=self.max_pairs_per_edge,
            subcluster_k=self.subcluster_k,
            eps=self.eps,
            dataset=self.dataset,
        )
        self.raw_features = self.raw_features.to(device=Z.device, dtype=Z.dtype)
        self.raw_features[:, _column_indices(MOE_SUBCLUSTER_FEATURE_COLUMNS)] = (
            subcluster_block
        )
        self.subcluster_initialized = True

    def get_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_features.detach()
        mean = raw.mean(dim=0, keepdim=True)
        std = raw.std(dim=0, unbiased=False, keepdim=True)
        normalized = (raw - mean) / (std + self.eps)
        return normalized.detach(), raw.detach()
