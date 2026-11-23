import math
import random
from typing import Optional

import torch

from src.structural_features import HyperedgeMoEFeatureCache


def infer_num_hyperedges(hyperedge_index: torch.Tensor) -> int:
    if hyperedge_index.numel() == 0 or hyperedge_index.shape[1] == 0:
        return 0
    return int(hyperedge_index[1].max().item()) + 1


def build_hyperedge_members(hyperedge_index: torch.Tensor, num_hyperedges: int) -> list:
    hyperedge_members = [[] for _ in range(num_hyperedges)]
    if num_hyperedges <= 0 or hyperedge_index.numel() == 0:
        return hyperedge_members
    node_idx = hyperedge_index[0].detach().cpu().tolist()
    edge_idx = hyperedge_index[1].detach().cpu().tolist()
    for node_id, edge_id in zip(node_idx, edge_idx):
        hyperedge_members[int(edge_id)].append(int(node_id))
    return hyperedge_members


def _sample_nodes_outside_original(
    num_nodes: int,
    original_set: set[int],
    num_samples: int,
) -> list[int]:
    available_count = int(num_nodes) - len(original_set)
    if num_samples > available_count:
        raise ValueError(
            f"cannot sample {num_samples} replacement nodes from "
            f"{available_count} available nodes"
        )
    forbidden = sorted(original_set)
    sampled_ranks = random.sample(range(available_count), num_samples)
    sampled_nodes = []
    for rank in sampled_ranks:
        node_id = int(rank)
        for forbidden_node in forbidden:
            if forbidden_node <= node_id:
                node_id += 1
            else:
                break
        sampled_nodes.append(node_id)
    return sampled_nodes


def _compute_negative_sampling_diagnostics(
    mode: str,
    pos_members: list[list[int]],
    neg_members: list[list[int]],
    source_edge_ids: torch.Tensor,
    replacement_counts: list[int],
    skipped_impossible_count: int,
) -> dict:
    source_ids = source_edge_ids.detach().cpu().tolist()
    identical_count = 0
    invalid_count = 0
    duplicate_member_count = 0
    overlap_ratios = []

    for neg_edge_id, source_edge_id in enumerate(source_ids):
        if source_edge_id < 0 or source_edge_id >= len(pos_members):
            invalid_count += 1
            continue
        if neg_edge_id >= len(neg_members):
            invalid_count += 1
            continue

        positive_members = pos_members[int(source_edge_id)]
        negative_members = neg_members[neg_edge_id]
        positive_set = set(positive_members)
        negative_set = set(negative_members)
        duplicate_members = len(negative_set) != len(negative_members)
        same_size = len(negative_members) == len(positive_members)
        identical = same_size and negative_set == positive_set

        if identical:
            identical_count += 1
        if duplicate_members:
            duplicate_member_count += 1
        if (not same_size) or duplicate_members or identical:
            invalid_count += 1

        overlap_denominator = max(1, len(positive_members))
        overlap_ratios.append(
            float(len(positive_set.intersection(negative_set)))
            / float(overlap_denominator)
        )

    replacement_distribution: dict[int, int] = {}
    for replacement_count in replacement_counts:
        key = int(replacement_count)
        replacement_distribution[key] = replacement_distribution.get(key, 0) + 1

    avg_replacement_count = (
        float(sum(replacement_counts)) / float(len(replacement_counts))
        if replacement_counts
        else 0.0
    )
    avg_overlap_ratio = (
        float(sum(overlap_ratios)) / float(len(overlap_ratios))
        if overlap_ratios
        else 0.0
    )
    return {
        "mode": mode,
        "positive_hyperedge_count": int(len(pos_members)),
        "negative_hyperedge_count": int(len(source_ids)),
        "identical_pos_neg_count": int(identical_count),
        "invalid_negative_count": int(invalid_count),
        "duplicate_member_negative_count": int(duplicate_member_count),
        "skipped_impossible_count": int(skipped_impossible_count),
        "avg_replacement_count": avg_replacement_count,
        "replacement_count_distribution": replacement_distribution,
        "avg_pos_neg_member_overlap_ratio": avg_overlap_ratio,
    }


def create_negative_samples_with_source_ids(
    hyperedge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    device = hyperedge_index.device
    dtype = hyperedge_index.dtype
    num_edges = infer_num_hyperedges(hyperedge_index)
    pos_members = build_hyperedge_members(hyperedge_index, num_edges)

    neg_nodes: list[int] = []
    neg_edges: list[int] = []
    source_edge_ids: list[int] = []
    replacement_counts: list[int] = []
    skipped_impossible_count = 0

    for edge_id, original_members in enumerate(pos_members):
        edge_size = len(original_members)
        if edge_size <= 0:
            skipped_impossible_count += 1
            continue

        original_set = set(original_members)
        num_replace = max(1, int(math.floor(math.log2(edge_size))))
        available_replacements = int(num_nodes) - len(original_set)
        if available_replacements < num_replace:
            skipped_impossible_count += 1
            continue

        seen_nodes: set[int] = set()
        duplicate_positions = []
        for position, node_id in enumerate(original_members):
            if node_id in seen_nodes:
                duplicate_positions.append(position)
            else:
                seen_nodes.add(node_id)
        if len(duplicate_positions) > num_replace:
            skipped_impossible_count += 1
            continue

        duplicate_position_set = set(duplicate_positions)
        remaining_positions = [
            position
            for position in range(edge_size)
            if position not in duplicate_position_set
        ]
        additional_replace_count = num_replace - len(duplicate_positions)
        sampled_positions = random.sample(
            remaining_positions,
            additional_replace_count,
        )
        replace_positions = duplicate_positions + sampled_positions
        random.shuffle(replace_positions)

        replacement_nodes = _sample_nodes_outside_original(
            num_nodes=num_nodes,
            original_set=original_set,
            num_samples=num_replace,
        )
        negative_members = list(original_members)
        for position, replacement_node in zip(replace_positions, replacement_nodes):
            negative_members[position] = replacement_node

        negative_set = set(negative_members)
        has_duplicate_members = len(negative_set) != len(negative_members)
        is_identical = negative_set == original_set
        is_valid = (
            len(negative_members) == edge_size
            and not has_duplicate_members
            and not is_identical
            and all(node_id not in original_set for node_id in replacement_nodes)
            and len(set(replacement_nodes)) == len(replacement_nodes)
        )
        if not is_valid:
            diagnostic_source_ids = torch.tensor(
                source_edge_ids + [edge_id],
                device=device,
                dtype=torch.long,
            )
            diagnostic_neg_members = build_hyperedge_members(
                torch.tensor(
                    [
                        neg_nodes + negative_members,
                        neg_edges + [len(source_edge_ids)] * len(negative_members),
                    ],
                    device=device,
                    dtype=dtype,
                ),
                len(source_edge_ids) + 1,
            )
            diagnostics = _compute_negative_sampling_diagnostics(
                mode="valid_log",
                pos_members=pos_members,
                neg_members=diagnostic_neg_members,
                source_edge_ids=diagnostic_source_ids,
                replacement_counts=replacement_counts + [num_replace],
                skipped_impossible_count=skipped_impossible_count,
            )
            raise RuntimeError(
                "valid_log negative sampling produced an invalid negative "
                f"hyperedge: {diagnostics}"
            )

        neg_edge_id = len(source_edge_ids)
        neg_nodes.extend(negative_members)
        neg_edges.extend([neg_edge_id] * edge_size)
        source_edge_ids.append(edge_id)
        replacement_counts.append(num_replace)

    if neg_nodes:
        neg_hyperedge_index = torch.tensor(
            [neg_nodes, neg_edges],
            device=device,
            dtype=dtype,
        )
    else:
        neg_hyperedge_index = hyperedge_index.new_empty((2, 0))
    neg_source_edge_ids = torch.tensor(
        source_edge_ids,
        device=device,
        dtype=torch.long,
    )
    neg_members = build_hyperedge_members(
        neg_hyperedge_index,
        len(source_edge_ids),
    )
    diagnostics = _compute_negative_sampling_diagnostics(
        mode="valid_log",
        pos_members=pos_members,
        neg_members=neg_members,
        source_edge_ids=neg_source_edge_ids,
        replacement_counts=replacement_counts,
        skipped_impossible_count=skipped_impossible_count,
    )
    if (
        diagnostics["identical_pos_neg_count"] != 0
        or diagnostics["invalid_negative_count"] != 0
        or diagnostics["duplicate_member_negative_count"] != 0
    ):
        raise RuntimeError(
            "valid_log negative sampling diagnostics failed: " f"{diagnostics}"
        )
    return neg_hyperedge_index, neg_source_edge_ids, diagnostics


def build_moe_features_for_hyperedge_index(
    X: torch.Tensor,
    hyperedge_index: torch.Tensor,
    num_nodes: int,
    reference_cache: HyperedgeMoEFeatureCache,
    node_embeds: torch.Tensor,
    epoch: int,
    stage: str,
    subcluster_embeds: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_hyperedges = infer_num_hyperedges(hyperedge_index)
    if num_hyperedges == 0:
        reference_features, reference_raw_features = reference_cache.get_features()
        return (
            reference_features.new_empty((0, reference_features.shape[1])),
            reference_raw_features.new_empty((0, reference_raw_features.shape[1])),
        )
    hyperedge_members = build_hyperedge_members(hyperedge_index, num_hyperedges)
    feature_cache = HyperedgeMoEFeatureCache(
        X=X,
        hyperedge_members=hyperedge_members,
        num_nodes=num_nodes,
        seed=reference_cache.seed,
        max_nodes_per_edge=reference_cache.max_nodes_per_edge,
        max_pairs_per_edge=reference_cache.max_pairs_per_edge,
        subcluster_k=reference_cache.subcluster_k,
        eps=reference_cache.eps,
        dataset=reference_cache.dataset,
    )
    feature_cache.maybe_update_dynamic(
        Z=node_embeds,
        epoch=epoch,
        stage=stage,
        update_interval=1,
        force=True,
    )
    if subcluster_embeds is not None:
        feature_cache.update_subcluster_once(subcluster_embeds)
    return feature_cache.get_features()


def init_negative_feature_cache_state() -> dict:
    return {
        "cached_neg_hyperedge_index": None,
        "cached_neg_source_edge_ids": None,
        "cached_neg_moe_features": None,
        "cached_neg_raw_moe_features": None,
        "cached_neg_sampling_diagnostics": None,
        "cached_epoch": None,
    }


def should_refresh_negative_feature_cache(
    cache_state: dict,
    epoch: int,
    neg_feature_update_interval: int,
    force_feature_update: bool,
    recompute_neg_features: bool,
) -> bool:
    if cache_state.get("cached_neg_hyperedge_index") is None:
        return True
    if cache_state.get("cached_neg_source_edge_ids") is None:
        return True
    if cache_state.get("cached_neg_sampling_diagnostics") is None:
        return True
    if recompute_neg_features:
        if cache_state.get("cached_neg_moe_features") is None:
            return True
        if cache_state.get("cached_neg_raw_moe_features") is None:
            return True
    if force_feature_update:
        return True
    update_interval = max(1, int(neg_feature_update_interval))
    return int(epoch) % update_interval == 0


def get_cached_negative_features(
    cache_state: dict,
    features: torch.Tensor,
    pos_hyperedge_index: torch.Tensor,
    num_nodes: int,
    feature_cache: HyperedgeMoEFeatureCache,
    node_embeds: torch.Tensor,
    epoch: int,
    stage: str,
    neg_feature_update_interval: int,
    force_feature_update: bool,
    recompute_neg_features: bool,
    subcluster_reference_embeds: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    dict,
    bool,
]:
    refreshed = should_refresh_negative_feature_cache(
        cache_state=cache_state,
        epoch=epoch,
        neg_feature_update_interval=neg_feature_update_interval,
        force_feature_update=force_feature_update,
        recompute_neg_features=recompute_neg_features,
    )
    if refreshed:
        (
            neg_hyperedge_index,
            neg_source_edge_ids,
            neg_sampling_diagnostics,
        ) = create_negative_samples_with_source_ids(
            pos_hyperedge_index,
            num_nodes,
        )
        if recompute_neg_features:
            if feature_cache.subcluster_initialized:
                neg_subcluster_embeds = (
                    subcluster_reference_embeds
                    if subcluster_reference_embeds is not None
                    else node_embeds
                )
            else:
                neg_subcluster_embeds = None
            neg_moe_features, neg_raw_moe_features = (
                build_moe_features_for_hyperedge_index(
                    X=features,
                    hyperedge_index=neg_hyperedge_index,
                    num_nodes=num_nodes,
                    reference_cache=feature_cache,
                    node_embeds=node_embeds,
                    epoch=epoch,
                    stage=stage,
                    subcluster_embeds=neg_subcluster_embeds,
                )
            )
        else:
            neg_moe_features = None
            neg_raw_moe_features = None
        cache_state["cached_neg_hyperedge_index"] = neg_hyperedge_index
        cache_state["cached_neg_source_edge_ids"] = neg_source_edge_ids
        cache_state["cached_neg_moe_features"] = neg_moe_features
        cache_state["cached_neg_raw_moe_features"] = neg_raw_moe_features
        cache_state["cached_neg_sampling_diagnostics"] = neg_sampling_diagnostics
        cache_state["cached_epoch"] = int(epoch)

    neg_hyperedge_index = cache_state["cached_neg_hyperedge_index"]
    neg_source_edge_ids = cache_state["cached_neg_source_edge_ids"]
    neg_moe_features = cache_state["cached_neg_moe_features"]
    neg_raw_moe_features = cache_state["cached_neg_raw_moe_features"]
    neg_sampling_diagnostics = cache_state["cached_neg_sampling_diagnostics"]
    required_values = [neg_hyperedge_index, neg_source_edge_ids]
    if recompute_neg_features:
        required_values.extend([neg_moe_features, neg_raw_moe_features])
    required_values.append(neg_sampling_diagnostics)
    if any(value is None for value in required_values):
        raise RuntimeError("negative feature cache is incomplete")
    return (
        neg_hyperedge_index,
        neg_source_edge_ids,
        neg_moe_features,
        neg_raw_moe_features,
        neg_sampling_diagnostics,
        refreshed,
    )
