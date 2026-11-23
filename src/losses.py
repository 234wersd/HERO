from typing import Optional

import torch
import torch.nn.functional as F

from src.confidence import (
    build_cluster_threshold_diagnostics,
    cluster_conf_weight_for_epoch,
    compute_cluster_threshold_state,
    threshold_flip_weight_for_epoch,
    _compute_flip_loss,
    _compute_reweighted_lclus,
    _validate_config,
    _zero_like_loss,
)
from src.score_moe import HyperedgeScoreMoELoss
from src.config import (
    NOISE_CORRECTION_MODES,
    NOISE_SELECTION_MODES,
    SCORE_LABEL_MODES,
    compute_required_ensemble_votes,
    validate_ensemble_config,
)
from src.sampling import infer_num_hyperedges


def compute_ensemble_low_mask(
    cluster_logits: torch.Tensor,
    ensemble_quantiles: list[float],
    cluster_conf_min_cluster_size: int,
    ensemble_vote_ratio: float,
    ensemble_vote_rule: str,
) -> torch.Tensor:
    num_quantiles = len(ensemble_quantiles)
    vote_count = torch.zeros(
        cluster_logits.shape[0],
        device=cluster_logits.device,
        dtype=torch.long,
    )
    if num_quantiles == 0:
        return vote_count.bool()
    for quantile in ensemble_quantiles:
        state_q = compute_cluster_threshold_state(
            cluster_logits=cluster_logits,
            cluster_conf_quantile=quantile,
            cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
        )
        vote_count += state_q["low_mask"].long()

    required_votes = compute_required_ensemble_votes(
        ensemble_vote_ratio=ensemble_vote_ratio,
        num_quantiles=num_quantiles,
        ensemble_vote_rule=ensemble_vote_rule,
    )
    return vote_count >= required_votes


def init_noise_selection_state() -> dict:
    """Create seed-local detached temporal state for Stage2 noise selection."""
    return {
        "previous_low_conf_mask": None,
        "previous_alternative_target": None,
        "persistence_count": None,
    }


def _inactive_cluster_threshold_state(
    cluster_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return shape-compatible neutral state without computing thresholds."""
    num_nodes, num_clusters = cluster_logits.shape
    pseudo_labels = pseudo_labels.detach()
    return {
        "probs": cluster_logits.new_zeros(cluster_logits.shape),
        "conf": cluster_logits.new_zeros(num_nodes),
        "pseudo_label": pseudo_labels,
        "global_threshold": cluster_logits.new_zeros(()),
        "thresholds": cluster_logits.new_zeros(num_clusters),
        "node_threshold": cluster_logits.new_zeros(num_nodes),
        "low_mask": torch.zeros_like(pseudo_labels, dtype=torch.bool),
        "per_cluster_counts": torch.bincount(
            pseudo_labels,
            minlength=num_clusters,
        ),
        "per_cluster_low_conf_counts": torch.zeros(
            num_clusters,
            device=cluster_logits.device,
            dtype=torch.long,
        ),
        "used_global_threshold": torch.zeros(
            num_clusters,
            device=cluster_logits.device,
            dtype=torch.bool,
        ),
    }


def _alternative_target(
    probabilities: torch.Tensor,
    old_cluster: torch.Tensor,
) -> torch.Tensor:
    """Return alternatives [N] from probabilities [N, K], excluding old [N]."""
    if probabilities.ndim != 2 or old_cluster.shape != (probabilities.shape[0],):
        raise ValueError("probabilities must be [N, K] and old_cluster must be [N]")
    if probabilities.shape[1] < 2:
        raise ValueError("alternative cluster selection requires at least two clusters")
    masked_probabilities = probabilities.clone()
    masked_probabilities.scatter_(1, old_cluster.unsqueeze(1), -1.0)
    return masked_probabilities.argmax(dim=1)


def compute_multi_evidence_selection(
    cluster_logits: torch.Tensor,
    view1_cluster_logits: torch.Tensor,
    view2_cluster_logits: torch.Tensor,
    low_conf_mask: torch.Tensor,
    temporal_state: dict,
    relocation_min_target_prob: float,
    temporal_persistence_steps: int,
    binary_support_required: bool = True,
) -> dict[str, torch.Tensor]:
    """Build detached evidence masks, requiring actual support for binary relocation."""
    if cluster_logits.ndim != 2:
        raise ValueError("cluster_logits must have shape [N, K]")
    if cluster_logits.shape[1] <= 1:
        raise ValueError("multi-evidence selection requires K > 1")
    if view1_cluster_logits.shape != cluster_logits.shape:
        raise ValueError("view1_cluster_logits must match cluster_logits shape [N, K]")
    if view2_cluster_logits.shape != cluster_logits.shape:
        raise ValueError("view2_cluster_logits must match cluster_logits shape [N, K]")
    if low_conf_mask.shape != (cluster_logits.shape[0],):
        raise ValueError("low_conf_mask must have shape [N]")
    if temporal_persistence_steps < 1:
        raise ValueError("temporal_persistence_steps must be at least 1")
    if not 0.0 <= float(relocation_min_target_prob) <= 1.0:
        raise ValueError("relocation_min_target_prob must be in [0, 1]")

    with torch.no_grad():
        old_cluster = cluster_logits.detach().argmax(dim=1)
        view1_probabilities = F.softmax(view1_cluster_logits.detach(), dim=1)
        view2_probabilities = F.softmax(view2_cluster_logits.detach(), dim=1)
        mean_view_probabilities = 0.5 * (view1_probabilities + view2_probabilities)
        alternative_view1 = _alternative_target(
            view1_probabilities,
            old_cluster,
        )
        alternative_view2 = _alternative_target(
            view2_probabilities,
            old_cluster,
        )
        candidate_target = _alternative_target(
            mean_view_probabilities,
            old_cluster,
        )
        target_probability = mean_view_probabilities.gather(
            1,
            candidate_target.unsqueeze(1),
        ).squeeze(1)
        target_reliable = target_probability >= float(relocation_min_target_prob)
        if binary_support_required and cluster_logits.shape[1] == 2:
            # Excluding the current cluster leaves only one binary candidate;
            # agreement on that forced choice provides no transfer evidence.
            cluster_pair_indices = torch.stack((old_cluster, candidate_target), dim=1)
            view1_pair_probabilities = view1_probabilities.gather(
                1, cluster_pair_indices
            )
            view2_pair_probabilities = view2_probabilities.gather(
                1, cluster_pair_indices
            )
            cross_view_agreement = (
                (view1_pair_probabilities[:, 1] > view1_pair_probabilities[:, 0])
                & (view2_pair_probabilities[:, 1] > view2_pair_probabilities[:, 0])
                & target_reliable
            )
            old_probability = mean_view_probabilities.gather(
                1, old_cluster.unsqueeze(1)
            ).squeeze(1)
            # Every counted check must support the candidate, including any
            # configured probability floor. Ties provide no support.
            persistence_eligible = (
                low_conf_mask & target_reliable & (target_probability > old_probability)
            )
        else:
            # Preserve the original multiclass and other correction rules.
            cross_view_agreement = (
                alternative_view1.eq(alternative_view2)
                & candidate_target.eq(alternative_view1)
                & target_reliable
            )
            persistence_eligible = low_conf_mask

        previous_low_conf_mask = temporal_state.get("previous_low_conf_mask")
        previous_alternative_target = temporal_state.get("previous_alternative_target")
        previous_persistence_count = temporal_state.get("persistence_count")
        state_is_compatible = (
            previous_low_conf_mask is not None
            and previous_alternative_target is not None
            and previous_persistence_count is not None
            and previous_low_conf_mask.shape == low_conf_mask.shape
            and previous_alternative_target.shape == candidate_target.shape
            and previous_persistence_count.shape == low_conf_mask.shape
        )
        if state_is_compatible:
            previous_low_conf_mask = previous_low_conf_mask.to(
                device=cluster_logits.device,
                dtype=torch.bool,
            )
            previous_alternative_target = previous_alternative_target.to(
                device=cluster_logits.device,
                dtype=torch.long,
            )
            previous_persistence_count = previous_persistence_count.to(
                device=cluster_logits.device,
                dtype=torch.long,
            )
            continues_previous = (
                persistence_eligible
                & previous_low_conf_mask
                # For K=2, the candidate uniquely determines the old cluster,
                # so this also preserves the complete source-target pair.
                & candidate_target.eq(previous_alternative_target)
            )
            persistence_count = torch.where(
                continues_previous,
                previous_persistence_count + 1,
                persistence_eligible.long(),
            )
        else:
            persistence_count = persistence_eligible.long()

        persistent_alternative = (
            low_conf_mask
            & target_reliable
            & (persistence_count >= int(temporal_persistence_steps))
        )
        reliable_evidence_mask = cross_view_agreement | persistent_alternative
        target_cluster = torch.full_like(old_cluster, -1)
        target_cluster[reliable_evidence_mask] = candidate_target[
            reliable_evidence_mask
        ]
        valid_target_mask = target_cluster >= 0
        correction_mask = low_conf_mask & reliable_evidence_mask & valid_target_mask
        if torch.any(correction_mask & ~low_conf_mask):
            raise AssertionError("correction_mask must be a subset of low_conf_mask")
        if correction_mask.any() and torch.any(
            target_cluster[correction_mask] == old_cluster[correction_mask]
        ):
            raise AssertionError("correction target must differ from old cluster")

        temporal_state["previous_low_conf_mask"] = low_conf_mask.detach().clone()
        temporal_state["previous_alternative_target"] = (
            candidate_target.detach().clone()
        )
        temporal_state["persistence_count"] = persistence_count.detach().clone()

    return {
        "low_conf_mask": low_conf_mask.detach(),
        "old_cluster": old_cluster.detach(),
        "target_cluster": target_cluster.detach(),
        "valid_target_mask": valid_target_mask.detach(),
        "reliable_evidence_mask": reliable_evidence_mask.detach(),
        "target_probability_views": target_probability.detach(),
        "cross_view_agreement": cross_view_agreement.detach(),
        "persistent_alternative": persistent_alternative.detach(),
        "correction_mask": correction_mask.detach(),
    }


def compute_soft_relocation_loss(
    cluster_logits: torch.Tensor,
    correction_mask: torch.Tensor,
    old_cluster: torch.Tensor,
    target_cluster: torch.Tensor,
    relocation_margin: float,
) -> torch.Tensor:
    """Compute logit-stable relocation loss from logits [N, K] and masks [N]."""
    num_nodes = cluster_logits.shape[0]
    expected_shape = (num_nodes,)
    if correction_mask.shape != expected_shape:
        raise ValueError("correction_mask must have shape [N]")
    if old_cluster.shape != expected_shape or target_cluster.shape != expected_shape:
        raise ValueError("old_cluster and target_cluster must have shape [N]")
    if relocation_margin < 0:
        raise ValueError("relocation_margin must be non-negative")
    if not correction_mask.any():
        return _zero_like_loss(cluster_logits)
    if torch.any(target_cluster[correction_mask] == old_cluster[correction_mask]):
        raise AssertionError("relocation target must differ from old cluster")

    selected_log_probabilities = F.log_softmax(
        cluster_logits[correction_mask],
        dim=1,
    )
    old_log_probability = selected_log_probabilities.gather(
        1,
        old_cluster[correction_mask].unsqueeze(1),
    ).squeeze(1)
    target_log_probability = selected_log_probabilities.gather(
        1,
        target_cluster[correction_mask].unsqueeze(1),
    ).squeeze(1)
    per_node_loss = F.relu(
        float(relocation_margin) + old_log_probability - target_log_probability
    )
    return per_node_loss.mean()


def _diagnostic_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def remove_label_free_difficulty_diagnostics(diagnostics: dict) -> None:
    for key in [
        "difficulty_gate_mean",
        "difficulty_gate_std",
        "difficulty_gate_min",
        "difficulty_gate_max",
        "signed_margin_mean",
        "signed_margin_std",
        "signed_margin_min",
        "signed_margin_max",
        "bce_difficulty_mean",
        "bce_difficulty_std",
    ]:
        diagnostics.pop(key, None)


def compute_total_cluster_loss(
    cluster_loss: torch.Tensor,
    flip_loss: torch.Tensor,
    gamma: float,
    flip_weight: float,
) -> torch.Tensor:
    return float(gamma) * (cluster_loss + float(flip_weight) * flip_loss)


def compute_score_moe_calibrated_loss_with_cluster_threshold(
    loss_fn: HyperedgeScoreMoELoss,
    pos_node_embeds: torch.Tensor,
    aug_node_embeds: torch.Tensor,
    other_aug_node_embeds: torch.Tensor,
    pos_hyperedge_index: torch.Tensor,
    neg_hyperedge_index: torch.Tensor,
    moe_features: torch.Tensor,
    raw_moe_features: torch.Tensor,
    neg_moe_features: torch.Tensor,
    neg_raw_moe_features: torch.Tensor,
    neg_source_edge_ids: torch.Tensor,
    clustering_logits: torch.Tensor,
    recompute_neg_features: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    tau: float = 0.5,
    current_epoch: Optional[int] = None,
    score_moe_start_stage: str = "both",
    score_moe_enabled_this_stage: bool = True,
    step2_enabled: bool = True,
    score_label_mode: str = "label_free",
    stage: str = "Finetune",
    finetune_epoch: int = 1,
    use_cluster_confidence_threshold: bool = False,
    use_threshold_flip: bool = False,
    disable_cluster_confidence_threshold: bool = False,
    cluster_conf_mode: str = "reweight",
    cluster_conf_quantile: float = 0.2,
    cluster_low_conf_weight: float = 0.5,
    cluster_conf_min_cluster_size: int = 10,
    cluster_conf_warmup_epochs: int = 20,
    cluster_conf_temperature: float = 0.05,
    ensemble_threshold: bool = False,
    ensemble_quantiles: Optional[list[float]] = None,
    ensemble_vote_ratio: float = 0.5,
    ensemble_vote_rule: str = "ceil",
    flip_weight: float = 0.01,
    flip_quantile: Optional[float] = None,
    flip_mode: str = "negative",
    flip_warmup_epochs: int = 20,
    flip_eps: float = 1e-6,
    view1_cluster_logits: Optional[torch.Tensor] = None,
    view2_cluster_logits: Optional[torch.Tensor] = None,
    noise_selection_state: Optional[dict] = None,
    noise_selection_mode: str = "legacy_quantile_vote",
    noise_correction_mode: str = "negative_top_penalty",
    relocation_weight: float = 0.01,
    relocation_margin: float = 0.05,
    relocation_min_target_prob: float = 0.0,
    temporal_persistence_steps: int = 2,
    return_score_moe_diagnostics: bool = False,
    return_cluster_threshold_diagnostics: bool = False,
    return_analysis_state: bool = False,
) -> tuple[torch.Tensor, dict]:
    _validate_config(
        cluster_conf_mode=cluster_conf_mode,
        cluster_conf_quantile=cluster_conf_quantile,
        cluster_low_conf_weight=cluster_low_conf_weight,
        cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
        cluster_conf_temperature=cluster_conf_temperature,
        flip_quantile=flip_quantile,
        flip_mode=flip_mode,
        flip_eps=flip_eps,
    )
    ensemble_quantiles = [] if ensemble_quantiles is None else ensemble_quantiles
    validate_ensemble_config(
        ensemble_quantiles=ensemble_quantiles,
        ensemble_vote_ratio=ensemble_vote_ratio,
        ensemble_vote_rule=ensemble_vote_rule,
    )
    if noise_selection_mode not in NOISE_SELECTION_MODES:
        raise ValueError(
            f"noise_selection_mode must be one of {NOISE_SELECTION_MODES}, "
            f"got {noise_selection_mode!r}"
        )
    if noise_correction_mode not in NOISE_CORRECTION_MODES:
        raise ValueError(
            f"noise_correction_mode must be one of {NOISE_CORRECTION_MODES}, "
            f"got {noise_correction_mode!r}"
        )
    if relocation_weight < 0:
        raise ValueError("relocation_weight must be non-negative")
    if relocation_margin < 0:
        raise ValueError("relocation_margin must be non-negative")
    if not 0.0 <= float(relocation_min_target_prob) <= 1.0:
        raise ValueError("relocation_min_target_prob must be in [0, 1]")
    if temporal_persistence_steps < 1:
        raise ValueError("temporal_persistence_steps must be at least 1")
    if score_label_mode not in SCORE_LABEL_MODES:
        raise ValueError(
            f"score_label_mode must be one of {SCORE_LABEL_MODES}, "
            f"got {score_label_mode!r}"
        )

    pos_diagnostics = None
    pos_num_hyperedges = infer_num_hyperedges(pos_hyperedge_index)
    neg_source_edge_ids = neg_source_edge_ids.to(
        device=pos_node_embeds.device,
        dtype=torch.long,
    )
    neg_num_hyperedges = int(neg_source_edge_ids.numel())
    neg_index_num_hyperedges = infer_num_hyperedges(neg_hyperedge_index)
    if neg_index_num_hyperedges != neg_num_hyperedges:
        raise ValueError(
            "neg_hyperedge_index edge ids must be compact and aligned with "
            "neg_source_edge_ids: "
            f"got {neg_index_num_hyperedges} indexed negative hyperedges and "
            f"{neg_num_hyperedges} source ids"
        )
    pos_labels = pos_node_embeds.new_ones(pos_num_hyperedges)
    neg_labels = pos_node_embeds.new_zeros(neg_num_hyperedges)
    if recompute_neg_features:
        if neg_moe_features.shape[0] != neg_num_hyperedges:
            raise ValueError(
                "neg_moe_features must align with neg_hyperedge_index: "
                f"got {neg_moe_features.shape[0]} feature rows for "
                f"{neg_num_hyperedges} negative hyperedges"
            )
        if neg_raw_moe_features.shape[0] != neg_num_hyperedges:
            raise ValueError(
                "neg_raw_moe_features must align with neg_hyperedge_index: "
                f"got {neg_raw_moe_features.shape[0]} feature rows for "
                f"{neg_num_hyperedges} negative hyperedges"
            )
        neg_scorer_source_edge_ids = None
    else:
        if neg_source_edge_ids.numel() > 0:
            max_source_edge_id = int(neg_source_edge_ids.max().item())
            if max_source_edge_id >= neg_moe_features.shape[0]:
                raise ValueError(
                    "neg_source_edge_ids references feature row "
                    f"{max_source_edge_id}, but only "
                    f"{neg_moe_features.shape[0]} feature rows are available"
                )
            if max_source_edge_id >= neg_raw_moe_features.shape[0]:
                raise ValueError(
                    "neg_source_edge_ids references raw feature row "
                    f"{max_source_edge_id}, but only "
                    f"{neg_raw_moe_features.shape[0]} raw feature rows are available"
                )
        neg_scorer_source_edge_ids = neg_source_edge_ids

    if score_label_mode == "legacy":
        if return_score_moe_diagnostics:
            pos_scores, _, pos_diagnostics = loss_fn.scorer(
                pos_node_embeds,
                pos_hyperedge_index,
                moe_features,
                raw_moe_features,
                labels=pos_labels,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
                return_diagnostics=True,
            )
        else:
            pos_scores, _ = loss_fn.scorer(
                pos_node_embeds,
                pos_hyperedge_index,
                moe_features,
                raw_moe_features,
                labels=pos_labels,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            )
        if neg_num_hyperedges > 0:
            neg_scores, _ = loss_fn.scorer(
                pos_node_embeds,
                neg_hyperedge_index,
                neg_moe_features,
                neg_raw_moe_features,
                source_edge_ids=neg_scorer_source_edge_ids,
                labels=neg_labels,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            )
        else:
            neg_scores = pos_node_embeds.new_empty(0)
    else:
        if return_score_moe_diagnostics:
            pos_scores, _, pos_diagnostics = loss_fn.scorer.forward_label_free(
                pos_node_embeds,
                pos_hyperedge_index,
                moe_features,
                raw_moe_features,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
                return_diagnostics=True,
            )
        else:
            pos_scores, _ = loss_fn.scorer.forward_label_free(
                pos_node_embeds,
                pos_hyperedge_index,
                moe_features,
                raw_moe_features,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            )
        if neg_num_hyperedges > 0:
            neg_scores, _ = loss_fn.scorer.forward_label_free(
                pos_node_embeds,
                neg_hyperedge_index,
                neg_moe_features,
                neg_raw_moe_features,
                source_edge_ids=neg_scorer_source_edge_ids,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            )
        else:
            neg_scores = pos_node_embeds.new_empty(0)

    pos_loss = loss_fn.bce_loss(pos_scores, pos_labels)
    if neg_num_hyperedges > 0:
        neg_loss = loss_fn.bce_loss(neg_scores, neg_labels)
    else:
        neg_loss = pos_loss.new_zeros(())
    structure_loss = pos_loss + neg_loss
    infonce_loss = loss_fn.compute_infonce_loss(
        aug_node_embeds,
        other_aug_node_embeds,
        tau,
    )

    with torch.no_grad():
        pseudo_labels = torch.argmax(clustering_logits, dim=1)
    per_node_lclus = F.cross_entropy(
        clustering_logits,
        pseudo_labels,
        reduction="none",
    )
    lclus_original = per_node_lclus.mean()
    lclus_reweighted = lclus_original
    effective_weight = torch.ones_like(per_node_lclus)

    stage_is_finetune = stage.lower() == "finetune"
    stage_step2_enabled = stage_is_finetune and bool(step2_enabled)
    step2_effects_enabled = (
        stage_step2_enabled and not disable_cluster_confidence_threshold
    )
    if stage_step2_enabled:
        threshold_state = compute_cluster_threshold_state(
            cluster_logits=clustering_logits,
            cluster_conf_quantile=cluster_conf_quantile,
            cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
        )
    else:
        threshold_state = _inactive_cluster_threshold_state(
            cluster_logits=clustering_logits,
            pseudo_labels=pseudo_labels,
        )
    low_conf_mask = threshold_state["low_mask"]
    ensemble_active = (
        step2_effects_enabled
        and noise_selection_mode == "legacy_quantile_vote"
        and bool(ensemble_threshold)
        and len(ensemble_quantiles) > 0
    )
    ensemble_low_mask = None
    if ensemble_active:
        ensemble_low_mask = compute_ensemble_low_mask(
            cluster_logits=clustering_logits,
            ensemble_quantiles=ensemble_quantiles,
            cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
            ensemble_vote_ratio=ensemble_vote_ratio,
            ensemble_vote_rule=ensemble_vote_rule,
        )

    if use_cluster_confidence_threshold and step2_effects_enabled:
        cluster_conf_weight_t = cluster_conf_weight_for_epoch(
            finetune_epoch=finetune_epoch,
            cluster_conf_warmup_epochs=cluster_conf_warmup_epochs,
        )
        lclus_reweighted, effective_weight = _compute_reweighted_lclus(
            per_node_lclus=per_node_lclus,
            state=threshold_state,
            cluster_conf_mode=cluster_conf_mode,
            cluster_low_conf_weight=cluster_low_conf_weight,
            cluster_conf_weight_t=cluster_conf_weight_t,
            cluster_conf_temperature=cluster_conf_temperature,
        )
    else:
        cluster_conf_weight_t = 0.0

    if not step2_effects_enabled or flip_quantile is None:
        flip_threshold_state = threshold_state
    else:
        flip_threshold_state = compute_cluster_threshold_state(
            cluster_logits=clustering_logits,
            cluster_conf_quantile=flip_quantile,
            cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
        )

    if noise_selection_mode == "legacy_quantile_vote":
        selection_mask = (
            ensemble_low_mask
            if ensemble_low_mask is not None
            else flip_threshold_state["low_mask"]
        )
    elif noise_selection_mode == "single_quantile":
        selection_mask = flip_threshold_state["low_mask"]
    else:
        selection_mask = low_conf_mask

    correction_warmup_active = step2_effects_enabled and finetune_epoch <= int(
        flip_warmup_epochs
    )
    correction_active = (
        step2_effects_enabled
        and bool(use_threshold_flip)
        and not correction_warmup_active
    )
    if noise_selection_state is not None and correction_warmup_active:
        noise_selection_state["previous_low_conf_mask"] = None
        noise_selection_state["previous_alternative_target"] = None
        noise_selection_state["persistence_count"] = None

    old_cluster = threshold_state["pseudo_label"].detach()
    target_cluster = torch.full_like(old_cluster, -1)
    cross_view_agreement = torch.zeros_like(low_conf_mask)
    persistent_alternative = torch.zeros_like(low_conf_mask)
    valid_target_mask = torch.zeros_like(low_conf_mask)
    correction_mask = torch.zeros_like(low_conf_mask)
    target_probability_views = threshold_state["conf"].new_zeros(
        threshold_state["conf"].shape
    )
    needs_alternative_evidence = correction_active and (
        noise_selection_mode == "multi_evidence"
        or noise_correction_mode == "soft_relocation"
    )
    if needs_alternative_evidence:
        if view1_cluster_logits is None or view2_cluster_logits is None:
            raise ValueError(
                "two augmented-view cluster logits are required for "
                "multi_evidence or soft_relocation"
            )
        if noise_selection_state is None:
            noise_selection_state = init_noise_selection_state()
        evidence = compute_multi_evidence_selection(
            cluster_logits=clustering_logits,
            view1_cluster_logits=view1_cluster_logits,
            view2_cluster_logits=view2_cluster_logits,
            low_conf_mask=low_conf_mask,
            temporal_state=noise_selection_state,
            relocation_min_target_prob=relocation_min_target_prob,
            temporal_persistence_steps=temporal_persistence_steps,
            binary_support_required=(noise_correction_mode == "soft_relocation"),
        )
        old_cluster = evidence["old_cluster"]
        target_cluster = evidence["target_cluster"]
        valid_target_mask = evidence["valid_target_mask"]
        target_probability_views = evidence["target_probability_views"]
        cross_view_agreement = evidence["cross_view_agreement"]
        persistent_alternative = evidence["persistent_alternative"]
        if noise_selection_mode == "multi_evidence":
            correction_mask = evidence["correction_mask"]
        else:
            correction_mask = (
                selection_mask
                & low_conf_mask
                & cross_view_agreement
                & valid_target_mask
            )
    elif correction_active:
        correction_mask = selection_mask.detach()

    requires_reliable_target = (
        noise_selection_mode == "multi_evidence"
        or noise_correction_mode == "soft_relocation"
    )
    if requires_reliable_target and torch.any(correction_mask & ~low_conf_mask):
        raise AssertionError("correction_mask must be a subset of low_conf_mask")
    if needs_alternative_evidence and correction_mask.any():
        if not torch.all(valid_target_mask[correction_mask]):
            raise AssertionError("every correction node must have a valid target")
        if torch.any(target_cluster[correction_mask] == old_cluster[correction_mask]):
            raise AssertionError("correction target must differ from old cluster")

    flip_loss = _zero_like_loss(clustering_logits)
    relocation_loss = _zero_like_loss(clustering_logits)
    flip_weight_t = 0.0
    relocation_weight_t = 0.0
    if correction_active:
        if noise_correction_mode == "negative_top_penalty":
            correction_state = dict(flip_threshold_state)
            correction_state["low_mask"] = correction_mask
            flip_loss = _compute_flip_loss(
                cluster_logits=clustering_logits,
                state=correction_state,
                flip_mode=flip_mode,
                flip_eps=flip_eps,
            )
            flip_weight_t = threshold_flip_weight_for_epoch(
                finetune_epoch=finetune_epoch,
                flip_weight=flip_weight,
                flip_warmup_epochs=flip_warmup_epochs,
            )
        else:
            relocation_loss = compute_soft_relocation_loss(
                cluster_logits=clustering_logits,
                correction_mask=correction_mask,
                old_cluster=old_cluster,
                target_cluster=target_cluster,
                relocation_margin=relocation_margin,
            )
            relocation_weight_t = float(relocation_weight)

    cluster_loss = lclus_reweighted
    weighted_flip_loss = float(flip_weight_t) * flip_loss
    weighted_relocation_loss = float(relocation_weight_t) * relocation_loss
    weighted_correction_loss = weighted_flip_loss + weighted_relocation_loss
    clustering_loss = cluster_loss + weighted_correction_loss
    if stage_is_finetune and not step2_effects_enabled:
        if not torch.allclose(
            clustering_loss,
            lclus_original,
            rtol=0.0,
            atol=0.0,
        ):
            raise AssertionError(
                "disabled Step2 requires effective Stage2 cluster loss "
                "to equal the original cluster loss"
            )
    active_correction_loss = (
        flip_loss
        if noise_correction_mode == "negative_top_penalty"
        else relocation_loss
    )
    active_correction_weight = (
        flip_weight_t
        if noise_correction_mode == "negative_top_penalty"
        else relocation_weight_t
    )
    total_cluster_loss = compute_total_cluster_loss(
        cluster_loss=cluster_loss,
        flip_loss=active_correction_loss,
        gamma=gamma,
        flip_weight=active_correction_weight,
    )
    total_loss = (
        float(alpha) * structure_loss + float(beta) * infonce_loss + total_cluster_loss
    )

    pos_score_mean = pos_scores.mean().item() if pos_scores.numel() > 0 else 0.0
    neg_score_mean = neg_scores.mean().item() if neg_scores.numel() > 0 else 0.0
    metrics: dict[str, object] = {
        "structure_loss": float(structure_loss.detach().item()),
        "bce_loss": float(structure_loss.detach().item()),
        "infonce_loss": float(infonce_loss.detach().item()),
        "clustering_loss": float(clustering_loss.detach().item()),
        "Lclus_original": float(lclus_original.detach().item()),
        "Lclus_reweighted": float(lclus_reweighted.detach().item()),
        "pos_loss": float(pos_loss.detach().item()),
        "neg_loss": float(neg_loss.detach().item()),
        "avg_pos_score": pos_score_mean,
        "avg_neg_score": neg_score_mean,
        "pos_neg_score_gap": pos_score_mean - neg_score_mean,
        "cluster_conf_weight_t": float(cluster_conf_weight_t),
        "cluster_conf_weight_mean": float(effective_weight.detach().mean().item()),
        "effective_quantile": float(cluster_conf_quantile),
        "effective_temperature": float(cluster_conf_temperature),
        "ensemble_threshold": bool(ensemble_active),
        "ensemble_threshold_target": (
            "correction only" if ensemble_active else "disabled"
        ),
        "ensemble_low_conf_ratio": (
            float(ensemble_low_mask.float().mean().item()) if ensemble_active else 0.0
        ),
        "flip_loss": float(flip_loss.detach().item()),
        "flip_weight_t": float(flip_weight_t),
        "weighted_flip_loss": float(weighted_flip_loss.detach().item()),
        "relocation_loss": float(relocation_loss.detach().item()),
        "relocation_weight_t": float(relocation_weight_t),
        "weighted_relocation_loss": float(weighted_relocation_loss.detach().item()),
        "correction_loss": float((flip_loss + relocation_loss).detach().item()),
        "correction_weight": float(flip_weight_t + relocation_weight_t),
        "noise_selection_mode": noise_selection_mode,
        "noise_correction_mode": noise_correction_mode,
        "disable_cluster_confidence_threshold": bool(
            disable_cluster_confidence_threshold
        ),
        "step2_enabled": bool(step2_enabled),
        "cluster_threshold_active": bool(
            step2_effects_enabled and use_cluster_confidence_threshold
        ),
        "correction_active": bool(correction_active),
        "correction_warmup_active": bool(correction_warmup_active),
        "temporal_state_updated": bool(needs_alternative_evidence),
        "low_conf_count": int(low_conf_mask.sum().item()),
        "correction_count": int(correction_mask.sum().item()),
        "total_cluster_loss": float(total_cluster_loss.detach().item()),
        "gradient_mod_loss": 0.0,
        "recompute_neg_features": bool(recompute_neg_features),
        "score_label_mode": score_label_mode,
    }

    if return_score_moe_diagnostics:
        diagnostics = pos_diagnostics if pos_diagnostics is not None else {}
        diagnostics.update(
            {
                "score_label_mode": score_label_mode,
                "score_moe_start_stage": score_moe_start_stage,
                "score_moe_enabled_this_stage": bool(score_moe_enabled_this_stage),
                "pos_score_mean": pos_score_mean,
                "neg_score_mean": neg_score_mean,
                "pos_neg_score_gap": metrics["pos_neg_score_gap"],
                "bce_loss": metrics["bce_loss"],
            }
        )
        if score_label_mode == "label_free":
            remove_label_free_difficulty_diagnostics(diagnostics)
        metrics["score_moe_diagnostics"] = diagnostics

    if return_cluster_threshold_diagnostics and stage_step2_enabled:
        diagnostics = build_cluster_threshold_diagnostics(
            cluster_logits=clustering_logits,
            stage=stage,
            use_cluster_confidence_threshold=(
                use_cluster_confidence_threshold and step2_effects_enabled
            ),
            use_threshold_flip=(use_threshold_flip and step2_effects_enabled),
            cluster_conf_mode=cluster_conf_mode,
            cluster_conf_weight_t=cluster_conf_weight_t,
            cluster_conf_quantile=cluster_conf_quantile,
            flip_quantile=flip_quantile,
            cluster_low_conf_weight=cluster_low_conf_weight,
            cluster_conf_temperature=cluster_conf_temperature,
            cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
            lclus_original=float(lclus_original.detach().item()),
            lclus_reweighted=float(lclus_reweighted.detach().item()),
            flip_loss=float(flip_loss.detach().item()),
            flip_weight_t=float(flip_weight_t),
            total_loss=float(clustering_loss.detach().item()),
        )
        if ensemble_active:
            low_mask = ensemble_low_mask
            num_nodes = max(1, int(low_mask.numel()))
            diagnostics.update(
                {
                    "ensemble_threshold": True,
                    "ensemble_threshold_target": "correction only",
                    "ensemble_quantiles": [
                        float(quantile) for quantile in ensemble_quantiles
                    ],
                    "ensemble_vote_ratio": float(ensemble_vote_ratio),
                    "ensemble_vote_rule": ensemble_vote_rule,
                    "ensemble_low_conf_count": int(low_mask.sum().item()),
                    "ensemble_low_conf_ratio": (
                        float(low_mask.float().sum().item()) / float(num_nodes)
                    ),
                }
            )
            if use_threshold_flip and step2_effects_enabled:
                diagnostics["flip_low_conf_count"] = int(low_mask.sum().item())
                diagnostics["flip_low_conf_ratio"] = float(
                    low_mask.float().sum().item()
                ) / float(num_nodes)
        with torch.no_grad():
            num_nodes = max(1, int(low_conf_mask.numel()))
            num_clusters = int(clustering_logits.shape[1])
            correction_count = int(correction_mask.sum().item())
            cross_view_count = int(cross_view_agreement.sum().item())
            persistent_count = int(persistent_alternative.sum().item())
            per_cluster_correction_counts = torch.bincount(
                old_cluster[correction_mask],
                minlength=num_clusters,
            )
            if needs_alternative_evidence:
                old_equals_target_count = int(
                    (old_cluster[correction_mask] == target_cluster[correction_mask])
                    .sum()
                    .item()
                )
                if old_equals_target_count != 0:
                    raise AssertionError(
                        "old cluster must differ from relocation target"
                    )
                target_counts = torch.bincount(
                    target_cluster[correction_mask],
                    minlength=num_clusters,
                )
                target_distribution = {
                    int(cluster_idx): int(count)
                    for cluster_idx, count in enumerate(target_counts.cpu().tolist())
                    if int(count) > 0
                }
                selected_main_probabilities = threshold_state["probs"].detach()[
                    correction_mask
                ]
                old_probability_values = selected_main_probabilities.gather(
                    1,
                    old_cluster[correction_mask].unsqueeze(1),
                ).squeeze(1)
                target_probability_values = selected_main_probabilities.gather(
                    1,
                    target_cluster[correction_mask].unsqueeze(1),
                ).squeeze(1)
                probability_margin_values = (
                    target_probability_values - old_probability_values
                )
                view_target_probability_values = target_probability_views[
                    correction_mask
                ]
            else:
                old_equals_target_count = 0
                target_distribution = {}
                empty_values = clustering_logits.new_empty(0)
                old_probability_values = empty_values
                target_probability_values = empty_values
                probability_margin_values = empty_values
                view_target_probability_values = empty_values

            correction_contribution = (
                float(gamma) * weighted_correction_loss.detach().abs()
            )
            correction_to_total_ratio = float(
                (
                    correction_contribution / total_loss.detach().abs().clamp_min(1e-12)
                ).item()
            )
            diagnostics.update(
                {
                    "noise_selection_mode": noise_selection_mode,
                    "noise_correction_mode": noise_correction_mode,
                    "disable_cluster_confidence_threshold": bool(
                        disable_cluster_confidence_threshold
                    ),
                    "step2_enabled": bool(step2_enabled),
                    "correction_active": bool(correction_active),
                    "correction_warmup_active": bool(correction_warmup_active),
                    "temporal_state_updated": bool(needs_alternative_evidence),
                    "low_conf_count": int(low_conf_mask.sum().item()),
                    "low_conf_ratio": float(low_conf_mask.float().mean().item()),
                    "cross_view_agreement_count": cross_view_count,
                    "cross_view_agreement_ratio": (
                        float(cross_view_count) / float(num_nodes)
                    ),
                    "persistent_count": persistent_count,
                    "persistent_ratio": float(persistent_count) / float(num_nodes),
                    "correction_count": correction_count,
                    "correction_ratio": float(correction_count) / float(num_nodes),
                    "per_cluster_correction_counts": [
                        int(value)
                        for value in per_cluster_correction_counts.cpu().tolist()
                    ],
                    "relocation_target_distribution": target_distribution,
                    "old_equals_target_count": old_equals_target_count,
                    "correction_mask_subset_low_conf": bool(
                        not torch.any(correction_mask & ~low_conf_mask).item()
                    ),
                    "L_cluster_original": float(lclus_original.detach().item()),
                    "L_cluster_reweighted": float(lclus_reweighted.detach().item()),
                    "L_negative_flip": float(flip_loss.detach().item()),
                    "L_relocation": float(relocation_loss.detach().item()),
                    "correction_weight": float(flip_weight_t + relocation_weight_t),
                    "correction_to_total_loss_ratio": correction_to_total_ratio,
                    "target_probability_stats": _diagnostic_stats(
                        target_probability_values
                    ),
                    "old_probability_stats": _diagnostic_stats(old_probability_values),
                    "target_minus_old_probability_stats": _diagnostic_stats(
                        probability_margin_values
                    ),
                    "view_mean_target_probability_stats": _diagnostic_stats(
                        view_target_probability_values
                    ),
                    "temporal_persistence_steps": int(temporal_persistence_steps),
                    "relocation_min_target_prob": float(relocation_min_target_prob),
                }
            )
        metrics["cluster_threshold_diagnostics"] = diagnostics

    if return_analysis_state:
        if not stage_step2_enabled:
            raise ValueError(
                "analysis state can only be captured from an enabled Stage 2"
            )
        relocation_is_active_mode = noise_correction_mode == "soft_relocation"
        metrics["analysis_state"] = {
            "predicted_cluster": threshold_state["pseudo_label"].detach(),
            "confidence": threshold_state["conf"].detach(),
            "cluster_threshold": threshold_state["node_threshold"].detach(),
            "cluster_thresholds": threshold_state["thresholds"].detach(),
            "global_threshold": threshold_state["global_threshold"].detach(),
            "low_confidence_mask": low_conf_mask.detach(),
            "relocation_selected": (
                correction_mask.detach()
                if relocation_is_active_mode
                else torch.zeros_like(correction_mask)
            ),
            "alternative_cluster": (
                target_cluster.detach()
                if relocation_is_active_mode
                else torch.full_like(target_cluster, -1)
            ),
            "finetune_epoch": int(finetune_epoch),
            "correction_active": bool(correction_active),
            "relocation_weight_t": float(relocation_weight_t),
        }

    return total_loss, metrics
