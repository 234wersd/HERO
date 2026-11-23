from typing import Optional

import torch
import torch.nn as nn
from torch import optim

from src.augmentation import Augmentor
from src.structural_features import HyperedgeMoEFeatureCache
from src.score_moe import HyperedgeScoreMoELoss
from src.config import NEGATIVE_SAMPLING_MODE, SCORE_MOE_START_STAGE
from src.losses import (
    compute_score_moe_calibrated_loss_with_cluster_threshold,
    init_noise_selection_state,
)
from src.sampling import (
    get_cached_negative_features,
    init_negative_feature_cache_state,
)


def train_step(
    encoder: nn.Module,
    project_head: nn.Module,
    cluster_module: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: HyperedgeScoreMoELoss,
    features: torch.Tensor,
    pos_hyperedge_index: torch.Tensor,
    negative_feature_cache_state: dict,
    feature_cache: HyperedgeMoEFeatureCache,
    epoch: int,
    stage: str,
    finetune_epoch: int,
    feature_update_interval: int,
    neg_feature_update_interval: int,
    recompute_neg_features: bool,
    force_feature_update: bool,
    features_aug1: torch.Tensor,
    hyperedge_index_aug1: torch.Tensor,
    features_aug2: torch.Tensor,
    hyperedge_index_aug2: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    tau: float,
    score_moe_start_stage: str,
    score_moe_enabled_this_stage: bool,
    step2_enabled: bool,
    score_label_mode: str,
    use_cluster_confidence_threshold: bool,
    use_threshold_flip: bool,
    disable_cluster_confidence_threshold: bool,
    cluster_conf_mode: str,
    cluster_conf_quantile: float,
    cluster_low_conf_weight: float,
    cluster_conf_min_cluster_size: int,
    cluster_conf_warmup_epochs: int,
    cluster_conf_temperature: float,
    ensemble_threshold: bool,
    ensemble_quantiles: list[float],
    ensemble_vote_ratio: float,
    ensemble_vote_rule: str,
    flip_weight: float,
    flip_quantile: Optional[float],
    flip_mode: str,
    flip_warmup_epochs: int,
    flip_eps: float,
    noise_selection_state: Optional[dict],
    noise_selection_mode: str,
    noise_correction_mode: str,
    relocation_weight: float,
    relocation_margin: float,
    relocation_min_target_prob: float,
    temporal_persistence_steps: int,
    return_score_moe_diagnostics: bool = False,
    return_cluster_threshold_diagnostics: bool = False,
    return_analysis_state: bool = False,
    subcluster_reference_embeds: Optional[torch.Tensor] = None,
) -> dict:
    encoder.train()
    project_head.train()
    cluster_module.train()
    loss_fn.train()
    optimizer.zero_grad()

    pos_node_emb, _ = encoder(features, pos_hyperedge_index)
    aug_emb_1, _ = encoder(features_aug1, hyperedge_index_aug1)
    aug_emb_2, _ = encoder(features_aug2, hyperedge_index_aug2)
    proj_emb_1 = project_head(aug_emb_1)
    proj_emb_2 = project_head(aug_emb_2)
    clustering_logits = cluster_module(pos_node_emb)
    needs_augmented_cluster_probabilities = (
        stage.lower() == "finetune"
        and step2_enabled
        and bool(use_threshold_flip)
        and not disable_cluster_confidence_threshold
        and finetune_epoch > int(flip_warmup_epochs)
        and (
            noise_selection_mode == "multi_evidence"
            or noise_correction_mode == "soft_relocation"
        )
    )
    if needs_augmented_cluster_probabilities:
        with torch.no_grad():
            view1_cluster_logits = cluster_module(aug_emb_1.detach())
            view2_cluster_logits = cluster_module(aug_emb_2.detach())
    else:
        view1_cluster_logits = None
        view2_cluster_logits = None

    with torch.no_grad():
        feature_cache.maybe_update_dynamic(
            Z=pos_node_emb,
            epoch=epoch,
            stage=stage,
            update_interval=feature_update_interval,
            force=force_feature_update,
        )
        moe_features, raw_moe_features = feature_cache.get_features()
        (
            neg_hyperedge_index,
            neg_source_edge_ids,
            cached_neg_moe_features,
            cached_neg_raw_moe_features,
            neg_sampling_diagnostics,
            neg_feature_cache_refreshed,
        ) = get_cached_negative_features(
            cache_state=negative_feature_cache_state,
            features=features,
            pos_hyperedge_index=pos_hyperedge_index,
            num_nodes=features.shape[0],
            feature_cache=feature_cache,
            node_embeds=pos_node_emb,
            epoch=epoch,
            stage=stage,
            neg_feature_update_interval=neg_feature_update_interval,
            force_feature_update=force_feature_update,
            recompute_neg_features=recompute_neg_features,
            subcluster_reference_embeds=subcluster_reference_embeds,
        )
        if recompute_neg_features:
            neg_score_moe_features = cached_neg_moe_features
            neg_score_raw_moe_features = cached_neg_raw_moe_features
        else:
            neg_score_moe_features = moe_features
            neg_score_raw_moe_features = raw_moe_features

    loss, metrics = compute_score_moe_calibrated_loss_with_cluster_threshold(
        loss_fn=loss_fn,
        pos_node_embeds=pos_node_emb,
        aug_node_embeds=proj_emb_1,
        other_aug_node_embeds=proj_emb_2,
        pos_hyperedge_index=pos_hyperedge_index,
        neg_hyperedge_index=neg_hyperedge_index,
        moe_features=moe_features,
        raw_moe_features=raw_moe_features,
        neg_moe_features=neg_score_moe_features,
        neg_raw_moe_features=neg_score_raw_moe_features,
        neg_source_edge_ids=neg_source_edge_ids,
        clustering_logits=clustering_logits,
        recompute_neg_features=recompute_neg_features,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        tau=tau,
        current_epoch=epoch,
        score_moe_start_stage=score_moe_start_stage,
        score_moe_enabled_this_stage=score_moe_enabled_this_stage,
        step2_enabled=step2_enabled,
        score_label_mode=score_label_mode,
        stage=stage,
        finetune_epoch=finetune_epoch,
        use_cluster_confidence_threshold=use_cluster_confidence_threshold,
        use_threshold_flip=use_threshold_flip,
        disable_cluster_confidence_threshold=disable_cluster_confidence_threshold,
        cluster_conf_mode=cluster_conf_mode,
        cluster_conf_quantile=cluster_conf_quantile,
        cluster_low_conf_weight=cluster_low_conf_weight,
        cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
        cluster_conf_warmup_epochs=cluster_conf_warmup_epochs,
        cluster_conf_temperature=cluster_conf_temperature,
        ensemble_threshold=ensemble_threshold,
        ensemble_quantiles=ensemble_quantiles,
        ensemble_vote_ratio=ensemble_vote_ratio,
        ensemble_vote_rule=ensemble_vote_rule,
        flip_weight=flip_weight,
        flip_quantile=flip_quantile,
        flip_mode=flip_mode,
        flip_warmup_epochs=flip_warmup_epochs,
        flip_eps=flip_eps,
        view1_cluster_logits=view1_cluster_logits,
        view2_cluster_logits=view2_cluster_logits,
        noise_selection_state=noise_selection_state,
        noise_selection_mode=noise_selection_mode,
        noise_correction_mode=noise_correction_mode,
        relocation_weight=relocation_weight,
        relocation_margin=relocation_margin,
        relocation_min_target_prob=relocation_min_target_prob,
        temporal_persistence_steps=temporal_persistence_steps,
        return_score_moe_diagnostics=return_score_moe_diagnostics,
        return_cluster_threshold_diagnostics=return_cluster_threshold_diagnostics,
        return_analysis_state=return_analysis_state,
    )
    loss.backward()
    optimizer.step()
    metrics["total_loss"] = loss.item()
    metrics["neg_feature_cache_refreshed"] = bool(neg_feature_cache_refreshed)
    metrics["neg_feature_cache_epoch"] = int(
        negative_feature_cache_state["cached_epoch"]
    )
    metrics["neg_feature_update_interval"] = int(neg_feature_update_interval)
    metrics["recompute_neg_features"] = bool(recompute_neg_features)
    metrics["negative_sampling_mode"] = NEGATIVE_SAMPLING_MODE
    metrics["negative_sampling_diagnostics"] = dict(neg_sampling_diagnostics)
    return metrics


def run_stage(
    stage_label: str,
    stage_name: str,
    start_epoch: int,
    num_epochs: int,
    clustering_gamma: float,
    encoder: nn.Module,
    project_head: nn.Module,
    cluster_module: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    loss_fn: HyperedgeScoreMoELoss,
    augmentor: Augmentor,
    features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    num_nodes: int,
    feature_cache: HyperedgeMoEFeatureCache,
    feature_update_interval: int,
    config: dict,
    subcluster_reference_embeds: Optional[torch.Tensor] = None,
) -> dict:
    if num_epochs <= 0:
        print(f"{stage_label} {stage_name} skipped")
        return {}

    first_epoch = start_epoch + 1
    last_epoch = start_epoch + num_epochs
    score_moe_start_stage = SCORE_MOE_START_STAGE
    stage_is_finetune = stage_name.lower() == "finetune"
    score_moe_enabled_this_stage = bool(config["score_moe_enabled"])
    step2_enabled = bool(config["step2_enabled"])
    cluster_threshold_diag_every = int(config["cluster_threshold_diag_every"])
    neg_feature_update_interval = int(config["neg_feature_update_interval"])
    recompute_neg_features = bool(config["recompute_neg_features"])
    score_label_mode = config["score_label_mode"]
    score_calibration_mode = config["score_calibration_mode"]
    negative_feature_cache_state = init_negative_feature_cache_state()
    noise_selection_state = (
        init_noise_selection_state()
        if (
            stage_is_finetune
            and step2_enabled
            and config["noise_selection_state_enabled"]
        )
        else None
    )
    print(f"{stage_label} {stage_name}: epochs {first_epoch}-{last_epoch}")
    final_metrics = {}
    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset + 1
        finetune_epoch = epoch_offset + 1 if stage_is_finetune else 0
        return_score_diagnostics = epoch_offset == num_epochs - 1
        return_cluster_threshold_diagnostics = (
            stage_is_finetune
            and step2_enabled
            and cluster_threshold_diag_every > 0
            and (
                finetune_epoch % cluster_threshold_diag_every == 0
                or epoch_offset == num_epochs - 1
            )
        )
        return_analysis_state = (
            bool(config.get("export_analysis_stats", False))
            and stage_is_finetune
            and step2_enabled
            and epoch_offset == num_epochs - 1
        )
        features_aug1, he_index_aug1, features_aug2, he_index_aug2 = augmentor.augment(
            features, hyperedge_index
        )
        final_metrics = train_step(
            encoder=encoder,
            project_head=project_head,
            cluster_module=cluster_module,
            optimizer=optimizer,
            loss_fn=loss_fn,
            features=features,
            pos_hyperedge_index=hyperedge_index,
            negative_feature_cache_state=negative_feature_cache_state,
            feature_cache=feature_cache,
            epoch=epoch,
            stage=stage_name,
            finetune_epoch=finetune_epoch,
            feature_update_interval=feature_update_interval,
            neg_feature_update_interval=neg_feature_update_interval,
            recompute_neg_features=recompute_neg_features,
            force_feature_update=epoch_offset == 0,
            features_aug1=features_aug1,
            hyperedge_index_aug1=he_index_aug1,
            features_aug2=features_aug2,
            hyperedge_index_aug2=he_index_aug2,
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=clustering_gamma,
            tau=config["tau"],
            score_moe_start_stage=score_moe_start_stage,
            score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            step2_enabled=step2_enabled,
            score_label_mode=score_label_mode,
            use_cluster_confidence_threshold=config["use_cluster_confidence_threshold"],
            use_threshold_flip=config["use_threshold_flip"],
            disable_cluster_confidence_threshold=config[
                "disable_cluster_confidence_threshold"
            ],
            cluster_conf_mode=config["cluster_conf_mode"],
            cluster_conf_quantile=config["cluster_conf_quantile"],
            cluster_low_conf_weight=config["cluster_low_conf_weight"],
            cluster_conf_min_cluster_size=config["cluster_conf_min_cluster_size"],
            cluster_conf_warmup_epochs=config["cluster_conf_warmup_epochs"],
            cluster_conf_temperature=config["cluster_conf_temperature"],
            ensemble_threshold=config["ensemble_threshold"],
            ensemble_quantiles=config["ensemble_quantiles"],
            ensemble_vote_ratio=config["ensemble_vote_ratio"],
            ensemble_vote_rule=config["ensemble_vote_rule"],
            flip_weight=config["flip_weight"],
            flip_quantile=config["flip_quantile"],
            flip_mode=config["flip_mode"],
            flip_warmup_epochs=config["flip_warmup_epochs"],
            flip_eps=config["flip_eps"],
            noise_selection_state=noise_selection_state,
            noise_selection_mode=config["noise_selection_mode"],
            noise_correction_mode=config["noise_correction_mode"],
            relocation_weight=config["effective_relocation_weight"],
            relocation_margin=config["relocation_margin"],
            relocation_min_target_prob=config["relocation_min_target_prob"],
            temporal_persistence_steps=config["temporal_persistence_steps"],
            return_score_moe_diagnostics=return_score_diagnostics,
            return_cluster_threshold_diagnostics=return_cluster_threshold_diagnostics,
            return_analysis_state=return_analysis_state,
            subcluster_reference_embeds=subcluster_reference_embeds,
        )
        if stage_is_finetune and epoch_offset == 0:
            prototype_state = optimizer.state.get(cluster_module.prototypes, {})
            if len(prototype_state) == 0:
                raise AssertionError(
                    "prototype optimizer state was not rebuilt after the "
                    "first Stage2 optimizer.step()"
                )
            final_metrics["prototype_optimizer_state_rebuilt_after_stage2_step"] = True
        if scheduler is not None:
            scheduler.step(final_metrics["total_loss"])

    print(
        f"{stage_label} {stage_name} end: final_loss={final_metrics['total_loss']:.6f}"
    )
    if "score_moe_diagnostics" in final_metrics:
        final_metrics["score_moe_diagnostics"][
            "score_moe_start_stage"
        ] = score_moe_start_stage
        final_metrics["score_moe_diagnostics"][
            "score_moe_enabled_this_stage"
        ] = score_moe_enabled_this_stage
    return final_metrics
