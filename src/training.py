import json
import os
import random
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
import platform
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "6"
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
parent_dir = str(REPO_ROOT)

from src.config import (
    CALIBRATION_CLIP,
    CALIBRATION_EPS,
    CLUSTER_CONF_MIN_CLUSTER_SIZE,
    CLUSTER_CONF_MODE,
    CLUSTER_CONF_QUANTILE,
    CLUSTER_CONF_TEMPERATURE,
    CLUSTER_CONF_WARMUP_EPOCHS,
    CLUSTER_LOW_CONF_WEIGHT,
    CLUSTER_THRESHOLD_DIAG_EVERY,
    CONTRASTIVE_MAX_NODES,
    DETACH_MOE_RESIDUAL,
    ENSEMBLE_QUANTILES,
    ENSEMBLE_THRESHOLD,
    ENSEMBLE_VOTE_RATIO,
    ENSEMBLE_VOTE_RULE,
    FLIP_EPS,
    FLIP_MODE,
    FLIP_QUANTILE,
    FLIP_WARMUP_EPOCHS,
    FLIP_WEIGHT,
    GATE_MAX,
    GATE_MIN,
    GATE_TEMPERATURE,
    GATE_THRESHOLD,
    GATE_TYPE,
    HERO_PRESETS,
    MARGIN_TEMPERATURE,
    MARGIN_THRESHOLD,
    MOE_FEATURE_UPDATE_INTERVAL,
    NEG_FEATURE_UPDATE_INTERVAL,
    NOISE_CORRECTION_MODE,
    NOISE_SELECTION_MODE,
    PROTOTYPE_OPTIMIZER_STATE,
    RELOCATION_MARGIN,
    RELOCATION_MIN_TARGET_PROB,
    RELOCATION_WEIGHT,
    SCORE_CALIBRATION_MODE,
    SCORE_LABEL_MODES,
    SCORE_MOE_MAX_NODES_PER_EDGE,
    SCORE_MOE_MAX_PAIRS_PER_EDGE,
    SCORE_MOE_NUM_EXPERTS,
    SCORE_MOE_PRIOR_WEIGHT,
    SCORE_MOE_ROUTER_HIDDEN,
    SCORE_MOE_ROUTER_TEMPERATURE,
    SCORE_MOE_SUBCLUSTER_K,
    SCORE_POOLING,
    SUPPORTED_DATASETS,
    TEMPORAL_PERSISTENCE_STEPS,
    USE_CLUSTER_CONFIDENCE_THRESHOLD,
    USE_DIFFICULTY_GATE,
    USE_THRESHOLD_FLIP,
    apply_ablation_config,
    parse_args,
    compute_score_lambdas,
    normalize_dataset_name,
    parse_float_list,
    validate_ensemble_config,
    validate_refined_config,
)

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from torch import optim

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.augmentation import Augmentor
from src.clustering import ClusteringPrototypes
from src.encoder import EnhancedHGNN
from src.evaluator import SimpleHypergraphEvaluator
from src.confidence import _validate_config
from src.data_loader import DatasetLoader
from src.structural_features import HyperedgeMoEFeatureCache
from src.score_moe import HyperedgeScoreMoELoss
from src.engine import run_stage
from src.sampling import build_hyperedge_members


def ensure_umap_artifact_args(args) -> None:
    if not hasattr(args, "save_umap_artifact"):
        args.save_umap_artifact = None
    if not hasattr(args, "overwrite_umap_artifact"):
        args.overwrite_umap_artifact = False


def ensure_analysis_args(args) -> None:
    if not hasattr(args, "export_analysis_stats"):
        args.export_analysis_stats = False
    if not hasattr(args, "analysis_output_dir"):
        args.analysis_output_dir = "results/hero_analysis"
    if not hasattr(args, "analysis_plot_seed"):
        args.analysis_plot_seed = 0


def save_umap_artifact(
    path,
    *,
    overwrite,
    method,
    seed,
    node_embedding,
    labels,
    predicted_labels,
    metrics,
    config,
    representation,
):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    node_embedding = node_embedding.detach().cpu()
    labels = labels.detach().cpu()
    predicted_labels = predicted_labels.detach().cpu().long()
    node_ids = torch.arange(labels.shape[0], dtype=torch.long)
    dataset_internal_name = normalize_dataset_name(str(config["dataset_name"]))
    dataset_display_name = {
        "cora": "Cora-C",
        "pubmed": "PubMed",
    }.get(dataset_internal_name, dataset_internal_name)
    num_nodes = labels.shape[0]

    assert node_embedding.ndim == 2
    assert node_embedding.shape[0] == num_nodes
    assert labels.shape == (num_nodes,)
    assert predicted_labels.shape == (num_nodes,)
    assert torch.equal(node_ids, torch.arange(num_nodes))
    assert torch.isfinite(node_embedding).all()

    artifact = {
        "method": method,
        "dataset": dataset_display_name,
        "dataset_internal_name": dataset_internal_name,
        "seed": int(seed),
        "node_embedding": node_embedding,
        "labels": labels,
        "predicted_labels": predicted_labels,
        "node_ids": node_ids,
        "metrics": {key: float(metrics[key]) for key in ["acc", "f1", "nmi", "ari"]},
        "config": dict(config),
        "representation": representation,
        "source_script": "src/training.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(artifact, path)
    print(f"Saved UMAP artifact: {path}")


def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
    encoder: nn.Module,
    cluster_module: nn.Module,
    features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    ground_truth: torch.Tensor,
    evaluator: SimpleHypergraphEvaluator,
) -> dict:
    encoder.eval()
    cluster_module.eval()
    with torch.no_grad():
        node_emb, _ = encoder(features, hyperedge_index)
        logits = cluster_module(node_emb)
        predicted_labels = torch.argmax(logits, dim=1)
    return evaluator.evaluate(predicted_labels, ground_truth)


def format_metric_line(results: dict) -> str:
    return (
        f"ACC={results['acc']:.4f}, F1={results['f1']:.4f}, "
        f"NMI={results['nmi']:.4f}, ARI={results['ari']:.4f}"
    )


def prototype_checksum(prototypes: torch.Tensor) -> float:
    return float(prototypes.detach().double().sum().item())


def optimizer_state_keys_for_param(
    optimizer: optim.Optimizer,
    param: nn.Parameter,
) -> list[str]:
    state = optimizer.state.get(param, {})
    return sorted(str(key) for key in state.keys())


def count_optimizer_state_params(
    optimizer: optim.Optimizer,
    excluded_params: Optional[set[nn.Parameter]] = None,
) -> int:
    excluded_params = set() if excluded_params is None else excluded_params
    count = 0
    for param, state in optimizer.state.items():
        if param in excluded_params:
            continue
        if len(state) > 0:
            count += 1
    return count


def handle_prototype_optimizer_state_after_kmeans(
    optimizer: optim.Optimizer,
    cluster_module: nn.Module,
    checksum_after_kmeans: float,
    prototype_state_existed_before_kmeans: bool,
    prototype_state_keys_before_reset: list[str],
    other_state_count_before: int,
) -> dict:
    cluster_params = list(cluster_module.parameters())
    cluster_param_set = set(cluster_params)
    prototype_param = cluster_module.prototypes

    for param in cluster_params:
        optimizer.state.pop(param, None)

    prototype_state_after = optimizer.state.get(prototype_param, {})
    prototype_state_exists_after = len(prototype_state_after) > 0
    other_state_count_after = count_optimizer_state_params(
        optimizer,
        excluded_params=cluster_param_set,
    )

    if prototype_state_exists_after:
        raise AssertionError(
            "prototype optimizer state must be empty after KMeans reset"
        )
    if other_state_count_after != other_state_count_before:
        raise AssertionError(
            "non-prototype optimizer state count changed during "
            "prototype reset: "
            f"before={other_state_count_before}, "
            f"after={other_state_count_after}"
        )

    diagnostics = {
        "mode": PROTOTYPE_OPTIMIZER_STATE,
        "prototype_state_existed_before_kmeans": (
            prototype_state_existed_before_kmeans
        ),
        "prototype_state_keys_before_reset": prototype_state_keys_before_reset,
        "prototype_state_exists_after_reset": prototype_state_exists_after,
        "other_state_count_before": other_state_count_before,
        "other_state_count_after": other_state_count_after,
        "prototype_checksum_after_kmeans": checksum_after_kmeans,
    }
    return diagnostics


def run_experiment(args, config: dict, run_index: int, total_runs: int) -> dict:
    start_time = time.time()
    fix_seed(config["seed"])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = False

    print(
        f"HERO | dataset={args.dataset} | seed={config['seed']} | device={device} | ablation={config['ablation']}"
    )
    print("Configuration: " + json.dumps(config, sort_keys=True))

    data = DatasetLoader().load(args.dataset, data_dir=args.data_dir).to(device)
    features, hyperedge_index, labels = data.features, data.hyperedge_index, data.labels
    node_dim = features.shape[1]
    num_nodes = features.shape[0]
    num_hyperedges = int(hyperedge_index[1].max().item()) + 1
    n_clusters = len(torch.unique(labels))
    hyperedge_members = build_hyperedge_members(hyperedge_index, num_hyperedges)
    feature_cache = HyperedgeMoEFeatureCache(
        X=features,
        hyperedge_members=hyperedge_members,
        num_nodes=num_nodes,
        seed=config["seed"],
        max_nodes_per_edge=SCORE_MOE_MAX_NODES_PER_EDGE,
        max_pairs_per_edge=SCORE_MOE_MAX_PAIRS_PER_EDGE,
        subcluster_k=SCORE_MOE_SUBCLUSTER_K,
        dataset=args.dataset,
    )

    augmentor = Augmentor(config["feat_mask_rate"], config["edge_drop_rate"])
    encoder = EnhancedHGNN(node_dim, config["emb_dim"], config["num_layers"]).to(device)

    cluster_module = ClusteringPrototypes(
        n_clusters, config["emb_dim"], config["tau"]
    ).to(device)
    project_head = nn.Sequential(
        nn.Linear(config["emb_dim"], config["emb_dim"]),
        nn.ReLU(),
        nn.Linear(config["emb_dim"], config["emb_dim"]),
    ).to(device)
    loss_fn = HyperedgeScoreMoELoss(
        embed_dim=config["emb_dim"],
        hidden_dim=config["emb_dim"],
        num_experts=SCORE_MOE_NUM_EXPERTS,
        router_hidden=SCORE_MOE_ROUTER_HIDDEN,
        lambda_compact_score=config["lambda_compact_score"],
        lambda_mixed_score=config["lambda_mixed_score"],
        lambda_diffuse_score=config["lambda_diffuse_score"],
        calibration_clip=config["score_calibration_scale"],
        calibration_eps=CALIBRATION_EPS,
        score_calibration_mode=config["score_calibration_mode"],
        use_difficulty_gate=USE_DIFFICULTY_GATE,
        gate_type=GATE_TYPE,
        gate_threshold=GATE_THRESHOLD,
        gate_temperature=GATE_TEMPERATURE,
        gate_min=GATE_MIN,
        gate_max=GATE_MAX,
        margin_threshold=MARGIN_THRESHOLD,
        margin_temperature=MARGIN_TEMPERATURE,
        detach_moe_residual=DETACH_MOE_RESIDUAL,
        router_temperature=SCORE_MOE_ROUTER_TEMPERATURE,
        router_type=config["effective_routing_mode"],
        prior_weight=SCORE_MOE_PRIOR_WEIGHT,
        score_pooling=SCORE_POOLING,
        contrastive_max_nodes=CONTRASTIVE_MAX_NODES,
    ).to(device)

    all_params = (
        list(encoder.parameters())
        + list(project_head.parameters())
        + list(loss_fn.parameters())
        + list(cluster_module.parameters())
    )
    optimizer = optim.Adam(
        all_params, lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)
    evaluator = SimpleHypergraphEvaluator()

    run_stage(
        stage_label="Stage 1",
        stage_name="Pretrain",
        start_epoch=0,
        num_epochs=config["pretrain_epochs"],
        clustering_gamma=0.0,
        encoder=encoder,
        project_head=project_head,
        cluster_module=cluster_module,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        augmentor=augmentor,
        features=features,
        hyperedge_index=hyperedge_index,
        num_nodes=num_nodes,
        feature_cache=feature_cache,
        feature_update_interval=MOE_FEATURE_UPDATE_INTERVAL,
        config=config,
    )

    encoder.eval()
    with torch.no_grad():
        initial_node_emb, _ = encoder(features, hyperedge_index)
        feature_cache.update_subcluster_once(initial_node_emb)
    cluster_param_set = set(cluster_module.parameters())
    prototype_state_before_kmeans = optimizer.state.get(
        cluster_module.prototypes,
        {},
    )
    prototype_state_existed_before_kmeans = len(prototype_state_before_kmeans) > 0
    prototype_state_keys_before_reset = optimizer_state_keys_for_param(
        optimizer,
        cluster_module.prototypes,
    )
    other_state_count_before_kmeans = count_optimizer_state_params(
        optimizer,
        excluded_params=cluster_param_set,
    )
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=config["seed"])
    kmeans.fit(initial_node_emb.cpu().numpy())
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, device=features.device
    )
    with torch.no_grad():
        cluster_module.prototypes.copy_(cluster_centers)
    if not torch.equal(cluster_module.prototypes.detach(), cluster_centers):
        raise AssertionError(
            "KMeans prototype initialization differs from cluster centers"
        )
    handle_prototype_optimizer_state_after_kmeans(
        optimizer=optimizer,
        cluster_module=cluster_module,
        checksum_after_kmeans=prototype_checksum(cluster_module.prototypes),
        prototype_state_existed_before_kmeans=(prototype_state_existed_before_kmeans),
        prototype_state_keys_before_reset=prototype_state_keys_before_reset,
        other_state_count_before=other_state_count_before_kmeans,
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] *= config["finetune_lr_scale"]

    stage2_metrics = run_stage(
        stage_label="Stage 2",
        stage_name="Finetune",
        start_epoch=config["pretrain_epochs"],
        num_epochs=config["finetune_epochs"],
        clustering_gamma=config["gamma"],
        encoder=encoder,
        project_head=project_head,
        cluster_module=cluster_module,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        augmentor=augmentor,
        features=features,
        hyperedge_index=hyperedge_index,
        num_nodes=num_nodes,
        feature_cache=feature_cache,
        feature_update_interval=MOE_FEATURE_UPDATE_INTERVAL,
        config=config,
        subcluster_reference_embeds=initial_node_emb,
    )

    final_results = evaluate(
        encoder, cluster_module, features, hyperedge_index, labels, evaluator
    )
    print(f"Final evaluation: {format_metric_line(final_results)}")
    if config["export_analysis_stats"]:
        from src.analysis import export_hero_analysis_stats

        export_hero_analysis_stats(
            output_root=config["analysis_output_dir"],
            dataset=config["dataset_name"],
            seed=config["seed"],
            feature_cache=feature_cache,
            scorer=loss_fn.scorer,
            analysis_state=stage2_metrics.get("analysis_state"),
            config=config,
            analysis_plot_seed=config["analysis_plot_seed"],
        )
    if args.save_umap_artifact:
        encoder.eval()
        cluster_module.eval()
        with torch.no_grad():
            node_embedding, _ = encoder(features, hyperedge_index)
            logits = cluster_module(node_embedding)
            predicted_labels = logits.argmax(dim=1)
        save_umap_artifact(
            args.save_umap_artifact,
            overwrite=args.overwrite_umap_artifact,
            method="HERO",
            seed=config["seed"],
            node_embedding=node_embedding,
            labels=labels,
            predicted_labels=predicted_labels,
            metrics=final_results,
            config=config,
            representation=(
                "final Stage-2 encoder output on raw "
                f"{'Cora-C' if config['dataset_name'] == 'cora' else 'PubMed' if config['dataset_name'] == 'pubmed' else config['dataset_name']} "
                "hypergraph"
            ),
        )
    print(f"Run time: {time.time() - start_time:.2f}s")
    return {"cluster_module": final_results}


def main(config_override=None, parsed_args=None):
    args = parsed_args if parsed_args is not None else parse_args()
    ensure_umap_artifact_args(args)
    ensure_analysis_args(args)
    raw_dataset = args.dataset
    args.dataset = normalize_dataset_name(args.dataset)
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"unsupported dataset '{raw_dataset}'. "
            f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
        )

    if args.moe_scale < 0:
        raise ValueError(f"moe_scale must be non-negative, got {args.moe_scale}")

    seeds = list(args.seeds)
    if args.save_umap_artifact and len(seeds) != 1:
        raise ValueError("--save_umap_artifact requires exactly one seed")

    preset_name = getattr(args, "preset", "hero_refined")
    if preset_name not in HERO_PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; expected one of {tuple(HERO_PRESETS)}"
        )

    config = {
        "dataset_name": args.dataset,
        "exp_name": args.dataset,
        "emb_dim": 512,
        "num_layers": 1,
        "lr": 5.0e-4,
        "finetune_lr_scale": 1,
        "weight_decay": 5.0e-04,
        "pretrain_epochs": 180,
        "finetune_epochs": 120,
        "seed": 42,
        "preset": preset_name,
        "ablation": getattr(args, "ablation", "full"),
        "alpha": 1,
        "beta": 1,
        "tau": 0.5,
        "gamma": 1,
        "feat_mask_rate": 0.4,
        "edge_drop_rate": 0.4,
        "score_label_mode": args.score_label_mode,
        "score_calibration_mode": SCORE_CALIBRATION_MODE,
        "score_calibration_scale": CALIBRATION_CLIP,
        "neg_feature_update_interval": NEG_FEATURE_UPDATE_INTERVAL,
        "recompute_neg_features": False,
        "use_cluster_confidence_threshold": USE_CLUSTER_CONFIDENCE_THRESHOLD,
        "use_threshold_flip": USE_THRESHOLD_FLIP,
        "disable_cluster_confidence_threshold": False,
        "cluster_conf_mode": CLUSTER_CONF_MODE,
        "cluster_conf_quantile": CLUSTER_CONF_QUANTILE,
        "cluster_low_conf_weight": CLUSTER_LOW_CONF_WEIGHT,
        "cluster_conf_min_cluster_size": CLUSTER_CONF_MIN_CLUSTER_SIZE,
        "cluster_conf_warmup_epochs": CLUSTER_CONF_WARMUP_EPOCHS,
        "cluster_conf_temperature": CLUSTER_CONF_TEMPERATURE,
        "ensemble_threshold": ENSEMBLE_THRESHOLD,
        "ensemble_quantiles": list(ENSEMBLE_QUANTILES),
        "ensemble_vote_ratio": ENSEMBLE_VOTE_RATIO,
        "ensemble_vote_rule": ENSEMBLE_VOTE_RULE,
        "flip_weight": FLIP_WEIGHT,
        "flip_quantile": FLIP_QUANTILE,
        "flip_mode": FLIP_MODE,
        "flip_warmup_epochs": FLIP_WARMUP_EPOCHS,
        "flip_eps": FLIP_EPS,
        "cluster_threshold_diag_every": CLUSTER_THRESHOLD_DIAG_EVERY,
        "noise_selection_mode": NOISE_SELECTION_MODE,
        "noise_correction_mode": NOISE_CORRECTION_MODE,
        "relocation_weight": RELOCATION_WEIGHT,
        "relocation_margin": RELOCATION_MARGIN,
        "relocation_min_target_prob": RELOCATION_MIN_TARGET_PROB,
        "temporal_persistence_steps": TEMPORAL_PERSISTENCE_STEPS,
        "moe_scale": float(args.moe_scale),
        "export_analysis_stats": bool(args.export_analysis_stats),
        "analysis_output_dir": str(args.analysis_output_dir),
        "analysis_plot_seed": int(args.analysis_plot_seed),
    }
    config.update(HERO_PRESETS[preset_name])

    cli_override_names = (
        "emb_dim",
        "lr",
        "weight_decay",
        "pretrain_epochs",
        "finetune_epochs",
        "finetune_lr_scale",
        "feat_mask_rate",
        "edge_drop_rate",
        "score_calibration_scale",
        "score_calibration_mode",
        "noise_selection_mode",
        "noise_correction_mode",
        "relocation_weight",
        "relocation_margin",
        "relocation_min_target_prob",
        "temporal_persistence_steps",
        "cluster_low_conf_weight",
        "cluster_conf_quantile",
        "disable_cluster_confidence_threshold",
        "ensemble_threshold",
        "ensemble_vote_ratio",
        "ensemble_vote_rule",
        "flip_weight",
        "flip_quantile",
        "flip_mode",
        "flip_eps",
    )
    for override_name in cli_override_names:
        override_value = getattr(args, override_name, None)
        if override_value is not None:
            config[override_name] = override_value
    cli_ensemble_quantiles = getattr(args, "ensemble_quantiles", None)
    if cli_ensemble_quantiles is not None:
        config["ensemble_quantiles"] = parse_float_list(cli_ensemble_quantiles)
    explicit_score_calibration_scale = (
        getattr(args, "score_calibration_scale", None) is not None
    )
    if config_override:
        config_override = dict(config_override)
        explicit_score_calibration_scale = explicit_score_calibration_scale or any(
            name in config_override
            for name in (
                "calibration_clip",
                "score_calibration_scale",
                "s_c",
                "sc",
            )
        )
        for alias, canonical_name in (
            ("calibration_clip", "score_calibration_scale"),
            ("s_c", "score_calibration_scale"),
            ("sc", "score_calibration_scale"),
            ("xi", "cluster_conf_quantile"),
        ):
            if alias not in config_override:
                continue
            alias_value = config_override.pop(alias)
            if canonical_name in config_override and float(
                config_override[canonical_name]
            ) != float(alias_value):
                raise ValueError(
                    f"conflicting values for {alias!r} and " f"{canonical_name!r}"
                )
            config_override[canonical_name] = alias_value
        if "dataset_name" in config_override:
            config_override["dataset_name"] = normalize_dataset_name(
                config_override["dataset_name"]
            )
        config.update(config_override)
        args.dataset = config["dataset_name"]
        config["exp_name"] = args.dataset
    if explicit_score_calibration_scale and config["score_calibration_mode"] == "none":
        raise ValueError(
            "s_c/score_calibration_scale has no effect when "
            "score_calibration_mode='none'; use --preset hero_refined or "
            "--score_calibration_mode scaled_tanh for the s_c sweep"
        )
    apply_ablation_config(config)
    if config["export_analysis_stats"] and config["ablation"] != "full":
        raise ValueError("--export_analysis_stats requires --ablation full")
    if (
        config["export_analysis_stats"]
        and not str(config["analysis_output_dir"]).strip()
    ):
        raise ValueError("--analysis_output_dir must not be empty")
    config["result_name_tag"] = (
        f"{config['dataset_name']}_{config['preset']}_{config['ablation']}"
    )
    config.update(compute_score_lambdas(config["moe_scale"]))
    config["ensemble_quantiles"] = parse_float_list(config["ensemble_quantiles"])
    if config["score_label_mode"] not in SCORE_LABEL_MODES:
        raise ValueError(
            "score_label_mode must be one of "
            f"{SCORE_LABEL_MODES}, got "
            f"{config['score_label_mode']!r}"
        )
    if int(config["neg_feature_update_interval"]) <= 0:
        raise ValueError(
            "neg_feature_update_interval must be positive, got "
            f"{config['neg_feature_update_interval']}"
        )
    for epoch_name in ("pretrain_epochs", "finetune_epochs"):
        if int(config[epoch_name]) <= 0:
            raise ValueError(f"{epoch_name} must be positive, got {config[epoch_name]}")
    if int(config["emb_dim"]) <= 0:
        raise ValueError(f"emb_dim must be positive, got {config['emb_dim']}")
    if float(config["lr"]) <= 0:
        raise ValueError(f"lr must be positive, got {config['lr']}")
    if float(config["weight_decay"]) < 0:
        raise ValueError(
            "weight_decay must be non-negative, got " f"{config['weight_decay']}"
        )
    if float(config["finetune_lr_scale"]) <= 0:
        raise ValueError(
            "finetune_lr_scale must be positive, got " f"{config['finetune_lr_scale']}"
        )
    for rate_name in ("feat_mask_rate", "edge_drop_rate"):
        if not 0.0 <= float(config[rate_name]) <= 1.0:
            raise ValueError(f"{rate_name} must be in [0, 1], got {config[rate_name]}")
    _validate_config(
        cluster_conf_mode=config["cluster_conf_mode"],
        cluster_conf_quantile=config["cluster_conf_quantile"],
        cluster_low_conf_weight=config["cluster_low_conf_weight"],
        cluster_conf_min_cluster_size=config["cluster_conf_min_cluster_size"],
        cluster_conf_temperature=config["cluster_conf_temperature"],
        flip_quantile=config["flip_quantile"],
        flip_mode=config["flip_mode"],
        flip_eps=config["flip_eps"],
    )
    validate_ensemble_config(
        ensemble_quantiles=config["ensemble_quantiles"],
        ensemble_vote_ratio=config["ensemble_vote_ratio"],
        ensemble_vote_rule=config["ensemble_vote_rule"],
    )
    validate_refined_config(config)

    all_results = {"acc": [], "f1": [], "nmi": [], "ari": []}
    for run_idx, seed in enumerate(seeds):
        print(f"\nRun {run_idx + 1}/{len(seeds)}")
        config["seed"] = seed
        run_results = run_experiment(args, config, run_idx, len(seeds))
        for metric in all_results:
            all_results[metric].append(run_results["cluster_module"][metric])

    if len(seeds) > 1:
        print("\nAggregate final evaluation:")
        for metric in ["acc", "f1", "nmi", "ari"]:
            values = np.asarray(all_results[metric], dtype=np.float64)
            print(f"  {metric.upper()}: {values.mean():.4f} +/- {values.std():.4f}")

    if all_results:
        result_dir = os.path.join(parent_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        seed_tag = "-".join(str(int(seed)) for seed in seeds)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result_filename = (
            f"{config['result_name_tag']}_seeds_{seed_tag}_{timestamp}.json"
        )
        result_path = os.path.join(result_dir, result_filename)
        aggregate = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric, values in all_results.items()
        }
        result_payload = {
            "config": dict(config),
            "environment": {
                "python": platform.python_version(),
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "torch-geometric",
                        "torch-scatter",
                        "torch-sparse",
                        "numpy",
                        "scipy",
                        "scikit-learn",
                    )
                },
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
                "torch_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_index": args.device,
            },
            "dataset": config["dataset_name"],
            "preset": config["preset"],
            "requested_ablation": config["ablation"],
            "effective_routing_mode": config["effective_routing_mode"],
            "confidence_reweighting_enabled": bool(config["cluster_reweight_enabled"]),
            "relocation_enabled": bool(config["relocation_enabled"]),
            "effective_relocation_weight": float(config["effective_relocation_weight"]),
            "seeds": [int(seed) for seed in seeds],
            "metrics": {
                metric: [float(value) for value in values]
                for metric, values in all_results.items()
            },
            "aggregate": aggregate,
        }
        with open(result_path, "x", encoding="utf-8") as result_file:
            json.dump(result_payload, result_file, indent=2, sort_keys=True)
            result_file.write("\n")
        print(f"Results file: {result_path}")

    return all_results
