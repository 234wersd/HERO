import argparse
import math


SCORE_LABEL_MODES = ("label_free", "legacy")
SCORE_CALIBRATION_MODES = (
    "separate_batch",
    "joint_batch",
    "none",
    "scaled_tanh",
)
NOISE_SELECTION_MODES = (
    "legacy_quantile_vote",
    "single_quantile",
    "multi_evidence",
)
NOISE_CORRECTION_MODES = (
    "negative_top_penalty",
    "soft_relocation",
)
ABLATION_MODES = (
    "step1_only",
    "step2_only",
    "uniform_routing",
    "no_relocation",
    "full",
)

SUPPORTED_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "cora_coauthor",
    "dblp_coauthor",
    "zoo",
    "20newsW100",
    "Mushroom",
    "NTU2012",
    "ModelNet40",
    "yelpRestaurant",
]

DATASET_ALIASES = {
    "cora": "cora",
    "cora-c": "cora",
    "cora_c": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "cora-a": "cora_coauthor",
    "cora_a": "cora_coauthor",
    "cora_coauthor": "cora_coauthor",
    "dblp": "dblp_coauthor",
    "dblp_coauthor": "dblp_coauthor",
    "zoo": "zoo",
    "20newsw100": "20newsW100",
    "mushroom": "Mushroom",
    "ntu2012": "NTU2012",
    "modelnet40": "ModelNet40",
    "yelprestaurant": "yelpRestaurant",
}

SCORE_MOE_NUM_EXPERTS = 3
SCORE_MOE_ROUTER_HIDDEN = 64
BASE_LAMBDA_COMPACT = 0.03
BASE_LAMBDA_MIXED = 0.05
BASE_LAMBDA_DIFFUSE = 0.10
CALIBRATION_CLIP = 2.0
CALIBRATION_EPS = 1e-6
USE_DIFFICULTY_GATE = True
GATE_TYPE = "bce_zscore"
GATE_THRESHOLD = 0.0
GATE_TEMPERATURE = 1.0
GATE_MIN = 0.0
GATE_MAX = 1.0
MARGIN_THRESHOLD = 1.0
MARGIN_TEMPERATURE = 1.0
DETACH_MOE_RESIDUAL = False
SCORE_MOE_ROUTER_TEMPERATURE = 1.0
SCORE_MOE_PRIOR_WEIGHT = 0.5
SCORE_CALIBRATION_MODE = "none"
SCORE_POOLING = "mean_max"
ROUTER_TYPE = "hybrid"
SCORE_MOE_START_STAGE = "both"
MOE_FEATURE_UPDATE_INTERVAL = 10
NEG_FEATURE_UPDATE_INTERVAL = 5
NEGATIVE_SAMPLING_MODE = "valid_log"
PROTOTYPE_OPTIMIZER_STATE = "reset_after_kmeans"
CONTRASTIVE_MAX_NODES = 4096
SCORE_MOE_MAX_NODES_PER_EDGE = 256
SCORE_MOE_MAX_PAIRS_PER_EDGE = 4096
SCORE_MOE_SUBCLUSTER_K = 2

USE_CLUSTER_CONFIDENCE_THRESHOLD = True
USE_THRESHOLD_FLIP = True
CLUSTER_CONF_MODE = "reweight"
CLUSTER_CONF_QUANTILE = 0.2
CLUSTER_LOW_CONF_WEIGHT = 0.5
CLUSTER_CONF_MIN_CLUSTER_SIZE = 10
CLUSTER_CONF_WARMUP_EPOCHS = 20
CLUSTER_CONF_TEMPERATURE = 0.05
ENSEMBLE_THRESHOLD = True
ENSEMBLE_QUANTILES = [0.1, 0.2, 0.3]
ENSEMBLE_VOTE_RATIO = 0.5
ENSEMBLE_VOTE_RULE = "ceil"
FLIP_WEIGHT = 0.01
FLIP_QUANTILE = None
FLIP_MODE = "negative"
FLIP_WARMUP_EPOCHS = 20
FLIP_EPS = 1e-6
CLUSTER_THRESHOLD_DIAG_EVERY = 20
NOISE_SELECTION_MODE = "legacy_quantile_vote"
NOISE_CORRECTION_MODE = "negative_top_penalty"
RELOCATION_WEIGHT = 0.01
RELOCATION_MARGIN = 0.05
RELOCATION_MIN_TARGET_PROB = 0.0
TEMPORAL_PERSISTENCE_STEPS = 2

HERO_PRESETS = {
    "hero_refined": {
        "score_calibration_mode": "scaled_tanh",
        "score_calibration_scale": 4.0,
        "noise_selection_mode": "multi_evidence",
        "noise_correction_mode": "soft_relocation",
        "relocation_weight": 0.01,
        "relocation_margin": 0.05,
        "relocation_min_target_prob": 0.0,
        "temporal_persistence_steps": 2,
    },
}


def parse_float_list(values) -> list[float]:
    if isinstance(values, str):
        values = [values]
    parsed = []
    for value in values:
        parsed.extend(
            float(item.strip()) for item in str(value).split(",") if item.strip()
        )
    return parsed


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HERO: two-stage attributed hypergraph clustering"
    )
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA index; uses CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root (default: data/dataset inside this repository).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "--score_label_mode",
        type=str,
        default="label_free",
        choices=SCORE_LABEL_MODES,
    )
    parser.add_argument("--moe_scale", type=float, default=2.0)
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=None,
        help="Override the configured embedding dimension.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the configured Stage-1 learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Override the configured optimizer weight decay.",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=None,
        help="Override the configured pretraining epoch count.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=None,
        help="Override the configured fine-tuning epoch count.",
    )
    parser.add_argument(
        "--finetune_lr_scale",
        type=float,
        default=None,
        help=(
            "Override the configured fine-tuning learning-rate scale. "
            "The effective Stage-2 learning rate is lr * finetune_lr_scale."
        ),
    )
    parser.add_argument(
        "--feat_mask_rate",
        type=float,
        default=None,
        help="Override the configured feature masking rate.",
    )
    parser.add_argument(
        "--edge_drop_rate",
        type=float,
        default=None,
        help="Override the configured incidence dropout rate.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=ABLATION_MODES,
        default="full",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="hero_refined",
        choices=tuple(HERO_PRESETS),
    )
    parser.add_argument(
        "--score_calibration_scale",
        "--calibration_clip",
        "--s_c",
        "--s-c",
        "--sc",
        dest="score_calibration_scale",
        type=float,
        default=None,
        help=(
            "Score calibration scale s_c. For the HERO sensitivity sweep, "
            "use with --preset hero_refined (scaled_tanh calibration)."
        ),
    )
    parser.add_argument(
        "--score_calibration_mode",
        type=str,
        choices=SCORE_CALIBRATION_MODES,
        default=None,
    )
    parser.add_argument(
        "--noise_selection_mode",
        type=str,
        choices=NOISE_SELECTION_MODES,
        default=None,
    )
    parser.add_argument(
        "--noise_correction_mode",
        type=str,
        choices=NOISE_CORRECTION_MODES,
        default=None,
    )
    parser.add_argument("--relocation_weight", type=float, default=None)
    parser.add_argument("--relocation_margin", type=float, default=None)
    parser.add_argument("--relocation_min_target_prob", type=float, default=None)
    parser.add_argument("--temporal_persistence_steps", type=int, default=None)
    parser.add_argument("--cluster_low_conf_weight", type=float, default=None)
    parser.add_argument(
        "--cluster_conf_quantile",
        "--xi",
        dest="cluster_conf_quantile",
        type=float,
        default=None,
        help="Cluster-confidence quantile xi; must be in [0, 1].",
    )
    parser.add_argument(
        "--disable_cluster_confidence_threshold",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--ensemble_threshold",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument("--ensemble_quantiles", nargs="+", default=None)
    parser.add_argument("--ensemble_vote_ratio", type=float, default=None)
    parser.add_argument(
        "--ensemble_vote_rule",
        type=str,
        choices=(ENSEMBLE_VOTE_RULE,),
        default=None,
    )
    parser.add_argument("--flip_weight", type=float, default=None)
    parser.add_argument("--flip_quantile", type=float, default=None)
    parser.add_argument(
        "--flip_mode",
        type=str,
        choices=("negative", "second_best"),
        default=None,
    )
    parser.add_argument("--flip_eps", type=float, default=None)
    return parser


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip()
    return DATASET_ALIASES.get(key, DATASET_ALIASES.get(key.lower(), key))


def compute_score_lambdas(moe_scale: float) -> dict:
    if moe_scale < 0:
        raise ValueError(f"moe_scale must be non-negative, got {moe_scale}")
    return {
        "lambda_compact_score": BASE_LAMBDA_COMPACT * float(moe_scale),
        "lambda_mixed_score": BASE_LAMBDA_MIXED * float(moe_scale),
        "lambda_diffuse_score": BASE_LAMBDA_DIFFUSE * float(moe_scale),
    }


def apply_ablation_config(config: dict) -> None:
    """Resolve component-ablation flags after preset and CLI overrides."""
    ablation = config.get("ablation", "full")
    if ablation not in ABLATION_MODES:
        raise ValueError(f"ablation must be one of {ABLATION_MODES}, got {ablation!r}")

    config["score_moe_enabled"] = ablation != "step2_only"
    config["step2_enabled"] = ablation != "step1_only"
    config["effective_routing_mode"] = (
        "uniform" if ablation == "uniform_routing" else ROUTER_TYPE
    )
    config["noise_selection_state_enabled"] = ablation != "no_relocation"

    if ablation == "step1_only":
        config["use_cluster_confidence_threshold"] = False
        config["use_threshold_flip"] = False
        config["disable_cluster_confidence_threshold"] = True
        config["relocation_weight"] = 0.0
    elif ablation == "step2_only":
        # step2_only denotes the complete refined Step2 mechanism.
        config["use_cluster_confidence_threshold"] = True
        config["use_threshold_flip"] = True
        config["disable_cluster_confidence_threshold"] = False
        config["noise_selection_mode"] = "multi_evidence"
        config["noise_correction_mode"] = "soft_relocation"
    elif ablation == "no_relocation":
        # Keep cluster-specific confidence thresholding and L_clus reweighting,
        # but disable every correction/relocation path.
        config["use_threshold_flip"] = False

    disable_cluster_threshold = bool(
        config.get("disable_cluster_confidence_threshold", False)
    )
    if disable_cluster_threshold:
        config["use_cluster_confidence_threshold"] = False
        config["use_threshold_flip"] = False

    config["cluster_threshold_active"] = bool(
        config["step2_enabled"]
        and config["use_cluster_confidence_threshold"]
        and not disable_cluster_threshold
    )
    config["cluster_reweight_enabled"] = bool(
        config["cluster_threshold_active"]
        and float(config["cluster_low_conf_weight"]) != 1.0
    )
    config["relocation_enabled"] = bool(
        config["step2_enabled"]
        and not disable_cluster_threshold
        and config["use_threshold_flip"]
        and config["noise_correction_mode"] == "soft_relocation"
        and float(config["relocation_weight"]) > 0.0
    )
    config["effective_relocation_weight"] = (
        float(config["relocation_weight"]) if config["relocation_enabled"] else 0.0
    )


def validate_ensemble_config(
    ensemble_quantiles: list[float],
    ensemble_vote_ratio: float,
    ensemble_vote_rule: str = ENSEMBLE_VOTE_RULE,
) -> None:
    for quantile in ensemble_quantiles:
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError(f"ensemble quantiles must be in [0, 1], got {quantile}")
    if not 0.0 < float(ensemble_vote_ratio) <= 1.0:
        raise ValueError(
            "ensemble_vote_ratio must be in (0, 1], " f"got {ensemble_vote_ratio}"
        )
    if ensemble_vote_rule != ENSEMBLE_VOTE_RULE:
        raise ValueError(
            "ensemble_vote_rule is fixed to "
            f"{ENSEMBLE_VOTE_RULE!r}, got {ensemble_vote_rule!r}"
        )


def validate_refined_config(config: dict) -> None:
    ablation = config.get("ablation", "full")
    if ablation not in ABLATION_MODES:
        raise ValueError(f"ablation must be one of {ABLATION_MODES}, got {ablation!r}")
    if config["score_calibration_mode"] not in SCORE_CALIBRATION_MODES:
        raise ValueError(
            "score_calibration_mode must be one of "
            f"{SCORE_CALIBRATION_MODES}, got "
            f"{config['score_calibration_mode']!r}"
        )
    if float(config["score_calibration_scale"]) <= 0:
        raise ValueError("score_calibration_scale must be positive")
    if config["noise_selection_mode"] not in NOISE_SELECTION_MODES:
        raise ValueError(
            "noise_selection_mode must be one of "
            f"{NOISE_SELECTION_MODES}, got {config['noise_selection_mode']!r}"
        )
    if config["noise_correction_mode"] not in NOISE_CORRECTION_MODES:
        raise ValueError(
            "noise_correction_mode must be one of "
            f"{NOISE_CORRECTION_MODES}, got {config['noise_correction_mode']!r}"
        )
    if float(config["relocation_weight"]) < 0:
        raise ValueError("relocation_weight must be non-negative")
    if float(config["relocation_margin"]) < 0:
        raise ValueError("relocation_margin must be non-negative")
    if not 0.0 <= float(config["relocation_min_target_prob"]) <= 1.0:
        raise ValueError("relocation_min_target_prob must be in [0, 1]")
    if int(config["temporal_persistence_steps"]) < 1:
        raise ValueError("temporal_persistence_steps must be at least 1")
    if config.get("disable_cluster_confidence_threshold", False):
        if config.get("cluster_threshold_active", True):
            raise ValueError(
                "disabled cluster confidence threshold cannot remain active"
            )
        if config.get("use_cluster_confidence_threshold", True):
            raise ValueError(
                "disabled cluster confidence threshold cannot reweight nodes"
            )
        if config.get("use_threshold_flip", True):
            raise ValueError(
                "disabled cluster confidence threshold cannot apply correction"
            )
        if float(config.get("effective_relocation_weight", 0.0)) != 0.0:
            raise ValueError(
                "effective relocation weight must be zero when threshold is disabled"
            )

    if ablation == "step1_only":
        if not config.get("score_moe_enabled", False):
            raise ValueError("step1_only must keep Score-MoE enabled")
        if config.get("step2_enabled", True):
            raise ValueError("step1_only must disable Step2")
        if config.get("cluster_threshold_active", True):
            raise ValueError("step1_only must disable cluster thresholding")
        if config.get("cluster_reweight_enabled", True):
            raise ValueError("step1_only must disable cluster reweighting")
        if config.get("relocation_enabled", True):
            raise ValueError("step1_only must disable relocation")
        if float(config.get("relocation_weight", 0.0)) != 0.0:
            raise ValueError("step1_only relocation_weight must be zero")
        if float(config.get("effective_relocation_weight", 0.0)) != 0.0:
            raise ValueError("step1_only effective relocation weight must be zero")
    elif ablation == "step2_only":
        if config.get("score_moe_enabled", True):
            raise ValueError("step2_only must disable Score-MoE")
        if not config.get("step2_enabled", False):
            raise ValueError("step2_only must enable Step2")
        if not config.get("cluster_threshold_active", False):
            raise ValueError("step2_only must enable cluster thresholding")
        if not config.get("cluster_reweight_enabled", False):
            raise ValueError("step2_only requires effective low-confidence reweighting")
        if config["noise_selection_mode"] != "multi_evidence":
            raise ValueError("step2_only must use multi_evidence selection")
        if config["noise_correction_mode"] != "soft_relocation":
            raise ValueError("step2_only must use soft_relocation correction")
        if not config.get("relocation_enabled", False):
            raise ValueError("step2_only must enable relocation")
    elif ablation == "uniform_routing":
        if not config.get("score_moe_enabled", False):
            raise ValueError("uniform_routing must keep Score-MoE enabled")
        if not config.get("step2_enabled", False):
            raise ValueError("uniform_routing must keep Step2 enabled")
        if config.get("effective_routing_mode") != "uniform":
            raise ValueError("uniform_routing must use uniform routing")
    elif ablation == "no_relocation":
        if not config.get("score_moe_enabled", False):
            raise ValueError("no_relocation must keep Score-MoE enabled")
        if not config.get("step2_enabled", False):
            raise ValueError("no_relocation must keep Step2 enabled")
        if not config.get("cluster_threshold_active", False):
            raise ValueError("no_relocation must keep cluster thresholding enabled")
        if not config.get("cluster_reweight_enabled", False):
            raise ValueError("no_relocation must keep confidence reweighting enabled")
        if config.get("relocation_enabled", True):
            raise ValueError("no_relocation must disable relocation")
        if config.get("noise_selection_state_enabled", True):
            raise ValueError("no_relocation must disable temporal state")
        if config.get("use_threshold_flip", True):
            raise ValueError("no_relocation must disable correction selection")
        if float(config.get("effective_relocation_weight", -1.0)) != 0.0:
            raise ValueError("no_relocation effective relocation weight must be zero")
    else:
        if not config.get("score_moe_enabled", False):
            raise ValueError("full ablation mode must enable Score-MoE")
        if not config.get("step2_enabled", False):
            raise ValueError("full ablation mode must enable Step2")
        if config.get("effective_routing_mode") != ROUTER_TYPE:
            raise ValueError(
                "full ablation mode must preserve the default routing mode"
            )


def compute_required_ensemble_votes(
    ensemble_vote_ratio: float,
    num_quantiles: int,
    ensemble_vote_rule: str,
) -> int:
    if num_quantiles <= 0:
        return 0
    if ensemble_vote_rule != ENSEMBLE_VOTE_RULE:
        raise ValueError(
            "ensemble_vote_rule is fixed to "
            f"{ENSEMBLE_VOTE_RULE!r}, got {ensemble_vote_rule!r}"
        )
    vote_value = float(ensemble_vote_ratio) * int(num_quantiles)
    required_votes = math.ceil(vote_value)
    return min(int(num_quantiles), max(1, required_votes))


def parse_args(argv=None):
    parser = build_arg_parser()
    parser.add_argument("--save_umap_artifact", type=str, default=None)
    parser.add_argument("--overwrite_umap_artifact", action="store_true")
    parser.add_argument("--export_analysis_stats", action="store_true")
    parser.add_argument("--analysis_output_dir", type=str, default="results/hero_analysis")
    parser.add_argument("--analysis_plot_seed", type=int, default=0)
    return parser.parse_args(argv)
