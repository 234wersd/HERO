import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.structural_features import MOE_FEATURE_COLUMNS


EXPERT_NAMES = ("Compact", "Mixed", "Diffuse")
ROUTING_STATISTIC_SOURCES = {
    "cardinality": "log_size",
    "internal_similarity": "mean_sim",
    "dispersion": "scatter_norm",
    "subcluster_tendency": "subcluster_score_norm",
}


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def _write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _validate_analysis_state(analysis_state: dict) -> int:
    required = {
        "predicted_cluster",
        "confidence",
        "cluster_threshold",
        "cluster_thresholds",
        "global_threshold",
        "low_confidence_mask",
        "relocation_selected",
        "alternative_cluster",
    }
    missing = sorted(required.difference(analysis_state))
    if missing:
        raise KeyError(
            "final Stage-2 analysis state is missing fields: " + ", ".join(missing)
        )

    node_fields = (
        "predicted_cluster",
        "confidence",
        "cluster_threshold",
        "low_confidence_mask",
        "relocation_selected",
        "alternative_cluster",
    )
    lengths = {
        field: int(_as_numpy(analysis_state[field]).reshape(-1).shape[0])
        for field in node_fields
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"final Stage-2 node fields have inconsistent lengths: {lengths}"
        )
    return next(iter(lengths.values()))


def export_hero_analysis_stats(
    *,
    output_root: str,
    dataset: str,
    seed: int,
    feature_cache,
    scorer,
    analysis_state: dict,
    config: dict,
    analysis_plot_seed: int = 0,
) -> dict[str, str]:


    if str(config.get("ablation", "full")) != "full":
        raise ValueError("--export_analysis_stats is only defined for --ablation full")
    if analysis_state is None:
        raise ValueError(
            "final Stage-2 analysis state was not captured; ensure Stage 2 ran"
        )

    num_nodes = _validate_analysis_state(analysis_state)
    normalized_features, raw_features = feature_cache.get_features()
    with torch.no_grad():
        routing_probabilities = scorer.route(
            normalized_features,
            raw_features,
        )

    normalized = _as_numpy(normalized_features).astype(np.float64, copy=False)
    raw = _as_numpy(raw_features).astype(np.float64, copy=False)
    pi = _as_numpy(routing_probabilities).astype(np.float64, copy=False)
    num_hyperedges = len(feature_cache.hyperedge_members)
    expected_feature_shape = (num_hyperedges, len(MOE_FEATURE_COLUMNS))
    if (
        raw.shape != expected_feature_shape
        or normalized.shape != expected_feature_shape
    ):
        raise ValueError(
            "router feature cache shape mismatch: "
            f"raw={raw.shape}, standardized={normalized.shape}, "
            f"expected={expected_feature_shape}"
        )
    if pi.shape != (num_hyperedges, len(EXPERT_NAMES)):
        raise ValueError(
            f"routing probabilities have shape {pi.shape}; "
            f"expected {(num_hyperedges, len(EXPERT_NAMES))}"
        )
    if not np.isfinite(pi).all():
        raise ValueError("routing probabilities contain non-finite values")
    if np.any(pi < 0.0) or not np.allclose(pi.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("routing probabilities are not valid distributions")

    feature_index = {
        feature_name: index for index, feature_name in enumerate(MOE_FEATURE_COLUMNS)
    }
    routing_rows = []
    for hyperedge_id in range(num_hyperedges):
        dominant_expert = int(np.argmax(pi[hyperedge_id]))
        cardinality = int(len(feature_cache.hyperedge_members[hyperedge_id]))
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "hyperedge_id": hyperedge_id,
            "pi_compact": float(pi[hyperedge_id, 0]),
            "pi_mixed": float(pi[hyperedge_id, 1]),
            "pi_diffuse": float(pi[hyperedge_id, 2]),
            "dominant_expert": dominant_expert,
            "dominant_expert_name": EXPERT_NAMES[dominant_expert],
            "cardinality": cardinality,
            "cardinality_log1p_raw": float(
                raw[hyperedge_id, feature_index["log_size"]]
            ),
            "cardinality_standardized": float(
                normalized[hyperedge_id, feature_index["log_size"]]
            ),
        }
        for exported_name, router_feature_name in (
            ("internal_similarity", "mean_sim"),
            ("dispersion", "scatter_norm"),
            ("subcluster_tendency", "subcluster_score_norm"),
        ):
            raw_value = float(raw[hyperedge_id, feature_index[router_feature_name]])
            standardized_value = float(
                normalized[hyperedge_id, feature_index[router_feature_name]]
            )
            row[exported_name] = raw_value
            row[f"{exported_name}_raw"] = raw_value
            row[f"{exported_name}_standardized"] = standardized_value
        for feature_name, column_index in feature_index.items():
            row[f"router_raw_{feature_name}"] = float(raw[hyperedge_id, column_index])
            row[f"router_standardized_{feature_name}"] = float(
                normalized[hyperedge_id, column_index]
            )
        routing_rows.append(row)

    predicted_cluster = (
        _as_numpy(analysis_state["predicted_cluster"])
        .reshape(-1)
        .astype(np.int64, copy=False)
    )
    confidence = (
        _as_numpy(analysis_state["confidence"])
        .reshape(-1)
        .astype(np.float64, copy=False)
    )
    cluster_threshold = (
        _as_numpy(analysis_state["cluster_threshold"])
        .reshape(-1)
        .astype(np.float64, copy=False)
    )
    low_confidence_mask = (
        _as_numpy(analysis_state["low_confidence_mask"])
        .reshape(-1)
        .astype(bool, copy=False)
    )
    relocation_selected = (
        _as_numpy(analysis_state["relocation_selected"])
        .reshape(-1)
        .astype(bool, copy=False)
    )
    alternative_cluster = (
        _as_numpy(analysis_state["alternative_cluster"])
        .reshape(-1)
        .astype(np.int64, copy=False)
    )
    cluster_thresholds = (
        _as_numpy(analysis_state["cluster_thresholds"])
        .reshape(-1)
        .astype(np.float64, copy=False)
    )
    global_threshold_values = _as_numpy(analysis_state["global_threshold"]).reshape(-1)
    if global_threshold_values.size != 1:
        raise ValueError(
            "final Stage-2 global threshold must be a scalar, got "
            f"shape {global_threshold_values.shape}"
        )
    global_threshold = float(global_threshold_values[0])

    if not np.isfinite(confidence).all():
        raise ValueError("final Stage-2 confidence contains non-finite values")
    if not np.isfinite(cluster_threshold).all():
        warnings.warn(
            "final Stage-2 cluster thresholds contain non-finite values",
            RuntimeWarning,
        )
    if not np.isfinite(global_threshold):
        warnings.warn(
            "final Stage-2 global threshold is non-finite",
            RuntimeWarning,
        )
    if np.any(relocation_selected & ~low_confidence_mask):
        raise ValueError("relocation_selected must be a subset of low_confidence_mask")
    if np.any(relocation_selected & (alternative_cluster == predicted_cluster)):
        raise ValueError("a selected relocation target equals its predicted cluster")
    if np.any(relocation_selected & (alternative_cluster < 0)):
        raise ValueError("a selected relocation node has no valid alternative cluster")

    confidence_rows = [
        {
            "dataset": dataset,
            "seed": int(seed),
            "node_id": node_id,
            "predicted_cluster": int(predicted_cluster[node_id]),
            "confidence": float(confidence[node_id]),
            "q_i": float(confidence[node_id]),
            "cluster_threshold": float(cluster_threshold[node_id]),
            "t_k": float(cluster_threshold[node_id]),
            "global_threshold": global_threshold,
            "low_confidence_mask": int(low_confidence_mask[node_id]),
            "relocation_selected": int(relocation_selected[node_id]),
            "alternative_cluster": int(alternative_cluster[node_id]),
        }
        for node_id in range(num_nodes)
    ]

    output_dir = Path(output_root).expanduser() / dataset / f"seed_{int(seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    routing_path = output_dir / "routing_stats.csv"
    confidence_path = output_dir / "confidence_stats.csv"
    metadata_path = output_dir / "metadata.json"
    _write_csv(routing_path, list(routing_rows[0]), routing_rows)
    _write_csv(confidence_path, list(confidence_rows[0]), confidence_rows)

    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": int(seed),
        "preset": str(config.get("preset", "")),
        "ablation": str(config.get("ablation", "")),
        "analysis_plot_seed": int(analysis_plot_seed),
        "ground_truth_labels_used": False,
        "capture_points": {
            "routing": ("final trained router applied to the final feature cache"),
            "confidence": (
                "last Stage-2 training loss state, before its optimizer update"
            ),
        },
        "files": {
            "routing_stats": routing_path.name,
            "confidence_stats": confidence_path.name,
        },
        "row_counts": {
            "hyperedges": num_hyperedges,
            "nodes": num_nodes,
        },
        "routing": {
            "effective_routing_mode": str(config.get("effective_routing_mode", "")),
            "expert_index_to_name": {
                str(index): name for index, name in enumerate(EXPERT_NAMES)
            },
            "router_feature_columns": list(MOE_FEATURE_COLUMNS),
            "statistic_sources": {
                **ROUTING_STATISTIC_SOURCES,
                "cardinality": (
                    "len(feature_cache.hyperedge_members[e]); router uses "
                    "log_size=log1p(cardinality)"
                ),
            },
            "raw_and_standardized_values_exported": True,
            "dynamic_cache_initialized": bool(feature_cache.dynamic_initialized),
            "subcluster_cache_initialized": bool(feature_cache.subcluster_initialized),
        },
        "confidence": {
            "cluster_thresholds": [
                float(value) for value in cluster_thresholds.tolist()
            ],
            "global_threshold": global_threshold,
            "cluster_conf_quantile": float(config.get("cluster_conf_quantile", 0.0)),
            "cluster_conf_min_cluster_size": int(
                config.get("cluster_conf_min_cluster_size", 0)
            ),
            "finetune_epoch": int(analysis_state.get("finetune_epoch", 0)),
            "correction_active": bool(analysis_state.get("correction_active", False)),
            "relocation_weight_t": float(
                analysis_state.get("relocation_weight_t", 0.0)
            ),
            "low_confidence_count": int(low_confidence_mask.sum()),
            "relocation_selected_count": int(relocation_selected.sum()),
        },
    }
    _write_json(metadata_path, metadata)

    paths = {
        "routing_stats": str(routing_path),
        "confidence_stats": str(confidence_path),
        "metadata": str(metadata_path),
    }
    print(f"HERO analysis export: {output_dir}")
    return paths
