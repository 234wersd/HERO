from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


CLUSTER_CONF_MODES = ("reweight", "soft")
FLIP_MODES = ("negative", "second_best")


def _zero_like_loss(cluster_logits: torch.Tensor) -> torch.Tensor:
    return cluster_logits.sum() * 0.0


def _stats(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    values = values.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _warmup_scale(current_epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(current_epoch)) / float(warmup_epochs))


def cluster_conf_weight_for_epoch(
    finetune_epoch: int,
    cluster_conf_warmup_epochs: int,
) -> float:
    return _warmup_scale(finetune_epoch, cluster_conf_warmup_epochs)


def threshold_flip_weight_for_epoch(
    finetune_epoch: int,
    flip_weight: float,
    flip_warmup_epochs: int,
) -> float:
    return float(flip_weight) * _warmup_scale(finetune_epoch, flip_warmup_epochs)


def _validate_config(
    cluster_conf_mode: str,
    cluster_conf_quantile: float,
    cluster_low_conf_weight: float,
    cluster_conf_min_cluster_size: int,
    cluster_conf_temperature: float,
    flip_quantile: Optional[float],
    flip_mode: str,
    flip_eps: float,
) -> None:
    if cluster_conf_mode not in CLUSTER_CONF_MODES:
        raise ValueError(
            f"cluster_conf_mode must be one of {CLUSTER_CONF_MODES}, "
            f"got {cluster_conf_mode!r}"
        )
    if not 0.0 <= float(cluster_conf_quantile) <= 1.0:
        raise ValueError(
            f"cluster_conf_quantile must be in [0, 1], got {cluster_conf_quantile}"
        )
    if not 0.0 <= float(cluster_low_conf_weight) <= 1.0:
        raise ValueError(
            "cluster_low_conf_weight must be in [0, 1], "
            f"got {cluster_low_conf_weight}"
        )
    if cluster_conf_min_cluster_size < 1:
        raise ValueError(
            "cluster_conf_min_cluster_size must be positive, "
            f"got {cluster_conf_min_cluster_size}"
        )
    if cluster_conf_temperature <= 0:
        raise ValueError(
            f"cluster_conf_temperature must be positive, got {cluster_conf_temperature}"
        )
    if flip_quantile is not None and not 0.0 <= float(flip_quantile) <= 1.0:
        raise ValueError(f"flip_quantile must be in [0, 1], got {flip_quantile}")
    if flip_mode not in FLIP_MODES:
        raise ValueError(f"flip_mode must be one of {FLIP_MODES}, got {flip_mode!r}")
    if flip_eps <= 0:
        raise ValueError(f"flip_eps must be positive, got {flip_eps}")


def compute_cluster_threshold_state(
    cluster_logits: torch.Tensor,
    cluster_conf_quantile: float = 0.2,
    cluster_conf_min_cluster_size: int = 10,
) -> Dict[str, torch.Tensor]:
    probs = F.softmax(cluster_logits, dim=1)
    num_nodes, num_clusters = probs.shape
    device = cluster_logits.device
    dtype = cluster_logits.dtype

    with torch.no_grad():
        conf, pseudo_label = probs.detach().max(dim=1)
        if conf.numel() == 0:
            global_threshold = cluster_logits.new_tensor(0.0)
        else:
            global_threshold = torch.quantile(
                conf.float(),
                float(cluster_conf_quantile),
            ).to(device=device, dtype=dtype)

        thresholds = torch.full(
            (num_clusters,),
            global_threshold.item(),
            device=device,
            dtype=dtype,
        )
        used_global_threshold = torch.ones(
            num_clusters,
            device=device,
            dtype=torch.bool,
        )
        per_cluster_counts = torch.zeros(
            num_clusters,
            device=device,
            dtype=torch.long,
        )

        for cluster_idx in range(num_clusters):
            mask_k = pseudo_label == cluster_idx
            count = int(mask_k.sum().item())
            per_cluster_counts[cluster_idx] = count
            if count < int(cluster_conf_min_cluster_size):
                continue
            threshold_k = torch.quantile(
                conf[mask_k].float(),
                float(cluster_conf_quantile),
            ).to(device=device, dtype=dtype)
            thresholds[cluster_idx] = threshold_k
            used_global_threshold[cluster_idx] = False

        node_threshold = thresholds[pseudo_label]
        low_mask = conf < node_threshold
        per_cluster_low_conf_counts = torch.bincount(
            pseudo_label[low_mask],
            minlength=num_clusters,
        )

    return {
        "probs": probs,
        "conf": conf,
        "pseudo_label": pseudo_label,
        "global_threshold": global_threshold,
        "thresholds": thresholds,
        "node_threshold": node_threshold,
        "low_mask": low_mask,
        "per_cluster_counts": per_cluster_counts,
        "per_cluster_low_conf_counts": per_cluster_low_conf_counts,
        "used_global_threshold": used_global_threshold,
    }


def _weighted_mean(loss_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum().clamp_min(1e-12)
    return (weights * loss_values).sum() / denominator


def _compute_reweighted_lclus(
    per_node_lclus: torch.Tensor,
    state: Dict[str, torch.Tensor],
    cluster_conf_mode: str,
    cluster_low_conf_weight: float,
    cluster_conf_weight_t: float,
    cluster_conf_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    conf = state["conf"]
    node_threshold = state["node_threshold"]
    low_mask = state["low_mask"]

    if cluster_conf_mode == "soft":
        soft_gate = torch.sigmoid(
            (conf - node_threshold) / float(cluster_conf_temperature)
        )
        raw_weight = (
            float(cluster_low_conf_weight)
            + (1.0 - float(cluster_low_conf_weight)) * soft_gate
        )
    else:
        raw_weight = torch.ones_like(per_node_lclus)
        raw_weight = torch.where(
            low_mask,
            raw_weight.new_full(raw_weight.shape, float(cluster_low_conf_weight)),
            raw_weight,
        )

    effective_weight = 1.0 - float(cluster_conf_weight_t) * (1.0 - raw_weight)
    effective_weight = effective_weight.detach().to(
        device=per_node_lclus.device,
        dtype=per_node_lclus.dtype,
    )
    return _weighted_mean(per_node_lclus, effective_weight), effective_weight


def _compute_flip_loss(
    cluster_logits: torch.Tensor,
    state: Dict[str, torch.Tensor],
    flip_mode: str,
    flip_eps: float,
) -> torch.Tensor:
    low_mask = state["low_mask"]
    if not low_mask.any():
        return _zero_like_loss(cluster_logits)

    probs = state["probs"]
    pseudo_label = state["pseudo_label"]
    if flip_mode == "second_best":
        if probs.shape[1] < 2:
            return _zero_like_loss(cluster_logits)
        with torch.no_grad():
            second_label = probs.detach().topk(k=2, dim=1).indices[:, 1]
        return F.cross_entropy(cluster_logits[low_mask], second_label[low_mask])

    pseudo_prob = (
        probs[low_mask]
        .gather(
            1,
            pseudo_label[low_mask].unsqueeze(1),
        )
        .squeeze(1)
    )
    return -torch.log(1.0 - pseudo_prob + float(flip_eps)).mean()


def build_cluster_threshold_diagnostics(
    cluster_logits: torch.Tensor,
    stage: str,
    use_cluster_confidence_threshold: bool,
    use_threshold_flip: bool,
    cluster_conf_mode: str,
    cluster_conf_weight_t: float,
    cluster_conf_quantile: float,
    flip_quantile: Optional[float],
    cluster_low_conf_weight: float,
    cluster_conf_temperature: float,
    cluster_conf_min_cluster_size: int,
    lclus_original: float,
    lclus_reweighted: float,
    flip_loss: float,
    flip_weight_t: float,
    total_loss: float,
) -> Dict[str, object]:
    with torch.no_grad():
        state = compute_cluster_threshold_state(
            cluster_logits=cluster_logits,
            cluster_conf_quantile=cluster_conf_quantile,
            cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
        )
        low_mask = state["low_mask"]
        num_nodes = max(1, int(low_mask.numel()))
        if use_threshold_flip:
            if flip_quantile is None:
                flip_state = state
            else:
                flip_state = compute_cluster_threshold_state(
                    cluster_logits=cluster_logits,
                    cluster_conf_quantile=flip_quantile,
                    cluster_conf_min_cluster_size=cluster_conf_min_cluster_size,
                )
            flip_low_mask = flip_state["low_mask"]
            flip_num_nodes = max(1, int(flip_low_mask.numel()))
            flip_low_conf_count = int(flip_low_mask.sum().item())
            flip_low_conf_ratio = float(flip_low_mask.float().sum().item()) / float(
                flip_num_nodes
            )
            flip_threshold_stats = _stats(flip_state["thresholds"])
        else:
            flip_low_conf_count = 0
            flip_low_conf_ratio = 0.0
            flip_threshold_stats = _stats(cluster_logits.new_empty(0))
        return {
            "stage": stage,
            "use_cluster_confidence_threshold": bool(use_cluster_confidence_threshold),
            "use_threshold_flip": bool(use_threshold_flip),
            "cluster_conf_mode": cluster_conf_mode,
            "cluster_conf_weight_t": float(cluster_conf_weight_t),
            "cluster_conf_quantile": float(cluster_conf_quantile),
            "flip_quantile": (None if flip_quantile is None else float(flip_quantile)),
            "cluster_low_conf_weight": float(cluster_low_conf_weight),
            "cluster_conf_temperature": float(cluster_conf_temperature),
            "confidence_stats": _stats(state["conf"]),
            "threshold_stats": _stats(state["thresholds"]),
            "low_conf_count": int(low_mask.sum().item()),
            "low_conf_ratio": float(low_mask.float().sum().item()) / float(num_nodes),
            "flip_low_conf_count": flip_low_conf_count,
            "flip_low_conf_ratio": flip_low_conf_ratio,
            "flip_threshold_stats": flip_threshold_stats,
            "Lclus_original": float(lclus_original),
            "Lclus_reweighted": float(lclus_reweighted),
            "flip_loss": float(flip_loss),
            "flip_weight_t": float(flip_weight_t),
            "total_loss": float(total_loss),
            "per_cluster_low_conf_counts": [
                int(value)
                for value in state["per_cluster_low_conf_counts"]
                .detach()
                .cpu()
                .tolist()
            ],
            "per_cluster_threshold": [
                float(value) for value in state["thresholds"].detach().cpu().tolist()
            ],
            "per_cluster_counts": [
                int(value)
                for value in state["per_cluster_counts"].detach().cpu().tolist()
            ],
            "per_cluster_uses_global_threshold": [
                bool(value)
                for value in state["used_global_threshold"].detach().cpu().tolist()
            ],
        }
