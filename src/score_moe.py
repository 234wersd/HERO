import inspect
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from src.structural_features import MOE_DIAGNOSTIC_FEATURES, MOE_FEATURE_COLUMNS


GATE_TYPES = ("bce_zscore", "margin")
SCORE_POOLING_MODES = {"mean", "mean_max"}
SCORE_LABEL_MODES = ("legacy", "label_free")
SCORE_CALIBRATION_MODES = (
    "separate_batch",
    "joint_batch",
    "none",
    "scaled_tanh",
)


def assert_label_free_scorer_interface(scorer: nn.Module) -> None:
    signature = inspect.signature(scorer.forward_label_free)
    if "labels" in signature.parameters:
        raise AssertionError("label-free scorer interface must not accept labels")


def _pool_hyperedge_embeddings(
    node_embeds: torch.Tensor,
    hyperedge_index: torch.Tensor,
    num_hyperedges: int,
    mode: str,
) -> torch.Tensor:
    if mode not in SCORE_POOLING_MODES:
        raise ValueError(
            f"score_pooling must be one of {sorted(SCORE_POOLING_MODES)}, got {mode!r}"
        )

    node_idx, edge_idx = hyperedge_index
    mean_pooled = scatter_mean(
        node_embeds[node_idx], edge_idx, dim=0, dim_size=num_hyperedges
    )
    if mode == "mean":
        return mean_pooled

    max_pooled = scatter_max(
        node_embeds[node_idx], edge_idx, dim=0, dim_size=num_hyperedges
    )[0]
    max_pooled = max_pooled.to(device=mean_pooled.device, dtype=mean_pooled.dtype)
    return torch.cat([mean_pooled, max_pooled], dim=1)


class ScalarScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hyperedge_embeds: torch.Tensor) -> torch.Tensor:
        if hyperedge_embeds.shape[-1] != self.input_dim:
            raise ValueError(
                f"scorer input dim mismatch: got {hyperedge_embeds.shape[-1]}, "
                f"expected {self.input_dim}"
            )
        return self.mlp(hyperedge_embeds).squeeze(-1)


class HyperedgeScoreMoEScorer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        feature_dim: int = len(MOE_FEATURE_COLUMNS),
        num_experts: int = 3,
        router_hidden: int = 64,
        lambda_compact_score: float = 0.03,
        lambda_mixed_score: float = 0.05,
        lambda_diffuse_score: float = 0.10,
        calibration_clip: float = 2.0,
        calibration_eps: float = 1e-6,
        score_calibration_mode: str = "separate_batch",
        use_difficulty_gate: bool = True,
        gate_type: str = "bce_zscore",
        gate_threshold: float = 0.0,
        gate_temperature: float = 1.0,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        margin_threshold: float = 1.0,
        margin_temperature: float = 1.0,
        detach_moe_residual: bool = False,
        router_temperature: float = 1.0,
        router_type: str = "hybrid",
        prior_weight: float = 0.5,
        score_pooling: str = "mean_max",
        contrastive_max_nodes: int = 0,
    ):
        super().__init__()
        if router_type not in {"learnable", "rule", "hybrid", "uniform"}:
            raise ValueError(
                "router_type must be one of "
                "['learnable', 'rule', 'hybrid', 'uniform'], "
                f"got {router_type!r}"
            )
        if router_temperature <= 0:
            raise ValueError(
                f"router_temperature must be positive, got {router_temperature}"
            )
        if num_experts != 3:
            raise ValueError("calibrated Score-MoE requires exactly 3 experts")
        if score_calibration_mode not in SCORE_CALIBRATION_MODES:
            raise ValueError(
                "score_calibration_mode must be one of "
                f"{SCORE_CALIBRATION_MODES}, got {score_calibration_mode!r}"
            )
        if calibration_clip <= 0:
            raise ValueError(
                f"calibration_clip must be positive, got {calibration_clip}"
            )
        if calibration_eps <= 0:
            raise ValueError(f"calibration_eps must be positive, got {calibration_eps}")
        if gate_type not in GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {GATE_TYPES}, got {gate_type!r}"
            )
        if gate_temperature <= 0:
            raise ValueError(
                f"gate_temperature must be positive, got {gate_temperature}"
            )
        if margin_temperature <= 0:
            raise ValueError(
                f"margin_temperature must be positive, got {margin_temperature}"
            )
        if not 0 <= gate_min <= gate_max <= 1:
            raise ValueError(
                "difficulty gate bounds must satisfy 0 <= gate_min <= gate_max <= 1"
            )
        if score_pooling not in SCORE_POOLING_MODES:
            raise ValueError(
                "score_pooling must be one of "
                f"{sorted(SCORE_POOLING_MODES)}, got {score_pooling!r}"
            )

        self.num_experts = num_experts
        self.score_pooling = score_pooling
        self.scorer_input_dim = embed_dim if score_pooling == "mean" else 2 * embed_dim
        self.register_buffer(
            "score_lambdas",
            torch.tensor(
                [
                    lambda_compact_score,
                    lambda_mixed_score,
                    lambda_diffuse_score,
                ],
                dtype=torch.float32,
            ),
        )
        self.calibration_clip = calibration_clip
        self.calibration_eps = calibration_eps
        self.score_calibration_mode = score_calibration_mode
        self.use_difficulty_gate = use_difficulty_gate
        self.gate_type = gate_type
        self.gate_threshold = gate_threshold
        self.gate_temperature = gate_temperature
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.margin_threshold = margin_threshold
        self.margin_temperature = margin_temperature
        self.detach_moe_residual = detach_moe_residual
        self.router_temperature = router_temperature
        self.router_type = router_type
        self.prior_weight = prior_weight

        self.shared_scorer = ScalarScorer(self.scorer_input_dim, hidden_dim)
        self.expert_scorers = nn.ModuleList(
            [
                ScalarScorer(self.scorer_input_dim, hidden_dim)
                for _ in range(num_experts)
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(feature_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_experts),
        )
        final_router_layer = self.router[-1]
        nn.init.zeros_(final_router_layer.weight)
        nn.init.zeros_(final_router_layer.bias)

    def rule_logits(
        self,
        moe_features: torch.Tensor,
        raw_moe_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        idx = {name: MOE_FEATURE_COLUMNS.index(name) for name in MOE_FEATURE_COLUMNS}
        z_log_size = moe_features[:, idx["log_size"]]
        z_mean_sim = moe_features[:, idx["mean_sim"]]
        z_var_sim = moe_features[:, idx["var_sim"]]
        z_scatter_norm = moe_features[:, idx["scatter_norm"]]
        z_subcluster_score_norm = moe_features[:, idx["subcluster_score_norm"]]

        if raw_moe_features is None:
            major_ratio = moe_features[:, idx["major_ratio"]]
        else:
            raw_moe_features = raw_moe_features.to(
                device=moe_features.device, dtype=moe_features.dtype
            )
            major_ratio = raw_moe_features[:, idx["major_ratio"]]

        score_compact = z_mean_sim - z_scatter_norm - 0.3 * z_log_size
        score_mixed = (
            z_subcluster_score_norm + 0.5 * z_var_sim - torch.abs(major_ratio - 0.5)
        )
        score_diffuse = z_log_size + z_scatter_norm + 0.5 * z_var_sim - z_mean_sim
        return torch.stack([score_compact, score_mixed, score_diffuse], dim=-1)

    def route(
        self,
        moe_features: torch.Tensor,
        raw_moe_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.router_type == "uniform":
            return torch.full(
                (moe_features.shape[0], self.num_experts),
                1.0 / float(self.num_experts),
                device=moe_features.device,
                dtype=moe_features.dtype,
            )
        mlp_logits = self.router(moe_features)
        if self.router_type == "learnable":
            logits = mlp_logits
        else:
            rule_logits = self.rule_logits(moe_features, raw_moe_features)
            if self.router_type == "rule":
                logits = rule_logits
            else:
                logits = self.prior_weight * rule_logits + mlp_logits
        return F.softmax(logits / self.router_temperature, dim=-1)

    def calibrate_deltas(self, raw_deltas: torch.Tensor) -> torch.Tensor:
        if self.score_calibration_mode == "scaled_tanh":
            return torch.tanh(raw_deltas / float(self.calibration_clip))
        if self.score_calibration_mode == "none":
            return torch.tanh(raw_deltas)
        delta_mean = raw_deltas.mean(dim=0, keepdim=True)
        delta_std = raw_deltas.std(dim=0, unbiased=False, keepdim=True)
        delta_norm = (raw_deltas - delta_mean) / (delta_std + self.calibration_eps)
        return delta_norm.clamp(-self.calibration_clip, self.calibration_clip)

    def compute_difficulty_gate(
        self,
        shared_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        labels = labels.to(device=shared_scores.device, dtype=shared_scores.dtype)
        bce_difficulty = F.binary_cross_entropy_with_logits(
            shared_scores.detach(),
            labels,
            reduction="none",
        )
        signed_margin = None
        if not self.use_difficulty_gate:
            difficulty_gate = torch.ones_like(bce_difficulty)
        elif self.gate_type == "margin":
            label_sign = 2.0 * labels - 1.0
            signed_margin = label_sign * shared_scores.detach()
            difficulty_gate = torch.sigmoid(
                (float(self.margin_threshold) - signed_margin)
                / float(self.margin_temperature)
            )
        else:
            bce_mean = bce_difficulty.mean()
            bce_std = bce_difficulty.std(unbiased=False)
            bce_z = (bce_difficulty - bce_mean) / (bce_std + self.calibration_eps)
            difficulty_gate = torch.sigmoid(
                (bce_z - self.gate_threshold) / self.gate_temperature
            )
        difficulty_gate = difficulty_gate.clamp(self.gate_min, self.gate_max)
        return difficulty_gate.detach(), bce_difficulty, signed_margin

    def summarize_diagnostics(
        self,
        pi: torch.Tensor,
        raw_features: torch.Tensor,
        shared_scores: torch.Tensor,
        raw_deltas: torch.Tensor,
        calibrated_deltas: torch.Tensor,
        score_delta: torch.Tensor,
        difficulty_gate: torch.Tensor,
        bce_difficulty: torch.Tensor,
        signed_margin: Optional[torch.Tensor],
        eps: float = 1e-12,
    ) -> Dict[str, object]:
        with torch.no_grad():
            pi = pi.detach().float()
            raw_features = raw_features.detach().float().to(device=pi.device)
            shared_scores = shared_scores.detach().float()
            raw_deltas = raw_deltas.detach().float()
            calibrated_deltas = calibrated_deltas.detach().float()
            score_delta = score_delta.detach().float()
            score_lambdas = self.score_lambdas.detach().float().to(device=pi.device)
            expert_contributions = pi * calibrated_deltas * score_lambdas
            difficulty_gate = difficulty_gate.detach().float()
            bce_difficulty = bce_difficulty.detach().float()
            if signed_margin is None:
                signed_margin_mean = float("nan")
                signed_margin_std = float("nan")
                signed_margin_min = float("nan")
                signed_margin_max = float("nan")
            else:
                signed_margin = signed_margin.detach().float()
                signed_margin_mean = float(signed_margin.mean().item())
                signed_margin_std = float(signed_margin.std(unbiased=False).item())
                signed_margin_min = float(signed_margin.min().item())
                signed_margin_max = float(signed_margin.max().item())

            entropy = -(pi * (pi + eps).log()).sum(dim=1)
            max_pi = pi.max(dim=1).values
            top1_experts = pi.argmax(dim=1)
            top1_counts = torch.bincount(top1_experts, minlength=pi.shape[1])
            delta_ratio = score_delta.abs() / shared_scores.abs().clamp_min(eps)
            avg_pi = [float(value) for value in pi.mean(dim=0).cpu().tolist()]
            top1_expert_counts = [int(value) for value in top1_counts.cpu().tolist()]
            router_entropy_mean = float(entropy.mean().item())
            router_entropy_std = float(entropy.std(unbiased=False).item())

            diagnostics: Dict[str, object] = {
                "avg_pi": avg_pi,
                "router_avg_probability": avg_pi,
                "routing_entropy_mean": router_entropy_mean,
                "routing_entropy_std": router_entropy_std,
                "router_entropy": router_entropy_mean,
                "max_pi_mean": float(max_pi.mean().item()),
                "top1_expert_counts": top1_expert_counts,
                "dominant_expert_counts": top1_expert_counts,
                "top1_feature_avgs": {},
                "score_pooling": self.score_pooling,
                "gate_type": self.gate_type,
                "margin_threshold": float(self.margin_threshold),
                "margin_temperature": float(self.margin_temperature),
                "gate_min": float(self.gate_min),
                "gate_max": float(self.gate_max),
                "detach_moe_residual": bool(self.detach_moe_residual),
                "score_calibration_mode": self.score_calibration_mode,
                "score_calibration_scale": float(self.calibration_clip),
                "routing_probability_row_sum_max_error": float(
                    (pi.sum(dim=1) - 1.0).abs().max().item()
                ),
                "shared_score_mean": float(shared_scores.mean().item()),
                "shared_score_std": float(shared_scores.std(unbiased=False).item()),
                "expert_score_mean": [
                    float(value) for value in raw_deltas.mean(dim=0).cpu().tolist()
                ],
                "expert_score_std": [
                    float(value)
                    for value in raw_deltas.std(dim=0, unbiased=False).cpu().tolist()
                ],
                "raw_delta_mean_per_expert": [
                    float(value) for value in raw_deltas.mean(dim=0).cpu().tolist()
                ],
                "raw_delta_std_per_expert": [
                    float(value)
                    for value in raw_deltas.std(dim=0, unbiased=False).cpu().tolist()
                ],
                "raw_delta_abs_mean_per_expert": [
                    float(value)
                    for value in raw_deltas.abs().mean(dim=0).cpu().tolist()
                ],
                "raw_delta_min_per_expert": [
                    float(value)
                    for value in raw_deltas.min(dim=0).values.cpu().tolist()
                ],
                "raw_delta_max_per_expert": [
                    float(value)
                    for value in raw_deltas.max(dim=0).values.cpu().tolist()
                ],
                "calibrated_delta_mean_per_expert": [
                    float(value)
                    for value in calibrated_deltas.mean(dim=0).cpu().tolist()
                ],
                "calibrated_delta_std_per_expert": [
                    float(value)
                    for value in calibrated_deltas.std(dim=0, unbiased=False)
                    .cpu()
                    .tolist()
                ],
                "calibrated_delta_abs_mean_per_expert": [
                    float(value)
                    for value in calibrated_deltas.abs().mean(dim=0).cpu().tolist()
                ],
                "calibrated_delta_min_per_expert": [
                    float(value)
                    for value in calibrated_deltas.min(dim=0).values.cpu().tolist()
                ],
                "calibrated_delta_max_per_expert": [
                    float(value)
                    for value in calibrated_deltas.max(dim=0).values.cpu().tolist()
                ],
                "saturation_rate_095_per_expert": [
                    float(value)
                    for value in (calibrated_deltas.abs() > 0.95)
                    .float()
                    .mean(dim=0)
                    .cpu()
                    .tolist()
                ],
                "saturation_rate_099_per_expert": [
                    float(value)
                    for value in (calibrated_deltas.abs() > 0.99)
                    .float()
                    .mean(dim=0)
                    .cpu()
                    .tolist()
                ],
                "expert_contribution_mean_per_expert": [
                    float(value)
                    for value in expert_contributions.mean(dim=0).cpu().tolist()
                ],
                "expert_contribution_abs_mean_per_expert": [
                    float(value)
                    for value in expert_contributions.abs().mean(dim=0).cpu().tolist()
                ],
                "expert_contribution_std_per_expert": [
                    float(value)
                    for value in expert_contributions.std(dim=0, unbiased=False)
                    .cpu()
                    .tolist()
                ],
                "score_delta_mean": float(score_delta.mean().item()),
                "score_delta_std": float(score_delta.std(unbiased=False).item()),
                "score_delta_abs_mean": float(score_delta.abs().mean().item()),
                "score_delta_ratio_mean": float(delta_ratio.mean().item()),
                "score_delta_ratio_max": float(delta_ratio.max().item()),
                "difficulty_gate_mean": float(difficulty_gate.mean().item()),
                "difficulty_gate_std": float(
                    difficulty_gate.std(unbiased=False).item()
                ),
                "difficulty_gate_min": float(difficulty_gate.min().item()),
                "difficulty_gate_max": float(difficulty_gate.max().item()),
                "signed_margin_mean": signed_margin_mean,
                "signed_margin_std": signed_margin_std,
                "signed_margin_min": signed_margin_min,
                "signed_margin_max": signed_margin_max,
                "bce_difficulty_mean": float(bce_difficulty.mean().item()),
                "bce_difficulty_std": float(bce_difficulty.std(unbiased=False).item()),
            }

            for feature_name in MOE_DIAGNOSTIC_FEATURES:
                feature_idx = MOE_FEATURE_COLUMNS.index(feature_name)
                values = raw_features[:, feature_idx]
                top1_avgs = []
                for expert_idx in range(pi.shape[1]):
                    mask = top1_experts == expert_idx
                    if mask.any():
                        top1_avgs.append(float(values[mask].mean().item()))
                    else:
                        top1_avgs.append(float("nan"))
                diagnostics["top1_feature_avgs"][feature_name] = top1_avgs

            return diagnostics

    def _compute_score_components(
        self,
        node_embeds: torch.Tensor,
        hyperedge_index: torch.Tensor,
        moe_features: torch.Tensor,
        raw_moe_features: Optional[torch.Tensor] = None,
        source_edge_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _, edge_idx = hyperedge_index
        num_hyperedges = edge_idx.max().item() + 1
        hyperedge_embeds = _pool_hyperedge_embeddings(
            node_embeds=node_embeds,
            hyperedge_index=hyperedge_index,
            num_hyperedges=num_hyperedges,
            mode=self.score_pooling,
        )
        if hyperedge_embeds.shape[1] != self.scorer_input_dim:
            raise ValueError(
                f"pooled scorer input dim mismatch: got {hyperedge_embeds.shape[1]}, "
                f"expected {self.scorer_input_dim}"
            )

        moe_input = (
            hyperedge_embeds.detach() if self.detach_moe_residual else hyperedge_embeds
        )
        router_features = (
            moe_features.detach() if self.detach_moe_residual else moe_features
        )
        if raw_moe_features is not None and self.detach_moe_residual:
            router_raw_features = raw_moe_features.detach()
        else:
            router_raw_features = raw_moe_features

        all_pi = self.route(router_features, router_raw_features)
        feature_values = (
            router_raw_features if router_raw_features is not None else router_features
        )
        if source_edge_ids is None:
            pi = all_pi[:num_hyperedges]
            selected_raw_features = feature_values[:num_hyperedges]
        else:
            source_edge_ids = source_edge_ids.to(
                device=edge_idx.device, dtype=torch.long
            )
            if source_edge_ids.numel() != num_hyperedges:
                raise ValueError(
                    f"source_edge_ids has {source_edge_ids.numel()} entries, "
                    f"but scorer produced {num_hyperedges} hyperedge scores"
                )
            pi = all_pi[source_edge_ids]
            selected_raw_features = feature_values[source_edge_ids]

        shared_scores = self.shared_scorer(hyperedge_embeds)
        raw_deltas = torch.stack(
            [expert(moe_input) for expert in self.expert_scorers], dim=1
        )
        calibrated_deltas = self.calibrate_deltas(raw_deltas)
        score_lambdas = self.score_lambdas.to(
            device=calibrated_deltas.device, dtype=calibrated_deltas.dtype
        )
        ungated_score_delta = (pi * calibrated_deltas * score_lambdas).sum(dim=1)
        return {
            "num_hyperedges": num_hyperedges,
            "selected_raw_features": selected_raw_features,
            "shared_scores": shared_scores,
            "raw_deltas": raw_deltas,
            "calibrated_deltas": calibrated_deltas,
            "ungated_score_delta": ungated_score_delta,
            "pi": pi,
        }

    def forward_label_free(
        self,
        node_embeds: torch.Tensor,
        hyperedge_index: torch.Tensor,
        moe_features: torch.Tensor,
        raw_moe_features: Optional[torch.Tensor] = None,
        source_edge_ids: Optional[torch.Tensor] = None,
        current_epoch: Optional[int] = None,
        score_moe_start_stage: str = "both",
        score_moe_enabled_this_stage: bool = True,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        del current_epoch
        del score_moe_start_stage
        components = self._compute_score_components(
            node_embeds=node_embeds,
            hyperedge_index=hyperedge_index,
            moe_features=moe_features,
            raw_moe_features=raw_moe_features,
            source_edge_ids=source_edge_ids,
        )
        shared_scores = components["shared_scores"]
        score_delta = components["ungated_score_delta"]
        if score_moe_enabled_this_stage:
            scores = shared_scores + score_delta
        else:
            score_delta = torch.zeros_like(shared_scores)
            scores = shared_scores

        if return_diagnostics:
            difficulty_gate = torch.ones_like(shared_scores)
            bce_difficulty = torch.zeros_like(shared_scores)
            diagnostics = self.summarize_diagnostics(
                pi=components["pi"],
                raw_features=components["selected_raw_features"],
                shared_scores=shared_scores,
                raw_deltas=components["raw_deltas"],
                calibrated_deltas=components["calibrated_deltas"],
                score_delta=score_delta,
                difficulty_gate=difficulty_gate,
                bce_difficulty=bce_difficulty,
                signed_margin=None,
            )
            diagnostics["lambda_compact_score"] = float(self.score_lambdas[0].item())
            diagnostics["lambda_mixed_score"] = float(self.score_lambdas[1].item())
            diagnostics["lambda_diffuse_score"] = float(self.score_lambdas[2].item())
            diagnostics["score_label_mode"] = "label_free"
            diagnostics["score_moe_enabled_this_stage"] = bool(
                score_moe_enabled_this_stage
            )
            diagnostics["score_pooling"] = self.score_pooling
            diagnostics["moe_score_identity_max_error"] = float(
                (scores - (shared_scores + score_delta)).detach().abs().max().item()
            )
            return scores, components["pi"], diagnostics
        return scores, components["pi"]

    def forward(
        self,
        node_embeds: torch.Tensor,
        hyperedge_index: torch.Tensor,
        moe_features: torch.Tensor,
        raw_moe_features: Optional[torch.Tensor] = None,
        source_edge_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        current_epoch: Optional[int] = None,
        score_moe_start_stage: str = "both",
        score_moe_enabled_this_stage: bool = True,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        del current_epoch
        components = self._compute_score_components(
            node_embeds=node_embeds,
            hyperedge_index=hyperedge_index,
            moe_features=moe_features,
            raw_moe_features=raw_moe_features,
            source_edge_ids=source_edge_ids,
        )
        shared_scores = components["shared_scores"]
        ungated_score_delta = components["ungated_score_delta"]
        if labels is None:
            labels = torch.ones_like(shared_scores)
        else:
            labels = labels.to(device=shared_scores.device, dtype=shared_scores.dtype)
            if labels.numel() != shared_scores.numel():
                raise ValueError(
                    f"labels has {labels.numel()} entries, "
                    f"expected {shared_scores.numel()}"
                )
        difficulty_gate, bce_difficulty, signed_margin = self.compute_difficulty_gate(
            shared_scores,
            labels,
        )
        score_delta = difficulty_gate * ungated_score_delta
        if score_moe_enabled_this_stage:
            scores = shared_scores + score_delta
        else:
            score_delta = torch.zeros_like(shared_scores)
            scores = shared_scores

        if return_diagnostics:
            diagnostics = self.summarize_diagnostics(
                pi=components["pi"],
                raw_features=components["selected_raw_features"],
                shared_scores=shared_scores,
                raw_deltas=components["raw_deltas"],
                calibrated_deltas=components["calibrated_deltas"],
                score_delta=score_delta,
                difficulty_gate=difficulty_gate,
                bce_difficulty=bce_difficulty,
                signed_margin=signed_margin,
            )
            diagnostics["lambda_compact_score"] = float(self.score_lambdas[0].item())
            diagnostics["lambda_mixed_score"] = float(self.score_lambdas[1].item())
            diagnostics["lambda_diffuse_score"] = float(self.score_lambdas[2].item())
            diagnostics["score_moe_start_stage"] = score_moe_start_stage
            diagnostics["score_moe_enabled_this_stage"] = bool(
                score_moe_enabled_this_stage
            )
            diagnostics["score_pooling"] = self.score_pooling
            diagnostics["score_label_mode"] = "legacy"
            diagnostics["moe_score_identity_max_error"] = float(
                (scores - (shared_scores + score_delta)).detach().abs().max().item()
            )
            return scores, components["pi"], diagnostics
        return scores, components["pi"]


class HyperedgeScoreMoELoss(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int = 3,
        router_hidden: int = 64,
        lambda_compact_score: float = 0.03,
        lambda_mixed_score: float = 0.05,
        lambda_diffuse_score: float = 0.10,
        calibration_clip: float = 2.0,
        calibration_eps: float = 1e-6,
        score_calibration_mode: str = "separate_batch",
        use_difficulty_gate: bool = True,
        gate_type: str = "bce_zscore",
        gate_threshold: float = 0.0,
        gate_temperature: float = 1.0,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        margin_threshold: float = 1.0,
        margin_temperature: float = 1.0,
        detach_moe_residual: bool = False,
        router_temperature: float = 1.0,
        router_type: str = "hybrid",
        prior_weight: float = 0.5,
        score_pooling: str = "mean_max",
        contrastive_max_nodes: int = 0,
    ):
        super().__init__()
        self.scorer = HyperedgeScoreMoEScorer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            router_hidden=router_hidden,
            lambda_compact_score=lambda_compact_score,
            lambda_mixed_score=lambda_mixed_score,
            lambda_diffuse_score=lambda_diffuse_score,
            calibration_clip=calibration_clip,
            calibration_eps=calibration_eps,
            score_calibration_mode=score_calibration_mode,
            use_difficulty_gate=use_difficulty_gate,
            gate_type=gate_type,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            gate_min=gate_min,
            gate_max=gate_max,
            margin_threshold=margin_threshold,
            margin_temperature=margin_temperature,
            detach_moe_residual=detach_moe_residual,
            router_temperature=router_temperature,
            router_type=router_type,
            prior_weight=prior_weight,
            score_pooling=score_pooling,
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.clustering_loss_fn = nn.CrossEntropyLoss()
        self.contrastive_max_nodes = int(contrastive_max_nodes)
        assert_label_free_scorer_interface(self.scorer)

    def compute_infonce_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        if self.contrastive_max_nodes > 0 and z1.shape[0] > self.contrastive_max_nodes:
            sampled_nodes = torch.randperm(z1.shape[0], device=z1.device)[
                : self.contrastive_max_nodes
            ]
            z1 = z1[sampled_nodes]
            z2 = z2[sampled_nodes]
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(sim_matrix.shape[0], device=z1.device)
        return (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

    def forward(
        self,
        pos_node_embeds: torch.Tensor,
        aug_node_embeds: torch.Tensor,
        other_aug_node_embeds: torch.Tensor,
        pos_hyperedge_index: torch.Tensor,
        neg_hyperedge_index: torch.Tensor,
        moe_features: torch.Tensor,
        raw_moe_features: torch.Tensor,
        neg_source_edge_ids: torch.Tensor,
        clustering_logits: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        tau: float = 0.5,
        current_epoch: Optional[int] = None,
        score_moe_start_stage: str = "both",
        score_moe_enabled_this_stage: bool = True,
        score_label_mode: str = "label_free",
        return_score_moe_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if score_label_mode not in SCORE_LABEL_MODES:
            raise ValueError(
                f"score_label_mode must be one of {SCORE_LABEL_MODES}, "
                f"got {score_label_mode!r}"
            )
        pos_diagnostics = None
        pos_num_hyperedges = int(pos_hyperedge_index[1].max().item()) + 1
        neg_num_hyperedges = int(neg_hyperedge_index[1].max().item()) + 1
        pos_labels = pos_node_embeds.new_ones(pos_num_hyperedges)
        neg_labels = pos_node_embeds.new_zeros(neg_num_hyperedges)
        if score_label_mode == "legacy":
            if return_score_moe_diagnostics:
                pos_scores, pos_pi, pos_diagnostics = self.scorer(
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
                pos_scores, pos_pi = self.scorer(
                    pos_node_embeds,
                    pos_hyperedge_index,
                    moe_features,
                    raw_moe_features,
                    labels=pos_labels,
                    current_epoch=current_epoch,
                    score_moe_start_stage=score_moe_start_stage,
                    score_moe_enabled_this_stage=score_moe_enabled_this_stage,
                )
            neg_scores, _ = self.scorer(
                pos_node_embeds,
                neg_hyperedge_index,
                moe_features,
                raw_moe_features,
                source_edge_ids=neg_source_edge_ids,
                labels=neg_labels,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            )
        else:
            if return_score_moe_diagnostics:
                pos_scores, pos_pi, pos_diagnostics = self.scorer.forward_label_free(
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
                pos_scores, pos_pi = self.scorer.forward_label_free(
                    pos_node_embeds,
                    pos_hyperedge_index,
                    moe_features,
                    raw_moe_features,
                    current_epoch=current_epoch,
                    score_moe_start_stage=score_moe_start_stage,
                    score_moe_enabled_this_stage=score_moe_enabled_this_stage,
                )
            neg_scores, _ = self.scorer.forward_label_free(
                pos_node_embeds,
                neg_hyperedge_index,
                moe_features,
                raw_moe_features,
                source_edge_ids=neg_source_edge_ids,
                current_epoch=current_epoch,
                score_moe_start_stage=score_moe_start_stage,
                score_moe_enabled_this_stage=score_moe_enabled_this_stage,
            )

        pos_loss = self.bce_loss(pos_scores, pos_labels)
        neg_loss = self.bce_loss(neg_scores, neg_labels)
        structure_loss = pos_loss + neg_loss

        infonce_loss = self.compute_infonce_loss(
            aug_node_embeds, other_aug_node_embeds, tau
        )
        with torch.no_grad():
            pseudo_labels = torch.argmax(clustering_logits, dim=1)
        clustering_loss = self.clustering_loss_fn(clustering_logits, pseudo_labels)
        total_loss = (
            alpha * structure_loss + beta * infonce_loss + gamma * clustering_loss
        )

        pos_score_mean = pos_scores.mean().item() if pos_scores.numel() > 0 else 0.0
        neg_score_mean = neg_scores.mean().item() if neg_scores.numel() > 0 else 0.0
        metrics: Dict[str, object] = {
            "structure_loss": structure_loss.item(),
            "bce_loss": structure_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "clustering_loss": clustering_loss.item(),
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "avg_pos_score": pos_score_mean,
            "avg_neg_score": neg_score_mean,
            "pos_neg_score_gap": pos_score_mean - neg_score_mean,
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
            metrics["score_moe_diagnostics"] = diagnostics

        return total_loss, metrics
