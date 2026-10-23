"""Trainable SENSA semantic and state-adaptation modules.

Only components used by the final 2.0Free predictive path are defined here.
"""

import torch
import torch.nn as nn


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class PredictiveSemanticFactorizer(nn.Module):
    """Map each frozen-Qwen event embedding to a bank of latent semantic factors."""

    def __init__(self, text_dim: int, d_model: int, n_factors: int, dropout: float):
        super().__init__()
        self.n_factors = int(n_factors)
        self.text_norm = nn.LayerNorm(int(text_dim))
        self.base = nn.Sequential(
            nn.Linear(int(text_dim), d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.factor_delta = nn.Linear(d_model, self.n_factors * d_model)
        self.factor_gate = nn.Linear(d_model, self.n_factors)
        self.factor_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.factor_delta.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.factor_delta.bias)
        nn.init.zeros_(self.factor_gate.weight)
        nn.init.zeros_(self.factor_gate.bias)

    def forward(self, event_embeddings: torch.Tensor) -> torch.Tensor:
        # [B,E,text_dim] -> [B,E,K,D]
        base = self.base(self.text_norm(event_embeddings))
        b, e, d = base.shape
        delta = self.factor_delta(base).view(b, e, self.n_factors, d)
        gate = torch.sigmoid(self.factor_gate(base)).unsqueeze(-1)
        return self.factor_norm(base.unsqueeze(2) + gate * delta)


class UniformHistoricalTextAggregator(nn.Module):
    """Timing-aware uniform aggregation over every valid historical event/factor item.

    There is no query-key retrieval, top-k selection, or NULL rejection. The aggregated
    context is shared across horizons; horizon dependence enters through the state adapter.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.timing = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(
        self,
        event_factors: torch.Tensor,
        event_timing: torch.Tensor,
        event_mask: torch.Tensor,
        horizon_count: int,
    ):
        if event_factors.ndim != 4:
            raise ValueError("event_factors must have shape [B,E,K,D]")
        b, e, k, d = event_factors.shape
        if event_timing.shape != (b, e, 5):
            raise ValueError("event_timing must have shape [B,E,5]")
        if event_mask.shape != (b, e):
            raise ValueError("event_mask must have shape [B,E]")
        if horizon_count <= 0:
            raise ValueError("horizon_count must be positive")

        items = event_factors + self.timing(event_timing).unsqueeze(2)
        flat = items.reshape(b, e * k, d)
        flat_mask = event_mask.unsqueeze(-1).expand(-1, -1, k).reshape(b, e * k)
        values = self.value(flat)
        valid = flat_mask.to(values.dtype).unsqueeze(-1)
        pooled = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        any_event = flat_mask.any(dim=1, keepdim=True)
        context_one = any_event.to(values.dtype) * self.output(pooled)
        context = context_one.unsqueeze(1).expand(-1, int(horizon_count), -1)
        semantic_mass = (
            any_event.unsqueeze(1)
            .expand(-1, int(horizon_count), -1)
            .to(context.dtype)
        )
        return context, semantic_mass


class MemoryFreeConditionalStateAdapter(nn.Module):
    """Produce a bounded latent-state correction and an intervention gate."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        adapter_scale: float,
        gate_max: float,
        gate_bias: float,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.adapter_scale = float(adapter_scale)
        self.gate_max = float(gate_max)
        if not 0.0 < self.gate_max <= 1.0:
            raise ValueError("gate_max must be in (0,1]")

        self.condition = nn.Sequential(
            nn.Linear(self.d_model * 5, self.d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 3, self.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(self.d_model * 2),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.d_model * 2 + 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        _zero_linear(self.delta_head[-1])
        _zero_linear(self.gate_head[-1])
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

    def forward(
        self,
        base_context,
        horizon_query,
        stats_embedding,
        event_context,
        regime_embedding,
        semantic_mass,
        candidate_disagreement,
    ):
        core = self.condition(
            torch.cat(
                (
                    base_context,
                    horizon_query,
                    stats_embedding,
                    event_context,
                    regime_embedding,
                ),
                dim=-1,
            )
        )
        raw_delta = self.delta_head(core)
        delta = self.adapter_scale * torch.tanh(
            raw_delta / max(self.adapter_scale, 1e-6)
        )
        gate_features = torch.cat(
            (core, semantic_mass, candidate_disagreement), dim=-1
        )
        raw_probability = torch.sigmoid(self.gate_head(gate_features))
        effective_probability = raw_probability
        alpha = self.gate_max * effective_probability
        return delta, alpha, raw_probability, effective_probability
