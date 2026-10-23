"""Final SENSA 2.0Free model.

This module contains only the predictive path used by the final architecture:
frozen numerical backbone, semantic factorization, timing-aware uniform text
aggregation, numerical-regime conditioning, intervention gate, and shared readout.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from configs.data_contract import EXPECTED_DATA_HASHES
from layers.sensa_modules import (
    MemoryFreeConditionalStateAdapter,
    PredictiveSemanticFactorizer,
    UniformHistoricalTextAggregator,
)
from models.numerical_backbone import Model as FrozenNumericalModel
from utils.text_cache import FrozenTextCache, build_event_batch


class SENSA(nn.Module):
    """SENSA model for one Time-MMD domain and prediction horizon."""

    def __init__(self, configs, load_numerical_checkpoint=True):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.label_len = int(configs.label_len)
        self.domain = str(configs.domain)
        self.frequency_name = str(configs.frequency_name)
        self.frozen_numerical_seed = int(configs.frozen_numerical_seed)
        self.frozen_numerical_root = str(configs.frozen_numerical_root)
        self._deployment_scale = float(getattr(configs, "deployment_scale", 1.0))
        self.total_correction_clip = float(configs.total_correction_clip)
        self.max_events = int(configs.max_events)

        d_model = int(configs.d_model)
        text_dim = int(configs.text_dim)
        dropout = float(configs.umsa_dropout)

        self.backbone = FrozenNumericalModel(configs)
        self._backbone_checkpoint_loaded = False
        if load_numerical_checkpoint:
            self._load_frozen_backbone()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        package_root = Path(__file__).resolve().parents[1]
        cache_dir = Path(str(configs.text_cache_dir)).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = package_root / cache_dir
        cache_path = cache_dir / (self.domain + ".last4.npz")
        meta_path = cache_path.with_suffix(".json")
        self.text_cache = FrozenTextCache(cache_path, expected_dim=text_dim)
        if not meta_path.is_file():
            raise FileNotFoundError("Frozen text cache metadata missing: {}".format(meta_path))

        def _sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        self.text_cache_signature = "sha256:" + hashlib.sha256(
            json.dumps(
                {"npz": _sha256(cache_path), "json": _sha256(meta_path)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        cache_meta = self.text_cache.metadata
        expected_csv_hash = EXPECTED_DATA_HASHES[
            "{}/{}.csv".format(self.domain, self.domain)
        ]
        required_contract = {
            "hidden_size": text_dim,
            "pooling": "mean_of_last4_layers_over_fact_tokens_only",
            "uses_fact": True,
            "uses_preds": False,
            "data_sha256": expected_csv_hash,
        }
        for key, value in required_contract.items():
            if cache_meta.get(key) != value:
                raise RuntimeError(
                    "Frozen text cache contract mismatch {}: {!r} != {!r} ({})".format(
                        key, cache_meta.get(key), value, cache_path
                    )
                )

        self.semantic_factorizer = PredictiveSemanticFactorizer(
            text_dim=text_dim,
            d_model=d_model,
            n_factors=int(configs.semantic_factors),
            dropout=dropout,
        )
        self.text_aggregator = UniformHistoricalTextAggregator(d_model=d_model)
        self.regime_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.adapter = MemoryFreeConditionalStateAdapter(
            d_model=d_model,
            dropout=dropout,
            adapter_scale=float(configs.adapter_scale),
            gate_max=float(configs.semantic_gate_max),
            gate_bias=float(configs.semantic_gate_bias),
        )

    def _load_frozen_backbone(self):
        frozen_root = Path(self.frozen_numerical_root).expanduser().resolve()
        checkpoint = frozen_root / "checkpoints_MWP_numerical" / (
            "MWP_numerical_Maoweipu_{}_pl{}_seed{}.pth".format(
                self.domain, self.pred_len, self.frozen_numerical_seed
            )
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Required frozen numerical checkpoint is missing: {}".format(checkpoint)
            )
        payload = torch.load(str(checkpoint), map_location="cpu")
        required = {
            "model_state", "model", "domain", "frequency", "seq_len",
            "label_len", "pred_len", "seed", "data_version",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise RuntimeError(
                "Frozen numerical checkpoint missing metadata {}".format(missing)
            )
        expected = {
            "model": "MWP_numerical_Maoweipu",
            "domain": self.domain,
            "frequency": self.frequency_name,
            "seq_len": self.seq_len,
            "label_len": self.label_len,
            "pred_len": self.pred_len,
            "seed": self.frozen_numerical_seed,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    "Frozen numerical checkpoint identity mismatch {}: {} != {}".format(
                        key, payload.get(key), value
                    )
                )
        self.backbone.load_state_dict(payload["model_state"], strict=True)
        self.frozen_numerical_data_version = str(payload["data_version"])
        self._backbone_checkpoint_loaded = True

    def set_deployment_scale(self, value):
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("deployment scale must be in [0,1]")
        self._deployment_scale = value

    @property
    def deployment_scale(self):
        return float(self._deployment_scale)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _future_mark(self, y_mark):
        return self.backbone._future_mark(y_mark)

    @torch.no_grad()
    def _frozen_components(self, x_num, y_mark):
        if x_num.ndim != 3 or tuple(x_num.shape[1:]) != (self.seq_len, 1):
            raise ValueError("Expected x_num [B, {}, 1]".format(self.seq_len))
        future_mark = self._future_mark(y_mark)
        state = self.backbone.state(x_num, self.pred_len)
        state = self.backbone._apply_shock_anchor(x_num, state)

        base_tokens = self.backbone.state_encoder(state["history"])
        base_residual, base_context, horizon_query = self.backbone.state_decoder(
            base_tokens, self.pred_len, future_mark
        )
        base_prediction = state["future_base"] + base_residual
        robust_scale = self.backbone.robust_relative.robust_scale(x_num).clamp_max(
            self.backbone.robust_relative.scale_cap
        )
        stats_embedding = self.backbone._stats_embedding(state, robust_scale)

        (
            seasonal_candidate,
            seasonal_gate,
            seasonal_probability,
            seasonal_context,
            seasonal_query,
        ) = self.backbone.seasonal_memory(
            base_tokens,
            base_context,
            stats_embedding,
            self.pred_len,
            y_mark,
            future_mark,
        )
        (
            relative_candidate,
            relative_gate,
            relative_probability,
            relative_context,
            relative_query,
            relative_scale,
        ) = self.backbone.robust_relative(
            x_num,
            state,
            base_context,
            stats_embedding,
            self.pred_len,
            future_mark,
        )
        total = seasonal_gate * seasonal_candidate + relative_gate * relative_candidate
        total = self.total_correction_clip * torch.tanh(
            total / max(self.total_correction_clip, 1e-6)
        )
        numerical_prediction = base_prediction + total
        return {
            "state": state,
            "base_tokens": base_tokens,
            "base_residual": base_residual,
            "base_context": base_context,
            "horizon_query": horizon_query,
            "future_mark": future_mark,
            "stats_embedding": stats_embedding,
            "robust_scale": robust_scale,
            "seasonal_candidate": seasonal_candidate,
            "seasonal_gate": seasonal_gate,
            "seasonal_probability": seasonal_probability,
            "seasonal_context": seasonal_context,
            "seasonal_query": seasonal_query,
            "relative_candidate": relative_candidate,
            "relative_gate": relative_gate,
            "relative_probability": relative_probability,
            "relative_context": relative_context,
            "relative_query": relative_query,
            "relative_scale": relative_scale,
            "numerical_prediction": numerical_prediction,
        }

    def _prediction_from_context(self, components, refined_context):
        base_residual = self.backbone.state_decoder.head(refined_context)
        base_prediction = components["state"]["future_base"] + base_residual
        seasonal_gate, _ = self.backbone.seasonal_memory.gate(
            components["seasonal_query"],
            components["seasonal_context"],
            refined_context,
            components["stats_embedding"],
            components["seasonal_candidate"],
        )
        relative_gate, _ = self.backbone.robust_relative.gate(
            components["relative_query"],
            components["relative_context"],
            refined_context,
            components["stats_embedding"],
            components["relative_candidate"],
        )
        total = (
            seasonal_gate * components["seasonal_candidate"]
            + relative_gate * components["relative_candidate"]
        )
        total = self.total_correction_clip * torch.tanh(
            total / max(self.total_correction_clip, 1e-6)
        )
        return base_prediction + total

    def _regime_embedding(self, components):
        scale = components["robust_scale"].expand(-1, self.pred_len, -1)
        disagreement = (
            components["seasonal_candidate"] - components["relative_candidate"]
        ).abs()
        regime = torch.cat(
            (
                torch.log1p(scale),
                components["seasonal_gate"],
                components["relative_gate"],
                disagreement,
                components["base_residual"].abs(),
            ),
            dim=-1,
        )
        return (
            self.regime_projection(regime),
            disagreement / scale.clamp_min(0.05),
        )

    def _validate_text_contract(self, batch):
        fact_text = batch.get("fact_text")
        if fact_text is None:
            raise RuntimeError(
                "SENSA requires historical fact text. Use include_text=True and text_mode='fact'."
            )
        pred_text = batch.get("pred_text")
        if pred_text is not None:
            nonempty = any(
                str(item or "").strip()
                for row in pred_text
                for item in (row if isinstance(row, (list, tuple)) else [row])
            )
            if nonempty:
                raise RuntimeError("SENSA refuses non-empty future-prediction text")
        return fact_text

    def _forward_state(self, batch):
        fact_text = self._validate_text_contract(batch)
        x_num = batch["x_num"]
        y_mark = batch.get("y_mark")
        components = self._frozen_components(x_num, y_mark)

        base_context = components["base_context"].detach()
        horizon_query = components["horizon_query"].detach()
        stats_embedding = components["stats_embedding"].detach()
        numerical_prediction = components["numerical_prediction"].detach()

        event_embedding, event_timing, event_mask, event_count = build_event_batch(
            fact_text,
            self.text_cache,
            device=x_num.device,
            dtype=x_num.dtype,
            max_events=self.max_events,
        )
        event_factors = self.semantic_factorizer(event_embedding)
        event_context, semantic_mass = self.text_aggregator(
            event_factors,
            event_timing,
            event_mask,
            horizon_count=self.pred_len,
        )
        regime_embedding, candidate_disagreement = self._regime_embedding(components)
        delta, learned_alpha, raw_probability, effective_probability = self.adapter(
            base_context,
            horizon_query,
            stats_embedding,
            event_context,
            regime_embedding,
            semantic_mass,
            candidate_disagreement.clamp(0.0, 10.0),
        )
        candidate_context = base_context + delta
        candidate_prediction = self._prediction_from_context(
            components, candidate_context
        )
        return {
            "components": components,
            "base_context": base_context,
            "numerical_prediction": numerical_prediction,
            "candidate_prediction": candidate_prediction,
            "delta": delta,
            "learned_alpha": learned_alpha,
            "raw_probability": raw_probability,
            "effective_probability": effective_probability,
            "semantic_mass": semantic_mass,
            "event_count": event_count,
        }

    def _prediction_at_scale(self, state, scale):
        refined_context = state["base_context"] + (
            float(scale) * state["learned_alpha"]
        ) * state["delta"]
        return self._prediction_from_context(state["components"], refined_context)

    def forward_batch(self, batch):
        state = self._forward_state(batch)
        prediction = self._prediction_at_scale(state, self._deployment_scale)
        if tuple(prediction.shape[1:]) != (self.pred_len, 1):
            raise ValueError("Invalid SENSA prediction shape {}".format(tuple(prediction.shape)))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("SENSA prediction contains NaN/Inf")
        diagnostics = {
            "semantic_mass": state["semantic_mass"].detach().mean(),
            "event_count": state["event_count"].detach().mean(),
            "intervention_gate": state["learned_alpha"].detach().mean(),
            "deployment_scale": prediction.new_tensor(self._deployment_scale),
            "latent_delta_abs": state["delta"].detach().abs().mean(),
            "prediction_change_abs": (
                prediction.detach() - state["numerical_prediction"]
            ).abs().mean(),
        }
        return {"prediction": prediction, "diagnostics": diagnostics}

    def forward(self, x_num, x_mark=None, decoder_input=None, y_mark=None):
        raise RuntimeError(
            "SENSA requires forward_batch() because historical fact text is mandatory."
        )


def load_sensa_checkpoint(model, checkpoint_path, device="cpu"):
    """Load a final 2.0Free checkpoint, ignoring only legacy unused-module weights."""
    payload = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError("SENSA checkpoint must be a mapping")
    for key, expected in (("domain", model.domain), ("pred_len", model.pred_len)):
        if key in payload and str(payload[key]) != str(expected):
            raise RuntimeError(
                "SENSA checkpoint {} mismatch: {} != {}".format(key, payload[key], expected)
            )
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise RuntimeError("No model_state/state_dict found in SENSA checkpoint")

    mapped = {}
    ignored_legacy = []
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("response_memory."):
            ignored_legacy.append(key)
            continue
        if key.startswith("event_aligner."):
            suffix = key[len("event_aligner."):]
            if suffix.startswith(("timing.", "value.", "output.")):
                key = "text_aggregator." + suffix
            else:
                ignored_legacy.append(key)
                continue
        mapped[key] = value

    incompatible = model.load_state_dict(mapped, strict=False)
    if any(key.startswith("backbone.") for key in mapped):
        model._backbone_checkpoint_loaded = True
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if model._backbone_checkpoint_loaded:
        missing = [key for key in missing if not key.startswith("backbone.")]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch; missing={}, unexpected={}".format(
                missing[:20], unexpected[:20]
            )
        )
    if "selected_alpha" in payload:
        model.set_deployment_scale(float(payload["selected_alpha"]))
    return payload
