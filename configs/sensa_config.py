"""Configuration helpers for the released SENSA model."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from configs.numerical_config import numerical_model_parameters
from protocol import get_frequency, validate_pred_len


def make_config(
    domain,
    pred_len,
    text_cache_dir,
    numerical_root=None,
    numerical_seed=2026,
    deployment_scale=1.0,
):
    """Build the frozen per-domain model configuration.

    A full SENSA checkpoint contains the frozen numerical backbone weights. Set
    ``load_numerical_checkpoint=True`` on SENSA only if you want to load the separate
    numerical checkpoint before loading an adapter-only checkpoint.
    """
    validate_pred_len(domain, int(pred_len))
    spec = get_frequency(domain)
    parameters = numerical_model_parameters(domain)

    package_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(text_cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = package_root / cache_dir
    numerical_dir = Path(numerical_root).expanduser() if numerical_root else package_root

    values = deepcopy(parameters)
    values.update(
        {
            "domain": domain,
            "frequency_name": spec.name,
            "seq_len": spec.seq_len,
            "label_len": spec.label_len,
            "pred_len": int(pred_len),
            "frozen_numerical_seed": int(numerical_seed),
            "frozen_numerical_root": str(numerical_dir.resolve()),
            "text_cache_dir": str(cache_dir.resolve()),
            "text_dim": 896,
            "umsa_dropout": 0.05,
            "max_events": 16,
            "semantic_factors": 4,
            "adapter_scale": 1.0,
            "semantic_gate_max": 1.0,
            "semantic_gate_bias": -2.0,
            "deployment_scale": float(deployment_scale),
        }
    )
    return SimpleNamespace(**values)
