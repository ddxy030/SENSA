"""Frozen numerical-backbone parameters used by SENSA."""

import json
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "reference" / "MWP_numerical_locked_parameters.json"

BASE_MODEL = {
    "d_model": 32,
    "n_heads": 4,
    "d_ff": 64,
    "dropout": 0.10,
    "state_layers": 2,
    "branch_state_layers": 1,
    "state_kernel_sizes": [3, 5],
    "state_dilations": [1, 2],
    "cosine_eta_min_ratio": 0.1,
    "gate_bias": -5.0,
    "gate_max": 0.35,
    "seasonal_clip_scale": 1.5,
    "relative_clip_factor": 3.0,
    "relative_scale_cap": 4.0,
    "total_correction_clip": 2.0,
    "candidate_loss_weight": 0.15,
    "reliability_loss_weight": 0.10,
    "safety_loss_weight": 0.25,
    "oracle_temperature": 0.20,
    "oracle_margin": 0.0,
    "safety_margin": 0.0,
    "correction_penalty": 0.0001,
    "gate_penalty": 0.0001,
    "seasonal_smoothness_penalty": 0.0001,
    "safety_mae_ratio": 0.50,
    "mae_safety_margin": 0.0,
}


def locked_parameters():
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("Frozen parameter file not found: {}".format(LOCK_PATH))
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    parameters = payload.get("parameters_by_domain")
    if not isinstance(parameters, dict):
        raise ValueError("Invalid frozen numerical parameter file: {}".format(LOCK_PATH))
    return parameters


def numerical_model_parameters(domain):
    parameters_by_domain = locked_parameters()
    if domain not in parameters_by_domain:
        raise ValueError("Unknown domain {!r}".format(domain))
    locked = deepcopy(parameters_by_domain[domain])
    locked.pop("learning_rate", None)
    locked.pop("weight_decay", None)
    parameters = deepcopy(BASE_MODEL)
    parameters.update(locked)
    parameters["active_domain_for_parameters"] = domain
    return parameters
