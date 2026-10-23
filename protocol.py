"""Minimal Time-MMD protocol required by the released SENSA model and example."""

from dataclasses import dataclass
from typing import Dict, Tuple

FEATURES = "S"
TARGET = "OT"
ENC_IN = 1
DEC_IN = 1
C_OUT = 1
SEEDS = (2024, 2025, 2026)
AMP = False
NUM_WORKERS = 0
SPLIT_RATIOS = (0.7, 0.1, 0.2)
METRICS = ("MSE", "MAE")
SCALER = "sklearn.preprocessing.StandardScaler(train_only, ddof=0)"
TEXT_MODES = ("none", "fact")

EXPECTED_DATA_HASHES = {
    "Agriculture/Agriculture.csv": "3e3051391f457c480c368f327e87cae1d100b521171c6a980e275e53991d1d9c",
    "Climate/Climate.csv": "bc6e792f5490f5919431f560036fd13a17d9db55d6bcb2f84c647df7f5395e21",
    "Economy/Economy.csv": "444da8dc328a4d695259e47d664186370fd9a24d09dcb18cc8931f81e33d3ab4",
    "Energy/Energy.csv": "3d55a4df628cd378c3d985cf2a49a560dd02109a136c326c65efd272dd9d4daf",
    "Environment/Environment.csv": "da99bc4e5f3540b81e4610d641c1b54da6b17b1b018acc6991e7f1ff1e030b5c",
    "Health/Health.csv": "c708a79668d6a2759bb7d465109477f15cdaf750650ba06f76eecb9ef029dad1",
    "Security/Security.csv": "b9c5c343b15e3f89c3d76bc71db5f0d6d863c0407e61b3fad7a6d6dbb10a2758",
    "SocialGood/SocialGood.csv": "f1e5d8a9527633f32e639463b05955ab248b82907a400ec340ae9813b5fbd311",
    "Traffic/Traffic.csv": "b0eff9606cd5aea729fa60650a73aa0208e742543986a55862884e86c57132d2",
}

@dataclass(frozen=True)
class FrequencyProtocol:
    name: str
    pandas_freq: str
    seq_len: int
    label_len: int
    pred_lens: Tuple[int, int, int, int]
    time_dim: int

FREQUENCIES: Dict[str, FrequencyProtocol] = {
    "monthly": FrequencyProtocol("monthly", "m", 8, 4, (6, 8, 10, 12), 1),
    "weekly": FrequencyProtocol("weekly", "w", 36, 18, (12, 24, 36, 48), 2),
    "daily": FrequencyProtocol("daily", "d", 96, 48, (48, 96, 192, 336), 3),
}

DOMAIN_FREQUENCY = {
    "Agriculture": "monthly",
    "Climate": "monthly",
    "Economy": "monthly",
    "Energy": "weekly",
    "Environment": "daily",
    "Health": "weekly",
    "Security": "monthly",
    "SocialGood": "monthly",
    "Traffic": "monthly",
}
DOMAINS = tuple(DOMAIN_FREQUENCY.keys())

def get_frequency(domain: str) -> FrequencyProtocol:
    if domain not in DOMAIN_FREQUENCY:
        raise ValueError("Unknown domain {!r}; expected one of {}".format(domain, list(DOMAINS)))
    return FREQUENCIES[DOMAIN_FREQUENCY[domain]]

def validate_pred_len(domain: str, pred_len: int) -> int:
    spec = get_frequency(domain)
    if int(pred_len) not in spec.pred_lens:
        raise ValueError(
            "Illegal pred_len={} for {} ({}); legal values are {}".format(
                pred_len, domain, spec.name, list(spec.pred_lens)
            )
        )
    return int(pred_len)

def split_boundaries(total_length: int):
    if total_length <= 0:
        raise ValueError("total_length must be positive")
    train_n = int(total_length * 0.7)
    test_n = int(total_length * 0.2)
    val_n = total_length - train_n - test_n
    val_end = train_n + val_n
    return {
        "train": (0, train_n),
        "val": (train_n, val_end),
        "test": (val_end, total_length),
    }
