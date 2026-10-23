"""Data identity contract for the corrected Time-MMD release data."""

from protocol import DOMAINS, DOMAIN_FREQUENCY, EXPECTED_DATA_HASHES, FREQUENCIES

TEXT_DIM = 896
TEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT_POOLING = "mean_of_last4_layers_over_fact_tokens_only"
