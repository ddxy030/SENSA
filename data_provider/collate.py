"""
作用：按统一字典合同批处理数值张量、元数据和可选历史文本。
冻结状态：正式 baseline 开始后硬冻结。
修改影响：任何修改都必须重跑全部 baseline 和后续模型。
"""

from typing import Dict, List

import torch


TENSOR_KEYS = ("x_num", "y_context", "target", "x_mark", "y_mark")
META_KEYS = (
    "sample_id", "domain", "frequency", "split", "input_start", "input_end",
    "forecast_start", "forecast_end",
)
TEXT_KEYS = ("fact_text", "pred_text", "text_mask")


def collate_samples(samples: List[Dict]):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in TENSOR_KEYS}
    for key in META_KEYS:
        batch[key] = [sample[key] for sample in samples]
    for key in ("fact_text", "pred_text"):
        values = [sample[key] for sample in samples]
        batch[key] = None if all(value is None for value in values) else values
    mask_values = [sample["text_mask"] for sample in samples]
    if all(value is None for value in mask_values):
        batch["text_mask"] = None
    elif any(value is None for value in mask_values):
        raise ValueError("text_mask must be present for every sample in a textual batch")
    else:
        batch["text_mask"] = torch.tensor(mask_values, dtype=torch.bool)
    return batch
