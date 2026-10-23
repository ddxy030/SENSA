"""
作用：创建与模型无关的统一 Dataset、Scaler 和可复现 DataLoader。
冻结状态：正式 baseline 开始后硬冻结。
修改影响：任何修改都必须重跑全部 baseline 和后续模型。
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.collate import collate_samples
from data_provider.data_loader import TimeMMDDataset
from protocol import NUM_WORKERS


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_bundle(data_root, domain, pred_len, include_text=False, text_mode="none"):
    train = TimeMMDDataset(data_root, domain, "train", pred_len, include_text, text_mode)
    val = TimeMMDDataset(data_root, domain, "val", pred_len, include_text, text_mode, scaler=train.scaler)
    test = TimeMMDDataset(data_root, domain, "test", pred_len, include_text, text_mode, scaler=train.scaler)
    return {"train": train, "val": val, "test": test, "scaler": train.scaler}


def create_data_loader(dataset, batch_size, seed, split=None):
    split = split or dataset.split
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split {!r}".format(split))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(split == "train"),
        drop_last=False,
        num_workers=NUM_WORKERS,
        generator=generator,
        worker_init_fn=_seed_worker,
        collate_fn=collate_samples,
        pin_memory=False,
    )
