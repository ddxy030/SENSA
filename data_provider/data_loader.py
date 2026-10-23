"""
作用：只读 Time-MMD CSV，拟合 Train-only scaler 并生成无越界统一滑动窗口。
冻结状态：正式 baseline 开始后硬冻结。
修改影响：任何修改都必须重跑全部 baseline 和后续模型。
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data_provider.time_features import time_features
from protocol import TEXT_MODES, get_frequency, split_boundaries, validate_pred_len


NUMERIC_COLUMNS = ("date", "start_date", "end_date", "OT")
TEXT_COLUMN_BY_MODE = {
    "none": (),
    "fact": ("fact",),
}


class TimeMMDDataset(Dataset):
    """One domain/split/horizon view with targets strictly inside its partition."""

    def __init__(
        self,
        data_root,
        domain,
        split,
        pred_len,
        include_text=False,
        text_mode="none",
        scaler=None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if text_mode not in TEXT_MODES:
            raise ValueError("Unknown text_mode {!r}".format(text_mode))
        if not include_text and text_mode != "none":
            raise ValueError("text_mode must be 'none' when include_text=False")
        if include_text and text_mode == "none":
            raise ValueError("include_text=True requires fact, pred, or fact_pred")

        self.data_root = Path(data_root)
        self.domain = domain
        self.split = split
        self.pred_len = validate_pred_len(domain, int(pred_len))
        self.frequency = get_frequency(domain)
        self.seq_len = self.frequency.seq_len
        self.label_len = self.frequency.label_len
        self.include_text = bool(include_text)
        self.text_mode = text_mode
        self.csv_path = self.data_root / domain / (domain + ".csv")
        if not self.csv_path.is_file():
            raise FileNotFoundError(str(self.csv_path))

        usecols = list(NUMERIC_COLUMNS) + list(TEXT_COLUMN_BY_MODE[text_mode] if include_text else ())
        self.frame = pd.read_csv(self.csv_path, usecols=usecols)
        self._validate_frame()
        self.dates = pd.to_datetime(self.frame["date"], errors="raise")
        # Formatting the full column is intentionally done once. Environment contains
        # many daily windows, so repeating this conversion in __getitem__ is prohibitive.
        self.date_strings = self.dates.dt.strftime("%Y-%m-%d").to_numpy()
        self.boundaries = split_boundaries(len(self.frame))
        train_end = self.boundaries["train"][1]

        raw_values = self.frame["OT"].to_numpy(dtype=np.float64).reshape(-1, 1)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(raw_values[:train_end])
        else:
            self._validate_scaler(scaler, raw_values[:train_end])
        self.scaler = scaler
        self.values = scaler.transform(raw_values).astype(np.float32, copy=False)
        self.marks = time_features(self.dates, self.frequency.name)
        self.anchors = self._build_anchors()
        if not self.anchors:
            raise ValueError(
                "No legal {} samples for {} with seq_len={} pred_len={}".format(
                    split, domain, self.seq_len, self.pred_len
                )
            )

    def _validate_frame(self):
        missing = [column for column in NUMERIC_COLUMNS if column not in self.frame.columns]
        if missing:
            raise ValueError("Missing required columns: {}".format(missing))
        parsed = pd.to_datetime(self.frame["date"], errors="raise")
        if parsed.duplicated().any():
            raise ValueError("date contains duplicates in {}".format(self.csv_path))
        if not parsed.is_monotonic_increasing or not (parsed.diff().dropna() > pd.Timedelta(0)).all():
            raise ValueError("date must be strictly increasing in {}".format(self.csv_path))
        numeric = pd.to_numeric(self.frame["OT"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("OT must contain only finite numeric values in {}".format(self.csv_path))

    @staticmethod
    def _validate_scaler(scaler, train_values):
        if not isinstance(scaler, StandardScaler):
            raise TypeError("scaler must be sklearn.preprocessing.StandardScaler")
        expected = StandardScaler().fit(train_values)
        if not np.allclose(scaler.mean_, expected.mean_) or not np.allclose(scaler.var_, expected.var_):
            raise ValueError("Provided scaler is not fitted exclusively on this domain's train partition")

    def _build_anchors(self) -> List[int]:
        split_start, split_end = self.boundaries[self.split]
        first = max(split_start, self.seq_len, self.label_len)
        last = split_end - self.pred_len
        return list(range(first, last + 1)) if last >= first else []

    def __len__(self):
        return len(self.anchors)

    @staticmethod
    def _text_value(value):
        return "" if pd.isna(value) else str(value)

    def _history_text(self, t: int, column: str) -> List[str]:
        # The half-open slice is the causality boundary: index t and later are never read.
        return [self._text_value(value) for value in self.frame[column].iloc[t - self.seq_len:t].tolist()]

    def __getitem__(self, index):
        t = self.anchors[index]
        x_slice = slice(t - self.seq_len, t)
        context_slice = slice(t - self.label_len, t)
        target_slice = slice(t, t + self.pred_len)
        y_mark_slice = slice(t - self.label_len, t + self.pred_len)

        fact_text = None
        pred_text = None
        text_mask = None
        if self.include_text:
            if "fact" in TEXT_COLUMN_BY_MODE[self.text_mode]:
                fact_text = self._history_text(t, "fact")
            if "preds" in TEXT_COLUMN_BY_MODE[self.text_mode]:
                pred_text = self._history_text(t, "preds")
            chosen = fact_text if fact_text is not None else pred_text
            if fact_text is not None and pred_text is not None:
                text_mask = [bool(a.strip() or b.strip()) for a, b in zip(fact_text, pred_text)]
            else:
                text_mask = [bool(value.strip()) for value in chosen]

        return {
            "x_num": torch.from_numpy(self.values[x_slice].copy()),
            "y_context": torch.from_numpy(self.values[context_slice].copy()),
            "target": torch.from_numpy(self.values[target_slice].copy()),
            "x_mark": torch.from_numpy(self.marks[x_slice].copy()),
            "y_mark": torch.from_numpy(self.marks[y_mark_slice].copy()),
            "fact_text": fact_text,
            "pred_text": pred_text,
            "text_mask": text_mask,
            "sample_id": "{}:{}:t{}:pl{}".format(self.domain, self.split, t, self.pred_len),
            "domain": self.domain,
            "frequency": self.frequency.name,
            "split": self.split,
            "input_start": self.date_strings[t - self.seq_len],
            "input_end": self.date_strings[t - 1],
            "forecast_start": self.date_strings[t],
            "forecast_end": self.date_strings[t + self.pred_len - 1],
        }

    def inverse_transform(self, array):
        shape = array.shape
        restored = self.scaler.inverse_transform(np.asarray(array).reshape(-1, 1))
        return restored.reshape(shape)
