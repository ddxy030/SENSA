"""Frozen-Qwen cache loading and causal historical-event compaction for SENSA.

No target values, future text, or test statistics are used here.  Cache entries are
functions of an individual historical ``fact`` string only.  Consecutive identical
facts inside each numerical history window are compacted into one event.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def text_key(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


class FrozenTextCache:
    """Read-only exact-text -> frozen embedding lookup."""

    def __init__(self, path, expected_dim=None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                "Frozen text cache not found: {}. Run tools/prepare_text_cache.py first.".format(
                    self.path
                )
            )
        payload = np.load(str(self.path), allow_pickle=False)
        keys = payload["keys"]
        embeddings = payload["embeddings"].astype(np.float32, copy=False)
        if embeddings.ndim != 2 or len(keys) != embeddings.shape[0]:
            raise ValueError("Invalid text cache arrays in {}".format(self.path))
        if expected_dim is not None and embeddings.shape[1] != int(expected_dim):
            raise ValueError(
                "Text cache dim {} != expected {} in {}".format(
                    embeddings.shape[1], expected_dim, self.path
                )
            )
        self.embedding_dim = int(embeddings.shape[1])
        self._vectors = torch.from_numpy(embeddings)
        self._index = {str(key): i for i, key in enumerate(keys.tolist())}
        meta_path = self.path.with_suffix(".json")
        self.metadata = {}
        if meta_path.is_file():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    def lookup(self, text: str) -> torch.Tensor:
        key = text_key(text)
        try:
            index = self._index[key]
        except KeyError:
            raise KeyError(
                "Text is absent from frozen cache {} (sha256={}). Rebuild the cache from the current frozen data.".format(
                    self.path, key
                )
            )
        return self._vectors[index]


def compact_events(row: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Compact consecutive identical non-empty facts in one causal history window.

    Returns tuples ``(fact, start_index, end_index)``; empty spans are ignored and
    break continuity.  This prevents semantic pseudo-replication when one event is
    repeated over multiple numerical timestamps.
    """
    events: List[Tuple[str, int, int]] = []
    current = None
    start = None
    for i, raw in enumerate(row):
        value = "" if raw is None else str(raw).strip()
        if not value or value.lower() == "nan":
            if current is not None:
                events.append((current, int(start), i - 1))
                current, start = None, None
            continue
        if current is None:
            current, start = value, i
        elif value != current:
            events.append((current, int(start), i - 1))
            current, start = value, i
    if current is not None:
        events.append((current, int(start), len(row) - 1))
    return events


def build_event_batch(
    fact_text: Sequence[Sequence[str]],
    cache: FrozenTextCache,
    device: torch.device,
    dtype: torch.dtype,
    max_events: int,
):
    """Build padded compacted-event tensors for a mini-batch.

    ``timing`` channels are all causal and normalized to [0,1] when possible:
      0) event start position in history,
      1) event end position in history,
      2) duration / history length,
      3) age from forecast origin / history length,
      4) recency = exp(-4 * age_ratio).
    The most recent ``max_events`` compacted events are retained if a window contains
    more events; this is one shared rule for every domain.
    """
    if fact_text is None:
        raise ValueError("SENSA requires fact_text; run with --include_text --text_mode fact")
    batch_size = len(fact_text)
    if batch_size == 0:
        raise ValueError("Empty textual batch")
    histories = [list(row) for row in fact_text]
    lengths = [len(row) for row in histories]
    if min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError("fact_text histories must have one non-zero shared length")
    history_len = lengths[0]
    if int(max_events) <= 0:
        raise ValueError("max_events must be positive")

    compacted = [compact_events(row)[-int(max_events):] for row in histories]
    event_dim = cache.embedding_dim
    emb = torch.zeros(batch_size, int(max_events), event_dim, dtype=dtype, device=device)
    timing = torch.zeros(batch_size, int(max_events), 5, dtype=dtype, device=device)
    mask = torch.zeros(batch_size, int(max_events), dtype=torch.bool, device=device)
    counts = torch.zeros(batch_size, dtype=dtype, device=device)

    denom = float(max(1, history_len - 1))
    for b, events in enumerate(compacted):
        counts[b] = float(len(events))
        offset = int(max_events) - len(events)  # right-align: newest event is always last
        for j, (text, start, end) in enumerate(events):
            slot = offset + j
            emb[b, slot] = cache.lookup(text).to(device=device, dtype=dtype)
            duration = float(end - start + 1)
            age = float(history_len - 1 - end)
            age_ratio = age / float(max(1, history_len))
            timing[b, slot, 0] = float(start) / denom
            timing[b, slot, 1] = float(end) / denom
            timing[b, slot, 2] = duration / float(max(1, history_len))
            timing[b, slot, 3] = age_ratio
            timing[b, slot, 4] = float(np.exp(-4.0 * age_ratio))
            mask[b, slot] = True
    return emb, timing, mask, counts
