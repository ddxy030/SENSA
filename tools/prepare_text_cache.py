#!/usr/bin/env python3
"""Build frozen Qwen last-four-layer event embeddings from Time-MMD ``fact`` only.

The language model is frozen and used only to prepare embeddings. SENSA inference reads the
resulting cache and does not instantiate Qwen. The prompt asks for predictive *event semantics* but
never asks the LLM to output future values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set cache roots before transformers import.
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from protocol import DOMAINS, EXPECTED_DATA_HASHES
from utils.text_cache import text_key


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

PROMPT_PREFIX = (
    "Encode the following historical event for time-series forecasting. "
    "Represent what happened, the affected quantity or population, direction and "
    "intensity when stated, temporal status, duration or persistence, scope, and "
    "uncertainty. Do not predict future numeric values and do not invent facts.\n"
    "Historical fact:\n"
)


def resolve_model(spec: str) -> str:
    p = Path(spec).expanduser()
    if not p.exists():
        return spec
    p = p.resolve()
    if (p / "config.json").is_file():
        return str(p)
    snapshots = p / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(x for x in snapshots.iterdir() if (x / "config.json").is_file())
        if candidates:
            return str(candidates[-1])
    raise FileNotFoundError("No config.json/snapshot under model path {}".format(p))


def unique_facts(csv_path: Path):
    frame = pd.read_csv(csv_path, usecols=["fact"])
    seen = set()
    facts = []
    for raw in frame["fact"].tolist():
        if pd.isna(raw):
            continue
        text = str(raw).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        facts.append(text)
    return facts


def encode_batch(tokenizer, model, texts, device, max_length):
    prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
    if len(prefix_ids) >= max_length - 1:
        raise RuntimeError("Predictive semantic prompt is too long for max_length={}".format(max_length))
    rows = []
    event_ranges = []
    reserve = max_length - len(prefix_ids)
    for text in texts:
        fact_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=reserve,
        )
        if not fact_ids:
            raise RuntimeError("Non-empty fact produced zero tokens")
        rows.append(prefix_ids + fact_ids)
        event_ranges.append((len(prefix_ids), len(prefix_ids) + len(fact_ids)))
    padded = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
    input_ids = padded["input_ids"].to(device)
    attention_mask = padded["attention_mask"].to(device)
    event_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for i, (start, end) in enumerate(event_ranges):
        event_mask[i, start:end] = True
    event_mask &= attention_mask.bool()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if out.hidden_states is None or len(out.hidden_states) < 4:
            raise RuntimeError("Frozen Qwen did not return at least four hidden layers")
        representation = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        weights = event_mask.unsqueeze(-1).to(representation.dtype)
        pooled = (representation * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if not torch.isfinite(pooled).all():
        raise FloatingPointError("Qwen cache embedding contains NaN/Inf")
    return pooled.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--domains", nargs="*", default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument("--model", default=os.environ.get("SENSA_TEXT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-online", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / "cache" / "text_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0 or args.max_length <= 32:
        raise ValueError("Invalid batch-size/max-length")

    local_only = not args.allow_online
    if args.allow_online:
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
    from transformers import AutoModel, AutoTokenizer
    model_spec = resolve_model(args.model)
    print("text_model={}".format(model_spec))
    print("local_files_only={}".format(local_only))
    tokenizer = AutoTokenizer.from_pretrained(model_spec, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_spec,
        local_files_only=local_only,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("Could not resolve Qwen hidden_size")
    prompt_sha = hashlib.sha256(PROMPT_PREFIX.encode("utf-8")).hexdigest()

    for domain in args.domains:
        csv_path = root / "data" / domain / (domain + ".csv")
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        actual_data_hash = sha256_file(csv_path)
        expected_data_hash = EXPECTED_DATA_HASHES["{}/{}.csv".format(domain, domain)]
        if actual_data_hash != expected_data_hash:
            raise RuntimeError(
                "Dataset hash mismatch for {}: expected {}, got {}. "
                "Use the corrected Time-MMD CSV version specified by the release.".format(
                    domain, expected_data_hash, actual_data_hash
                )
            )
        out_path = output_dir / (domain + ".last4.npz")
        meta_path = output_dir / (domain + ".last4.json")
        if out_path.is_file() and meta_path.is_file() and not args.force:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                int(meta.get("hidden_size", -1)) == hidden_size
                and meta.get("prompt_sha256") == prompt_sha
                and meta.get("data_sha256") == sha256_file(csv_path)
                and meta.get("pooling") == "mean_of_last4_layers_over_fact_tokens_only"
            ):
                print("CACHE_SKIP {} {}".format(domain, out_path))
                continue
            raise RuntimeError(
                "Existing cache metadata differs for {}; use --force to rebuild".format(domain)
            )

        facts = unique_facts(csv_path)
        if not facts:
            raise RuntimeError("No non-empty fact strings in {}".format(csv_path))
        chunks = []
        for start in range(0, len(facts), args.batch_size):
            part = facts[start:start + args.batch_size]
            chunks.append(encode_batch(tokenizer, model, part, device, args.max_length))
            if start == 0 or (start // args.batch_size) % 25 == 0:
                print("CACHE_PROGRESS {} {}/{}".format(domain, min(start + len(part), len(facts)), len(facts)))
        embeddings = np.concatenate(chunks, axis=0).astype(np.float16, copy=False)
        keys = np.asarray([text_key(x) for x in facts], dtype="<U64")
        np.savez_compressed(out_path, keys=keys, embeddings=embeddings)
        meta = {
            "schema_version": 1,
            "domain": domain,
            "model": args.model,
            "resolved_model": model_spec,
            "hidden_size": hidden_size,
            "pooling": "mean_of_last4_layers_over_fact_tokens_only",
            "prompt": PROMPT_PREFIX,
            "prompt_sha256": prompt_sha,
            "max_length": args.max_length,
            "fact_count": len(facts),
            "data_sha256": sha256_file(csv_path),
            "uses_fact": True,
            "uses_preds": False,
            "future_numeric_prediction_by_llm": False,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("CACHE_OK {} facts={} dim={} {}".format(domain, len(facts), hidden_size, out_path))

    print("SENSA_TEXT_CACHE_OK=true")


if __name__ == "__main__":
    main()
