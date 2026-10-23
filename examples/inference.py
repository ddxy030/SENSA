#!/usr/bin/env python3
"""Run one SENSA forecast from the corrected Time-MMD test split."""

import argparse

import torch

# Support direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parents[1])
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from configs.sensa_config import make_config
from data_provider.data_factory import build_data_bundle, create_data_loader
from models.sensa import SENSA, load_sensa_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data",
                        help="Directory containing <domain>/<domain>.csv")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--pred-len", required=True, type=int)
    parser.add_argument("--sensa-checkpoint", required=True,
                        help="Final SENSA checkpoint containing model_state")
    parser.add_argument("--text-cache-dir", default="cache/text_embeddings")
    parser.add_argument("--numerical-root", default=None,
                        help="Directory containing checkpoints_MWP_numerical/")
    parser.add_argument("--load-numerical-backbone", action="store_true",
                        help="Load the separate numerical checkpoint before the SENSA checkpoint")
    parser.add_argument("--deployment-scale", type=float, default=None,
                        help="Override selected_alpha stored in the SENSA checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    bundle = build_data_bundle(
        data_root=args.data_root,
        domain=args.domain,
        pred_len=args.pred_len,
        include_text=True,
        text_mode="fact",
    )
    loader = create_data_loader(
        bundle["test"],
        batch_size=args.batch_size,
        seed=args.seed,
        split="test",
    )

    config = make_config(
        domain=args.domain,
        pred_len=args.pred_len,
        text_cache_dir=args.text_cache_dir,
        numerical_root=args.numerical_root,
        numerical_seed=2026,
    )
    model = SENSA(
        config,
        load_numerical_checkpoint=args.load_numerical_backbone,
    ).to(device)
    checkpoint = load_sensa_checkpoint(model, args.sensa_checkpoint, device=device)
    if args.deployment_scale is not None:
        model.set_deployment_scale(args.deployment_scale)
    model.eval()

    batch = next(iter(loader))
    batch["x_num"] = batch["x_num"].to(device)
    batch["y_mark"] = batch["y_mark"].to(device)

    with torch.no_grad():
        output = model.forward_batch(batch)
    prediction_scaled = output["prediction"].detach().cpu().numpy().reshape(-1, 1)
    prediction_original = bundle["scaler"].inverse_transform(prediction_scaled).reshape(-1)

    print("domain={}".format(args.domain))
    print("pred_len={}".format(args.pred_len))
    print("deployment_scale={:.6f}".format(model.deployment_scale))
    print("checkpoint_selected_alpha={}".format(checkpoint.get("selected_alpha", "not recorded")))
    print("prediction_original_scale={}".format(prediction_original.tolist()))
    print("diagnostics={}".format({
        key: float(value.detach().cpu()) if torch.is_tensor(value) else value
        for key, value in output["diagnostics"].items()
    }))


if __name__ == "__main__":
    main()
