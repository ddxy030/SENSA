# SENSA: When Should Text Change a Forecast?

This repository contains a cleaned model implementation for **SENSA (Semantic Evidence-guided Numerical State Adaptation)** for multimodal time-series forecasting.

The release contains the final SENSA predictive implementation, a minimal inference example, and the corrected nine-domain dataset under `data/`. Experiment launchers, hyperparameter-search scripts, training logs, intermediate results, baseline implementations, frozen text caches, and trained model checkpoints are not included.

## Final model

SENSA keeps the numerical forecaster responsible for forecasting. Historical text is used as auxiliary evidence to adapt the numerical latent state; the language model does not generate future numeric values.

The released predictive path contains:

1. A frozen numerical backbone with a horizon-aware state and shared numerical readout.
2. A frozen-Qwen text cache built from historical `fact` fields only.
3. A semantic factorizer that maps each event embedding to a small latent factor bank.
4. Timing-aware uniform aggregation over valid historical event/factor items. The final model does not use query-key retrieval, top-k semantic selection, NULL rejection, or response-prototype memory.
5. Numerical-information conditioning and an intervention gate.
6. A bounded latent-state correction, read out by the same frozen numerical prediction head.

The model's deployment scale is read from the final SENSA checkpoint's `selected_alpha` field when present. Setting the deployment scale to zero gives the numerical-only fallback.

## Repository layout

```text
SENSA/
 data/
   ├── Agriculture/Agriculture.csv
   ├── Climate/Climate.csv
   ├── Economy/Economy.csv
   ├── Energy/Energy.csv
   ├── Environment/Environment.csv
   ├── Health/Health.csv
   ├── Security/Security.csv
   ├── SocialGood/SocialGood.csv
   └── Traffic/Traffic.csv
├── configs/
│   ├── data_contract.py
│   ├── numerical_config.py
│   └── sensa_config.py
├── data_provider/
│   ├── collate.py
│   ├── data_factory.py
│   ├── data_loader.py
│   └── time_features.py
├── examples/
│   └── inference.py
├── layers/
│   ├── numerical_core.py
│   ├── numerical_branches.py
│   └── sensa_modules.py
├── models/
│   ├── numerical_backbone.py
│   └── sensa.py
├── reference/
│   └── MWP_numerical_locked_parameters.json
├── tools/
│   └── prepare_text_cache.py
├── utils/
│   └── text_cache.py
├── protocol.py
├── requirements.txt
└── .gitignore
```

## Environment

The development environment used for the source implementation was Python 3.8.12 and PyTorch 2.1.2 with CUDA 12.1. Install a PyTorch build appropriate for your system first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The text-cache preparation tool additionally requires the frozen Qwen model files to be available locally. It uses `transformers==4.40.2`.

## Data contract

This repository includes the corrected nine-domain Time-MMD-derived CSV files under `data/`. The files are organized as follows:

```text
data/
├── Agriculture/Agriculture.csv
├── Climate/Climate.csv
├── Economy/Economy.csv
├── Energy/Energy.csv
├── Environment/Environment.csv
├── Health/Health.csv
├── Security/Security.csv
├── SocialGood/SocialGood.csv
└── Traffic/Traffic.csv
```

Each CSV contains `date`, `start_date`, `end_date`, `OT`, `fact`, and `preds`. The main SENSA protocol uses historical `fact` text only and does not use future-horizon text to predict numerical values.

The fixed windows are:

| Frequency | Input length | Label length | Prediction lengths |
|---|---:|---:|---|
| Monthly | 8 | 4 | 6, 8, 10, 12 |
| Weekly | 36 | 18 | 12, 24, 36, 48 |
| Daily | 96 | 48 | 48, 96, 192, 336 |

## Prepare the frozen text cache

The model reads cached embeddings; it does not instantiate Qwen during forecasting. The cache is built from unique historical `fact` strings and uses the mean of the last four Qwen hidden layers over fact tokens only. Future `preds` text is not used.

Make sure the local `Qwen/Qwen2.5-0.5B-Instruct` model is available, then run:

```bash
python tools/prepare_text_cache.py \
  --root . \
  --domains Agriculture Climate Economy Energy Environment Health Security SocialGood Traffic
```

By default the tool runs in offline mode and writes the cache to `cache/text_embeddings/`. Use `--allow-online` only if you intentionally want Transformers to access the network.

## Inference example

The example expects a final SENSA checkpoint containing `model_state`. The checkpoint should correspond to the requested domain and prediction length; when available, its `selected_alpha` value is used automatically.

```bash
python examples/inference.py \
  --data-root data \
  --domain Agriculture \
  --pred-len 6 \
  --text-cache-dir cache/text_embeddings \
  --sensa-checkpoint /path/to/final_sensa_checkpoint.pth \
  --device cuda:0
```

The example reads one batch from the chronological test split and prints the forecast in the original `OT` scale. To override the checkpoint's deployment scale, add `--deployment-scale VALUE`, where `VALUE` is in `[0, 1]`.

A full SENSA checkpoint contains the frozen numerical backbone weights. If you instead have an adapter-only checkpoint, provide the numerical checkpoint root and add `--load-numerical-backbone`:

```bash
python examples/inference.py \
  --data-root data \
  --domain Agriculture \
  --pred-len 6 \
  --text-cache-dir cache/text_embeddings \
  --numerical-root /path/to/numerical_project \
  --load-numerical-backbone \
  --sensa-checkpoint /path/to/adapter_checkpoint.pth
```

The numerical checkpoint root must contain `checkpoints_MWP_numerical/`.

## What is and is not included

Included:
- final numerical-backbone modules and frozen per-domain parameter configuration;
- the final memory-free SENSA predictive modules;
- causal historical fact processing and frozen-Qwen cache preparation;
- a minimal inference example.

Not included:
- training and batch-launch scripts;
- hyperparameter-search or ablation orchestration;
- experiment logs and intermediate outputs;
- Qwen weights, text-cache files, numerical checkpoints, or SENSA checkpoints.

Therefore, this repository provides the model implementation and an inference interface, rather than a self-contained reproduction bundle. Inference requires the included corrected dataset, a generated text cache, and a compatible trained SENSA checkpoint. Training from scratch requires a separate training pipeline, which is not included in this release.

## Paper and citation

The paper title and citation details should be updated here once the submission version and public bibliographic information are finalized.

## License

Original SENSA source code authored for this project is released under the
MIT License; see [LICENSE](LICENSE).

The code license does not automatically apply to the dataset files under
`data/`, third-party components, or pretrained model weights. Dataset
provenance and applicable source-specific terms are documented separately
in [`data/README.md`](data/README.md) and
[`data/DATA_LICENSE.md`](data/DATA_LICENSE.md).

The repository does not redistribute Qwen weights or trained model
checkpoints. Third-party code, if included, remains subject to its applicable
upstream license and attribution requirements.
