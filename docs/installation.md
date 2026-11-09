# Installation

## CPU selector

Python 3.10 or later is sufficient for frame selection. The core package depends only on NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
```

## Scoring and model inference

Use Linux, Python 3.11, and GPUs with sufficient memory for the model and video context. The inference runner uses vLLM; the dependency file installs its matching PyTorch runtime. Install in a fresh environment with a compatible GPU driver:

```bash
python -m pip install -r requirements/inference.txt
python -m pip install -e .
```

[`requirements/inference-lock.txt`](../requirements/inference-lock.txt) records a complete Linux x86_64 / Python 3.11 dependency resolution and can replace `inference.txt` when a fully pinned environment is desired. Consult [vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for platform-specific requirements.

CPU tests and reference selection checks are available. The upstream vLLM inference path still needs an end-to-end smoke test on the target GPU environment; dependency resolution alone is not GPU validation. Start with `--limit 8` before a full benchmark.

## Models

Answering models download from ModelScope by default:

```bash
python scripts/download_models.py --models qwen7b internvl minicpm
```

For scale ablations, add `--models qwen3b qwen32b`. Frozen scoring encoders have separate snapshots; explicitly select their Hugging Face repositories with:

```bash
python scripts/download_models.py --provider huggingface --models blip clip blip2
```

Existing local snapshots can be used directly: pass `--model` to `aps.infer` or `aps.score`, or edit `model_path` in a copied experiment config. Download revisions can be pinned with `--revision`.

Continue with [data preparation](data.md), then [evaluation](reproducibility.md).
