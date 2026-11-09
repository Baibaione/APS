# APS

**English** | [Chinese](README.zh-CN.md)

APS selects video frames for long-video question answering under a fixed frame budget, using a question-conditioned relevance curve:

**Relevance scoring → adaptive temporal partitioning (COMPS) → budget allocation → within-segment Top-K selection → video question answering.**

The historical method key `dps` refers to the same method, now named **APS**. This repository contains inference experiments; none of the three answering models is trained here.

The repository covers **APS, its ablations, and reproduction tools**. APS is the default method. Uniform sampling is retained as an ablation control and as the fallback when the candidate pool is smaller than the frame budget. Implementations, integration frameworks, and run configurations for other frame-selection methods are excluded.

> Historical results were obtained on MetaX GPUs with FlagOS. The inference entry point below targets **NVIDIA CUDA and upstream vLLM**. Code checks and CPU protocol tests have passed, but end-to-end NVIDIA inference has not yet been validated. The reported historical results are not NVIDIA measurements.

## Repository layout

```text
src/aps/                    APS, scoring, NVIDIA inference, metrics, and scheduling
configs/main_table.json     Main experiments: three models × three benchmarks
configs/ablations.json      Parameter, encoder, size, frame, partition, and penalty ablations
assets/selections/          Six reference APS selections: three benchmarks × 16/32 frames
assets/question_ids/        Reference question order, without question text or answers
assets/question_manifest/  Video identifiers and question hashes for identity checks
assets/input_manifest.json SHA-256 hashes of historical annotations, scores, and candidates
reports/                   Sanitized aggregate results, source hashes, and validation records
scripts/                   Model downloads, data import, MLVU conversion, and release checks
examples/                  CPU example requiring no models or benchmark data
third_party/               Third-party provenance and licenses
```

Model weights, videos, full datasets, score caches, raw predictions, and runtime logs are excluded from Git. Score caches can be imported from existing experiments or generated using the instructions below.

## 1. Run the CPU example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python examples/quickstart.py
pytest -q
```

Minimal API usage:

```python
from aps import APSConfig, select_frames

selected = select_frames(
    scores,                # Candidate relevance scores for one question, in temporal order
    original_frame_ids,    # Original video frame indices, not timestamps
    APSConfig(budget=32, alpha=1.0, beta=0.6, penalty=0.5),
)
```

## 2. Set up NVIDIA inference

Use Linux x86_64, Python 3.11, and an NVIDIA GPU with BF16 support. Check that `nvidia-smi` works. Create a fresh environment and let vLLM install its compatible CUDA PyTorch dependencies; do not copy PyTorch, Triton, or platform plugins from the original MetaX container.

```bash
conda create -n aps-nvidia python=3.11 -y
conda activate aps-nvidia
python -m pip install --upgrade pip
python -m pip install -r requirements-nvidia-lock.txt
python -m pip install -e '.[test,download]'
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

The candidate environment pins upstream `vllm==0.20.2`, `transformers==4.57.1`, and `decord==0.6.0`. Direct dependencies are listed in `requirements-nvidia.txt`; `requirements-nvidia-lock.txt` contains 189 dependencies resolved for Linux x86_64 / Python 3.11. This is a dependency set for NVIDIA validation, not the original FlagOS environment or evidence of GPU runtime compatibility. The installed driver must meet the requirements of the PyTorch/vLLM wheels; see the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

The historical Qwen protocol preserves native video resolution. Qwen–MLVU uses a maximum context length of 127,800 and requires substantially more memory than the other combinations. Use a GPU with more memory or explicit tensor parallelism if needed. **Reducing resolution or frame count changes the evaluation protocol.** See [evaluation and reproducibility](docs/reproducibility.md).

## 3. Download the three answering models

ModelScope is the default download provider:

```bash
python scripts/download_models.py --models qwen7b internvl minicpm
```

| Model alias | Answering model | Main-table frames |
|---|---|---:|
| qwen25vl | Qwen2.5-VL-7B-Instruct | 32 |
| internvl | InternVL3-8B | 32 |
| minicpm | MiniCPM-V-4 | 16 |

Existing model snapshots can be placed or linked under the corresponding `models/` directories. Use the download script's `--revision` option to pin a model revision. Inference records model configuration hashes, weight-file metadata, and dependency versions.

The APS main experiments share BLIP-ITM Large relevance scores.

## 4. Prepare data and score caches

The reference protocol uses 1,337 LongVideoBench questions, 2,700 VideoMME questions, and 2,174 questions across the seven MLVU multiple-choice tasks. Subtitles are not supplied to the models.

If the historical experiment directory is available, import it to preserve the exact annotations, question order, and floating-point scores:

```bash
python scripts/import_legacy.py --source /path/to/previous/AKS
```

The resulting layout is:

```text
data/<dataset>/questions.json
data/<dataset>/videos/...
cache/<dataset>/blip/scores.json
cache/<dataset>/blip/frames.json
```

`<dataset>` is `longvideobench`, `videomme`, or `mlvu`. Video directories are linked rather than copied. The importer verifies historical SHA-256 hashes and stops on a mismatch.

See **[data preparation](docs/data.md)** for public data sources, MLVU conversion, and fresh scoring. Reusing selections requires matching annotations, question order, candidate frames, and the decoded video frame grid. Matching the question count alone is insufficient.

## 5. Select frames with APS

For LongVideoBench with a 32-frame budget:

```bash
python -m aps.select \
  --scores cache/longvideobench/blip/scores.json \
  --frames cache/longvideobench/blip/frames.json \
  --annotation data/longvideobench/questions.json \
  --output cache/selected/lvb_aps_k32.json \
  --budget 32 --alpha 1.0 --beta 0.6 --penalty 0.5 \
  --workers 8 --details
```

`--details` also saves partition boundaries, per-segment budgets, and candidate positions for inspection and visualization. The main-table parameters are:

| Benchmark | α | β | λ | Smoothing radius | Minimum segment length | Maximum linear segment length |
|---|---:|---:|---:|---:|---:|---:|
| LVB | 1.0 | 0.6 | 0.5 | 2 | 3 | 80 |
| VideoMME | 1.0 | 0.4 | 0.5 | 2 | 3 | 80 |
| MLVU | 1.0 | 0.4 | 0.5 | 2 | 3 | 80 |

Qwen and InternVL share scores, parameters, and 32-frame selections. MiniCPM uses 16-frame selections. Thus, the main table requires **six selection files and nine APS question-answering evaluations**.

After importing the historical caches, run `python scripts/verify_selections.py --workers 8` to repeat all 12,422 selection comparisons. The recorded validation is in `reports/selector_parity.json`.

## 6. Run one evaluation

Run preflight checks, then a small GPU smoke test, and finally the full benchmark. This example uses the reference LongVideoBench selections:

```bash
python -m aps.infer check \
  --dataset longvideobench --model models/Qwen2.5-VL-7B-Instruct \
  --candidates cache/longvideobench/blip/frames.json \
  --selected assets/selections/longvideobench/aps_k32.json \
  --methods aps --nframes 32 --gpus '0' --limit 8 \
  --run-dir outputs/qwen_lvb_smoke

python -m aps.infer run \
  --dataset longvideobench --model models/Qwen2.5-VL-7B-Instruct \
  --candidates cache/longvideobench/blip/frames.json \
  --selected assets/selections/longvideobench/aps_k32.json \
  --methods aps --nframes 32 --gpus '0' --limit 8 \
  --run-dir outputs/qwen_lvb_smoke
```

Preflight checks decode videos in the requested question range and verify frame grids without loading the answering-model weights. They do load the processor, and `check` does not replace a GPU smoke test. For a full run, remove `--limit 8` and use a separate directory such as `--run-dir outputs/qwen_lvb_full`.

Use `--gpus '0;1;2;3'` for data parallelism, with an independent model replica and question shard on each GPU. For two-GPU tensor parallelism, use `--gpus '0,1' --tensor-parallel-size 2`; for two such groups, use `--gpus '0,1;2,3' --tensor-parallel-size 2`. Keep the GPU-group string quoted.

Outputs under `--run-dir` include `manifest.json`, `rank*.jsonl`, `rank*.log`, and `summary.json`. Rerun the **same command** to resume. Use a new output directory when parameters, shard counts, or inputs change.

## 7. Run the nine main experiments

Print the execution plan without using GPUs:

```bash
python -m aps.experiments --config configs/main_table.json --gpus '0;1;2;3' --methods aps
```

Run eight-question smoke tests for all nine combinations:

```bash
python -m aps.experiments --config configs/main_table.json \
  --gpus '0;1;2;3' --methods aps --limit 8 --execute
```

After checking the logs, run the full evaluations:

```bash
python -m aps.experiments --config configs/main_table.json \
  --gpus '0;1;2;3' --methods aps --execute
```

Use `--job qwen25vl_longvideobench` to select one job; see the JSON configuration for other IDs. To include the Uniform control, use `--methods aps,uniform`, producing 18 results across nine model/benchmark combinations. The scheduler runs combinations sequentially, with parallel workers inside each combination. No machine names or IP addresses are fixed in the configuration.

## 8. Ablations

Configurations are in [configs/ablations.json](configs/ablations.json), with instructions in [docs/ablations.md](docs/ablations.md):

- **α/β:** α∈{0.5,1.0,1.5}, β∈{0.4,0.5,0.6,0.7}; Qwen7B on LVB/VideoMME.
- **Frame budget:** 8/16/32/64; Qwen7B, APS/Uniform, LVB/VideoMME.
- **Model size:** Qwen2.5-VL 3B/7B/32B; APS/Uniform, LVB/VideoMME.
- **Scoring encoder:** BLIP, CLIP, and BLIP-2; Qwen7B on LVB/VideoMME.
- **Partitioning:** `full`, `equal_length`, `linear_only`, and `constant_only`; Qwen and MiniCPM on all three benchmarks.
- **Complexity penalty λ:** 0/0.25/0.5/0.75 for InternVL on all three benchmarks; λ=0 for Qwen on LVB/VideoMME.

## 9. Metrics and historical results

LVB and VideoMME use question-level accuracy. **The main MLVU metric is the unweighted mean of the seven task accuracies.** Do not substitute MLVU micro accuracy in the main table.

Historical APS results are shown below. [reports/main_table.json](reports/main_table.json) contains nine APS aggregates and nine Uniform controls; results for other frame-selection methods are excluded. These measurements come from the original MetaX/FlagOS experiments, not the NVIDIA port.

| Model | Frames | LVB | VideoMME | MLVU Macro |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 32 | 63.80% | 62.33% | 68.05% |
| InternVL3-8B | 32 | 63.58% | 66.85% | 74.17% |
| MiniCPM-V-4 | 16 | 58.12% | 57.63% | 58.21% |

Answer parsing follows the historical experiments. LVB uses the original random fallback with a fixed per-question seed when parsing fails. VideoMME/MLVU count unparsed answers as incorrect and report their count separately. Some MiniCPM responses contain answer text rather than an option letter; parser changes must be applied consistently to all compared methods. InternVL's original Chinese system prompt is preserved verbatim because it is part of the evaluation input; code comments, docstrings, and runtime messages are in English.

## 10. Release preparation

```bash
pytest -q
python scripts/check_release.py
python -m build
```

This directory is a standalone Git repository. See [release instructions](docs/release.md) for setup and publishing steps. The final paper title, authors, citation, and license for the original APS code remain to be supplied by the owner.

Third-party provenance and retained licenses are documented in [third_party/NOTICE.md](third_party/NOTICE.md). Historical directory names and source references support data import and traceability; installing the old experiment framework is not required.
