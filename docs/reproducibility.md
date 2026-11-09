# Evaluation

## Main experiments

[`configs/main_table.json`](../configs/main_table.json) defines nine model/benchmark combinations. The models share a BLIP relevance cache. Qwen2.5-VL-7B and InternVL3-8B use 32 frames; MiniCPM-V-4 uses 16.

```bash
# Preview all commands.
python -m aps.experiments --config configs/main_table.json --gpus '0;1;2;3'

# Evaluate both methods.
python -m aps.experiments --config configs/main_table.json \
  --methods aps,uniform --gpus '0;1;2;3' --execute
```

Semicolons separate worker groups. Each group holds a model replica and processes a question shard. Experiments run sequentially, with the selected groups working together within each experiment. Quote the argument to prevent shell interpretation.

Use `--job qwen25vl_longvideobench` to select one experiment. Add `--limit 8` for a smoke test, whose outputs are stored separately from full evaluations.

Direct inference accepts a selected-frame file:

```bash
python -m aps.infer run --dataset longvideobench --data-root data \
  --model models/Qwen2.5-VL-7B-Instruct \
  --candidates cache/longvideobench/blip/frames.json \
  --selected assets/selections/longvideobench/aps_k32.json \
  --methods aps,uniform --nframes 32 --gpus '0;1;2;3' \
  --max-model-len 32768 --kv-cache-gib 4 --seed 0 \
  --run-dir outputs/qwen_lvb_k32
```

Reference selections require matching annotations and candidate caches; see [data preparation](data.md).

## Protocol

- BF16, greedy decoding (`temperature=0`), seed 0.
- Maximum generated tokens: 32 for LongVideoBench, 16 for Video-MME/MLVU.
- No subtitles. The Video-MME prompt retains the benchmark template's phrase “video and the subtitles,” but no subtitle content is loaded.
- Qwen uses native video dimensions aligned to multiples of 28 with bicubic resizing; processor spatial resizing and extra frame sampling are disabled. InternVL receives raw frames through the video channel. MiniCPM receives K PIL images with spaces between image placeholders.
- Qwen Uniform uses float32 linspace followed by rounding. InternVL/MiniCPM use equal-segment center sampling. Short-candidate fallback uses the same model-specific rule over the whole video.
- InternVL's Chinese system prompt is preserved verbatim as model input.

## Accuracy

The main table uses **question-level micro accuracy** on all three benchmarks: correct answers divided by total questions. Full MLVU evaluation also computes **task macro accuracy**, the unweighted mean across seven MCQ tasks.

| Output field | Meaning |
|---|---|
| `accuracy_percent` | Question-level micro accuracy |
| `task_macro_accuracy_percent` | MLVU seven-task macro accuracy |
| `main_table_accuracy_percent` | Selected reporting metric; micro in the paper configs |
| `main_table_metric` | Metric selected for the run |

Use `--mlvu-metric macro` with direct inference to select macro reporting; both statistics remain available. Small MLVU smoke tests do not report a full-benchmark main-table score. Aggregation does not affect model outputs or selected frames.

Parsing failures remain in the denominator. Empty parsed choices are incorrect for MLVU/Video-MME. LongVideoBench uses its benchmark parser's random fallback with a fixed per-question seed. Parsing failures and Uniform frame fallbacks are counted separately.

## Memory and parallelism

Qwen–MLVU uses a context limit of 127,800 and a 10 GiB KV cache per GPU. Most other main-table jobs use 32,768 and 4 GiB. Weights, visual encoding, and temporary tensors consume additional memory; KV cache values are not total GPU requirements.

Main-table jobs use TP=1. The Qwen32B scale ablation uses TP=2: commas join devices within a group, as in `--gpus '0,1;2,3'`. With direct inference, also set `--tensor-parallel-size 2`. The launcher validates group sizes against the selected config. Use a new output directory when changing parallelism or numerical settings.

## Outputs and resumption

Each run writes `manifest.json`, per-worker logs and JSONL predictions, and `summary.json`. The manifest binds annotations, candidates, selections, code, parser functions, model configuration, weight-file metadata, dependencies, and run parameters.

Rerun the same command to resume. Interrupted JSONL tails are repaired while complete records are retained. Summarization verifies expected question/method coverage, IDs, ground truth, and correctness flags; an incomplete run cannot report a complete benchmark result. Changed inputs or settings require a separate output directory.
