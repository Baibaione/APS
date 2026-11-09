# Paper ablations

[`configs/ablations.json`](../configs/ablations.json) contains 63 configuration entries, including shared main-table controls. Commands preview the plan by default; append `--execute` to run. Output paths depend on experiment parameters, so identical configurations with matching inputs and runtime settings can reuse completed outputs.

| Experiment | Axis | Model | Benchmarks | Settings |
|---|---|---|---|---|
| Parameters, Fig. 3 | `alpha_beta` | Qwen7B, 32 frames | LVB, Video-MME | α ∈ {0.5, 1.0, 1.5}; β ∈ {0.4, 0.5, 0.6, 0.7} |
| Frame budget, Fig. 3 | `frames` | Qwen7B | LVB, Video-MME | 8, 16, 32, 64; APS and Uniform |
| Model scale, Table 4 | `scale` | Qwen2.5-VL | LVB, Video-MME | 3B, 7B, 32B; APS and Uniform |
| Encoder, Table 5 | `encoder` | Qwen7B, 32 frames | LVB, Video-MME | BLIP, CLIP, BLIP-2 |
| Partition design, Table 2 | `partition` | Qwen7B, 32 frames | LVB, Video-MME | Full, equal-length, linear-only, constant-only, no penalty |
| Regularization, Table 3 | `penalty` | InternVL3-8B, 32 frames | All three | η ∈ {0.25, 0.5, 0.75} |

Other parameters follow the [main settings](method.md#reference-settings). The config field `penalty` denotes η; the segment cost uses λ=(η/2) log N.

## Parameters and frame budgets

```bash
python -m aps.experiments --config configs/ablations.json \
  --axis alpha_beta --gpus '0;1;2;3'
python -m aps.experiments --config configs/ablations.json \
  --axis frames --gpus '0;1;2;3'
```

The 64-frame configuration uses a context limit of 49,152 and a 4 GiB KV cache. The 32-frame configuration is the main-table control.

## Model scale

Choose a job with matching GPU group sizes:

```bash
python scripts/download_models.py --models qwen3b qwen32b
python -m aps.experiments --config configs/ablations.json \
  --job scale_qwen25vl_longvideobench_3b --gpus '0;1;2;3'
python -m aps.experiments --config configs/ablations.json \
  --job scale_qwen25vl_longvideobench_32b --gpus '0,1;2,3'
```

Qwen32B uses TP=2 and a 6 GiB KV cache per GPU; the smaller models use TP=1. Replace `longvideobench` with `videomme` for the other benchmark. A single invocation cannot mix jobs requiring different group sizes.

## Scoring encoder

Generate CLIP and BLIP-2 caches on the same candidate frames used by BLIP:

```bash
python scripts/download_models.py --provider huggingface --models clip blip2
CUDA_VISIBLE_DEVICES=0 python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/clip-vit-base-patch32 \
  --scorer clip --candidates cache/longvideobench/blip/frames.json \
  --output-dir cache/longvideobench/clip --num-shards 1
```

Repeat with `--merge` to assemble `scores.json` and `frames.json`. For BLIP-2, use `--scorer blip2 --model models/blip2-itm-vit-g --output-dir cache/longvideobench/blip2`. For Video-MME, change the dataset and paths accordingly.

```bash
python -m aps.experiments --config configs/ablations.json \
  --axis encoder --gpus '0;1;2;3'
```

## Partition design

- **Full:** DP with constant and linear segment models.
- **Equal-length:** obtain the full method's segment count J, then split candidates into J contiguous intervals of equal size up to integer rounding.
- **Linear-only:** retain only linear fits, including their maximum segment length.
- **Constant-only:** retain only constant fits, with no maximum segment length.
- **No penalty:** set η=0, retaining both models, smoothing, noise normalization, and segment-length constraints.

All variants retain the same allocation and within-segment Top-K selection.

```bash
python -m aps.experiments --config configs/ablations.json \
  --axis partition --gpus '0;1;2;3'
```

## Regularization

```bash
python -m aps.experiments --config configs/ablations.json \
  --axis penalty --gpus '0;1;2;3'
```

InternVL's η=0.5 setting is the main-table control. MLVU uses β=0.4 and reports micro accuracy, with task macro accuracy also saved.
