# Reproducing the ablations

Commands are printed without execution unless `--execute` is explicitly supplied. Defaults follow the main table; only the selected ablation factor changes. `configs/ablations.json` contains 82 configuration entries, including reusable main-table controls, rather than 82 distinct experiments that must all be rerun.

The scheduler derives output directories from the full experiment parameters. Identical configurations can be reused across ablation axes; after checking the manifest, a job resumes or summarizes completed results. A historical directory name alone does not establish completion: importing old results also requires validating their parameters and inputs. Historical aggregates are stored separately under `reports/ablations/`.

## α/β

Qwen7B, 32 frames, BLIP, and λ=0.5 on LVB/VideoMME only: 3×4×2=24 combinations.

```bash
python -m aps.experiments --config configs/ablations.json --axis alpha_beta --gpus '0;1;2;3'
```

## Frame budget

Qwen7B with 8/16/32/64 frames, APS and Uniform, on LVB/VideoMME. The 64-frame configuration uses a context length of 49,152 and a 4 GiB KV cache, following the historical manifest while preserving the original image preprocessing. The 32-frame configuration is the main-table control.

```bash
python -m aps.experiments --config configs/ablations.json --axis frames --gpus '0;1;2;3'
```

## Model size

```bash
python scripts/download_models.py --models qwen3b qwen32b
python -m aps.experiments --config configs/ablations.json --job <EXACT_JOB_ID> --gpus '0,1;2,3'
```

The 32B configuration uses TP=2 and a 6 GiB KV cache per GPU, so select its exact job ID and use two-GPU groups. The 3B/7B configurations use TP=1, with groups such as `--gpus '0;1;2;3'`. A single invocation cannot use the same GPU grouping for mixed TP=1 and TP=2 jobs; the scheduler rejects incompatible groups.

## Scoring encoder

```bash
python scripts/download_models.py --provider huggingface --models clip blip2
CUDA_VISIBLE_DEVICES=0 python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/clip-vit-base-patch32 \
  --scorer clip --candidates cache/longvideobench/blip/frames.json \
  --output-dir cache/longvideobench/clip --num-shards 1
```

Run the same command with `--merge` to produce `scores.json` and `frames.json`. For BLIP-2, use `--scorer blip2 --model models/blip2-itm-vit-g --output-dir cache/longvideobench/blip2`. For VideoMME, change the dataset and paths accordingly. All encoders must reuse and validate the BLIP candidate grid.

```bash
python -m aps.experiments --config configs/ablations.json --axis encoder --gpus '0;1;2;3'
```

## Temporal partitioning

- `full`: dynamic programming with both constant and linear models.
- `equal_length`: first obtain the segment count J from full COMPS, then divide the candidate positions into J contiguous, equally sized segments up to integer rounding. Budget allocation and within-segment Top-K selection remain unchanged.
- `linear_only`: retain only linear fits in the DP, including the original maximum segment length.
- `constant_only`: retain only constant fits in the DP, with no maximum segment length.

Qwen7B and MiniCPM use their main-table budgets of 32 and 16 frames, respectively, on all three benchmarks. Each model has nine new evaluations: three variants × three benchmarks. The `full` variant is the main-table control.

```bash
python -m aps.experiments --config configs/ablations.json --axis partition --gpus '0;1;2;3'
```

## Complexity penalty λ

InternVL uses λ∈{0,0.25,0.5,0.75} on all three benchmarks. Qwen uses λ=0 on LVB/VideoMME only, with 32 frames. Keep α and β at the corresponding main-table values.

```bash
python -m aps.experiments --config configs/ablations.json --axis penalty --gpus '0;1;2;3'
python -m aps.experiments --config configs/ablations.json --axis penalty_zero --gpus '0;1;2;3'
```

Setting λ=0 removes only the complexity penalty. Smoothing, noise normalization, minimum segment length, maximum linear segment length, DP fitting costs, budget allocation, and Top-K selection remain active. It does not reduce APS to Uniform or enforce a fixed segment count.
