# Data and score caches

## Exact inputs from the historical experiments

Use `python scripts/import_legacy.py --source /path/to/previous/AKS` to import an existing experiment directory. For each benchmark, the script copies only `include_frame_idx.json`, BLIP `scores.json`, and `frames.json`, and links local videos.

The separately prepared `APS-reproduction-inputs.tar.gz` is a local backup outside the Git repository. If available, extract it and pass its `AKS/` directory to the same importer; videos must be obtained and linked separately. The archive contains historical annotations and relevance caches, but no model weights or videos. Any future distribution must retain the data provenance and comply with the applicable licenses; third-party data is not original APS code.

## Prepare data from public sources

- [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench): obtain the validation annotations and corresponding videos.
- [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME): obtain the test annotations and videos. The historical protocol uses AKS's `include_frame_idx.json`; replacing it with another annotation version changes the protocol.
- [Official MLVU data](https://huggingface.co/datasets/MLVU/MVLU): use the seven MCQ tasks, from `MLVU/json/1_plotQA.json` through `7_topic_reasoning.json`, and their videos. Open-ended generation tasks are excluded. Follow the provider's access and download requirements.
- Historical annotation source: [ncTimTang/AKS](https://github.com/ncTimTang/AKS), revision `b0b8a58fedf1d05d78151e2969cecde60e83721d`. One historical LVB question differs from the original validation set, so the hashes recorded in this repository define the reference inputs.

Organize the videos as follows:

```text
data/longvideobench/videos/<video_path>
data/videomme/videos/<videoID>.mp4
data/mlvu/videos/1_plotQA/<video_name>
data/mlvu/videos/2_needle/<video_name>
... through 7_topic_reasoning/
```

This repository does not automatically download full video datasets, allowing existing downloads to be reused.

Convert the original MLVU JSON files:

```bash
python scripts/convert_mlvu.py --json-dir /path/to/MLVU/json \
  --output data/mlvu/questions.json
```

The converter reconstructs the reference question order using the recorded question-text hashes and video names. It fails if the source data changes rather than assuming row positions match. LVB/VideoMME use the historical evaluation JSON schemas. When exporting from Parquet, preserve the following fields and the question order:

| Benchmark | ID | Video fields | Question/options/answer | Grouping fields |
|---|---|---|---|---|
| LVB | `id` | `video_path` | `question`, `candidates`, zero-based `correct_choice` | `duration_group`, `question_category` |
| VideoMME | `question_id` | `videoID` without extension | `question`, `options` with A./B. labels, `answer` as a letter | `duration`, `domain`, `sub_category`, `task_type` |
| MLVU | `question_id` | `video_relpath`, `video_name` | `question` including options, `candidates`, `answer` as a letter | `task_type` |

Validate and install newly downloaded annotations:

```bash
python scripts/prepare_annotations.py --dataset longvideobench \
  --input /path/to/include_frame_idx.json --output data/longvideobench/questions.json
```

To deliberately use a different annotation version, pass `--allow-new-protocol`. Then regenerate scores and selections; the reference selections and claims of exact reproduction no longer apply. Even a change in JSON serialization changes the file hash. Import the original bytes for exact reproduction.

## Generate fresh scores

The scoring model is separate from the answering model. The three main-table models share one BLIP-ITM Large score cache. Encoder ablations change the scoring model while keeping the other selection parameters fixed. To generate fresh scores:

```bash
python scripts/download_models.py --provider huggingface --models blip
CUDA_VISIBLE_DEVICES=0 python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/blip-itm-large-coco \
  --scorer blip --output-dir cache/longvideobench/blip \
  --shard-id 0 --num-shards 1
python -m aps.score \
  --dataset longvideobench --annotation data/longvideobench/questions.json \
  --video-root data/longvideobench/videos --model models/blip-itm-large-coco \
  --scorer blip --output-dir cache/longvideobench/blip \
  --num-shards 1 --merge
```

Answering models are downloaded from ModelScope by default; the scoring example explicitly selects their official Hugging Face repositories. If a ModelScope mirror or a local snapshot is already available, pass its local directory with `--model`. The scripts do not silently switch models or download providers.

For multi-GPU scoring, launch shards with `--shard-id 0..N-1 --num-shards N`, assigning each process its own `CUDA_VISIBLE_DEVICES`. Keep shard count, batch size, model, and annotations fixed within an output directory. Run `--merge` after all shards complete; merging rejects missing or duplicate questions.

The candidate grid follows the historical implementation exactly: `stride=int(fps)` and candidate indices `j*stride` for `j=0..total_frames//stride-1`. This approximates one frame per second; a 29.97 FPS video uses stride 29, not exact one-second timestamps. Do not reuse a cache if decoding produces a different frame count or FPS.

Fresh BLIP scoring uses HF `BlipForImageTextRetrieval`, FP32, 384×384 PIL bicubic resizing, CLIP mean/std normalization, 35-token text truncation, and the positive-class ITM probability. MLVU retains the historical six-decimal score rounding. Historical LVB/VideoMME caches came from the AKS/LAVIS pipeline. **Fresh HF scores have not been proven numerically equivalent to those caches on every frame.** Import the historical caches for exact reproduction; a matching model name alone does not justify replacing them.

CLIP uses image/text cosine similarity from `openai/clip-vit-base-patch32`. BLIP-2 uses `Salesforce/blip2-itm-vit-g`, positive-class ITM probabilities, FP32, and 32-token question truncation. Encoder ablations cover LVB/VideoMME only, without appending answer candidates to the question. MLVU already includes options in its `question` field; the scorer does not append them again.
