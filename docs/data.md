# Data and relevance scores

## Download benchmarks

Obtain videos and annotations from the dataset providers:

| Benchmark | Source | Evaluation subset |
|---|---|---|
| LongVideoBench | [Official dataset](https://huggingface.co/datasets/longvideobench/LongVideoBench) | Validation, 1,337 questions |
| Video-MME | [Dataset](https://huggingface.co/datasets/lmms-lab/Video-MME) | 2,700 multiple-choice questions, without subtitles |
| MLVU | [Official dataset](https://huggingface.co/datasets/MLVU/MVLU) | Seven MCQ tasks, 2,174 questions |

Model weights, videos, and full annotation/score caches are not included in this repository. Existing video downloads can be linked into the following layout:

```text
data/
  longvideobench/questions.json
  longvideobench/videos/<video_path>
  videomme/questions.json
  videomme/videos/<videoID>.mp4
  mlvu/questions.json
  mlvu/videos/1_plotQA/<video_name>
  mlvu/videos/2_needle/<video_name>
  ...
cache/
  <dataset>/blip/frames.json
  <dataset>/blip/scores.json
```

## Prepare annotations

| Benchmark | ID | Video fields | Question and answer fields | Grouping fields |
|---|---|---|---|---|
| LVB | `id` | `video_path` | `question`, `candidates`, zero-based `correct_choice` | `duration_group`, `question_category` |
| Video-MME | `question_id` | `videoID` without extension | `question`, labeled `options`, letter `answer` | `duration`, `domain`, `sub_category`, `task_type` |
| MLVU | `question_id` | `video_relpath`, `video_name` | `question` including options, `candidates`, letter `answer` | `task_type` |

Convert MLVU's seven MCQ JSON files, from `1_plotQA.json` through `7_topic_reasoning.json`:

```bash
python scripts/convert_mlvu.py --json-dir /path/to/MLVU/json \
  --output data/mlvu/questions.json
```

The converter reconstructs the reference question order using recorded question-text hashes and video names. It rejects changed source data rather than assuming that row positions match. Generation tasks are excluded.

For LVB/Video-MME, use the evaluation JSON schema above and validate the annotation:

```bash
python scripts/prepare_annotations.py --dataset longvideobench \
  --input /path/to/include_frame_idx.json --output data/longvideobench/questions.json
```

The reference annotation snapshot comes from [AKS, revision b0b8a58](https://github.com/ncTimTang/AKS/tree/b0b8a58fedf1d05d78151e2969cecde60e83721d/datasets). Input hashes are recorded in [`assets/input_manifest.json`](../assets/input_manifest.json). This snapshot includes one LVB question variant relative to the original validation annotations; use the matching snapshot to compare directly with the reference results.

For another annotation version, add `--allow-new-protocol` and regenerate scores and selections. Reference selections must not be mixed with changed questions or row orders. File hashes include JSON serialization, so preserve the original bytes when importing reference inputs.

## Generate relevance scores

The scoring encoder is separate from the answering model. Main experiments share one BLIP-ITM Large score cache across all three answering models.

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

For multi-GPU scoring, launch shards with `--shard-id 0..N-1 --num-shards N`, assigning each process its own `CUDA_VISIBLE_DEVICES`. Keep model, annotations, shard count, and batch size fixed within one output directory. Merge after all shards finish; missing or duplicate questions are rejected.

Candidate indices are `j * int(fps)` for `j = 0..total_frames//int(fps)-1`. This approximates 1 FPS: at 29.97 FPS, the stride is 29 decoded frames, not exactly one second. Decoder frame counts and FPS must agree with any reused candidate cache.

BLIP scoring uses `BlipForImageTextRetrieval`, FP32, 384×384 PIL bicubic resizing, CLIP normalization, 35-token text truncation, and positive-class ITM probability. MLVU scores are rounded to six decimals. The reference LVB/Video-MME caches use the AKS/LAVIS scoring implementation; fresh Hugging Face scores have not been verified as numerically identical for every frame. Use matching reference caches when exact selection reproduction is required.

Encoder ablations use CLIP cosine similarity (`openai/clip-vit-base-patch32`) or BLIP-2 ITM probabilities (`Salesforce/blip2-itm-vit-g`, FP32, 32-token text truncation). All encoders use the same candidate grid. LVB/Video-MME scoring uses the question without appended answer candidates; MLVU's question field already includes options. See [ablations](ablations.md#scoring-encoder).

## Reuse matching caches

If a reference input bundle is available in the `data/` and `cache/` layout above, import it with checksum validation:

```bash
python scripts/import_cache.py --source /path/to/reference-inputs
python scripts/verify_selections.py --cache-root cache
```

The importer copies annotations and score/frame JSONs and links videos when present. For an AKS-format input bundle, use `--layout aks --source /path/to/AKS`. Importing caches does not require installing another frame-selection framework.

The six files in `assets/selections/` cover three benchmarks at 16 and 32 frames. They contain original video frame indices in annotation order. They require matching annotation, score, and candidate hashes; they are not standalone video data.
