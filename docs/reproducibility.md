# Evaluation protocol and NVIDIA port status

## Preserved experimental settings

- Qwen2.5-VL-7B and InternVL3-8B use 32 frames; MiniCPM-V-4 uses 16 frames.
- BF16, greedy decoding (`temperature=0`), and `seed=0`.
- At most 32 generated tokens for LVB and 16 for VideoMME/MLVU.
- No subtitles are supplied. The historical VideoMME prompt still contains the phrase “video and the subtitles”; this is template wording, not evidence that subtitles are loaded. It is retained consistently across methods.
- Qwen preserves the historical preprocessing: native video dimensions, alignment to multiples of 28, and bicubic resizing, with processor spatial resizing and additional frame sampling disabled. InternVL receives raw frames through the video channel. MiniCPM receives K PIL images with spaces between image placeholders.
- Qwen Uniform uses float32 linspace followed by rounding. InternVL/MiniCPM use equal-segment center sampling. When fewer than K candidates exist, the same model-specific Uniform rule replaces the entire input.
- Historical Qwen/InternVL experiments share BLIP scores and K=32 APS selections. Scores are not recomputed separately for each answering model.
- InternVL's original Chinese system prompt is retained verbatim. It is model input, not a code comment; translating it would change the evaluation protocol. Repository comments, docstrings, and runtime messages use English.

## Metrics

LVB/VideoMME accuracy is the number of correct answers divided by the total question count. MLVU uses the unweighted arithmetic mean of the seven MCQ task accuracies. Outputs also retain micro accuracy for auditing, but `main_table_accuracy_percent` is the main-table metric. Small MLVU smoke tests do not report the full-benchmark macro metric.

Completed generation does not guarantee successful parsing. Parsing failures are counted separately. Empty parsed choices are incorrect on MLVU/VideoMME; LVB preserves the original parser's random fallback with a fixed per-question seed. Such failures occur relatively often in the historical MiniCPM–MLVU outputs. A parser change requires consistent reevaluation across all compared methods.

## NVIDIA port

The port removes platform-specific device variables, MetaX GPU plugins, platform Triton paths, custom visual-encoder chunking injections, and Transformers compatibility shims. It uses upstream vLLM `LLM`/`SamplingParams` and `CUDA_VISIBLE_DEVICES`. Paths are supplied through the CLI, without fixed hostnames, containers, or NFS directories.

Dependency selection follows [vLLM 0.20.2 package metadata](https://pypi.org/project/vllm/0.20.2/) and the [upstream supported-model list](https://docs.vllm.ai/en/latest/models/supported_models/). NVIDIA wheels have different PyTorch dependencies from the original platform wheels, so the old version strings are not reused as CUDA installation instructions.

GPU weight loading, processor behavior, vLLM visual-token budgets, long-video memory usage, and attention backends on specific NVIDIA GPUs still require actual smoke tests. The original MetaX hardware does not provide an NVIDIA validation environment. CPU CI, static checks, and frame-selection parity do not establish CUDA inference compatibility.

## Memory and parallelism

The historical Qwen–MLVU context limit is 127,800, with a 10 GiB KV cache per GPU. Most other main-table combinations use a context limit of 32,768 and a 4 GiB KV cache; `configs/main_table.json` is authoritative. These are experiment settings, not total GPU memory requirements: weights, visual encoding, and temporary tensors consume additional memory.

The Qwen32B size ablation uses TP=2. Other jobs default to TP=1. To evaluate a different parallel configuration, pass `--tensor-parallel-size 2 --gpus '0,1'` to `aps.infer` and use a separate output directory. Changing parallelism or compute backends can change floating-point results; token-for-token equivalence is not guaranteed.

## Resumption and integrity

Each output manifest records annotations, candidates, selections, parser functions, code, model configuration, weight-file metadata, dependency versions, and run parameters. The same command resumes completed questions. After acquiring the directory lock, the launcher repairs an interrupted JSONL tail lacking a newline while preserving complete records. Workers read only their own shard files to avoid concurrent partial-line reads.

Summarization requires every expected `(question_index, method)` record and verifies question IDs, ground-truth answers, and correctness flags. Missing questions prevent a complete result from being reported. Use a new directory when changing the model, method, frame budget, or sharding configuration.
