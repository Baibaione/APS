# Reference results

[`reference.json`](reference.json) contains full-precision APS and Uniform accuracies, correct-answer counts, and sample counts for the nine model/benchmark combinations in the main table.

| Benchmark | Questions | Main-table metric |
|---|---:|---|
| LongVideoBench | 1,337 | Micro accuracy |
| Video-MME | 2,700 | Micro accuracy |
| MLVU | 2,174 | Micro accuracy |

MLVU also retains `task_macro_accuracy_percent`: the unweighted mean of its seven task accuracies. Micro and macro have different denominators and should not be interchanged. Unparsed responses remain in the evaluated sample count.

Qwen2.5-VL-7B and InternVL3-8B use 32 frames; MiniCPM-V-4 uses 16. All APS runs share BLIP scores, with α=1.0, η=0.5, and β=0.6 for LVB or β=0.4 for Video-MME/MLVU. The exact parameters are in [`configs/main_table.json`](../configs/main_table.json).

Reference measurements were obtained with the MetaX/FlagOS inference environment. Results from another runtime or freshly generated scoring caches may differ numerically. Follow the [evaluation protocol](../docs/reproducibility.md) and [input preparation](../docs/data.md) to keep comparisons consistent.
