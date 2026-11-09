# Third-party provenance

The APS numerical kernels originate from the author's experiment implementation
(previous method key: `dps`), developed within an AKS-based experiment tree.
The original project-level license for APS is not selected in this preparation.

`src/aps/eval_utils/` contains the prompt and multiple-choice answer extraction
functions used by the historical experiments. These trace to
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) benchmark utilities
(including the MLVU 0.7.3 protocol); LongVideoBench parsing also credits
[MMMU](https://github.com/MMMU-Benchmark/MMMU). See `lmms-eval-LICENSE.txt` and
`Apache-2.0.txt` for the retained licensing text. Adaptations include preserving
the historical VideoMME template, the LVB letter-only fallback behavior, removing
unused subtitle-loading branches and packaging pure functions without importing
the entire evaluation framework. Source-file hashes are in `reports/source_hashes.json`.

The historical data/scoring pipeline was based on
[AKS](https://github.com/ncTimTang/AKS), BLIP/LAVIS and benchmark utilities.
This repository does not redistribute the complete AKS implementation or claim
ownership of its code, annotations or precomputed scores. Follow the respective
upstream terms when obtaining third-party data or caches.

Model snapshots, benchmark videos, annotations and separately distributed
reproduction caches retain their respective rights and terms. Aggregated
experiment results and APS-selected frame indices do not include video pixels.
