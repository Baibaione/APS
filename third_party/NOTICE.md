# Third-party provenance

APS numerical kernels implement the method described in the accompanying paper.
A project-level license for original APS code has not yet been selected.

`src/aps/eval_utils/` contains the prompt and multiple-choice answer extraction
functions used for benchmark evaluation. These trace to
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) benchmark utilities
(including the MLVU 0.7.3 protocol); LongVideoBench parsing also credits
[MMMU](https://github.com/MMMU-Benchmark/MMMU). See `lmms-eval-LICENSE.txt` and
`Apache-2.0.txt` for the retained licensing text. Adaptations include preserving
the VideoMME template, the LVB letter-only fallback behavior, removing
unused subtitle-loading branches and packaging pure functions without importing
the entire evaluation framework.

The reference data/scoring pipeline uses
[AKS](https://github.com/ncTimTang/AKS), BLIP/LAVIS and benchmark utilities.
This repository does not redistribute the complete AKS implementation or claim
ownership of its code, annotations or precomputed scores. Follow the respective
upstream terms when obtaining third-party data or caches.

Model snapshots, benchmark videos, annotations and separately distributed
reproduction caches retain their respective rights and terms. Aggregated
experiment results and APS-selected frame indices do not include video pixels.
