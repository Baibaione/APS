# Method

APS selects a fixed number of frames using a query-conditioned relevance curve. The answering model remains frozen. The implementation separates scoring, temporal partitioning, budget allocation, and within-partition selection.

## Temporal partitioning

Per-question min-max normalization maps candidate relevance scores to $x_i\in[0,1]$. Constant sequences map to zero. Partitioning uses a centered moving average with radius $h=2$, normalized by the available window length at the boundaries. The noise scale is computed from the **unsmoothed** scores:

$$\sigma=\max\left(\frac{1.4826\,\operatorname{MAD}(\Delta x)}{\sqrt{2}},\;0.03\right).$$

Dynamic programming chooses contiguous segments and a constant or linear model for each segment. A segment fitted by a degree-$d$ model has cost

$$C(I,d)=\frac{\operatorname{RSS}(I,d)}{2\sigma^2}+(d+1)\lambda,\qquad\lambda=\frac{\eta}{2}\log N,\quad d\in\{0,1\}.$$

The configuration field `penalty` and CLI flag `--eta` represent **η**, the paper's regularization strength. `--penalty` is an equivalent CLI spelling. The main setting is η=0.5; λ itself depends on the candidate count N.

The minimum segment length is 3 candidates. Linear segments have a maximum length of 80; constant segments have no maximum length. Cost ties favor a constant fit and then the earlier left boundary. Intervals use half-open candidate positions `[left, right)`.

## Allocation and selection

For segment j, let $\bar{x}_j$ be its mean unsmoothed relevance and $n_j$ its length. Its allocation weight is

$$w_j=(\bar{x}_j+10^{-6})^{\alpha}\,n_j^{\beta}.$$

When the budget permits, every segment initially receives one frame. Otherwise, the K highest-weight segments receive one each. Remaining frames are distributed using weighted integer quotas, respecting segment capacities; weight ties favor earlier segments.

Within each segment, select the highest unsmoothed scores, breaking ties by earlier candidate position. Return the selected original video frame indices in chronological order.

## Reference settings

| Benchmark | α | β | η | Candidate sampling |
|---|---:|---:|---:|---|
| LongVideoBench | 1.0 | 0.6 | 0.5 | Approximately 1 FPS |
| Video-MME | 1.0 | 0.4 | 0.5 | Approximately 1 FPS |
| MLVU | 1.0 | 0.4 | 0.5 | Approximately 1 FPS |

These settings correspond to the released main-table selections and results. Qwen2.5-VL-7B and InternVL3-8B use K=32; MiniCPM-V-4 uses K=16. See [data preparation](data.md) for the exact sampling rule and [ablations](ablations.md) for controlled variants.

## Short videos and complexity

For N=K, APS returns the candidate pool unchanged. For N<K, the selector returns all candidates and flags a fallback: the inference runner replaces the whole selection with K Uniform samples from the **original video**, using the answering model's sampling rule. It does not merely fill spare slots. Evaluation stops if the video itself has fewer than K decoded frames.

The partitioning DP takes $O(N^2)$ time and $O(N)$ working memory through prefix statistics, DP states, and backtracking. Scoring and answering-model inference are separate stages.
