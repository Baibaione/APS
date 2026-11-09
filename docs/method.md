# Algorithm and correspondence to the historical implementation

APS was previously named DPS. `kernels.py` preserves the six numerical functions from the original experiments. `selector.py` adds input validation, a common parameter interface, partition ablations, and diagnostic output. Computation order, floating-point formulas, and tie-breaking are preserved.

1. Apply per-question min-max normalization to the raw scores, yielding values in [0,1]. Return all zeros if the range is below 1e-12.
2. For partitioning, use a centered moving average with radius h=2. At the boundaries, divide by the actual window length.
3. Estimate the noise scale from first differences of the **unsmoothed** normalized scores: `sigma=max(1.4826*MAD(diff(x))/sqrt(2),0.03)`.
4. Dynamic programming minimizes the sum of fitting residuals and complexity penalties over all segments. For degree d∈{0,1}, a segment costs `SSE/(2*sigma^2) + (d+1)*0.5*log(max(N,2))*lambda`.
5. The minimum segment length is 3. In the historical implementation, **the maximum length of 80 applies only to linear segments; constant segments have no maximum length**. This asymmetric constraint is retained for reproduction.
6. At each right endpoint, first choose the lowest-cost constant candidate. Replace it with a linear candidate only if the latter has strictly lower cost. Cost ties favor the earlier left boundary.
7. Allocate using segment means of the **unsmoothed** scores: `weight_j=(mean_j+1e-6)^alpha * length_j^beta`. Assign one frame to every segment if the budget permits; otherwise assign one to each of the K highest-weight segments. Iteratively distribute the remaining budget using weights, integer quotas, and segment capacities. Weight ties favor earlier segments.
8. Within each segment, take the highest unsmoothed scores, breaking ties by earlier candidate position. Sort the final selections temporally and map them back to original video frame indices.

Segments use half-open intervals `[left,right)` over candidate positions, not video timestamps.

If N≤K, return all N candidates. If N<K, the question-answering entry point replaces the entire input with K Uniform samples from the **whole original video**, using the answering model's sampling rule. N=K does not trigger this fallback. The implementation neither pads with repeated frames nor merely fills the unallocated slots. Evaluation stops if the original video itself has fewer than K frames.

The partitioning DP enumerates left endpoints and takes O(N²) time. Prefix statistics, DP states, and backtracking use O(N) space. Efficiency measurements must distinguish scoring, partitioning, budget allocation, within-segment selection, and model inference. Historical hardware timings must not be presented as NVIDIA measurements.
