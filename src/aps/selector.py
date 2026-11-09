"""Portable APS and controlled partition ablations (CPU / NumPy)."""
from dataclasses import asdict, dataclass
import numpy as np
from .kernels import allocate_budget, fast_costs, minmax, partition, prefix_statistics, smooth


@dataclass(frozen=True)
class APSConfig:
    budget: int = 32
    alpha: float = 1.0
    beta: float = 0.6
    penalty: float = 0.5
    variant: str = "full"
    smoothing_radius: int = 2
    noise_floor: float = 0.03
    min_length: int = 3
    max_linear_length: int = 80

    def __post_init__(self):
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("budget must be a positive integer")
        for key in ("alpha", "beta", "penalty", "noise_floor"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.noise_floor == 0 or self.min_length < 2:
            raise ValueError("noise_floor > 0 and min_length >= 2 are required")
        if self.max_linear_length < self.min_length or self.smoothing_radius < 0:
            raise ValueError("Invalid segment length / smoothing radius")
        if self.variant not in ("full", "equal_length", "linear_only", "constant_only"):
            raise ValueError(f"Unknown partition variant: {self.variant}")


def single_model_partition(y, sigma, degree, cfg):
    """Ablation: remove one model without changing any remaining costs."""
    n, pref = len(y), prefix_statistics(y)
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    prev = np.full(n + 1, -1, dtype=int)
    base = 0.5 * np.log(max(n, 2)) * cfg.penalty
    for r in range(cfg.min_length, n + 1):
        lefts = np.arange(r - cfg.min_length + 1)
        lefts = lefts[np.isfinite(dp[lefts])]
        if degree == 1:
            lefts = lefts[r - lefts <= cfg.max_linear_length]
        if not len(lefts):
            continue
        costs = dp[lefts] + fast_costs(pref, r, lefts, degree) / (2 * sigma * sigma) + (degree + 1) * base
        j = int(np.argmin(costs))
        dp[r], prev[r] = float(costs[j]), int(lefts[j])
    if not np.isfinite(dp[n]):
        raise ValueError("No feasible partition under the specified length constraints")
    intervals, r = [], n
    while r:
        l = int(prev[r])
        intervals.append((l, r))
        r = l
    return intervals[::-1]


def validate_inputs(scores, frames):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("scores must be a nonempty finite 1D sequence")
    if len(x) != len(frames):
        raise ValueError("scores and frames must have equal lengths")
    if any(isinstance(f, (bool, np.bool_)) or not np.isfinite(f) or int(f) != f or f < 0 for f in frames):
        raise ValueError("frames must be nonnegative integer video frame indices")
    indices = [int(f) for f in frames]
    if indices != sorted(set(indices)):
        raise ValueError("candidate frames must be unique and ordered")
    return x, indices


def select_frames(scores, frames, config=None, *, return_details=False):
    """Return original video frame indices, chronologically sorted.

    If N < K, return all candidates and flag the downstream whole-video uniform
    fallback. The selector cannot reconstruct a full video from a score cache.
    N == K returns the candidate pool unchanged, without fallback.
    """
    cfg = config or APSConfig()
    scores, frames = validate_inputs(scores, frames)
    n = len(scores)
    details = {"config": asdict(cfg), "candidate_count": n,
               "uniform_fallback": n < cfg.budget, "partition_used": n > cfg.budget,
               "intervals": None, "allocation": None}
    if n <= cfg.budget:
        selected = frames
    else:
        x = minmax(scores)
        y = smooth(x, cfg.smoothing_radius)
        delta = np.diff(x)
        mad = np.median(np.abs(delta - np.median(delta))) if len(delta) else 0.0
        sigma = max(float(1.4826 * mad / np.sqrt(2)), cfg.noise_floor)
        if cfg.variant in ("full", "equal_length"):
            intervals = partition(y, sigma, cfg.min_length, cfg.max_linear_length, cfg.penalty)
            if cfg.variant == "equal_length":
                j = len(intervals)
                bounds = [k * n // j for k in range(j + 1)]
                intervals = list(zip(bounds[:-1], bounds[1:]))
                details["full_partition_count"] = j
        else:
            intervals = single_model_partition(y, sigma, int(cfg.variant == "linear_only"), cfg)
        assigned = allocate_budget(intervals, x, cfg.budget, cfg.alpha, cfg.beta)
        positions = sorted(i for (l, r), k in zip(intervals, assigned)
                           for i in sorted(range(l, r), key=lambda i: (-float(x[i]), i))[:int(k)])
        selected = [frames[i] for i in positions]
        if len(selected) != cfg.budget or len(set(selected)) != cfg.budget:
            raise RuntimeError("APS budget conservation failed")
        details.update(intervals=intervals, allocation=assigned.tolist(), sigma=sigma,
                       selected_positions=positions)
    return (selected, details) if return_details else selected
