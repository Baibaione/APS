"""APS numerical kernels, preserved from the experiment implementation.

Intervals are half-open candidate positions, not original video frame numbers.
Only LINEAR segments have a maximum length; constants are unbounded.
"""
import numpy as np

def minmax(x):
    x = np.asarray(x, dtype=float)
    d = np.ptp(x)
    return np.zeros_like(x) if d < 1e-12 else (x - x.min()) / d

def smooth(x, h=2):
    if h <= 0:
        return x.copy()
    c = np.r_[0.0, np.cumsum(x)]
    return np.array([(c[min(len(x), i + h + 1)] - c[max(0, i - h)]) / (min(len(x), i + h + 1) - max(0, i - h)) for i in range(len(x))])

def prefix_statistics(y):
    t = np.arange(len(y), dtype=float)
    return [np.r_[0.0, np.cumsum(a)] for a in (np.ones(len(y)), t, t * t, y, t * y, y * y)]

def fast_costs(pref, r, lefts, degree):
    n, sx, sxx, sy, sxy, syy = [p[r] - p[lefts] for p in pref]
    c = np.maximum(0.0, syy - sy * sy / n)
    if degree == 0:
        return c
    det = n * sxx - sx * sx
    slope = np.divide(n * sxy - sx * sy, det, out=np.zeros_like(det), where=det > 1e-10)
    intercept = (sy - slope * sx) / n
    return np.where(n >= 2, np.maximum(0.0, syy - intercept * sy - slope * sxy), c)

def partition(y, sigma, min_length=3, max_length=80, penalty_scale=0.5):
    """带惩罚的最优分段：degree-0（常数拟合）代价 + base，degree-1（线性拟合）
        代价 + 2*base，仅线性拟合受 max_length 上限约束，DP 求全局最优。"""
    n, pref = (len(y), prefix_statistics(y))
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    prev = np.full(n + 1, -1, dtype=int)
    base = 0.5 * np.log(max(n, 2)) * penalty_scale
    for r in range(min_length, n + 1):
        lefts = np.arange(r - min_length + 1)
        lefts = lefts[np.isfinite(dp[lefts])]
        if not len(lefts):
            continue
        scores0 = dp[lefts] + fast_costs(pref, r, lefts, 0) / (2 * sigma * sigma) + base
        j = int(np.argmin(scores0))
        best, best_l = (float(scores0[j]), int(lefts[j]))
        short = lefts[r - lefts <= max_length]
        if len(short):
            scores1 = dp[short] + fast_costs(pref, r, short, 1) / (2 * sigma * sigma) + 2 * base
            j = int(np.argmin(scores1))
            if scores1[j] < best:
                best, best_l = (float(scores1[j]), int(short[j]))
        dp[r], prev[r] = (best, best_l)
    if not np.isfinite(dp[n]):
        raise RuntimeError('No feasible partition')
    intervals, r = ([], n)
    while r:
        l = int(prev[r])
        intervals.append((l, r))
        r = l
    return intervals[::-1]

def allocate_budget(intervals, x, budget, alpha=1.0, beta=0.5):
    """段预算分配：权重 = 段均值^alpha * 段长^beta；每段保底1帧（预算够时），
        剩余按权重水线填充，容量受段长约束，装不下的自然再分配。"""
    means = np.array([np.mean(x[l:r]) for l, r in intervals])
    lengths = np.array([r - l for l, r in intervals], dtype=int)
    weights = (means + 1e-06) ** alpha * lengths.astype(float) ** beta
    assigned = np.zeros(len(intervals), dtype=int)
    if budget >= len(intervals):
        assigned[:] = 1
    else:
        order = np.lexsort((np.arange(len(intervals)), -weights))
        assigned[order[:budget]] = 1
    remaining = budget - int(assigned.sum())
    while remaining:
        available = np.where(assigned < lengths)[0]
        if not len(available):
            break
        w = weights[available]
        if w.sum() <= 0:
            w = np.ones(len(available))
        quotas = remaining * w / w.sum()
        floor = np.minimum(np.floor(quotas).astype(int), lengths[available] - assigned[available])
        if floor.sum():
            assigned[available] += floor
            remaining -= int(floor.sum())
        else:
            order = np.lexsort((available, -quotas))
            assigned[int(available[order[0]])] += 1
            remaining -= 1
    return assigned
