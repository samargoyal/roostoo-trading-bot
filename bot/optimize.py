"""Small convex portfolio optimisers in plain Python (no numpy), for a handful of assets.

  covariance       sample covariance of aligned return series, shrunk towards its diagonal
  erc_weights      equal risk contribution: each asset adds the same share of portfolio
                   variance. Solved as the strictly convex problem
                   min 1/2 y'Cy - (1/n) sum(log y_i), then y normalised (Spinu, 2013),
                   by cyclical coordinate descent.
  erc_weights_fixed  the same when some weights are fixed (coins whose trading is halted):
                   the free weights share the remaining budget with equal risk contributions,
                   counting their covariance with the fixed holdings.
  min_variance     minimum variance with weights summing to 1 and 0 <= w_i <= cap, by
                   projected gradient descent with an exact projection onto that set.
"""
import math
from typing import Dict, List, Sequence

Matrix = List[List[float]]


def covariance(series: Sequence[Sequence[float]], shrink: float = 0.1) -> Matrix:
    """Covariance of equally long return series (one per asset), shrunk towards the diagonal."""
    n = len(series)
    t = len(series[0])
    means = [sum(s) / t for s in series]
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            c = sum((series[i][k] - means[i]) * (series[j][k] - means[j]) for k in range(t)) / (t - 1)
            if i != j:
                c *= 1.0 - shrink
            cov[i][j] = cov[j][i] = c
    return cov


def _matvec(m: Matrix, v: Sequence[float]) -> List[float]:
    return [sum(row[k] * v[k] for k in range(len(v))) for row in m]


def erc_weights(cov: Matrix, iters: int = 1000, tol: float = 1e-12) -> List[float]:
    """Weights (summing to 1) with equal risk contributions w_i (Cw)_i."""
    n = len(cov)
    b = 1.0 / n
    y = [1.0 / math.sqrt(cov[i][i]) for i in range(n)]
    for _ in range(iters):
        change = 0.0
        for i in range(n):
            c = sum(cov[i][k] * y[k] for k in range(n)) - cov[i][i] * y[i]
            new = (-c + math.sqrt(c * c + 4.0 * cov[i][i] * b)) / (2.0 * cov[i][i])
            change = max(change, abs(new - y[i]))
            y[i] = new
        if change < tol:
            break
    total = sum(y)
    return [v / total for v in y]


def erc_weights_fixed(cov: Matrix, fixed: Dict[int, float], budget: float,
                      iters: int = 1000, tol: float = 1e-12) -> List[float]:
    """Weights for every asset: the `fixed` ones (index -> weight) as given, the free ones
    summing to `budget` with equal risk contributions w_i (Cw)_i, Cw including the fixed ones.

    For a risk budget b the free weights solve the strictly convex problem
        min 1/2 y'C_ff y + y'C_fx x - b sum(log y_i)        (x: the fixed weights)
    whose optimum has y_i (Cw)_i = b for every free asset; b is then set by bisection so
    that the free weights sum to the budget (their sum rises with b).
    """
    n = len(cov)
    free = [i for i in range(n) if i not in fixed]
    w = [fixed.get(i, 0.0) for i in range(n)]
    if not free or budget <= 0:
        return w
    for i in free:
        w[i] = budget / len(free)

    def total(b: float) -> float:
        for _ in range(iters):
            change = 0.0
            for i in free:
                c = sum(cov[i][k] * w[k] for k in range(n)) - cov[i][i] * w[i]
                new = (-c + math.sqrt(c * c + 4.0 * cov[i][i] * b)) / (2.0 * cov[i][i])
                change = max(change, abs(new - w[i]))
                w[i] = new
            if change < tol * budget:
                break
        return sum(w[i] for i in free)

    lo, hi = 0.0, budget * budget * max(cov[i][i] for i in free)
    while total(hi) < budget:
        lo, hi = hi, hi * 4.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if total(mid) < budget:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    scale = budget / total(hi)
    for i in free:
        w[i] *= scale
    return w


def project_capped_simplex(v: Sequence[float], cap: float) -> List[float]:
    """Euclidean projection onto {sum(w) = 1, 0 <= w_i <= cap} (needs cap * n >= 1)."""
    lo, hi = min(v) - cap, max(v)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if sum(min(max(x - mid, 0.0), cap) for x in v) > 1.0:
            lo = mid
        else:
            hi = mid
    return [min(max(x - hi, 0.0), cap) for x in v]


def min_variance(cov: Matrix, cap: float = 1.0, iters: int = 2000, tol: float = 1e-12) -> List[float]:
    """argmin w'Cw subject to sum(w) = 1 and 0 <= w_i <= cap."""
    n = len(cov)
    cap = max(cap, 1.0 / n)
    # Step size from a bound on the largest eigenvalue (the largest absolute row sum).
    step = 1.0 / max(sum(abs(x) for x in row) for row in cov)
    w = project_capped_simplex([1.0 / n] * n, cap)
    for _ in range(iters):
        grad = _matvec(cov, w)
        new = project_capped_simplex([w[i] - step * grad[i] for i in range(n)], cap)
        if max(abs(new[i] - w[i]) for i in range(n)) < tol:
            w = new
            break
        w = new
    return w


def mean_variance(alpha: Sequence[float], cov: Matrix, risk_aversion: float = 1.0, cap: float = 1.0,
                  iters: int = 2000, tol: float = 1e-12) -> List[float]:
    """argmax alpha'w - risk_aversion / 2 x w'Cw subject to sum(w) = 1 and 0 <= w_i <= cap, with C
    scaled to a mean variance of 1 so that alpha (normal scores) and risk are on one scale; a
    concave problem, solved by projected gradient ascent (round 83)."""
    n = len(cov)
    cap = max(cap, 1.0 / n)
    scale = sum(cov[i][i] for i in range(n)) / n or 1.0
    c = [[x / scale for x in row] for row in cov]
    bound = max(sum(abs(x) for x in row) for row in c)
    step = 1.0 / (risk_aversion * bound) if risk_aversion > 0 and bound > 0 else 0.1
    w = project_capped_simplex([1.0 / n] * n, cap)
    for _ in range(iters):
        cw = _matvec(c, w)
        new = project_capped_simplex([w[i] + step * (alpha[i] - risk_aversion * cw[i]) for i in range(n)], cap)
        if max(abs(new[i] - w[i]) for i in range(n)) < tol:
            return new
        w = new
    return w


def hrp_weights(cov: Matrix) -> List[float]:
    """Hierarchical risk parity (Lopez de Prado, 2016), split along the cluster tree: assets
    are clustered by correlation distance (single linkage), and from the top of the tree down
    each cluster's budget goes to its two halves in inverse proportion to their variance (each
    half held by inverse variance), so similar assets share one budget and no matrix is
    inverted (round 83)."""
    n = len(cov)
    if n == 1:
        return [1.0]
    sd = [math.sqrt(max(cov[i][i], 1e-18)) for i in range(n)]
    dist = [[math.sqrt(max(0.0, 0.5 * (1 - cov[i][j] / (sd[i] * sd[j])))) for j in range(n)] for i in range(n)]
    clusters = {i: [i] for i in range(n)}
    children = {}                                   # a merged cluster -> its two parts
    while len(clusters) > 1:
        keys = list(clusters)
        best = None
        for a_i, a in enumerate(keys):
            for b in keys[a_i + 1:]:
                d = min(dist[x][y] for x in clusters[a] for y in clusters[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        merged = tuple(clusters[a] + clusters[b])
        children[merged] = (tuple(clusters[a]), tuple(clusters[b]))
        clusters[a] = list(merged)
        del clusters[b]

    def variance(items):
        ivp = [1.0 / cov[i][i] if cov[i][i] > 0 else 0.0 for i in items]
        total = sum(ivp) or 1.0
        w = [x / total for x in ivp]
        return sum(w[a] * w[b] * cov[items[a]][items[b]] for a in range(len(items)) for b in range(len(items)))

    weights = [1.0] * n
    stack = [tuple(next(iter(clusters.values())))]
    while stack:
        node = stack.pop()
        if node not in children:
            continue
        left, right = children[node]
        vl, vr = variance(left), variance(right)
        share = 1.0 - vl / (vl + vr) if vl + vr > 0 else 0.5
        for i in left:
            weights[i] *= share
        for i in right:
            weights[i] *= 1.0 - share
        stack += [left, right]
    return weights


def risk_contributions(cov: Matrix, w: Sequence[float]) -> List[float]:
    cw = _matvec(cov, w)
    total = sum(w[i] * cw[i] for i in range(len(w)))
    return [w[i] * cw[i] / total for i in range(len(w))]
