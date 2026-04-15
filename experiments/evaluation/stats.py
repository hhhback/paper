from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float


def paired_bootstrap_ci(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap confidence interval for mean(metric_a) - mean(metric_b).

    For each bootstrap sample: resample N user-metric pairs with replacement,
    compute mean difference.
    CI = [alpha/2 percentile, 1-alpha/2 percentile] of B differences.
    p-value = 2 * min(proportion of diffs >= 0, proportion of diffs <= 0)
    """
    metric_a = np.asarray(metric_a, dtype=float)
    metric_b = np.asarray(metric_b, dtype=float)

    if len(metric_a) == 0 or len(metric_b) == 0:
        return BootstrapResult(mean_diff=0.0, ci_lower=0.0, ci_upper=0.0, p_value=1.0)

    n = len(metric_a)
    diffs = metric_a - metric_b
    observed_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diffs = diffs[indices].mean(axis=1)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    prop_ge_zero = np.mean(boot_diffs >= 0)
    prop_le_zero = np.mean(boot_diffs <= 0)
    p_value = float(2.0 * min(prop_ge_zero, prop_le_zero))
    p_value = min(p_value, 1.0)

    return BootstrapResult(
        mean_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni multiple comparison correction.

    Sort p-values p_(1) <= ... <= p_(m).
    Reject H_i if p_(i) <= alpha / (m - i + 1).
    Returns list of bool (True = reject null hypothesis = significant).
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * m

    for rank, (original_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            results[original_idx] = True
        else:
            break

    return results


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation).

    d = (mean_a - mean_b) / s_pooled
    s_pooled = sqrt(((n_a-1)*s_a^2 + (n_b-1)*s_b^2) / (n_a + n_b - 2))
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    if len(group_a) == 0 or len(group_b) == 0:
        return 0.0

    n_a, n_b = len(group_a), len(group_b)
    var_a = float(np.var(group_a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(group_b, ddof=1)) if n_b > 1 else 0.0

    denom_df = n_a + n_b - 2
    if denom_df <= 0:
        return 0.0

    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / denom_df)

    if s_pooled == 0.0:
        return 0.0

    return float((np.mean(group_a) - np.mean(group_b)) / s_pooled)
