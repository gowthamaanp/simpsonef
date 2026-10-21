"""Small statistics helpers shared by the uncertainty and explanation scripts (numpy only)."""
import numpy as np


def r2(p, t):
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion k / n."""
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def qhat(scores, alpha):
    """Conformal quantile: the ceil((n + 1)(1 - alpha)) / n empirical quantile of the calibration scores."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def auroc(positive, score):
    """Probability that a random positive case has a higher score than a random negative case (Mann-Whitney U)."""
    positive = np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([score[positive], score[~positive]]))) + 1.0
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
