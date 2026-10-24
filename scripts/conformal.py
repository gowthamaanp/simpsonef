"""Conformal 90% intervals for the formula EF (normalised split conformal, Lei et al. 2018): calibrate on VAL, score on TEST.

Scale of each case: s = (q95 - q05) / 2 + |EF - EF_BeatFinder|, BeatFinder's EF quantiles and median from FrameTracer's chords,
so s grows when the clip is uncertain or the two routes to EF disagree. Score_i = |y_i - EF_i| / s_i; Q is its
ceil((n + 1)(1 - alpha)) / n quantile on VAL and the interval is EF +- Q s. A constant-width interval (s = 1) is the baseline.
Writes checkpoints/conformal.json (read by src/pipeline.py) and results/conformal_TEST.csv.

Usage: python scripts/conformal.py --val results/echonet_VAL.csv --test results/echonet_TEST.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
CATS = [40.0, 50.0]
NAMES = ["EF < 40", "EF 40-49", "EF >= 50"]


def qhat(scores, alpha):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def auroc(pos, score):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = int(pos.sum()); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scale(df):
    return (df.q95 - df.q05).values / 2 + np.abs(df.ef - df.ef_beatfinder).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="results/echonet_VAL.csv"); ap.add_argument("--test", default="results/echonet_TEST.csv")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    v, t = pd.read_csv(os.path.join(ROOT, a.val)), pd.read_csv(os.path.join(ROOT, a.test))
    print(f"calibrated on VAL (n={len(v)}), scored on TEST (n={len(t)}), alpha {a.alpha}")
    for name, sv, st in (("constant width", np.ones(len(v)), np.ones(len(t))), ("normalised (ours)", scale(v), scale(t))):
        Q = qhat(np.abs(v.y.values - v.ef.values) / sv, a.alpha)
        lo, hi = t.ef.values - Q * st, t.ef.values + Q * st; cov = (t.y.values >= lo) & (t.y.values <= hi); w = hi - lo
        err = np.abs(t.y.values - t.ef.values); ci = wilson(int(cov.sum()), len(cov))
        print(f"  {name}: Q {Q:.4f}; coverage {100 * cov.mean():.1f}% (95% CI {100 * ci[0]:.1f}-{100 * ci[1]:.1f}), mean width {w.mean():.1f}, "
              f"AUROC of width for |err|>10 {auroc(err > 10, w):.2f}")
        for nm, m in zip(NAMES, (t.y < 40, (t.y >= 40) & (t.y < 50), t.y >= 50)):
            m = m.values; print(f"     {nm:9s} n={m.sum():4d} coverage {100 * cov[m].mean():5.1f}%  width {w[m].mean():5.1f}")
    order = np.argsort(w)
    for fr in (0.1, 0.2, 0.3):
        keep = order[:int(round(len(t) * (1 - fr)))]
        print(f"  referring the widest {100 * fr:.0f}%: MAE of the rest {err[keep].mean():.2f} (all {err.mean():.2f})")
    w80 = float(np.quantile(2 * Q * scale(v), 0.8))
    json.dump(dict(Q90=Q, width90_p80=w80, alpha=a.alpha, scale="(q95 - q05) / 2 + |EF - EF_BeatFinder|"),
              open(os.path.join(ROOT, "checkpoints", "conformal.json"), "w"), indent=1)
    straddle = np.zeros(len(t), bool)
    for c in CATS:
        straddle |= (lo < c) & (hi >= c)
    t.assign(lo90=lo, hi90=hi, covered=cov, width=w, straddles=straddle).to_csv(os.path.join(ROOT, "results", "conformal_TEST.csv"), index=False)


if __name__ == "__main__":
    main()
