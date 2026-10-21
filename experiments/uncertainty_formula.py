"""Conformal intervals for the formula EF (normalised split conformal, Lei et al. 2018), calibrated on VAL, scored on TEST.

The scale of each case combines the BiGRU's own uncertainty with the disagreement between two independent EF routes:
s = (q95 - q05) / 2 + |EF_formula - EF_head|, where q05 / q95 / EF_head are the BiGRU head's EF quantiles on Network A's chords.
(The beat-to-beat spread of the formula EF was tested as a scale and detected errors less well.) Score_i = |y_i - EF_i| / s_i;
Q is its ceil((n + 1)(1 - alpha)) / n quantile on VAL and the interval is EF +- Q s. A constant-width interval (s = 1) is the
baseline. Reports coverage (overall and by EF category), width, error detection by width, referral of the widest intervals and
EF-category commit / defer, and writes results/uncertainty_formula.md and checkpoints/conformal_formula.json.

Usage: python experiments/uncertainty_formula.py --val results/formula_ef_VAL_d2s3.csv --test results/formula_ef_TEST_d2s3.csv \
           --features data_cache/features.npz --head checkpoints/ef_head.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from uncertainty_utils import auroc, qhat, spearman, wilson  # noqa: E402
from uncertainty import head_outputs  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CATS = [40.0, 50.0]
NAMES = ["EF < 40", "EF 40-49", "EF >= 50"]


def scale(df, c=None):
    return (df.q95 - df.q05).values / 2 + np.abs(df.ef - df.hmed).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True); ap.add_argument("--test", required=True); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--features", default="data_cache/features.npz", help="Network A chord cache the BiGRU reads")
    ap.add_argument("--head", default="checkpoints/ef_head.pt")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(ROOT, a.features), allow_pickle=True); z = {k: z[k] for k in z.files}
    hd = {}
    for sp in ("VAL", "TEST"):
        q, _, keys = head_outputs(z, np.where(z["splits"] == sp)[0], a.head, dev)
        hd[sp] = pd.DataFrame(dict(key=keys, hmed=q[:, 0], q05=q[:, 1], q95=q[:, 2]))
    v = pd.read_csv(os.path.join(ROOT, a.val)).merge(hd["VAL"], on="key"); t = pd.read_csv(os.path.join(ROOT, a.test)).merge(hd["TEST"], on="key")
    c = None
    L = [f"Formula EF intervals: calibrated on VAL (n={len(v)}), scored on TEST (n={len(t)}); alpha={a.alpha}; "
         f"scale = BiGRU half-width + |formula EF - BiGRU EF|", ""]
    out = {}
    for name, sv, st in (("constant width", np.ones(len(v)), np.ones(len(t))), ("BiGRU + disagreement scaled (ours)", scale(v), scale(t))):
        Q = qhat(np.abs(v.y.values - v.ef.values) / sv, a.alpha)
        lo, hi = t.ef.values - Q * st, t.ef.values + Q * st; cov = (t.y.values >= lo) & (t.y.values <= hi); w = hi - lo
        err = np.abs(t.y.values - t.ef.values); ci = wilson(int(cov.sum()), len(cov))
        L.append(f"{name}: coverage {100 * cov.mean():.1f}% (95% CI {100 * ci[0]:.1f}-{100 * ci[1]:.1f}), mean width {w.mean():.1f}, "
                 f"AUROC of width for |err|>10 {auroc(err > 10, w):.2f}, Spearman(width, |err|) {spearman(w, err):+.2f}")
        for nm, m in zip(NAMES, (t.y < 40, (t.y >= 40) & (t.y < 50), t.y >= 50)):
            m = m.values; L.append(f"   {nm:9s} n={m.sum():4d} coverage {100 * cov[m].mean():5.1f}%  width {w[m].mean():5.1f}")
        out[name] = (lo, hi, cov, w, Q)
    lo, hi, cov, w, Q = out["BiGRU + disagreement scaled (ours)"]; ef, y = t.ef.values, t.y.values; err = np.abs(y - ef)
    wv = 2 * Q * scale(v); w80 = float(np.quantile(wv, 0.8))
    L += ["", "REFERRAL of the widest intervals (TEST):", f"  {'referred':>8s} {'MAE kept':>9s} {'kept |err|>10':>14s}"]
    order = np.argsort(w)
    for fr in (0.0, 0.1, 0.2, 0.3):
        keep = order[:int(round(len(y) * (1 - fr)))]
        L.append(f"  {100 * fr:7.0f}% {err[keep].mean():9.2f} {100 * (err[keep] > 10).mean():13.1f}%")
    straddle = np.zeros(len(y), bool)
    for th in CATS:
        straddle |= (lo < th) & (hi >= th)
    ok = np.digitize(ef, CATS) == np.digitize(y, CATS); ci = wilson(int(ok[~straddle].sum()), int((~straddle).sum()))
    refer = w > w80
    L += ["", "DECISIONS (EF category < 40 / 40-49 / >= 50):",
          f"  all cases: category correct {100 * ok.mean():.1f}%",
          f"  COMMIT (interval inside one category): {100 * (~straddle).mean():.1f}% of cases, correct {100 * ok[~straddle].mean():.1f}% (95% CI {100 * ci[0]:.1f}-{100 * ci[1]:.1f})",
          f"  DEFER (interval crosses 40 or 50): {100 * straddle.mean():.1f}%, correct {100 * ok[straddle].mean():.1f}%",
          f"  REFER (width above the VAL 80th percentile, {w80:.1f}): {100 * refer.mean():.1f}% of cases; MAE of the rest {err[~refer].mean():.2f}"]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    open(os.path.join(ROOT, "results", "uncertainty_formula.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(Q90=float(Q), width90_p80=w80, alpha=a.alpha, scale="(q95-q05)/2 + |EF_formula - EF_head|"), open(os.path.join(ROOT, "checkpoints", "conformal_formula.json"), "w"), indent=1)
    t.assign(lo90=lo, hi90=hi, covered=cov, width=w, straddles=straddle).to_csv(os.path.join(ROOT, "results", "formula_ef_intervals_TEST.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
