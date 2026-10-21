"""EF uncertainty: conformalised quantile regression (CQR, Romano et al. 2019) on the head's 5% / 50% / 95% outputs.

Score_i = max(q05_i - y_i, y_i - q95_i) on the calibration cases; Q is its ceil((n + 1)(1 - alpha)) / n empirical quantile and the
interval is [q05 - Q, q95 + Q]. The evaluation cases only measure coverage, width, conditional coverage, error detection by
interval width, referral, and EF-category commit / defer. A class-conditional (Mondrian) variant calibrates Q inside each
predicted EF category.

  --mode dev     calibrate on a random half of VAL, evaluate on the other half (TEST is not read)
  --mode final   calibrate on all of VAL, evaluate on TEST (run once, for the final model)

Writes results/uncertainty_<mode>.md, results/ef_intervals_<mode>.csv and checkpoints/conformal_<mode>.json (the corrections the
inference script applies).

Usage: python experiments/uncertainty.py --features data_cache/features.npz --head checkpoints/ef_head.pt --mode dev
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
import train_heads as th  # noqa: E402
from temporal_detector import TemporalEDESDetector  # noqa: E402
from uncertainty_utils import auroc, qhat, r2, spearman, wilson  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CATS = [40.0, 50.0]
NAMES = ["EF < 40", "EF 40-49", "EF >= 50"]


def head_outputs(z, idx, head_path, dev):
    """(n, 3) [median, q05, q95] EF in points, labels, and video keys for the clips `idx`."""
    ck = torch.load(os.path.join(ROOT, head_path), map_location=dev, weights_only=False)
    net = TemporalEDESDetector(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    X, L, ed, es, y = th.build(z, idx)
    X = torch.from_numpy((X - ck["mu"]) / ck["sd"]) * (torch.arange(th.T)[None, :, None] < torch.from_numpy(L)[:, None, None])
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out.append(net.forward_all(X[i:i + 64].to(dev), torch.from_numpy(L[i:i + 64]))[1].cpu().numpy() * 100)
    o = np.concatenate(out)
    return np.stack([o[:, 0], np.minimum(o[:, 1], o[:, 2]), np.maximum(o[:, 1], o[:, 2])], 1), y.astype(float), z["keys"][idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data_cache/features.npz")
    ap.add_argument("--head", default="checkpoints/ef_head.pt")
    ap.add_argument("--mode", choices=["dev", "final"], default="dev")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(ROOT, args.features), allow_pickle=True); z = {k: z[k] for k in z.files}
    val = np.where(z["splits"] == "VAL")[0]
    if args.mode == "dev":
        perm = np.random.RandomState(0).permutation(len(val)); cal_ix, ev_ix, ev_name = val[perm[:len(val) // 2]], val[perm[len(val) // 2:]], "VAL (held-out half)"
    else:
        cal_ix, ev_ix, ev_name = val, np.where(z["splits"] == "TEST")[0], "TEST"
    qc, yc, _ = head_outputs(z, cal_ix, args.head, dev)
    qe, ye, keys = head_outputs(z, ev_ix, args.head, dev)
    med, lo5, hi5 = qe[:, 0], qe[:, 1], qe[:, 2]; err = np.abs(ye - med)
    L = [f"CQR on the quantile head. Calibrated on {'a random half of VAL' if args.mode == 'dev' else 'all of VAL'} (n={len(yc)}); scored on {ev_name} (n={len(ye)}).", "",
         f"point estimate (median output): R2 {r2(med, ye):.3f}, MAE {err.mean():.2f}, r {np.corrcoef(med, ye)[0, 1]:.3f}"]
    out = pd.DataFrame(dict(key=keys, y=ye, yhat=med)); cal = {"mode": args.mode, "n_cal": int(len(yc))}
    res = {}
    for alpha in (0.1, 0.2):
        if alpha == 0.1:
            Q = qhat(np.maximum(qc[:, 1] - yc, yc - qc[:, 2]), alpha); lo, hi = lo5 - Q, hi5 + Q
        else:                                       # the 5/95% pair is a 90% interval, so 80% uses a symmetric band around the median
            Q = qhat(np.abs(yc - qc[:, 0]), alpha); lo, hi = med - Q, med + Q
        cov = (ye >= lo) & (ye <= hi); res[alpha] = (lo, hi, cov); a, b = wilson(int(cov.sum()), len(cov))
        L.append(f"{int(100 * (1 - alpha))}% intervals: Q = {Q:.2f}; realised coverage {100 * cov.mean():.1f}% (95% CI {100 * a:.1f}-{100 * b:.1f}); "
                 f"mean width {(hi - lo).mean():.1f}, median {np.median(hi - lo):.1f}, range {(hi - lo).min():.1f}-{(hi - lo).max():.1f}")
        out[f"lo{int(100 * (1 - alpha))}"], out[f"hi{int(100 * (1 - alpha))}"] = lo, hi; cal[f"Q{int(100 * (1 - alpha))}"] = Q
    lo, hi, cov = res[0.1]; width = hi - lo
    wc = (qc[:, 2] - qc[:, 1]) + 2 * cal["Q90"]                                       # widths of the 90% intervals on the CALIBRATION cases
    cal["width90_p80"] = float(np.quantile(wc, 0.8)); cal["width90_p90"] = float(np.quantile(wc, 0.9))
    L += ["", "CONDITIONAL coverage of the 90% intervals (guaranteed only on average, so reported by slice):"]
    for nm, m in zip(NAMES, (ye < 40, (ye >= 40) & (ye < 50), ye >= 50)):
        if m.sum() > 0:
            a, b = wilson(int(cov[m].sum()), int(m.sum()))
            L.append(f"  {nm:9s} n={int(m.sum()):4d}  coverage {100 * cov[m].mean():5.1f}% (95% CI {100 * a:.1f}-{100 * b:.1f})  mean width {width[m].mean():5.1f}")
    tb = np.digitize(width, np.quantile(width, [1 / 3, 2 / 3]))
    for nm, k in zip(("narrowest third", "middle third", "widest third"), range(3)):
        m = tb == k
        L.append(f"  {nm:15s} n={int(m.sum()):4d}  coverage {100 * cov[m].mean():5.1f}%  mean width {width[m].mean():5.1f}  MAE {err[m].mean():.2f}")
    L += ["", f"ERROR DETECTION by interval width: Spearman(width, |error|) {spearman(width, err):+.2f}; AUROC |error|>7.5: {auroc(err > 7.5, width):.3f}, "
              f">10: {auroc(err > 10, width):.3f}, >15: {auroc(err > 15, width):.3f}",
          "", "REFERRAL: hand the widest-interval x% of cases to a human; metrics on the kept cases",
          f"  {'referred':>8s} {'MAE kept':>9s} {'MAE random':>11s} {'MAE oracle':>11s} {'kept |err|>10':>14s} {'kept |err|>15':>14s}"]
    order = np.argsort(width); rg = np.random.RandomState(1)
    for frac in (0.0, 0.1, 0.2, 0.3, 0.5):
        keep = order[:int(round(len(ye) * (1 - frac)))]
        rnd = np.mean([err[rg.permutation(len(err))[:len(keep)]].mean() for _ in range(100)]); orc = np.sort(err)[:len(keep)].mean()
        L.append(f"  {100 * frac:7.0f}% {err[keep].mean():9.2f} {rnd:11.2f} {orc:11.2f} {100 * (err[keep] > 10).mean():13.1f}% {100 * (err[keep] > 15).mean():13.1f}%")
    straddle = np.zeros(len(ye), bool)
    for t in CATS:
        straddle |= (lo < t) & (hi >= t)
    ok = np.digitize(med, CATS) == np.digitize(ye, CATS); a, b = wilson(int(ok[~straddle].sum()), int((~straddle).sum()))
    L += ["", "EF CATEGORY (<40 / 40-49 / >=50) with deferral on the 90% interval:",
          f"  all cases                         : category correct {100 * ok.mean():.1f}%",
          f"  COMMIT (interval inside one category): {100 * (~straddle).mean():.1f}% of cases, correct {100 * ok[~straddle].mean():.1f}% (95% CI {100 * a:.1f}-{100 * b:.1f})",
          f"  DEFER  (interval straddles a threshold): {100 * straddle.mean():.1f}% of cases, correct {100 * ok[straddle].mean():.1f}%"]
    bc, be = np.digitize(qc[:, 0], CATS), np.digitize(med, CATS); sc = np.maximum(qc[:, 1] - yc, yc - qc[:, 2])
    Qb = [qhat(sc[bc == k], 0.1) if (bc == k).sum() > 20 else qhat(sc, 0.1) for k in range(3)]; cal["Q90_by_predicted_category"] = Qb
    lo_m, hi_m = lo5 - np.array([Qb[k] for k in be]), hi5 + np.array([Qb[k] for k in be]); cov_m = (ye >= lo_m) & (ye <= hi_m)
    L += ["", "CLASS-CONDITIONAL (Mondrian) variant, calibrated inside each PREDICTED EF category:",
          f"  calibration sizes {[int((bc == k).sum()) for k in range(3)]}; Q = {[round(float(q), 2) for q in Qb]}; overall coverage {100 * cov_m.mean():.1f}%, mean width {(hi_m - lo_m).mean():.1f}"]
    for k, nm in enumerate(("predicted EF < 40", "predicted 40-49", "predicted >= 50")):
        m = be == k
        if m.sum() > 0:
            a1, b1 = wilson(int(cov_m[m].sum()), int(m.sum()))
            L.append(f"  {nm:18s} n={int(m.sum()):4d}  coverage {100 * cov_m[m].mean():5.1f}% (95% CI {100 * a1:.1f}-{100 * b1:.1f})  mean width {(hi_m - lo_m)[m].mean():5.1f}")
    out["lo90_mondrian"], out["hi90_mondrian"] = lo_m, hi_m; out["width90"] = width; out["covered90"] = cov; out["straddles_threshold"] = straddle; out["err"] = ye - med
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "results", f"ef_intervals_{args.mode}.csv"), index=False)
    open(os.path.join(ROOT, "results", f"uncertainty_{args.mode}.md"), "w").write("\n".join(L) + "\n")
    json.dump(cal, open(os.path.join(ROOT, "checkpoints", f"conformal_{args.mode}.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
