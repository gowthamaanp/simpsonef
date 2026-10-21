"""EF from the formula only, with ED/ES frames found by the detector (no annotated frame is used).

For every TEST video: chord features -> detector heatmaps -> every beat's ED/next ES -> V = pi/4 * sum d_i^2 * axis / 20 -> EF
per beat -> median over beats. A video with no detected pair falls back to its global ED and ES peaks. All videos are scored.

Usage: python experiments/eval_formula_ef.py --features data_cache/features.npz --head checkpoints/ef_head.pt --split TEST
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import train_heads as th  # noqa: E402
from temporal_detector import TemporalEDESDetector  # noqa: E402
from uncertainty_utils import r2  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def vol(f):
    return (np.pi / 4.0) * (f[..., :20] ** 2).sum(-1) * f[..., 20] / 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data_cache/features.npz")
    ap.add_argument("--head", default="checkpoints/ef_head.pt")
    ap.add_argument("--split", default="TEST", choices=["VAL", "TEST"])
    ap.add_argument("--prom", type=float, default=0.1)
    ap.add_argument("--avg", type=int, default=0, help="average the chords over +-avg frames around each detected ED / ES")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(ROOT, a.features), allow_pickle=True); z = {k: z[k] for k in z.files}
    ix = np.where(z["splits"] == a.split)[0]
    ck = torch.load(os.path.join(ROOT, a.head), map_location=dev, weights_only=False)
    net = TemporalEDESDetector(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    X, L, ed_a, es_a, y = th.build(z, ix)
    Xn = torch.from_numpy((X - ck["mu"]) / ck["sd"]) * (torch.arange(th.T)[None, :, None] < torch.from_numpy(L)[:, None, None])
    heat = []
    with torch.no_grad():
        for i in range(0, len(Xn), 64):
            heat.append(torch.sigmoid(net.forward_all(Xn[i:i + 64].to(dev), torch.from_numpy(L[i:i + 64]))[0]).cpu().numpy())
    heat = np.concatenate(heat)
    ef_med, ef_first, nb, fb = [], [], [], 0
    for k in range(len(ix)):
        n = L[k]; v = vol(X[k, :n]); he, hs = heat[k, :n, 0], heat[k, :n, 1]
        pk = lambda h: find_peaks(np.concatenate([[0.0], h, [0.0]]), prominence=a.prom)[0] - 1
        ed, es = pk(he), pk(hs); efs = []
        for i, e in enumerate(ed):
            nxt = ed[i + 1] if i + 1 < len(ed) else 10 ** 9; c = es[(es > e) & (es < nxt)]
            if len(c):
                lo = lambda t: slice(max(0, t - a.avg), t + a.avg + 1)
                efs.append(100 * (1 - v[lo(c[0])].mean() / v[lo(e)].mean()))
        if not efs:
            fb += 1; efs = [100 * (1 - v[int(hs.argmax())] / v[int(he.argmax())])]
        ef_med.append(np.median(efs)); ef_first.append(efs[0]); nb.append(len(efs))
    ef_med, ef_first = np.array(ef_med), np.array(ef_first)
    for name, p in (("median over beats", ef_med), ("first beat", ef_first)):
        print(f"{a.split} all {len(y)} videos, formula EF, {name}: R2 {r2(p, y):.3f}  MAE {np.abs(p - y).mean():.2f}  r {np.corrcoef(p, y)[0, 1]:.3f}  bias {(p - y).mean():+.2f}")
    print(f"beats per video: median {int(np.median(nb))}, range {min(nb)}-{max(nb)}; videos with no detected pair (global-peak fallback): {fb}")


if __name__ == "__main__":
    main()
