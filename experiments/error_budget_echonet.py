"""Where does the formula-EF error on EchoNet TEST come from? Counterfactual swaps against the sonographer's tracing.

Final pipeline: chords of every frame (data_cache/features.npz) -> detector ED/ES of every beat -> formula EF per beat -> median.
Each row below replaces one ingredient with its ground truth (the tracing at the annotated ED / ES frame) and reports R2 against
the EF label, so the R2 gained by the swap is the share of the error that ingredient carries. All TEST videos.

Usage: python experiments/error_budget_echonet.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import train_heads as th  # noqa: E402
from build_dataset import canonical_polygon  # noqa: E402
from geometry import chord_widths  # noqa: E402
from temporal_detector import TemporalEDESDetector  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")


def r2(p, y):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ef(d_ed, a_ed, d_es, a_es):
    return 100 * (1 - (d_es ** 2).sum(-1) * a_es / ((d_ed ** 2).sum(-1) * a_ed))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(ROOT, "data_cache/features.npz"), allow_pickle=True); z = {k: z[k] for k in z.files}
    ix = np.where(z["splits"] == "TEST")[0]
    vt = pd.read_csv(os.path.join(D, "VolumeTracings.csv")); vt["key"] = vt.FileName.str.replace(".avi", "", regex=False)
    groups = {k: g[["X1", "Y1", "X2", "Y2"]].values for k, g in vt.groupby(["key", "Frame"])}

    ck = torch.load(os.path.join(ROOT, "checkpoints/ef_head.pt"), map_location=dev, weights_only=False)
    net = TemporalEDESDetector(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    X, L, ed_a, es_a, y = th.build(z, ix)                                   # clip (<= T frames) around the labelled beat
    Xn = torch.from_numpy((X - ck["mu"]) / ck["sd"]) * (torch.arange(th.T)[None, :, None] < torch.from_numpy(L)[:, None, None])
    with torch.no_grad():
        heat = np.concatenate([torch.sigmoid(net.forward_all(Xn[i:i + 64].to(dev), torch.from_numpy(L[i:i + 64]))[0]).cpu().numpy() for i in range(0, len(Xn), 64)])
    pk = lambda h: find_peaks(np.concatenate([[0.0], h, [0.0]]), prominence=0.1)[0] - 1

    R = []
    for k in range(len(ix)):
        key = z["keys"][ix[k]]; n = L[k]; f = X[k, :n].astype(np.float64); d, a = f[:, :20], f[:, 20]
        v = (d ** 2).sum(1) * a
        ed, es = pk(heat[k, :n, 0]), pk(heat[k, :n, 1]); beats = []
        for i, e in enumerate(ed):
            nxt = ed[i + 1] if i + 1 < len(ed) else 10 ** 9; c = es[(es > e) & (es < nxt)]
            if len(c):
                beats.append((int(e), int(c[0])))
        if not beats:
            beats = [(int(heat[k, :n, 0].argmax()), int(heat[k, :n, 1].argmax()))]
        efs = np.array([100 * (1 - v[s] / v[e]) for e, s in beats])
        lab = min(beats, key=lambda b: abs(b[0] - ed_a[k]) + abs(b[1] - es_a[k]))      # detected beat that matches the labelled one
        s0 = z["ed"][ix[k]] - ed_a[k]                                                     # clip start in the video
        g = {}
        for ph, fi in (("ED", ed_a[k]), ("ES", es_a[k])):
            rows = groups.get((key, int(fi + s0)))
            g[ph] = (None if rows is None else canonical_polygon(rows[:21]), rows is not None and len(rows) == 21)
        if g["ED"][0] is None or g["ES"][0] is None:
            continue
        gd = {ph: chord_widths(g[ph][0]) for ph in ("ED", "ES")}
        R.append(dict(key=key, y=y[k], final=np.median(efs), nbeats=len(efs), beat_sd=efs.std(),
                      det_lab=100 * (1 - v[lab[1]] / v[lab[0]]), frame_off=abs(lab[0] - ed_a[k]) + abs(lab[1] - es_a[k]),
                      clean=g["ED"][1] and g["ES"][1],
                      pd_ed=d[ed_a[k]], pa_ed=a[ed_a[k]], pd_es=d[es_a[k]], pa_es=a[es_a[k]],
                      gd_ed=gd["ED"][0], ga_ed=gd["ED"][1], gd_es=gd["ES"][0], ga_es=gd["ES"][1]))
    df = pd.DataFrame(R); yv = df.y.values
    st = lambda c: np.stack(df[c].values)
    pd_ed, pd_es, gd_ed, gd_es = st("pd_ed"), st("pd_es"), st("gd_ed"), st("gd_es")
    pa_ed, pa_es, ga_ed, ga_es = df.pa_ed.values, df.pa_es.values, df.ga_ed.values, df.ga_es.values
    rep = lambda p, nm: print(f"  {nm:62s} R2 {r2(p, yv):6.3f}   r {np.corrcoef(p, yv)[0, 1]:.3f} (linear-cal cap {np.corrcoef(p, yv)[0, 1] ** 2:.3f})   MAE {np.abs(p - yv).mean():5.2f}   bias {(p - yv).mean():+5.2f}")
    print(f"EchoNet TEST, {len(df)} videos with both labelled tracings readable (of {len(ix)})\n")
    print("A. Frames and aggregation (predicted tracings throughout)")
    rep(df.final.values, "final: detected ED/ES, median over beats")
    rep(df.det_lab.values, "detected frames, labelled beat only")
    pa = ef(pd_ed, pa_ed, pd_es, pa_es); rep(pa, "annotated frames, labelled beat")
    print("\nB. Tracing at the annotated frames: swap in the sonographer's tracing")
    rep(ef(gd_ed, ga_ed, gd_es, ga_es), "both phases ground truth (label ceiling)")
    rep(ef(gd_ed, ga_ed, pd_es, pa_es), "ED ground truth, ES predicted")
    rep(ef(pd_ed, pa_ed, gd_es, ga_es), "ED predicted, ES ground truth")
    rep(ef(pd_ed, ga_ed, pd_es, ga_es), "predicted widths, ground-truth axis length")
    rep(ef(gd_ed, pa_ed, gd_es, pa_es), "ground-truth widths, predicted axis length")
    for nm, sl in (("apex chords 1-5", slice(0, 5)), ("mid chords 6-15", slice(5, 15)), ("base chords 16-20", slice(15, 20))):
        a1, a2 = pd_ed.copy(), pd_es.copy(); a1[:, sl], a2[:, sl] = gd_ed[:, sl], gd_es[:, sl]
        rep(ef(a1, pa_ed, a2, pa_es), f"ground-truth {nm} only")
    for nm, sl in (("basal chords 19-20", slice(18, 20)), ("chords 17-20", slice(16, 20)), ("chord 20", slice(19, 20))):
        a1, a2 = pd_ed.copy(), pd_es.copy(); a1[:, sl], a2[:, sl] = gd_ed[:, sl], gd_es[:, sl]
        rep(ef(a1, pa_ed, a2, pa_es), f"ground-truth {nm} only")
    yb = df.y.values
    print("\n   log-volume error by label EF (annotated frames): ED mean / ES mean / ES-ED (positive ES-ED = EF underestimated)")
    eV = lambda p, pa, g, ga: np.log((p ** 2).sum(1) * pa) - np.log((g ** 2).sum(1) * ga)
    e1, e2 = eV(pd_ed, pa_ed, gd_ed, ga_ed), eV(pd_es, pa_es, gd_es, ga_es)
    for lo, hi in ((0, 40), (40, 50), (50, 60), (60, 70), (70, 100)):
        m = (yb >= lo) & (yb < hi)
        print(f"   EF {lo:3d}-{hi:3d}: n {m.sum():4d}   ED {e1[m].mean():+.3f}   ES {e2[m].mean():+.3f}   ES-ED {(e2 - e1)[m].mean():+.3f}")
    print("\nC. Systematic part of the tracing error (annotated frames)")
    lr = lambda p, g: np.log((p ** 2).sum(1)) - np.log((g ** 2).sum(1))
    for nm, e in (("log ED volume", np.log((pd_ed ** 2).sum(1) * pa_ed) - np.log((gd_ed ** 2).sum(1) * ga_ed)),
                  ("log ES volume", np.log((pd_es ** 2).sum(1) * pa_es) - np.log((gd_es ** 2).sum(1) * ga_es)),
                  ("log ED axis", np.log(pa_ed / ga_ed)), ("log ES axis", np.log(pa_es / ga_es)),
                  ("log ED sum d^2", lr(pd_ed, gd_ed)), ("log ES sum d^2", lr(pd_es, gd_es))):
        print(f"  {nm:16s} mean {e.mean():+.3f}  sd {e.std():.3f}")
    eed = np.log((pd_ed ** 2).sum(1) * pa_ed / ((gd_ed ** 2).sum(1) * ga_ed)); ees = np.log((pd_es ** 2).sum(1) * pa_es / ((gd_es ** 2).sum(1) * ga_es))
    print(f"  correlation of the ED and ES log-volume errors: {np.corrcoef(eed, ees)[0, 1]:+.2f} (1 = shared error that cancels in EF)")
    w = (gd_ed ** 2) / (gd_ed ** 2).sum(1, keepdims=True)
    for ph, p_, g_ in (("ED", pd_ed, gd_ed), ("ES", pd_es, gd_es)):
        e = (p_ - g_)
        print(f"  {ph} width error by chord (px, mean / sd), apex -> base:")
        print("     mean " + " ".join(f"{v:+4.1f}" for v in e.mean(0)))
        print("     sd   " + " ".join(f"{v:4.1f}" for v in e.std(0)))
    print("\nD. Concentration of the final pipeline's squared error")
    err = df.final.values - yv; se = err ** 2; o = np.argsort(-se); tot = ((yv - yv.mean()) ** 2).sum()
    print(f"  SSE {se.sum():.0f}; to reach R2 0.83 the SSE must fall to {0.17 * tot:.0f} (a cut of {se.sum() - 0.17 * tot:.0f}, {100 * (1 - 0.17 * tot / se.sum()):.0f}%)")
    for fr in (0.01, 0.02, 0.05, 0.10, 0.20):
        m = int(round(fr * len(se)))
        print(f"  worst {100 * fr:4.0f}% ({m:4d} videos) hold {100 * se[o[:m]].sum() / se.sum():4.1f}% of the SSE; R2 without them {1 - se[o[m:]].sum() / ((yv[o[m:]] - yv[o[m:]].mean()) ** 2).sum():.3f}")
    gt_ef = ef(gd_ed, ga_ed, gd_es, ga_es)
    top = o[:int(round(0.05 * len(se)))]
    print(f"  worst 5%: unclean tracing {int((~df.clean.values[top]).sum())}, |GT-tracing formula - label| > 5 in {int((np.abs(gt_ef - yv)[top] > 5).sum())}, "
          f"median beats {np.median(df.nbeats.values[top]):.0f} (all {np.median(df.nbeats):.0f}), beat-EF sd {np.median(df.beat_sd.values[top]):.1f} (all {np.median(df.beat_sd):.1f}), "
          f"label EF median {np.median(yv[top]):.0f} (all {np.median(yv):.0f}); sign: {int((err[top] > 0).sum())} over, {int((err[top] < 0).sum())} under")
    print("\nE. Error by label EF")
    bins = [0, 30, 40, 50, 60, 70, 100]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (yv >= lo) & (yv < hi)
        print(f"  EF {lo:3d}-{hi:3d}: n {m.sum():4d}  bias {err[m].mean():+5.2f}  MAE {np.abs(err[m]).mean():5.2f}  share of SSE {100 * se[m].sum() / se.sum():4.1f}%")
    b = np.polyfit(yv, df.final.values, 1)
    print(f"  slope of prediction on label {b[0]:.2f} (1 = no regression to the mean)")
    print("\nF. Beat averaging")
    for lo, hi in ((1, 2), (3, 3), (4, 4), (5, 6), (7, 99)):
        m = (df.nbeats.values >= lo) & (df.nbeats.values <= hi)
        if m.sum() > 20:
            print(f"  {lo}-{hi} beats: n {m.sum():4d}  MAE final {np.abs(err[m]).mean():.2f}  MAE labelled beat {np.abs(df.det_lab.values[m] - yv[m]).mean():.2f}")
    df.drop(columns=[c for c in df.columns if c.startswith(("pd_", "gd_"))]).assign(err=err, gt_ef=gt_ef).to_csv(os.path.join(ROOT, "results/error_budget_echonet.csv"), index=False)


if __name__ == "__main__":
    main()
