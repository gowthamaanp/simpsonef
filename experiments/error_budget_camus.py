"""Where does the zero-shot formula-EF error on CAMUS come from? Counterfactual swaps against the CAMUS reference.

Final pipeline: half-sequence as released -> model -> chords of every frame -> detector ED/ES -> biplane Simpson with one disk
height (CAMUS protocol; the common height cancels in EF, so only the chord widths and the frames matter). Each row replaces one
ingredient with the reference: the annotated frames, the reference diameters (official CAMUS code on the reference masks), or the
official code run on OUR predicted contour (separates contour error from chord-extraction convention). All 50 test patients.

Usage: python experiments/error_budget_camus.py
"""
import os
import sys

import cv2
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import camus_zeroshot as cz  # noqa: E402
from geometry import chord_widths  # noqa: E402
from inference import Pipeline  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def r2(p, y):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def native_mask(P, shape):
    """Predicted 42-point polygon on the 112 padded canvas -> filled mask on the native image grid."""
    h, w = shape; side = max(h, w); k = side / cz.S
    col = (P[1:41, 0] - 1 + 0.5) * k - 0.5 - (side - w) // 2
    row = (cz.S - P[1:41, 1] + 0.5) * k - 0.5 - (side - h) // 2
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(np.stack([col, row], 1)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def main():
    pipe = Pipeline(os.path.join(ROOT, "checkpoints/shape_estimator.pt"), os.path.join(ROOT, "checkpoints/ef_head.pt"), None)
    nd = os.path.join(cz.CAMUS, "database_nifti"); diam = cz.reference_simpson()
    pats = [l.strip() for l in open(os.path.join(cz.CAMUS, "database_split", "subgroup_testing.txt")) if l.strip()]
    rows = []
    for p in pats:
        r = {"patient": p}
        for view in ("4CH", "2CH"):
            cfg = cz.info(os.path.join(nd, p, f"Info_{view}.cfg")); ea, sa = int(cfg["ED"]) - 1, int(cfg["ES"]) - 1
            im = nib.load(os.path.join(nd, p, f"{p}_{view}_half_sequence.nii.gz")); vol = im.get_fdata().astype(np.float32)
            sp = tuple(float(v) for v in im.header.get_zooms()[:2][::-1]); side = max(vol.shape[0], vol.shape[1])
            scale = side / cz.S * float(np.mean(im.header.get_zooms()[:2]))
            frames = np.stack([cz.pad_to_square_resize(vol[:, :, t].T, cz.S, cv2.INTER_LINEAR) for t in range(vol.shape[2])]).clip(0, 255).astype(np.uint8)
            P = pipe.points(frames); d, a = chord_widths(P); d = np.asarray(d) * scale
            v = (d ** 2).sum(1) * np.asarray(a)
            bt = pipe.analyse(frames)["beats"]
            ed, es = (bt[0]["ed"], bt[0]["es"]) if bt else (int(v.argmax()), int(v.argmin()))
            gts = {ph: (nib.load(os.path.join(nd, p, f"{p}_{view}_{ph}_gt.nii.gz")).get_fdata().T == 1).astype(np.uint8) for ph in ("ED", "ES")}
            ref = {ph: np.asarray(diam(gts[ph], sp)[0]) for ph in ("ED", "ES")}
            hs = vol[:, :, 0].T.shape
            tool_pred = {ph: np.asarray(diam(native_mask(P[t], hs), sp)[0]) for ph, t in (("ED", ea), ("ES", sa))}
            r[view] = dict(det=(d[ed], d[es]), ann=(d[ea], d[sa]), ref=(ref["ED"], ref["ES"]), tool=(tool_pred["ED"], tool_pred["ES"]),
                           frame_off=abs(ed - ea) + abs(es - sa), n=len(frames), quality=cfg.get("ImageQuality", "?"))
        r["y"] = float(cz.info(os.path.join(nd, p, "Info_4CH.cfg"))["EF"]); rows.append(r)
    y = np.array([r["y"] for r in rows])

    def bi(src4, src2, ph_src=None):
        out = []
        for r in rows:
            if ph_src is not None:                                         # mix phases: (ED source, ES source)
                (e4, _), (e2, _) = r["4CH"][ph_src[0]], r["2CH"][ph_src[0]]
                (_, s4), (_, s2) = r["4CH"][ph_src[1]], r["2CH"][ph_src[1]]
            else:
                (e4, s4), (e2, s2) = r["4CH"][src4], r["2CH"][src2]
            out.append(100 * (1 - (s4 * s2).sum() / (e4 * e2).sum()))
        return np.array(out)

    def rep(pr, nm):
        print(f"  {nm:64s} R2 {r2(pr, y):6.3f}   r {np.corrcoef(pr, y)[0, 1]:.3f} (cap {np.corrcoef(pr, y)[0, 1] ** 2:.3f})   MAE {np.abs(pr - y).mean():5.2f}   bias {(pr - y).mean():+5.2f}")
        return pr
    print(f"CAMUS test, {len(rows)} patients, biplane Simpson with one disk height (CAMUS protocol)\n")
    print("A. Frames")
    fin = rep(bi("det", "det"), "final: detector frames, predicted chords")
    rep(bi("ann", "ann"), "annotated frames, predicted chords")
    print(f"     frame offset detector vs annotation (ED + ES, frames): median {np.median([r[v]['frame_off'] for r in rows for v in ('4CH', '2CH')]):.0f}, "
          f"mean {np.mean([r[v]['frame_off'] for r in rows for v in ('4CH', '2CH')]):.1f}; sequence length median {np.median([r[v]['n'] for r in rows for v in ('4CH', '2CH')]):.0f}")
    print("\nB. Contour vs chord-extraction convention (annotated frames)")
    rep(bi("ref", "ref"), "reference diameters (official code on reference masks) = label")
    rep(bi("tool", "tool"), "official code on OUR predicted contour")
    rep(bi("ref", "ann"), "4CH reference, 2CH predicted")
    rep(bi("ann", "ref"), "4CH predicted, 2CH reference")
    rep(bi(None, None, ("ref", "ann")), "ED reference, ES predicted")
    rep(bi(None, None, ("ann", "ref")), "ED predicted, ES reference")
    print("\nC. Chord-by-chord ratio predicted / reference (annotated frames; official code ordering, both 20 chords)")
    for v in ("4CH", "2CH"):
        for k, ph in enumerate(("ED", "ES")):
            for src, nm in (("ann", "our chords "), ("tool", "tool on ours")):
                rr = np.stack([r[v][src][k] / np.maximum(r[v]["ref"][k], 1e-3) for r in rows])
                print(f"  {v} {ph} {nm}: " + " ".join(f"{np.median(rr[:, i]):4.2f}" for i in range(rr.shape[1])))
    for v in ("4CH", "2CH"):
        for k, ph in enumerate(("ED", "ES")):
            lv = np.array([np.log((r[v]["ann"][k] ** 2).sum() / (r[v]["ref"][k] ** 2).sum()) for r in rows])
            lt = np.array([np.log((r[v]["tool"][k] ** 2).sum() / (r[v]["ref"][k] ** 2).sum()) for r in rows])
            print(f"  {v} {ph}: log sum d^2 error, our chords mean {lv.mean():+.3f} sd {lv.std():.3f}; tool on our contour mean {lt.mean():+.3f} sd {lt.std():.3f}")
    print("\nD. Error concentration and image quality (final)")
    err = fin - y; se = err ** 2; o = np.argsort(-se); tot = ((y - y.mean()) ** 2).sum()
    print(f"  SSE {se.sum():.0f}; R2 0.83 needs SSE {0.17 * tot:.0f} (a cut of {100 * (1 - 0.17 * tot / se.sum()):.0f}%)")
    for m in (1, 3, 5, 10):
        print(f"  worst {m:2d} patients hold {100 * se[o[:m]].sum() / se.sum():4.1f}% of the SSE; R2 without them {1 - se[o[m:]].sum() / ((y[o[m:]] - y[o[m:]].mean()) ** 2).sum():.3f}")
    print("  worst 5: " + "; ".join(f"{rows[i]['patient']} label {y[i]:.0f} pred {fin[i]:.0f} quality {rows[i]['4CH']['quality']}/{rows[i]['2CH']['quality']}" for i in o[:5]))
    q = np.array([rows[i]["4CH"]["quality"] for i in range(len(rows))])
    for g in np.unique(q):
        m = q == g
        print(f"  4CH quality {g:8s} n {m.sum():2d}  MAE {np.abs(err[m]).mean():.2f}  bias {err[m].mean():+.2f}  share of SSE {100 * se[m].sum() / se.sum():4.1f}%")
    b = np.polyfit(y, fin, 1)
    print(f"  slope of prediction on label {b[0]:.2f}")
    for lo, hi in ((0, 45), (45, 55), (55, 100)):
        m = (y >= lo) & (y < hi)
        print(f"  EF {lo}-{hi}: n {m.sum():2d}  bias {err[m].mean():+.2f}  MAE {np.abs(err[m]).mean():.2f}  share of SSE {100 * se[m].sum() / se.sum():4.1f}%")


if __name__ == "__main__":
    main()
