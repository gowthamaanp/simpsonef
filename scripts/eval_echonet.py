"""Evaluate the released pipeline (src/pipeline.py, exactly as run.py uses it) on EchoNet-Dynamic VAL or TEST.

Every video of the split with an EF label is scored (TEST: 1277), with no annotation at inference. Per video: the formula EF
(median over the detected beats), BeatFinder's EF quantiles (for conformal.py), and, on the annotated videos, ED / ES detection
error (the BeatFinder peak nearest each annotated frame), the formula EF at the annotated frames, and the Dice of BeatTracer's
tracings at the annotated frames in both mask conventions:
  png   cv2.fillPoly of all 42 points, truncated coordinates   = the released annotation/Masks PNGs
  repo  skimage polygon of the 40 chord ends, rounded           = the masks the EchoNet-Dynamic code draws from VolumeTracings.csv

Writes results/echonet_<split>.csv (per video; contains EchoNet labels, do not redistribute) and prints the summary.
Usage: python scripts/eval_echonet.py --split VAL       (TEST only once, for the frozen pipeline)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beatfinder import peaks  # noqa: E402
from pipeline import SimpsonEF, read_video  # noqa: E402
from simpson import contour_to_mask, contour_to_mask_repo, ef_single_plane  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def r2(p, y):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def repo_reference_mask(rows, size=112):
    """The EchoNet-Dynamic code's mask of a reference tracing: polygon of the 20 chords' X1 ends then the X2 ends reversed."""
    import skimage.draw
    t = np.asarray(rows); x = np.concatenate((t[1:, 0], t[1:, 2][::-1])); y = np.concatenate((t[1:, 1], t[1:, 3][::-1]))
    r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (size, size))
    m = np.zeros((size, size), bool); m[r, c] = True
    return m


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def nearest(sig, annot, prominence):
    p = peaks(sig, prominence)
    return int(np.argmax(sig)) if len(p) == 0 else int(p[np.argmin(np.abs(p - annot))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["VAL", "TEST"], default="VAL")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"), help="EchoNet-Dynamic: Videos/, FileList.csv, master.csv, VolumeTracings.csv, annotation/Masks/")
    ap.add_argument("--frametracer", default="checkpoints/frametracer.pt")
    ap.add_argument("--beatfinder", default="checkpoints/beatfinder.pt")
    ap.add_argument("--beattracer", default="checkpoints/beattracer.pt")
    ap.add_argument("--no-copies", action="store_true", help="ablation: no 15-copy test-time averaging")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    J = lambda p: os.path.join(ROOT, p)
    model = SimpsonEF(J(a.frametracer), J(a.beatfinder), J(a.beattracer), conformal=None, copies=not a.no_copies)
    fl = pd.read_csv(os.path.join(a.data, "FileList.csv")); fl["key"] = fl.FileName.str.replace(".avi", "", regex=False)
    fl = fl[(fl.Split == a.split) & fl.EF.notna()]
    ms = pd.read_csv(os.path.join(a.data, "master.csv")).set_index("FileName")
    vt = pd.read_csv(os.path.join(a.data, "VolumeTracings.csv")); vt["key"] = vt.FileName.str.replace(".avi", "", regex=False)
    vt = {k: g[["X1", "Y1", "X2", "Y2"]].values for k, g in vt.groupby(["key", "Frame"])}
    rows, t0 = [], time.time()
    for n, r in enumerate(fl.itertuples()):
        frames, _ = read_video(os.path.join(a.data, "Videos", f"{r.key}.avi"))
        res = model.analyse(frames)
        row = dict(key=r.key, y=float(r.EF), ef=res["ef"], n_beats=len(res["beats"]), ef_beatfinder=res["ef_beatfinder"], q05=res["q05"], q95=res["q95"])
        if r.key in ms.index:
            m = ms.loc[r.key]; fe, fs = int(m.ED_Frame), int(m.ES_Frame)
            row["afd_ed"] = abs(nearest(res["heat_ed"], fe, model.prominence) - fe); row["afd_es"] = abs(nearest(res["heat_es"], fs, model.prominence) - fs)
            Pa = model.beat_tracings(frames, [(fe, fs)])[0]
            row["ef_annotated"] = ef_single_plane(Pa[0], Pa[1])
            for j, (ph, fr) in enumerate((("ED", fe), ("ES", fs))):
                mp = os.path.join(a.data, "annotation", "Masks", ph, f"{r.key}.png")
                if os.path.exists(mp):
                    row[f"dice_png_{ph}"] = dice(contour_to_mask(Pa[j]), np.array(Image.open(mp)) > 0)
                t = vt.get((r.key, fr))
                if t is not None and len(t) >= 21:
                    row[f"dice_repo_{ph}"] = dice(contour_to_mask_repo(Pa[j]), repo_reference_mask(t))
        rows.append(row)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(fl)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = a.out or os.path.join("results", f"echonet_{a.split}{'_nocopies' if a.no_copies else ''}.csv")
    os.makedirs(os.path.dirname(J(out)), exist_ok=True); df.to_csv(J(out), index=False)
    p, y = df.ef.values, df.y.values
    rng = np.random.RandomState(0); bs = [r2(p[j], y[j]) for j in (rng.randint(0, len(y), len(y)) for _ in range(2000))]
    print(f"EchoNet {a.split}, all {len(df)} videos ({time.time() - t0:.0f}s)")
    print(f"  formula EF, detected beats, median: R2 {r2(p, y):.3f} (95% CI {np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f})  "
          f"MAE {np.abs(p - y).mean():.2f}  r {np.corrcoef(p, y)[0, 1]:.3f}  bias {(p - y).mean():+.2f}  slope {np.polyfit(y, p, 1)[0]:.2f}")
    an = df.dropna(subset=["ef_annotated"])
    print(f"  formula EF at the annotated frames (diagnostic, n={len(an)}): R2 {r2(an.ef_annotated.values, an.y.values):.3f}  MAE {np.abs(an.ef_annotated - an.y).mean():.2f}")
    print(f"  BeatFinder's own EF (not reported, conformal scale only): R2 {r2(df.ef_beatfinder.values, y):.3f}")
    for c in ("png", "repo"):
        print(f"  Dice ({c} masks): ED {df[f'dice_{c}_ED'].mean():.4f} (n={df[f'dice_{c}_ED'].notna().sum()})  ES {df[f'dice_{c}_ES'].mean():.4f}")
    fd = np.r_[an.afd_ed.values, an.afd_es.values]
    print(f"  ED/ES detection: mean {fd.mean():.2f} frames (ED {an.afd_ed.mean():.2f}, ES {an.afd_es.mean():.2f}); within 2 frames {100 * (fd <= 2).mean():.1f}%")


if __name__ == "__main__":
    main()
