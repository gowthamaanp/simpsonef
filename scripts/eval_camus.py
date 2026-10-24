"""Zero-shot CAMUS: the EchoNet-trained pipeline, unchanged, on the 50 CAMUS test patients, biplane Simpson with the CAMUS label
definition (src/pipeline.py SimpsonEF.analyse_biplane, as run.py --biplane uses it). No CAMUS data in training, no annotated frame.

Each view's half-sequence is zero-padded to a square and resized to 112 x 112 (aspect ratio kept). Reported: biplane EF (one disk
height for ED and ES), 4- and 2-chamber single-plane EF, and the Dice of FrameTracer's tracing at the annotated ED / ES frames
against the CAMUS LV cavity (label 1). The tracer for the biplane EF (FrameTracer, not BeatTracer) was chosen on the 400 CAMUS
training patients (--subset subgroup_training), whose labels serve only this choice.

Usage: python scripts/eval_camus.py [--camus data/camus/CAMUS_public] [--tracer frame|beat] [--subset subgroup_testing]
"""
import argparse
import os
import sys

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import S, SimpsonEF  # noqa: E402
from simpson import contour_to_mask, ef_single_plane  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def pad_to_square_resize(arr, size, interp):
    """Zero-pad (H, W) to a square canvas, then resize to (size, size)."""
    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), arr.dtype)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr
    return cv2.resize(canvas, (size, size), interpolation=interp)


def info(path):
    d = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1); d[k.strip()] = v.strip()
    return d


def report(nm, p, y):
    r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.RandomState(0); bs = []
    for _ in range(2000):
        j = rng.randint(0, len(y), len(y)); bs.append(1 - ((p[j] - y[j]) ** 2).sum() / ((y[j] - y[j].mean()) ** 2).sum())
    print(f"  {nm:34s} R2 {r2:.3f} (95% CI {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f})  MAE {np.abs(p - y).mean():.2f}  "
          f"r {np.corrcoef(p, y)[0, 1]:.3f}  bias {(p - y).mean():+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camus", default=os.environ.get("CAMUS_DIR", os.path.join(ROOT, "data", "camus", "CAMUS_public")))
    ap.add_argument("--subset", default="subgroup_testing", choices=["subgroup_testing", "subgroup_training"])
    ap.add_argument("--tracer", default="frame", choices=["frame", "beat"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model = SimpsonEF(conformal=None)
    nd = os.path.join(a.camus, "database_nifti")
    pats = [l.strip() for l in open(os.path.join(a.camus, "database_split", f"{a.subset}.txt")) if l.strip()]
    rows = []
    for p in pats:
        fr, cfg = {}, {}
        for view in ("4CH", "2CH"):
            cfg[view] = info(os.path.join(nd, p, f"Info_{view}.cfg"))
            vol = nib.load(os.path.join(nd, p, f"{p}_{view}_half_sequence.nii.gz")).get_fdata().astype(np.float32)
            fr[view] = np.stack([pad_to_square_resize(vol[:, :, t].T, S, cv2.INTER_LINEAR) for t in range(vol.shape[2])]).clip(0, 255).astype(np.uint8)
        res = model.analyse_biplane(fr["4CH"], fr["2CH"], tracer=a.tracer)
        r = dict(patient=p, y=float(cfg["4CH"]["EF"]), ef_biplane=res["ef"])
        for view in ("4CH", "2CH"):
            v = res["views"][view]; b = v["beat_points"][0]
            r[f"ef_{view}"] = ef_single_plane(b[0], b[1], common_height=True); r[f"beats_{view}"] = len(v["beats"])
            if a.subset == "subgroup_testing":
                for ph in ("ED", "ES"):
                    t = int(cfg[view][ph]) - 1
                    g = nib.load(os.path.join(nd, p, f"{p}_{view}_{ph}_gt.nii.gz")).get_fdata().T == 1
                    g = pad_to_square_resize(g.astype(np.uint8), S, cv2.INTER_NEAREST) > 0
                    m = contour_to_mask(v["frame_points"][t])
                    r[f"dice_{ph}_{view}"] = 2 * (m & g).sum() / (m.sum() + g.sum() + 1e-9)
        rows.append(r)
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    y = df.y.values
    print(f"CAMUS {a.subset}, all {len(df)} patients, zero-shot, {a.tracer.capitalize()}Tracer chords")
    report("biplane, one disk height (CAMUS)", df.ef_biplane.values, y)
    report("4-chamber only, one disk height", df.ef_4CH.values, y)
    report("2-chamber only, one disk height", df.ef_2CH.values, y)
    if "dice_ED_4CH" in df:
        print("  Dice " + "  ".join(f"{v}: ED {df[f'dice_ED_{v}'].mean():.3f} ES {df[f'dice_ES_{v}'].mean():.3f}" for v in ("4CH", "2CH")))


if __name__ == "__main__":
    main()
