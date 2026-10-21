"""Zero-shot evaluation on CAMUS (never used for training or selection): 50 test patients, apical 4- and 2-chamber views.

Frames are zero-padded to a square and resized to 112 x 112 (the aspect ratio is kept), then run through the EchoNet-trained
pipeline unchanged. Reported:
  Dice       mask drawn from the predicted points against the CAMUS LV cavity (label 1) at ED and ES, in the 112 x 112 frame
  EF, two disk-height conventions, from the predicted chords at the annotated ED and ES frames (4CH, 2CH, and Simpson's biplane rule with mm from the
                 CAMUS pixel spacing). EchoNet-Dynamic labels use each phase's own long-axis length ('per-phase'); the CAMUS reference tool uses one disk
                 height for ED and ES ('common step'), which ignores long-axis shortening and lowers EF by several points.
  reference      the CAMUS reference Simpson code (loaded from the notebook shipped with the dataset) applied to the reference masks: the biplane EF
                 (reproduces the label) and the single-plane per-phase EF, i.e. the EchoNet convention applied to CAMUS masks
  head           the EchoNet-trained temporal head on a synthetic beat (ED -> ES -> back to ED, since CAMUS sequences cover only ED -> ES):
                 EF error, coverage of the 90% interval calibrated on EchoNet VAL, and the share of cases deferred / referred
References: the CAMUS EF label (Info_*.cfg).

Usage: python experiments/camus_zeroshot.py --ckpt checkpoints/shape_estimator.pt --head checkpoints/ef_head.pt --conformal checkpoints/conformal_final.json
"""
import argparse
import os
import sys

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geometry import chord_widths, contour_to_mask  # noqa: E402
from inference import Pipeline  # noqa: E402
from uncertainty_utils import auroc, r2, wilson  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAMUS = os.environ.get("CAMUS_DIR", os.path.join(ROOT, "data", "camus", "CAMUS_public"))
S = 112


def pad_to_square_resize(arr, size, interp):
    """Zero-pad (H, W) to a square canvas, then resize to (size, size)."""
    h, w = arr.shape
    side = max(h, w)
    canvas = np.zeros((side, side), arr.dtype)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr
    return cv2.resize(canvas, (size, size), interpolation=interp)


def reference_simpson():
    """The CAMUS reference diameter code, read from the notebook released with the dataset (cells that define the functions)."""
    import json
    from typing import Tuple

    import PIL
    from PIL.Image import Resampling
    from skimage.measure import find_contours
    nb = json.load(open(os.path.join(CAMUS, "jupyter", "script_camus_ef.ipynb")))
    ns = dict(np=np, PIL=PIL, Resampling=Resampling, Tuple=Tuple, find_contours=find_contours)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and (src.startswith("def resize_image") or src.startswith("def compute_left_ventricle_volumes")):
            exec(src, ns)
    return ns["_compute_diameters"]


def sector_crop(vol, ed):
    """Bounding box of the ultrasound sector (pixels above a small threshold) in the ED frame: (y0, y1, x0, x1)."""
    ys, xs = np.nonzero(vol[:, :, ed].T > 8)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def info(path):
    d = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1); d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/shape_estimator.pt")
    ap.add_argument("--head", default="checkpoints/ef_head.pt")
    ap.add_argument("--conformal", default="checkpoints/conformal_final.json")
    ap.add_argument("--prep", choices=["pad", "sector"], default="pad", help="pad: zero-pad to a square; sector: crop to the sector's bounding box and stretch to 112 x 112 (as EchoNet frames are)")
    ap.add_argument("--split", default="testing", choices=["testing"], help="only the CAMUS test split is used")
    ap.add_argument("--out", default="results/camus_zeroshot")
    args = ap.parse_args()
    pipe = Pipeline(os.path.join(ROOT, args.ckpt), os.path.join(ROOT, args.head), os.path.join(ROOT, args.conformal))
    nd = os.path.join(CAMUS, "database_nifti"); diam = reference_simpson()
    pats = [l.strip() for l in open(os.path.join(CAMUS, "database_split", f"subgroup_{args.split}.txt")) if l.strip()]
    rows = []
    for p in pats:
        r = {"patient": p}; vols, mm, refd = {}, {}, {}
        for view in ("4CH", "2CH"):
            cfg = info(os.path.join(nd, p, f"Info_{view}.cfg")); ed, es = int(cfg["ED"]) - 1, int(cfg["ES"]) - 1
            im = nib.load(os.path.join(nd, p, f"{p}_{view}_half_sequence.nii.gz"))
            vol = im.get_fdata().astype(np.float32)
            spacing = float(np.mean(im.header.get_zooms()[:2]))
            side = max(vol.shape[0], vol.shape[1])
            if args.prep == "pad":
                frames = np.stack([pad_to_square_resize(vol[:, :, t].T, S, cv2.INTER_LINEAR) for t in range(vol.shape[2])]).clip(0, 255).astype(np.uint8)
                tf = lambda m: pad_to_square_resize(m, S, cv2.INTER_NEAREST) > 0
                scale = side / S * spacing                                           # mm per 112-frame pixel
            else:
                y0, y1, x0, x1 = sector_crop(vol, ed)
                frames = np.stack([cv2.resize(vol[:, :, t].T[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_LINEAR) for t in range(vol.shape[2])]).clip(0, 255).astype(np.uint8)
                tf = lambda m, y0=y0, y1=y1, x0=x0, x1=x1: cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
                scale = float(np.sqrt((y1 - y0) * (x1 - x0)) / S * spacing)           # mm per 112-frame pixel (geometric mean of the two stretch factors)
            P = pipe.points(frames)
            d, axis = chord_widths(P[[ed, es]])
            vols[view] = (d * scale, axis * scale)
            vp = (d ** 2).sum(1) * axis; vc = (d ** 2).sum(1) * axis.max()
            r[f"ef_{view}"] = float(100 * (1 - vp[1] / vp[0])); r[f"ef_{view}_common"] = float(100 * (1 - vc[1] / vc[0]))
            r["ef_label"] = float(cfg["EF"]); r["quality"] = cfg.get("ImageQuality", "")
            gts = {ph: (nib.load(os.path.join(nd, p, f"{p}_{view}_{ph}_gt.nii.gz")).get_fdata().T == 1).astype(np.uint8) for ph in ("ED", "ES")}
            ref = {ph: diam(gts[ph], tuple(float(v) for v in im.header.get_zooms()[:2][::-1])) for ph in ("ED", "ES")}
            refd[view] = ref
            vs = {ph: np.sum(ref[ph][0] ** 2) * ref[ph][1] for ph in ("ED", "ES")}
            r[f"ef_ref_{view}"] = float(100 * (1 - vs["ES"] / vs["ED"]))             # the EchoNet convention applied to the reference masks
            dice = []
            for k, (ph, t) in enumerate((("ED", ed), ("ES", es))):
                g = tf(gts[ph])
                m = contour_to_mask(P[t], S)
                dice.append(2 * (m & g).sum() / (m.sum() + g.sum() + 1e-9))
            r[f"dice_ed_{view}"], r[f"dice_es_{view}"] = float(dice[0]), float(dice[1])
            beat = np.concatenate([frames, frames[-2::-1]])                      # ED -> ES -> ED
            res = pipe.analyse(np.concatenate([beat, beat[1:]]))                 # two beats, like a short clip
            r[f"head_ef_{view}"], r[f"head_lo_{view}"], r[f"head_hi_{view}"] = res["ef_median"], res["ef_lo90"], res["ef_hi90"]
            r[f"head_action_{view}"] = res["action"].split(":")[0]
        a, b = vols["4CH"], vols["2CH"]
        v = (a[0] * b[0]).sum(1) * np.maximum(a[1], b[1]); r["ef_biplane"] = float(100 * (1 - v[1] / v[0]))          # per-phase height (EchoNet convention)
        v = (a[0] * b[0]).sum(1) * max(a[1].max(), b[1].max()); r["ef_biplane_common"] = float(100 * (1 - v[1] / v[0]))   # CAMUS protocol
        st = max(refd["4CH"]["ED"][1], refd["4CH"]["ES"][1], refd["2CH"]["ED"][1], refd["2CH"]["ES"][1])
        vr = {ph: np.sum(refd["4CH"][ph][0] * refd["2CH"][ph][0]) * st for ph in ("ED", "ES")}; r["ef_ref_biplane"] = float(100 * (1 - vr["ES"] / vr["ED"]))
        rows.append(r); print(p, {k: (round(x, 1) if isinstance(x, float) else x) for k, x in r.items() if k in ("ef_label", "ef_4CH", "ef_biplane", "head_ef_4CH")}, flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    df.to_csv(os.path.join(ROOT, args.out + f"_{args.prep}.csv"), index=False)
    y = df.ef_label.values
    L = [f"CAMUS zero-shot, {len(df)} test patients (model and head trained on EchoNet-Dynamic only), preprocessing '{args.prep}'.", "",
         "EF against the CAMUS label (biplane Simpson with one disk height for ED and ES):"]
    for nm, col in (("reference tool on the reference masks (sanity: reproduces the label)", "ef_ref_biplane"),
                    ("predicted chords, 4CH, per-phase height (EchoNet convention)", "ef_4CH"), ("predicted chords, 4CH, common height (CAMUS)", "ef_4CH_common"),
                    ("predicted chords, 2CH, per-phase height", "ef_2CH"), ("predicted chords, 2CH, common height", "ef_2CH_common"),
                    ("predicted chords, biplane, per-phase height", "ef_biplane"), ("predicted chords, biplane, common height (CAMUS protocol)", "ef_biplane_common"),
                    ("EchoNet head on a synthetic beat, 4CH", "head_ef_4CH"), ("EchoNet head on a synthetic beat, 2CH", "head_ef_2CH")):
        p_ = df[col].values
        L.append(f"  {nm:62s}: R2 {r2(p_, y):+.3f}  MAE {np.abs(p_ - y).mean():5.2f}  r {np.corrcoef(p_, y)[0, 1]:.3f}  bias {np.mean(p_ - y):+6.2f}")
    L += ["", "The head predicts EF in the EchoNet convention, so it is compared with the single-plane per-phase EF of the reference masks (same convention):"]
    for view in ("4CH", "2CH"):
        t = df[f"ef_ref_{view}"].values; p_ = df[f"head_ef_{view}"].values; w = (df[f"head_hi_{view}"] - df[f"head_lo_{view}"]).values; e = np.abs(p_ - t)
        cov = (t >= df[f"head_lo_{view}"]) & (t <= df[f"head_hi_{view}"]); lo, hi = wilson(int(cov.sum()), len(cov)); covl = ((y >= df[f"head_lo_{view}"]) & (y <= df[f"head_hi_{view}"])).mean()
        L.append(f"  {view}: R2 {r2(p_, t):+.3f}  MAE {e.mean():.2f}  r {np.corrcoef(p_, t)[0, 1]:.3f}  bias {np.mean(p_ - t):+.2f}; 90% interval coverage {100 * cov.mean():.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}) "
                 f"against {100 * covl:.1f}% against the label; width AUROC for |error|>10: {auroc(e > 10, w):.2f}; decisions {df[f'head_action_{view}'].value_counts().to_dict()}")
    L.append("")
    for view in ("4CH", "2CH"):
        L.append(f"Dice {view}: ED {df[f'dice_ed_{view}'].mean():.3f}  ES {df[f'dice_es_{view}'].mean():.3f}")
    open(os.path.join(ROOT, args.out + f"_{args.prep}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
