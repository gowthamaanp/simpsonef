"""CAMUS zero-shot EF from the image sequence only: no annotated ED/ES frame, no mask, no EF head.

Per view: half-sequence frames -> model -> tracings -> chords. ED and ES are the frames of largest and smallest predicted
volume (sum d_i^2 * axis) in the sequence. Biplane modified Simpson with the CAMUS protocol (one disk height: the longest axis over
views and phases), and the single-view EF, against the CAMUS EF label. All 50 test patients.

Usage: python experiments/camus_blind_ef.py
"""
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import camus_zeroshot as cz  # noqa: E402
from geometry import chord_widths  # noqa: E402
from inference import Pipeline  # noqa: E402
from uncertainty_utils import r2  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


CKPT = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else "checkpoints/shape_estimator.pt"
HEAD = sys.argv[sys.argv.index("--head") + 1] if "--head" in sys.argv else "checkpoints/ef_head.pt"
DETECTOR = "--detector" in sys.argv
NOPAIR = []
MODE = "detector" if DETECTOR else "volume extremes"


def main():
    pipe = Pipeline(os.path.join(ROOT, CKPT), os.path.join(ROOT, HEAD), os.path.join(ROOT, "checkpoints/conformal_final.json"))
    nd = os.path.join(cz.CAMUS, "database_nifti")
    pats = [l.strip() for l in open(os.path.join(cz.CAMUS, "database_split", "subgroup_testing.txt")) if l.strip()]
    rows = []
    for p in pats:
        v = {}
        for view in ("4CH", "2CH"):
            cfg = cz.info(os.path.join(nd, p, f"Info_{view}.cfg"))
            im = nib.load(os.path.join(nd, p, f"{p}_{view}_half_sequence.nii.gz")); vol = im.get_fdata().astype(np.float32)
            side = max(vol.shape[0], vol.shape[1]); scale = side / cz.S * float(np.mean(im.header.get_zooms()[:2]))
            frames = np.stack([cz.pad_to_square_resize(vol[:, :, t].T, cz.S, cz.cv2.INTER_LINEAR) for t in range(vol.shape[2])]).clip(0, 255).astype(np.uint8)
            d, axis = chord_widths(pipe.points(frames))
            d, axis = np.asarray(d) * scale, np.asarray(axis) * scale
            vp = (d ** 2).sum(1) * axis
            e, t = int(vp.argmax()), int(vp.argmin())
            if DETECTOR:                                                    # detector on the original half-sequence, as released
                bt = pipe.analyse(frames)["beats"]
                if bt:
                    e, t = bt[0]["ed"], bt[0]["es"]
                else:
                    NOPAIR.append((p, view))
            v[view] = (d, axis, e, t, float(cfg["EF"]))
        r = {"patient": p, "label": v["4CH"][4]}
        for view in ("4CH", "2CH"):
            d, axis, e, s, _ = v[view]; h = max(v["4CH"][1].max(), v["2CH"][1].max())
            r[f"ef_{view}"] = 100 * (1 - (d[s] ** 2).sum() / (d[e] ** 2).sum())                       # one disk height cancels
        (d4, a4, e4, s4, _), (d2, a2, e2, s2, _) = v["4CH"], v["2CH"]
        r["ef_biplane"] = 100 * (1 - (d4[s4] * d2[s2]).sum() / (d4[e4] * d2[e2]).sum())               # same common height in both phases cancels
        rows.append(r)
    y = np.array([r["label"] for r in rows])
    for k, nm in (("ef_biplane", "biplane, CAMUS protocol"), ("ef_4CH", "4CH only"), ("ef_2CH", "2CH only")):
        pr = np.array([r[k] for r in rows])
        print(f"CAMUS zero-shot, all {len(y)} test patients, blind ED/ES ({MODE}), {nm}: R2 {r2(pr, y):.3f}  MAE {np.abs(pr - y).mean():.2f}  r {np.corrcoef(pr, y)[0, 1]:.3f}  bias {(pr - y).mean():+.2f}")


if __name__ == "__main__":
    main()
    if DETECTOR:
        print(f"views with no detected beat (fell back to volume extremes): {len(NOPAIR)}")
