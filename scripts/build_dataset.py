"""Training records from the contour tracings (VolumeTracings.csv): data_cache/records.npz.

One record per annotated frame (ED and ES of every TRAIN and VAL video): key, split, phase, frame index, EF label and the 42-point
tracing polygon [A1, L1..L20, R20..R1, A2] in decoder coordinates (x = column + 1, y = 112 - row). A tracing has 21 rows
(the long axis and 20 chords); in a few frames several complete tracings are stacked (42, 63, ... rows). Those records have
valid = False and are never used for training.

Usage: python scripts/build_dataset.py
"""
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")
S = 112


def canonical_polygon(a):
    """a (21, 4) tracing rows -> (42, 2) polygon in decoder coordinates.

    Row 0 is the long axis, rows 1-20 are the chord end points. Axis end A1 is the one nearest the first chord; 'left' is the
    positive side of the oriented axis A1 -> A2.
    """
    A1, A2, L, R = a[0, :2].copy(), a[0, 2:].copy(), a[1:, :2].copy(), a[1:, 2:].copy()
    mid = (L + R) / 2
    if np.linalg.norm(mid[0] - A1) > np.linalg.norm(mid[0] - A2):
        A1, A2 = A2, A1
    u = (A2 - A1) / np.linalg.norm(A2 - A1)
    nrm = np.array([-u[1], u[0]])
    swap = ((L - R) @ nrm) < 0
    L2, R2 = L.copy(), R.copy()
    L2[swap], R2[swap] = R[swap], L[swap]
    corner = np.vstack([A1, L2, R2[::-1], A2])
    return np.stack([corner[:, 0] + 0.5, S + 0.5 - corner[:, 1]], 1)


def main():
    master = pd.read_csv(os.path.join(D, "master.csv")).dropna(subset=["ED_Frame", "ES_Frame", "EF"])
    master = master[master.Split.isin(["TRAIN", "VAL"])]
    vt = pd.read_csv(os.path.join(D, "VolumeTracings.csv"))
    vt["key"] = vt.FileName.str.replace(".avi", "", regex=False)
    groups = {k: g[["X1", "Y1", "X2", "Y2"]].values for k, g in vt.groupby(["key", "Frame"])}

    key, split, phase, frame, ef, poly, valid = [], [], [], [], [], [], []
    for _, r in master.iterrows():
        cap = cv2.VideoCapture(os.path.join(D, "Videos", f"{r.FileName}.avi"))
        n, w, h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if (w, h) != (S, S):
            continue
        for ph in ("ED", "ES"):
            fi = int(r[f"{ph}_Frame"])
            if fi >= n:
                continue
            a = groups.get((r.FileName, fi))
            ok = a is not None and len(a) == 21
            key.append(r.FileName); split.append(r.Split); phase.append(ph); frame.append(fi); ef.append(float(r.EF))
            poly.append(canonical_polygon(a) if ok else np.zeros((42, 2)))
            valid.append(ok)
    np.savez(os.path.join(ROOT, "data_cache", "records.npz"), key=np.array(key), split=np.array(split), phase=np.array(phase),
             frame=np.array(frame), ef=np.array(ef, np.float32), poly=np.array(poly, np.float32), valid=np.array(valid))
    s = np.array(split)
    print(f"{len(key)} records ({(s == 'TRAIN').sum()} TRAIN, {(s == 'VAL').sum()} VAL); {int(np.sum(valid))} with a single complete tracing")


if __name__ == "__main__":
    main()
