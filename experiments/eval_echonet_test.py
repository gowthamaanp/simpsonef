"""Evaluate ShapeEstimator on VAL or TEST at the annotated ED / ES frames (no head involved).

  Dice      mask drawn from the predicted points with the dataset's polygon convention, against the released annotation masks
  EF raw    EF from the chord volumes of the predicted points at the annotated frames, against the EF label (R2, MAE, correlation)
Everything is reported for all videos and for the videos whose two labelled frames each hold a single complete tracing
(21 rows; a few frames carry several stacked tracings, which also corrupts the released masks).

Usage: python experiments/eval_echonet_test.py --split VAL --ckpt checkpoints/shape_estimator.pt
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geometry import chord_volume, contour_to_mask  # noqa: E402
from shape_estimator import checkpoint_offsets, count_params, load_shape_estimator  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")
S = 112


def r2(p, y):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"])
    ap.add_argument("--ckpt", default="checkpoints/shape_estimator.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = torch.device("cuda")
    net = load_shape_estimator(os.path.join(ROOT, args.ckpt), dev)
    offsets = checkpoint_offsets(os.path.join(ROOT, args.ckpt))
    print(f"ShapeEstimator {count_params(net):,} parameters, {len(offsets)} input frames\n")

    vt = pd.read_csv(os.path.join(D, "VolumeTracings.csv"))
    rows = vt.groupby(["FileName", "Frame"]).size()
    master = pd.read_csv(os.path.join(D, "master.csv"))
    sel = master[master.Split == args.split].dropna(subset=["ED_Frame", "ES_Frame", "EF"])

    frames, keys, phases, gts, clean = [], [], [], [], {}
    for _, r in sel.iterrows():
        cap = cv2.VideoCapture(os.path.join(D, "Videos", f"{r.FileName}.avi")); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok_video = True
        for phase, fi in (("ED", int(r.ED_Frame)), ("ES", int(r.ES_Frame))):
            mp = os.path.join(D, "annotation", "Masks", phase, f"{r.FileName}.png")
            stack = []
            for o in offsets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(fi + o, 0, nfr - 1)))
                ok, fr = cap.read()
                if not ok:
                    break
                stack.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            if not os.path.exists(mp) or len(stack) != len(offsets) or stack[0].shape != (S, S):
                ok_video = False
                break
            frames.append(np.stack(stack).astype(np.float32) / 255.0); keys.append(r.FileName); phases.append(phase)
            gts.append(np.array(Image.open(mp)) > 0)
            clean[(r.FileName, phase)] = rows.get((r.FileName + ".avi", fi), 0) == 21
        cap.release()
        if not ok_video:                                   # keep videos whole: drop a half-read pair
            while keys and keys[-1] == r.FileName:
                keys.pop(); phases.pop(); frames.pop(); gts.pop()

    frames = torch.tensor(np.stack(frames), dtype=torch.float32)
    P = []
    with torch.no_grad():
        for i in range(0, len(frames), args.batch_size):
            P.append(net(frames[i:i + args.batch_size].to(dev)).cpu().numpy())
    P = np.concatenate(P)
    masks = np.stack([contour_to_mask(p, S) for p in P])
    dice = np.array([2 * (m & g).sum() / (m.sum() + g.sum() + 1e-9) for m, g in zip(masks, gts)])
    df = pd.DataFrame({"key": keys, "phase": phases, "dice": dice, "logv": np.log(chord_volume(P)),
                       "clean": [clean[(k, p)] for k, p in zip(keys, phases)]})
    lv = df.pivot_table(index="key", columns="phase", values="logv").dropna()
    cl = df.pivot_table(index="key", columns="phase", values="clean").loc[lv.index].min(axis=1).astype(bool)
    ef_pred = 100 * (1 - np.exp(lv["ES"] - lv["ED"]))
    ef_true = master.set_index("FileName").loc[lv.index, "EF"]

    for name, vsel in (("all videos", np.ones(len(lv), bool)), ("clean tracings only", cl.values)):
        keep = set(lv.index[vsel]); d = df[df.key.isin(keep)]
        p, y = ef_pred[vsel].values, ef_true[vsel].values
        print(f"=== {args.split}, {name} (n={vsel.sum()} videos) ===")
        for ph in ("ED", "ES"):
            print(f"  Dice {ph}: mean {d[d.phase == ph].dice.mean():.4f}  median {d[d.phase == ph].dice.median():.4f}")
        print(f"  Dice pooled {d.dice.mean():.4f}")
        print(f"  EF at the annotated frames (raw): R2 {r2(p, y):.3f}  MAE {np.abs(p - y).mean():.2f}  r {np.corrcoef(p, y)[0, 1]:.3f}\n")


if __name__ == "__main__":
    main()
