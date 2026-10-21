"""Cache the chord widths of every frame of every labelled EchoNet video (TRAIN, VAL, TEST) for the temporal head.

Per frame: the 20 chord widths and the long-axis length, in pixels, from the points predicted by ShapeEstimator (a checkpoint
that was trained with neighbouring frames gets them as input channels, clipped at the clip ends). Flat storage: feats
(total_frames, 21) float16; offsets (n_videos + 1); video i occupies rows offsets[i]:offsets[i + 1].

Usage: python experiments/cache_features.py --ckpt checkpoints/shape_estimator.pt --out data_cache/features.npz
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geometry import chord_widths  # noqa: E402
from shape_estimator import checkpoint_offsets, load_shape_estimator  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")
S = 112


def decode(key):
    """All grayscale 112x112 frames of one video (empty if the frame size differs)."""
    cap = cv2.VideoCapture(os.path.join(D, "Videos", f"{key}.avi"))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if g.shape != (S, S):
            break
        fr.append(g)
    cap.release()
    return key, (np.stack(fr) if fr else np.zeros((0, S, S), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/shape_estimator.pt")
    ap.add_argument("--out", default="data_cache/features.npz")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--pair-self", action="store_true", help="the checkpoint is a pair network trained with single-frame mode: feed (window, window) and read the first tracing")
    args = ap.parse_args()
    dev = torch.device("cuda")
    net = load_shape_estimator(os.path.join(ROOT, args.ckpt), dev)
    offsets = np.array(checkpoint_offsets(os.path.join(ROOT, args.ckpt)))
    trace = (lambda x: net(torch.cat([x, x], 1))[:, :42]) if args.pair_self else net

    m = pd.read_csv(os.path.join(D, "master.csv")).dropna(subset=["ED_Frame", "ES_Frame", "EF"])
    m = m[m.Split.isin(["TRAIN", "VAL", "TEST"])]
    meta = {r.FileName: r for r in m.itertuples()}
    keys, splits, ed, es, ef, feats, lens = [], [], [], [], [], [], []
    t0 = time.time()
    with Pool(8) as pool, torch.no_grad():
        for n, (key, fr) in enumerate(pool.imap(decode, m.FileName.values, chunksize=4)):
            r = meta[key]
            if len(fr) < 20 or max(int(r.ED_Frame), int(r.ES_Frame)) >= len(fr):
                continue
            ctx = np.clip(np.arange(len(fr))[:, None] + offsets[None, :], 0, len(fr) - 1)
            x = torch.from_numpy(fr[ctx]).float().div_(255.0)
            fo = []
            for j in range(0, len(x), args.chunk):
                d, axis = chord_widths(trace(x[j:j + args.chunk].to(dev)))
                fo.append(torch.cat([d, axis[:, None]], 1).half().cpu())
            feats.append(torch.cat(fo).numpy()); lens.append(len(fr))
            keys.append(key); splits.append(r.Split); ed.append(int(r.ED_Frame)); es.append(int(r.ES_Frame)); ef.append(float(r.EF))
            if (n + 1) % 500 == 0:
                print(f"  {n + 1}/{len(m)}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(ROOT, args.out), keys=np.array(keys), splits=np.array(splits), ed=np.array(ed), es=np.array(es),
             ef=np.array(ef, np.float32), offsets=np.concatenate([[0], np.cumsum(lens)]), feats=np.concatenate(feats))
    print(f"saved {args.out}: {len(keys)} videos, {sum(lens)} frames ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
