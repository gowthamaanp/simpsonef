"""Cache the frames around every annotated frame of data_cache/records.npz, aligned record for record.

For each record the frames at (annotated index + offset) are stored, clipped to the video, so the network sees motion. Writes
data_cache/context_frames.npz: frames (N, len(offsets), 112, 112) uint8 and the offsets; the offset-0 channel is the annotated frame.

Usage: python experiments/build_context_bank.py [--offsets=-24,-20,...,24]
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")
DEFAULT_OFFSETS = ",".join(str(o) for o in range(-24, 25, 4))


def read_video(key):
    cap = cv2.VideoCapture(os.path.join(D, "Videos", f"{key}.avi")); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(fr)


def worker(job):
    key, items, offsets = job
    v = read_video(key); n = len(v)
    return [(rec, np.stack([v[int(np.clip(f + o, 0, n - 1))] for o in offsets])) for rec, f in items]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--offsets", default=DEFAULT_OFFSETS)
    ap.add_argument("--out", default="data_cache/context_frames.npz"); args = ap.parse_args()
    offsets = [int(o) for o in args.offsets.split(",")]
    z = np.load(os.path.join(ROOT, "data_cache", "records.npz"), allow_pickle=True)
    jobs = {}
    for rec, (k, f) in enumerate(zip(z["key"], z["frame"])):
        jobs.setdefault(k, []).append((rec, int(f)))
    out = np.zeros((len(z["key"]), len(offsets), 112, 112), np.uint8)
    with Pool(8) as pool:
        for res in pool.imap_unordered(worker, [(k, v, offsets) for k, v in jobs.items()], chunksize=8):
            for rec, st in res:
                out[rec] = st
    np.savez(os.path.join(ROOT, args.out), frames=out, offsets=np.array(offsets))
    print(f"{len(out)} records, offsets {offsets}")


if __name__ == "__main__":
    main()
