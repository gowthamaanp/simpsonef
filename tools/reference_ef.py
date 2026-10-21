"""Reference outputs for testing the browser pipeline (docs/js/pipeline.js) against the PyTorch models.

For the first --n TEST videos: frames are written to <out>/<key>.u8 (T x 112 x 112 uint8) and the PyTorch result of the same
pipeline as the demo (FrameTracer + BeatFinder beats, BeatTracer without input-copy averaging, median over beats, conformal
interval) to <out>/reference.json.

Usage: python tools/reference_ef.py --n 20 --out /tmp/simpsonef_ref
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import build_context_bank as bank  # noqa: E402
import eval_pair as ep  # noqa: E402
from inference import Pipeline  # noqa: E402
from shape_estimator import checkpoint_offsets, load_shape_estimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Pipeline(os.path.join(ROOT, "checkpoints/shape_estimator.pt"), os.path.join(ROOT, "checkpoints/ef_head.pt"), None, device=str(dev))
    net = load_shape_estimator(os.path.join(ROOT, "checkpoints/d2s3.pt"), dev); off = checkpoint_offsets(os.path.join(ROOT, "checkpoints/d2s3.pt"))
    Q = json.load(open(os.path.join(ROOT, "checkpoints/conformal_formula.json")))["Q90"]
    import pandas as pd
    m = pd.read_csv(os.path.join(ROOT, "data", "master.csv")); keys = m[m.Split == "TEST"].FileName.values[:a.n]
    out = {}
    for k in keys:
        fr = bank.read_video(k)
        if fr.shape[1:] != (112, 112):
            continue
        fr.astype(np.uint8).tofile(os.path.join(a.out, f"{k}.u8"))
        res = pipe.analyse(fr); beats = [(b["ed"], b["es"]) for b in res["beats"]]
        if not beats:
            beats = [(int(np.argmax(res["heat_ed"])) + res["head_offset"], int(np.argmax(res["heat_es"])) + res["head_offset"])]
        Pp = ep.pair_points(net, fr, beats, off, dev)
        d, l = ep.chord_widths(Pp); v = (np.asarray(d) ** 2).sum(-1) * np.asarray(l); efs = 100 * (1 - v[:, 1] / v[:, 0])
        ef = float(np.median(efs)); s = (res["ef_q95"] - res["ef_q05"]) / 2 + abs(ef - res["ef_median"])
        out[k] = dict(T=len(fr), ef=ef, lo=ef - Q * s, hi=ef + Q * s, beats=[list(map(int, b)) for b in beats], label=float(m.set_index("FileName").loc[k, "EF"]))
        print(k, len(fr), round(ef, 2), [round(e, 1) for e in efs], flush=True)
    json.dump(out, open(os.path.join(a.out, "reference.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
