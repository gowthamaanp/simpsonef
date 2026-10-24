"""Reference outputs for testing the browser pipeline (docs/js/pipeline.js) against the PyTorch pipeline.

For the first --n TEST videos: frames are written to <out>/<key>.u8 (T x 112 x 112 uint8) and the result of src/pipeline.py with
the demo's settings (no input-copy averaging) to <out>/reference.json.

Usage: python tools/reference_ef.py --n 20 --data data --out /tmp/simpsonef_ref
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from pipeline import SimpsonEF, read_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = SimpsonEF(copies=False)
    m = pd.read_csv(os.path.join(a.data, "FileList.csv")); m = m[m.Split == "TEST"].head(a.n)
    out = {}
    for r in m.itertuples():
        k = r.FileName.replace(".avi", "")
        fr, _ = read_video(os.path.join(a.data, "Videos", f"{k}.avi"))
        fr.astype(np.uint8).tofile(os.path.join(a.out, f"{k}.u8"))
        res = model.analyse(fr)
        out[k] = dict(T=len(fr), ef=res["ef"], lo=res.get("lo90"), hi=res.get("hi90"), beats=[list(map(int, b)) for b in res["beats"]], label=float(r.EF))
        print(k, len(fr), round(res["ef"], 2), [round(e, 1) for e in res["beat_ef"]], flush=True)
    json.dump(out, open(os.path.join(a.out, "reference.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
