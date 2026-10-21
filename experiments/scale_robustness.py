"""How does a trained ShapeEstimator behave when the frames are resized by a factor t (the LV becomes t times larger in pixels)?

The network is fully convolutional, so it runs at any input size; whether it is accurate depends on how much scale variation it saw. For each t the
VAL frames are resized, run, the points mapped back to the 112 x 112 frame, and Dice against the reference masks (dataset convention) is reported;
with --gate the LV long axis predicted in a first pass is used to rescale the frame to the trained LV size when it lies outside [0.7, 1.45] x that size.

Usage: python experiments/scale_robustness.py --ckpt checkpoints/shape_estimator.pt [--gate]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geometry import contour_to_mask  # noqa: E402
from shape_estimator import load_shape_estimator  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/shape_estimator.pt")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--every", type=int, default=4, help="use every n-th clean VAL record")
    ap.add_argument("--scales", default="0.5,0.7,1.0,1.4,2.0,3.0")
    args = ap.parse_args()
    dev = torch.device("cuda")
    net = load_shape_estimator(os.path.join(ROOT, args.ckpt), dev)
    z = np.load(os.path.join(ROOT, "data_cache", "records.npz"), allow_pickle=True); cb = np.load(os.path.join(ROOT, "data_cache", "context_frames.npz"))["frames"]
    idx = np.where((z["split"] == "VAL") & z["valid"])[0][::args.every]; G = z["poly"][idx]; ph = z["phase"][idx]
    gm = np.stack([contour_to_mask(g) for g in G])
    target = float(np.mean([np.linalg.norm(g[0] - g[41]) for g in z["poly"][np.where((z["split"] == "TRAIN") & z["valid"])[0][::20]]]))

    def predict(x, tot):
        H = max(32, int(round(112 * tot)))
        xs = F.interpolate(x[None], size=(H, H), mode="bilinear", align_corners=False) if H != 112 else x[None]
        with torch.no_grad():
            p = net(xs)[0].cpu().numpy()
        k = H / 112; col = (p[:, 0] - 1 + 0.5) / k - 0.5; row = (H - p[:, 1] + 0.5) / k - 0.5
        return np.stack([col + 1, 112 - row], -1), float(np.linalg.norm(p[0] - p[41]))

    print(f"{'t':>5s} {'Dice ED':>8s} {'Dice ES':>8s}   (LV long axis {target:.0f} px in training; gate {'on' if args.gate else 'off'})")
    for t in (float(v) for v in args.scales.split(",")):
        P = []
        for i in range(len(idx)):
            x = torch.tensor(cb[idx[i]], dtype=torch.float32, device=dev).div(255)
            q, ell = predict(x, t); tot = t
            if args.gate:
                for _ in range(2):
                    r = ell / target
                    if r < 0.7 or r > 1.45:
                        tot *= float(np.clip(1 / r, 0.35, 4.0)); q, ell = predict(x, tot)
                    else:
                        break
            P.append(q)
        pm = np.stack([contour_to_mask(q) for q in P]); d = 2 * (pm & gm).sum((1, 2)) / (pm.sum((1, 2)) + gm.sum((1, 2)))
        print(f"{t:5.1f} {d[ph == 'ED'].mean():8.4f} {d[ph == 'ES'].mean():8.4f}")


if __name__ == "__main__":
    main()
