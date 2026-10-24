"""Check that augment_similarity (scripts/train_tracer.py) moves the image and the tracing points by the same transform.

Draw the mask of real tracing polygons, warp it with the augmentation, and compare with the mask drawn from the transformed points
(Dice > 0.95); also check that the transform is the stated one (rotation about the image centre, isotropic scale, shift).

Usage: python tests/test_augmentation.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simpson import contour_to_mask  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
_spec = importlib.util.spec_from_file_location("tr", os.path.join(ROOT, "scripts", "train_tracer.py"))
tr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tr)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    z = np.load(os.path.join(ROOT, "data_cache", "records.npz"), allow_pickle=True)
    ix = np.where(z["valid"])[0][::200][:60]
    pts = torch.tensor(z["poly"][ix], device=dev)
    mask = torch.tensor(np.stack([contour_to_mask(p) for p in pts.cpu().numpy()]), dtype=torch.float32, device=dev)[:, None]

    warped, pts2 = tr.augment_similarity(mask, pts)
    a = warped.reshape(len(ix), -1) > 0.5
    m2 = torch.tensor(np.stack([contour_to_mask(p) for p in pts2.cpu().numpy()]), device=dev).reshape(len(ix), -1)
    dice = (2 * (a & m2).sum(1) / (a.sum(1) + m2.sum(1)).clamp(min=1)).float()
    print(f"Dice(warped mask, mask of the transformed points): mean {float(dice.mean()):.4f}, min {float(dice.min()):.4f}")
    assert float(dice.min()) > 0.95, "image warp and point transform disagree"

    # the transform on the points is a similarity: pairwise distances scale by one factor per sample
    d0 = (pts[:, :, None] - pts[:, None]).norm(dim=-1); d1 = (pts2[:, :, None] - pts2[:, None]).norm(dim=-1)
    k = (d1.sum((1, 2)) / d0.sum((1, 2)))
    spread = float(((d1 / k[:, None, None]) - d0).abs().max())
    print(f"scale factors {float(k.min()):.3f}-{float(k.max()):.3f}; max deviation from a pure similarity {spread:.2e} px")
    assert spread < 1e-3
    print("OK")


if __name__ == "__main__":
    main()
