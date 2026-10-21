"""Train ShapeEstimator: a heatmap network that finds the 42 tracing points of the LV from a window of frames.

Targets are the tracing points of the annotated ED and ES frames (data_cache/records.npz); no mask is learned. Losses:
  points      L1 on the 42 point coordinates
  heatmap     cross-entropy of every point's heatmap against a Gaussian at its target
  volume      L1 on the chord log-volume (method of disks straight from the points)
  differential  L1 on the ED -> ES change of the chord log-volume (what EF depends on)
  width       L1 on log chord widths, each weighted by its share of the volume
ES frames are weighted --es-weight against ED. Optional --w-dice / --w-boundary add mask losses (default off).
Augmentation: one similarity transform (rotation, scale, shift; image and points transformed together, see
tests/test_augmentation.py), edge occlusion, and scanner-style degradations (augment_scanner). No flips.
Validation: chord log-volume error and ED/ES differential error against the tracings, and the Dice (dataset mask convention) of the mask drawn from
the points; the checkpoint with the lowest summed volume errors is kept. TEST is never touched.

Usage: python experiments/train_shape_estimator.py --epochs 80 --ema 0.999      (-> checkpoints/shape_estimator.pt)
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geometry import chord_widths, contour_to_mask, pixel_grid, soft_membership  # noqa: E402
from shape_estimator import RegressionEstimator, ScaleSpaceEstimator, ShapeEstimator, count_params, heat_loss  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def augment_similarity(x, pts, max_shift_px=18.0, max_rot_deg=20.0, scale_range=(0.8, 1.2)):
    """Random rotation about the image centre, isotropic scaling and shift of the frames, with the same transform on the points.

    Image content p -> c0 + k R(phi) (p - c0) + t, in decoder coordinates. x: (N, C, H, W); pts: (N, P, 2).
    """
    N, _, H, W = x.shape
    dev, dt = x.device, x.dtype
    phi = (torch.rand(N, device=dev) * 2 - 1) * (np.pi * max_rot_deg / 180.0)
    k = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(N, device=dev)
    t = (torch.rand(N, 2, device=dev) * 2 - 1) * max_shift_px
    c0 = torch.tensor([(W + 1) / 2.0, (H + 1) / 2.0], device=dev, dtype=dt)
    cols, rows = torch.arange(W, device=dev, dtype=dt), torch.arange(H, device=dev, dtype=dt)
    P = torch.stack([(cols + 1)[None, :].expand(H, W), (H - rows)[:, None].expand(H, W)], -1).reshape(1, -1, 2)
    q = P - t[:, None, :] - c0
    cs, sn = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
    px = (cs * q[..., 0] + sn * q[..., 1]) / k[:, None] + c0[0]
    py = (-sn * q[..., 0] + cs * q[..., 1]) / k[:, None] + c0[1]
    grid = torch.stack([(px - 0.5) / W * 2 - 1, ((H - py) + 0.5) / H * 2 - 1], -1).reshape(N, H, W, 2)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    R = torch.stack([torch.stack([cs[:, 0], -sn[:, 0]], -1), torch.stack([sn[:, 0], cs[:, 0]], -1)], -2)
    pts2 = c0 + k[:, None, None] * torch.einsum("nij,npj->npi", R, pts - c0) + t[:, None, :]
    return out, pts2


def augment_occlude(x, gt_mask, p=0.35, min_width_frac=0.15, margin=2):
    """Black out one image edge, never cutting into the bounding box of the target mask. gt_mask: (N, H, W) bool."""
    N, _, H, W = x.shape
    out = x.clone()
    min_w = max(1, int(min_width_frac * W))
    gt_np = gt_mask.detach().cpu().numpy()
    draw = np.random.rand(N)
    for i in range(N):
        if draw[i] > p:
            continue
        ys, xs = np.nonzero(gt_np[i])
        if len(ys) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        widths = {"left": max(0, x0 - margin), "right": max(0, W - (x1 + margin)),
                  "top": max(0, y0 - margin), "bottom": max(0, H - (y1 + margin))}
        cands = [s for s, w in widths.items() if w >= min_w]
        if not cands:
            continue
        side = cands[np.random.randint(len(cands))]
        w = int(np.random.randint(min_w, min(widths[side], W // 2) + 1))
        if side == "left":
            out[i, :, :, :w] = 0
        elif side == "right":
            out[i, :, :, W - w:] = 0
        elif side == "top":
            out[i, :, :w, :] = 0
        else:
            out[i, :, H - w:, :] = 0
    return out


def augment_scanner(x):
    """Scanner-style degradations for a (N, C, H, W) batch of 0-1 frames: depth-dependent gain (TGC), gamma / contrast / brightness,
    spatially correlated multiplicative speckle, additive Gaussian noise, anisotropic blur, resolution loss, and radial attenuation
    shadows. Gain is shared by all channels of a sample; the noise is drawn independently per channel."""
    N, C, H, W = x.shape
    dev = x.device
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*(s or (N, 1, 1, 1)), device=dev)
    on = lambda p: (torch.rand(N, 1, 1, 1, device=dev) < p).float()
    rows = torch.linspace(-0.5, 0.5, H, device=dev).view(1, 1, H, 1)
    ramp = on(0.5)
    x = x * (1 - ramp + ramp * torch.exp(u(-1.0, 1.0) * rows))                              # TGC: top-to-bottom gain ramp
    g = torch.where(on(0.7) > 0, u(0.6, 1.6), torch.ones(N, 1, 1, 1, device=dev))
    x = (x.clamp(min=1e-6) ** g * u(0.7, 1.4) + u(-0.05, 0.05)).clamp(0, 1)                 # gamma, contrast, brightness
    p99 = torch.quantile(x.flatten(1), 0.99, dim=1).view(N, 1, 1, 1)                          # keep every sample readable
    x = (x * (0.5 / p99.clamp(min=1e-3)).clamp(min=1.0)).clamp(0, 1)
    n = torch.randn(N, C, H, W, device=dev)                                                 # speckle: correlated multiplicative noise
    sg = float(0.6 + 0.8 * torch.rand(()))
    r = max(1, int(3 * sg)); k = torch.exp(-(torch.arange(-r, r + 1, device=dev, dtype=x.dtype) ** 2) / (2 * sg ** 2)); k = (k / k.sum()).view(1, 1, 1, -1)
    n = F.conv2d(F.pad(n.reshape(N * C, 1, H, W), (r, r, 0, 0), mode="reflect"), k)
    n = F.conv2d(F.pad(n, (0, 0, r, r), mode="reflect"), k.transpose(2, 3)).reshape(N, C, H, W)
    n = n / n.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    x = (x * (1 + on(0.7) * u(0.1, 0.5) * n)).clamp(0, 1)
    x = (x + on(0.5) * u(0.01, 0.06) * torch.randn_like(x)).clamp(0, 1)                      # additive Gaussian noise
    sub = torch.rand(N, device=dev) < 0.5                                                   # anisotropic blur on a random half
    if sub.any():
        sx, sy = float(0.3 + 1.5 * torch.rand(())), float(0.3 + 1.5 * torch.rand(()))
        def k1(s):
            rr = max(1, int(3 * s)); kk = torch.exp(-(torch.arange(-rr, rr + 1, device=dev, dtype=x.dtype) ** 2) / (2 * s ** 2)); return rr, kk / kk.sum()
        rx, kx = k1(sx); ry, ky = k1(sy); xs_ = x[sub]; M = xs_.shape[0]
        b = F.conv2d(F.pad(xs_.reshape(M * C, 1, H, W), (rx, rx, 0, 0), mode="reflect"), kx.view(1, 1, 1, -1))
        b = F.conv2d(F.pad(b, (0, 0, ry, ry), mode="reflect"), ky.view(1, 1, -1, 1)).reshape(M, C, H, W)
        x = x.clone(); x[sub] = b
    if torch.rand(()).item() < 0.3:                                                         # resolution loss
        f = float(0.6 + 0.25 * torch.rand(()))
        lo_ = F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)
        x = F.interpolate(lo_, size=(H, W), mode="bilinear", align_corners=False)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=x.dtype), torch.arange(W, device=dev, dtype=x.dtype), indexing="ij")
    ang = torch.atan2(xx[None] - W / 2, yy[None] + 1.0)                                      # angle from the apex at the top centre
    c0 = u(-0.5, 0.5, N, 1, 1); wd = u(0.06, 0.2, N, 1, 1)
    shadow = torch.exp(-((ang - c0) ** 2) / (2 * wd ** 2)) * on(0.25).view(N, 1, 1) * u(0.4, 0.8, N, 1, 1)
    return (x * (1 - shadow.unsqueeze(1))).clamp(0, 1)


def bbox_mask(pts, size):
    """(N, size, size) bool: the bounding box of each polygon (decoder coordinates), enough to keep edge occlusion off the ventricle."""
    out = torch.zeros(len(pts), size, size, dtype=torch.bool)
    c0, c1 = (pts[..., 0] - 1).amin(1).floor().long().clamp(0, size - 1), (pts[..., 0] - 1).amax(1).ceil().long().clamp(0, size - 1)
    r0, r1 = (size - pts[..., 1]).amin(1).floor().long().clamp(0, size - 1), (size - pts[..., 1]).amax(1).ceil().long().clamp(0, size - 1)
    for i in range(len(pts)):
        out[i, r0[i]:r1[i] + 1, c0[i]:c1[i] + 1] = True
    return out


def seed_everything(seed):
    """Fix every random source (Python, NumPy, PyTorch CPU / GPU). GPU kernels stay in their fast (not bit-exact) mode: reruns
    match up to normal run-to-run variation."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class U8Frames:
    """Frame bank kept as uint8; indexing returns float32 in [0, 1] (a 25-frame bank would not fit in RAM as float)."""

    def __init__(self, a):
        self.a = torch.from_numpy(np.ascontiguousarray(a)); self.shape = self.a.shape

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i].float().div_(255.0)


def log_volume(P):
    """(N, 42, 2) points -> log chord volume (N,), method of disks straight from the points."""
    d, axis = chord_widths(P)
    return torch.log((np.pi / 4 * d ** 2).sum(1) * axis / 20.0 + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64, help="frames per step (half ED, half ES)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--ema", type=float, default=0.0, help="decay per step of an exponential moving average of the weights (0 = off)")
    ap.add_argument("--es-weight", type=float, default=1.25, help="loss weight of ES frames relative to ED")
    ap.add_argument("--w-pts", type=float, default=1.5)
    ap.add_argument("--w-heat", type=float, default=1.0)
    ap.add_argument("--w-vol", type=float, default=1.06, help="chord log-volume")
    ap.add_argument("--w-diff", type=float, default=4.0, help="ED -> ES change of the chord log-volume")
    ap.add_argument("--w-width", type=float, default=2.0, help="log chord widths weighted by their share of the volume")
    ap.add_argument("--arch", choices=["heatmap", "regress", "scalespace"], default="heatmap", help="regress: ablation baseline that regresses the 84 coordinates from a pooled feature")
    ap.add_argument("--heat-sigma", type=float, default=1.5, help="width (px) of the Gaussian heatmap target")
    ap.add_argument("--dropout", type=float, default=0.0, help="spatial dropout inside the residual blocks")
    ap.add_argument("--teacher", default="", help="checkpoint whose predictions on the TRAIN frames are blended into the targets (self-distillation)")
    ap.add_argument("--distill", type=float, default=0.5, help="weight of the teacher points in the target: (1 - distill) * tracing + distill * teacher")
    ap.add_argument("--w-dice", type=float, default=0.0, help="optional soft Dice between the mask drawn from the points and from the tracing")
    ap.add_argument("--w-boundary", type=float, default=0.0)
    ap.add_argument("--geo-rot", type=float, default=20.0)
    ap.add_argument("--geo-scale", default="0.8,1.2")
    ap.add_argument("--scale-range", default="", help="lo,hi: each batch is resized by a log-uniform factor in this range (canvas 112 x factor), image and points together; the network is fully convolutional")
    ap.add_argument("--geo-shift", type=float, default=18.0)
    ap.add_argument("--ef-balance", type=float, default=0.0, help="sample TRAIN pairs with probability ~ (1 / frequency of their 10-point EF bin) ** ef_balance (0 = uniform, 1 = flat over EF)")
    ap.add_argument("--pair", action="store_true", help="joint ED + ES model: input = ED window and ES window stacked (2 x frames), output = 84 points (ED 42, then ES 42)")
    ap.add_argument("--w-ef", type=float, default=0.0, help="with --pair: |formula EF of the predicted ED/ES tracings - EF label| in EF units (fraction), weighted by EF-bin rarity")
    ap.add_argument("--ef-rarity", type=float, default=0.5, help="exponent of the inverse EF-bin frequency weighting of the EF term (0 = unweighted)")
    ap.add_argument("--scanner-prob", type=float, default=1.0, help="share of training samples that get the scanner-style degradations (noise, speckle, blur, gain)")
    ap.add_argument("--ef-bmc", type=float, default=0.0, help="if > 0: the EF term is Balanced MSE (batch Monte-Carlo, Ren et al. 2022) with this noise sd (EF fraction) instead of the absolute error")
    ap.add_argument("--self-pair-prob", type=float, default=0.0, help="with --pair: share of samples given the same window in both slots (ED,ED or ES,ES), so the pair network also traces single frames")
    ap.add_argument("--tail-oversample", type=float, default=1.0, help="sampling weight of TRAIN pairs with EF < 45 or >= 70 relative to the others (1 = natural mix)")
    ap.add_argument("--half-prob", type=float, default=0.0, help="with --pair: share of pairs whose windows are clipped to the ED..ES half-cycle (frames before ED -> ED frame, after ES -> ES frame), as in a released half-sequence")
    ap.add_argument("--init", default="", help="checkpoint whose weights initialise the network (and are the pseudo-label teacher with --unlabeled)")
    ap.add_argument("--unlabeled", default="", help="bank of unlabelled TRAIN windows (build_unlabeled_bank.py); targets are the --init teacher's points")
    ap.add_argument("--w-unl", type=float, default=0.5, help="weight of the pseudo-label loss")
    ap.add_argument("--unl-batch", type=int, default=32)
    ap.add_argument("--context-bank", default="data_cache/context_frames.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints/shape_estimator.pt")
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = torch.device(args.device)

    z = np.load(os.path.join(ROOT, "data_cache", "records.npz"), allow_pickle=True)
    cb = np.load(os.path.join(ROOT, args.context_bank))
    ftr = U8Frames(cb["frames"])                                              # uint8 in memory, float in [0, 1] per batch
    offsets = [int(o) for o in cb["offsets"]]
    poly = torch.tensor(z["poly"], dtype=torch.float32)
    split, phase, keys, valid = z["split"], z["phase"], z["key"], z["valid"]
    frame_no = z["frame"].astype(np.int64)
    H = W = ftr.shape[-1]

    net = {"regress": RegressionEstimator, "scalespace": ScaleSpaceEstimator}.get(args.arch, ShapeEstimator)(
        width=args.width, in_ch=len(offsets) * (2 if args.pair else 1), dropout=args.dropout, **({"n_points": 84} if args.pair else {})).to(dev)
    if args.init:
        net.load_state_dict(torch.load(os.path.join(ROOT, args.init), map_location=dev, weights_only=False)["state"])
    print(f"ShapeEstimator: {count_params(net):,} parameters, {len(offsets)} input frames, offsets {offsets}", flush=True)
    grid = pixel_grid(H, dev)

    def make_pairs(ix):
        by = {}
        for i in ix:
            by.setdefault(keys[i], {})[phase[i]] = i
        pr = [(v["ED"], v["ES"]) for v in by.values() if "ED" in v and "ES" in v]
        return np.array([a for a, _ in pr]), np.array([b for _, b in pr])

    tr_ed, tr_es = make_pairs(np.where((split == "TRAIN") & valid)[0])
    va_ed, va_es = make_pairs(np.where((split == "VAL") & valid)[0])
    print(f"TRAIN {len(tr_ed)} ED/ES pairs   VAL {len(va_ed)} pairs", flush=True)
    va_idx = np.concatenate([va_ed, va_es]); nv = len(va_ed)
    fva, pva = ftr[va_idx], poly[va_idx].to(dev)
    with torch.no_grad():
        lv_ref = log_volume(pva)
    va_ref = np.stack([contour_to_mask(p) for p in pva.cpu().numpy()])         # the dataset's mask of each reference tracing

    train_poly = poly
    if args.teacher:
        from shape_estimator import load_shape_estimator
        tnet = load_shape_estimator(os.path.join(ROOT, args.teacher), dev)
        with torch.no_grad():
            tp = torch.cat([tnet(ftr[i:i + 64].to(dev)).cpu() for i in range(0, len(ftr), 64)])
        train_poly = (1 - args.distill) * poly + args.distill * tp
        print(f"self-distillation: targets = {1 - args.distill:.2f} x tracing + {args.distill:.2f} x teacher ({args.teacher})", flush=True)
        del tnet
    ufr = upts = None
    if args.unlabeled:
        from shape_estimator import load_shape_estimator
        ub = np.load(os.path.join(ROOT, args.unlabeled)); assert [int(o) for o in ub["offsets"]] == offsets
        ufr = torch.from_numpy(ub["frames"])                                              # uint8 (N, C, 112, 112), converted per batch
        tnet = load_shape_estimator(os.path.join(ROOT, args.init), dev)
        with torch.no_grad():
            upts = torch.cat([tnet(ufr[i:i + 64].to(dev).float().div_(255.0)).cpu() for i in range(0, len(ufr), 64)])
        dd, aa = chord_widths(upts); good = (aa > 20) & (dd.min(1).values > 1.0)           # drop windows where the teacher found no plausible LV
        ufr, upts = ufr[good], upts[good]
        print(f"pseudo-labels: {int(good.sum())} of {len(good)} windows kept, teacher {args.init}", flush=True)
        del tnet
    pw = None
    if args.ef_balance > 0:
        efb = np.clip((z["ef"][tr_ed] // 10).astype(int), 1, 8)                            # bins <20, 20-30, ..., >=80
        cnt = np.bincount(efb, minlength=9).astype(float)
        pw = (1.0 / cnt[efb]) ** args.ef_balance; pw /= pw.sum()
        eff = np.bincount(efb, weights=pw, minlength=9)[1:]
        print("EF-balanced sampling: share per EF bin " + " ".join(f"{100 * v:.0f}%" for v in eff) + " (bins <20, 20s, ..., >=80)", flush=True)
    ef_lab = torch.tensor(z["ef"], dtype=torch.float32) / 100.0
    efbin = lambda e: np.clip((np.asarray(e) * 10).astype(int), 1, 8)
    cnt_bin = np.bincount(efbin(ef_lab.numpy()[tr_ed]), minlength=9).astype(float)
    wbin = torch.tensor((1.0 / np.maximum(cnt_bin, 1)) ** args.ef_rarity, dtype=torch.float32)
    wbin = wbin / (wbin[efbin(ef_lab.numpy()[tr_ed])].mean())
    half = args.batch_size // 2
    scale_range = tuple(float(v) for v in args.geo_scale.split(","))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * max(1, len(tr_ed) // half)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + np.cos(np.pi * min(t, total) / total)))
    ema_net = copy.deepcopy(net).eval() if args.ema > 0 else None
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    gstep, best = 0, float("inf")

    def save(path, model):
        torch.save({"state": model.state_dict(), "width": args.width, "in_ch": len(offsets) * (2 if args.pair else 1), "offsets": offsets, "dropout": args.dropout, "arch": args.arch,
                    "n_points": 84 if args.pair else 42, "pair": args.pair, "config": vars(args)}, os.path.join(ROOT, path))

    for ep in range(args.epochs):
        net.train()
        if args.tail_oversample != 1.0 and pw is None:
            e_ = z["ef"][tr_ed]; wt = np.where((e_ < 45) | (e_ >= 70), args.tail_oversample, 1.0); wt = wt / wt.sum()
            perm = np.random.choice(len(tr_ed), len(tr_ed), p=wt)
        else:
            perm = np.random.permutation(len(tr_ed)) if pw is None else np.random.choice(len(tr_ed), len(tr_ed), p=pw)
        t0, nb = time.time(), 0
        for i in range(0, len(perm) - half + 1, half):
            pp = perm[i:i + half]
            idx = np.concatenate([tr_ed[pp], tr_es[pp]])
            x, pts = ftr[idx].to(dev), train_poly[idx].to(dev)
            if args.pair:                                                     # one sample per video: [ED window | ES window], [ED points | ES points]
                if args.half_prob > 0:
                    x = x.clone(); C = len(offsets); o0 = offsets.index(0); off_t = torch.tensor(offsets, device=dev)
                    gap = torch.from_numpy(frame_no[tr_es[pp]] - frame_no[tr_ed[pp]]).to(dev)
                    for j in torch.nonzero((gap > 0) & (torch.rand(half, device=dev) < args.half_prob))[:, 0].tolist():
                        g = int(gap[j]); fe, fs = x[j, o0].clone(), x[half + j, o0].clone()
                        x[j, off_t < 0] = fe; x[j, off_t >= g] = fs                       # ED window: before ED -> ED, at/after ES -> ES
                        x[half + j, off_t > 0] = fs; x[half + j, off_t <= -g] = fe       # ES window: after ES -> ES, at/before ED -> ED
                selfp = torch.zeros(half, dtype=torch.bool, device=dev)
                if args.self_pair_prob > 0:
                    x, pts = x.clone(), pts.clone()
                    selfp = torch.rand(half, device=dev) < args.self_pair_prob; use_es = torch.rand(half, device=dev) < 0.5
                    for j in torch.nonzero(selfp)[:, 0].tolist():                # same frame in both slots, same target twice
                        a_, b_ = (half + j, j) if use_es[j] else (j, half + j)
                        x[b_], pts[b_] = x[a_], pts[a_]
                x, pts = torch.cat([x[:half], x[half:]], 1), torch.cat([pts[:half], pts[half:]], 1)
            Hc = H
            if args.scale_range:
                lo_, hi_ = (float(v) for v in args.scale_range.split(","))
                Hc = int(round(H * float(np.exp(np.random.uniform(np.log(lo_), np.log(hi_))))))
                if Hc != H:
                    x = F.interpolate(x, size=(Hc, Hc), mode="bilinear", align_corners=False); kk = Hc / H
                    pts = torch.stack([((pts[..., 0] - 1 + 0.5) * kk - 0.5) + 1, Hc - ((H - pts[..., 1] + 0.5) * kk - 0.5)], -1)
            x, pts = augment_similarity(x, pts, args.geo_shift * Hc / H, args.geo_rot, scale_range)
            with torch.no_grad():
                x = augment_occlude(x, bbox_mask(pts, Hc))
                if args.scanner_prob >= 1.0:
                    x = augment_scanner(x)
                elif args.scanner_prob > 0:
                    sel = torch.rand(len(x), device=x.device) < args.scanner_prob
                    if sel.any():
                        x = x.clone(); x[sel] = augment_scanner(x[sel])
            if args.arch == "regress":
                pred = net(x); logit = None
            else:
                logit = net.heatmaps(x)
                pred = net.decode(logit)
                if args.pair:                                                 # back to the per-frame layout [ED frames; ES frames] used by the losses
                    logit, pred, pts = (torch.cat([t[:, :42], t[:, 42:]], 0) for t in (logit, pred, pts))
            nh = len(idx) // 2
            sw = torch.cat([torch.ones(nh, device=dev), torch.full((nh,), args.es_weight, device=dev)])
            sw = sw / sw.mean()
            loss = args.w_pts * ((pred - pts).abs().mean((1, 2)) * sw).mean()
            if logit is not None:
                loss = loss + args.w_heat * (heat_loss(logit, pts, args.heat_sigma) * sw).mean()
            dp, ap_ = chord_widths(pred); dg, ag = chord_widths(pts)
            lvp = torch.log((np.pi / 4 * dp ** 2).sum(1) * ap_ / 20.0 + 1.0); lvg = torch.log((np.pi / 4 * dg ** 2).sum(1) * ag / 20.0 + 1.0)
            loss = loss + args.w_vol * ((lvp - lvg).abs() * sw).mean()
            loss = loss + args.w_diff * (((lvp[nh:] - lvp[:nh]) - (lvg[nh:] - lvg[:nh])).abs().mean())
            if args.w_ef > 0:
                vp_ = (dp ** 2).sum(1) * ap_; efp = 1 - vp_[nh:] / vp_[:nh].clamp(min=1e-6)
                ey = ef_lab[tr_ed[pp]].to(dev); ew = wbin[efbin(ef_lab.numpy()[tr_ed[pp]])].to(dev)
                if args.pair:
                    ew = ew * (~selfp).float()                                       # self pairs carry no EF label
                if args.ef_bmc > 0:                                           # each prediction must sit nearer its own label than the batch's other labels
                    lg = -((efp[:, None] - ey[None, :]) ** 2) / (2 * args.ef_bmc ** 2)
                    l_bmc = F.cross_entropy(lg, torch.arange(len(efp), device=dev), reduction="none") * (2 * args.ef_bmc ** 2)
                    loss = loss + args.w_ef * (ew * l_bmc).mean()
                else:
                    loss = loss + args.w_ef * (ew * (efp - ey).abs()).mean()
            share = dg ** 2 / (dg ** 2).sum(1, keepdim=True)
            loss = loss + args.w_width * ((share * (torch.log(dp.clamp(min=0.5)) - torch.log(dg.clamp(min=0.5))).abs()).sum(1) * sw).mean()
            if args.w_dice > 0 or args.w_boundary > 0:
                chi = soft_membership(pred, grid)
                gt = soft_membership(pts, grid)
                dice = 1 - (2 * (chi * gt).sum(1) + 1e-6) / (chi.sum(1) + gt.sum(1) + 1e-6)
                loss = loss + args.w_dice * (dice * sw).mean()
                if args.w_boundary > 0:
                    loss = loss + args.w_boundary * ((pred - pts).norm(dim=-1).mean(1) * sw).mean()
            if ufr is not None:
                ui = torch.randint(0, len(ufr), (args.unl_batch,))
                ux, up = ufr[ui].to(dev).float().div_(255.0), upts[ui].to(dev)
                Hu = H
                if args.scale_range:
                    Hu = int(round(H * float(np.exp(np.random.uniform(np.log(lo_), np.log(hi_))))))
                    if Hu != H:
                        ux = F.interpolate(ux, size=(Hu, Hu), mode="bilinear", align_corners=False); kk = Hu / H
                        up = torch.stack([((up[..., 0] - 1 + 0.5) * kk - 0.5) + 1, Hu - ((H - up[..., 1] + 0.5) * kk - 0.5)], -1)
                ux, up = augment_similarity(ux, up, args.geo_shift * Hu / H, args.geo_rot, scale_range)
                with torch.no_grad():
                    ux = augment_scanner(augment_occlude(ux, bbox_mask(up, Hu)))
                ul = net.heatmaps(ux); upred = net.decode(ul)
                lu = args.w_pts * (upred - up).abs().mean((1, 2)).mean() + args.w_heat * heat_loss(ul, up, args.heat_sigma).mean()
                dp_, ap2 = chord_widths(upred); dg_, ag2 = chord_widths(up)
                lu = lu + args.w_vol * (torch.log((np.pi / 4 * dp_ ** 2).sum(1) * ap2 / 20.0 + 1.0) - torch.log((np.pi / 4 * dg_ ** 2).sum(1) * ag2 / 20.0 + 1.0)).abs().mean()
                loss = loss + args.w_unl * lu
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            gstep += 1
            if ema_net is not None:
                d = min(args.ema, (1.0 + gstep) / (10.0 + gstep))
                with torch.no_grad():
                    for qe, q in zip(ema_net.parameters(), net.parameters()):
                        qe.mul_(d).add_(q.detach(), alpha=1.0 - d)
            nb += 1

        net.eval()
        ev = ema_net if ema_net is not None else net
        with torch.no_grad():
            if args.pair:
                fpair = torch.cat([fva[:nv], fva[nv:]], 1)
                pv = torch.cat([ev(fpair[i:i + 32].to(dev)) for i in range(0, nv, 32)]); pv = torch.cat([pv[:, :42], pv[:, 42:]], 0)
            else:
                pv = torch.cat([ev(fva[i:i + 64].to(dev)) for i in range(0, len(fva), 64)])
        pm = np.stack([contour_to_mask(p) for p in pv.cpu().numpy()])
        dice = 2 * (pm & va_ref).sum((1, 2)) / (pm.sum((1, 2)) + va_ref.sum((1, 2)) + 1e-9)
        lv = log_volume(pv)
        vol_err = float((lv - lv_ref).abs().mean())
        diff_err = float((((lv[nv:] - lv[:nv]) - (lv_ref[nv:] - lv_ref[:nv])).abs()).mean())
        score = vol_err + diff_err
        vv = torch.exp(lv) - 1.0                                              # chord volume (log_volume adds 1)
        ef_va = (1 - vv[nv:] / vv[:nv]).cpu().numpy() * 100; ef_y = z["ef"][va_ed]
        ef_r2 = 1 - ((ef_va - ef_y) ** 2).sum() / ((ef_y - ef_y.mean()) ** 2).sum()
        if args.w_ef > 0:
            score = 1.0 - ef_r2
        tag = ""
        if score < best:
            best = score; tag = "  *"
            save(args.out, ev)
        print(f"[ep {ep:3d}] loss={float(loss):.3f}  VAL Dice ED={dice[:nv].mean():.4f} ES={dice[nv:].mean():.4f}  "
              f"|log-vol err|={vol_err:.4f} |diff err|={diff_err:.4f}  VAL formula EF R2={ef_r2:.3f}  score={score:.4f}  ({time.time() - t0:.0f}s){tag}", flush=True)
        if ep == args.epochs - 1:
            save(args.out.replace(".pt", "_last.pt"), ev)
    print(f"\nbest VAL |log-vol err| + |diff err| = {best:.4f}")


if __name__ == "__main__":
    main()
