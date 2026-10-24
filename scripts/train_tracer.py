"""Train a tracer: FrameTracer (one 13-frame window -> 42 points) or, with --pair, BeatTracer (ED and ES windows -> 84 points).

Targets are the sonographer's tracings of the annotated ED and ES frames (data_cache/records.npz). Losses, ES frames weighted
--es-weight against ED:
  points        L1 on the point coordinates                                                     (--w-pts)
  heatmap       cross-entropy of every point's heatmap against a Gaussian (sigma 1.5 px)        (--w-heat)
  volume        L1 on log(V + 1), V from Simpson's rule on the points                               (--w-vol)
  ED -> ES      L1 on the ED -> ES change of log(V + 1), ~ log(1 - EF)                         (--w-diff)
  widths        L1 on log chord widths, each weighted by its share of V                         (--w-width)
  EF (--pair)   w_b |EF(ED tracing, ES tracing) - EF label| in EF units, w_b ~ (1 / TRAIN frequency of the label's EF bin) ** 0.5, bins <20, 20-29, ..., >=80, mean 1
Augmentation: one similarity transform (rotation +-20 deg, scale 0.8-1.2, shift +-18 px; image and points together, see
tests/test_augmentation.py), the frame rescaled 0.5-2.5x on its canvas, edge occlusion, and scanner-style degradations (gain,
gamma, speckle, noise, blur, resolution loss, shadows) on --scanner-prob of the samples. No flips.
Optimiser: AdamW (lr 3e-4, weight decay 1e-4), cosine decay per step, gradient clipping 5, EMA of the weights (0.999).
The checkpoint kept is the epoch with the lowest VAL log-volume + ED -> ES errors (with --w-ef: the highest VAL formula-EF R2 at
the annotated frames). TEST is never read.

  FrameTracer: python scripts/train_tracer.py --out checkpoints/frametracer.pt
  BeatTracer:  python scripts/train_tracer.py --pair --width 32 --scanner-prob 0 --out checkpoints/beattracer_pre.pt
               python scripts/train_tracer.py --pair --width 32 --scanner-prob 0 --w-ef 40 --epochs 12 --lr 1e-4 --dropout 0 \
                   --init checkpoints/beattracer_pre.pt --out checkpoints/beattracer.pt
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
from frametracer import Tracer, count_params, heat_loss  # noqa: E402
from simpson import chord_widths, contour_to_mask  # noqa: E402

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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32, help="frames per step (half ED, half ES); with --pair, half as many ED+ES samples")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--dropout", type=float, default=0.1, help="spatial dropout inside the residual blocks")
    ap.add_argument("--ema", type=float, default=0.999, help="decay per step of an exponential moving average of the weights (0 = off)")
    ap.add_argument("--es-weight", type=float, default=1.25, help="loss weight of ES frames relative to ED")
    ap.add_argument("--w-pts", type=float, default=1.5)
    ap.add_argument("--w-heat", type=float, default=1.0)
    ap.add_argument("--w-vol", type=float, default=1.06, help="log V")
    ap.add_argument("--w-diff", type=float, default=4.0, help="ED -> ES change of log V")
    ap.add_argument("--w-width", type=float, default=2.0, help="log chord widths weighted by their share of V")
    ap.add_argument("--heat-sigma", type=float, default=1.5, help="width (px) of the Gaussian heatmap target")
    ap.add_argument("--geo-rot", type=float, default=20.0)
    ap.add_argument("--geo-scale", default="0.8,1.2")
    ap.add_argument("--geo-shift", type=float, default=18.0)
    ap.add_argument("--scale-range", default="0.5,2.5", help="lo,hi: each batch is resized by a log-uniform factor in this range (canvas 112 x factor), image and points together")
    ap.add_argument("--scanner-prob", type=float, default=1.0, help="share of training samples that get the scanner-style degradations")
    ap.add_argument("--pair", action="store_true", help="BeatTracer: input = ED window and ES window stacked, output = 84 points (ED 42, then ES 42)")
    ap.add_argument("--w-ef", type=float, default=0.0, help="with --pair: weight of the EF-unit formula loss")
    ap.add_argument("--ef-rarity", type=float, default=0.5, help="exponent of the inverse EF-bin frequency weighting of the EF loss")
    ap.add_argument("--init", default="", help="checkpoint whose weights initialise the network (the EF stage starts from the pretrained BeatTracer)")
    ap.add_argument("--context-bank", default="data_cache/context_frames.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints/frametracer.pt")
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = torch.device(args.device)

    z = np.load(os.path.join(ROOT, "data_cache", "records.npz"), allow_pickle=True)
    cb = np.load(os.path.join(ROOT, args.context_bank))
    ftr = U8Frames(cb["frames"])                                              # uint8 in memory, float in [0, 1] per batch
    offsets = [int(o) for o in cb["offsets"]]
    poly = torch.tensor(z["poly"], dtype=torch.float32)
    split, phase, keys, valid = z["split"], z["phase"], z["key"], z["valid"]
    H = ftr.shape[-1]
    in_ch, n_points = len(offsets) * (2 if args.pair else 1), 84 if args.pair else 42

    net = Tracer(width=args.width, in_ch=in_ch, n_points=n_points, dropout=args.dropout).to(dev)
    if args.init:
        net.load_state_dict(torch.load(os.path.join(ROOT, args.init), map_location=dev, weights_only=False)["state"])
    print(f"{'BeatTracer' if args.pair else 'FrameTracer'}: {count_params(net):,} parameters, {in_ch} input frames, offsets {offsets}", flush=True)

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

    ef_lab = torch.tensor(z["ef"], dtype=torch.float32) / 100.0
    efbin = lambda e: np.clip((np.asarray(e) * 10).astype(int), 1, 8)          # bins <20, 20-30, ..., >=80
    cnt_bin = np.bincount(efbin(ef_lab.numpy()[tr_ed]), minlength=9).astype(float)
    wbin = torch.tensor((1.0 / np.maximum(cnt_bin, 1)) ** args.ef_rarity, dtype=torch.float32)
    wbin = wbin / (wbin[efbin(ef_lab.numpy()[tr_ed])].mean())
    half = args.batch_size // 2
    geo_scale = tuple(float(v) for v in args.geo_scale.split(","))
    lo_, hi_ = (float(v) for v in args.scale_range.split(","))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * max(1, len(tr_ed) // half)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + np.cos(np.pi * min(t, total) / total)))
    ema_net = copy.deepcopy(net).eval() if args.ema > 0 else None
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.out)), exist_ok=True)
    gstep, best = 0, float("inf")

    def save(path, model):
        torch.save({"state": model.state_dict(), "width": args.width, "in_ch": in_ch, "offsets": offsets, "dropout": args.dropout,
                    "n_points": n_points, "pair": args.pair, "config": vars(args)}, os.path.join(ROOT, path))

    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(tr_ed))
        t0 = time.time()
        for i in range(0, len(perm) - half + 1, half):
            pp = perm[i:i + half]
            idx = np.concatenate([tr_ed[pp], tr_es[pp]])
            x, pts = ftr[idx].to(dev), poly[idx].to(dev)
            if args.pair:                                                     # one sample per video: [ED window | ES window], [ED points | ES points]
                x, pts = torch.cat([x[:half], x[half:]], 1), torch.cat([pts[:half], pts[half:]], 1)
            Hc = int(round(H * float(np.exp(np.random.uniform(np.log(lo_), np.log(hi_))))))
            if Hc != H:
                x = F.interpolate(x, size=(Hc, Hc), mode="bilinear", align_corners=False); kk = Hc / H
                pts = torch.stack([((pts[..., 0] - 1 + 0.5) * kk - 0.5) + 1, Hc - ((H - pts[..., 1] + 0.5) * kk - 0.5)], -1)
            x, pts = augment_similarity(x, pts, args.geo_shift * Hc / H, args.geo_rot, geo_scale)
            with torch.no_grad():
                x = augment_occlude(x, bbox_mask(pts, Hc))
                if args.scanner_prob >= 1.0:
                    x = augment_scanner(x)
                elif args.scanner_prob > 0:
                    sel = torch.rand(len(x), device=x.device) < args.scanner_prob
                    if sel.any():
                        x = x.clone(); x[sel] = augment_scanner(x[sel])
            logit = net.heatmaps(x)
            pred = net.decode(logit)
            if args.pair:                                                     # back to the per-frame layout [ED frames; ES frames] used by the losses
                logit, pred, pts = (torch.cat([t[:, :42], t[:, 42:]], 0) for t in (logit, pred, pts))
            nh = len(idx) // 2
            sw = torch.cat([torch.ones(nh, device=dev), torch.full((nh,), args.es_weight, device=dev)])
            sw = sw / sw.mean()
            loss = args.w_pts * ((pred - pts).abs().mean((1, 2)) * sw).mean()
            loss = loss + args.w_heat * (heat_loss(logit, pts, args.heat_sigma) * sw).mean()
            dp, ap_ = chord_widths(pred); dg, ag = chord_widths(pts)
            lvp = torch.log((np.pi / 4 * dp ** 2).sum(1) * ap_ / 20.0 + 1.0); lvg = torch.log((np.pi / 4 * dg ** 2).sum(1) * ag / 20.0 + 1.0)
            loss = loss + args.w_vol * ((lvp - lvg).abs() * sw).mean()
            loss = loss + args.w_diff * (((lvp[nh:] - lvp[:nh]) - (lvg[nh:] - lvg[:nh])).abs().mean())
            if args.w_ef > 0:
                vp_ = (dp ** 2).sum(1) * ap_; efp = 1 - vp_[nh:] / vp_[:nh].clamp(min=1e-6)
                ey = ef_lab[tr_ed[pp]].to(dev); ew = wbin[efbin(ef_lab.numpy()[tr_ed[pp]])].to(dev)
                loss = loss + args.w_ef * (ew * (efp - ey).abs()).mean()
            share = dg ** 2 / (dg ** 2).sum(1, keepdim=True)
            loss = loss + args.w_width * ((share * (torch.log(dp.clamp(min=0.5)) - torch.log(dg.clamp(min=0.5))).abs()).sum(1) * sw).mean()
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
        vv = torch.exp(lv) - 1.0                                              # V (log_volume adds 1)
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
    print(f"\nbest VAL score = {best:.4f}")


if __name__ == "__main__":
    main()
