"""Formula EF from the joint ED + ES (pair) tracing model on EchoNet TEST and CAMUS test, all videos, no annotated frame.

ED / ES frames come from the frame detector (EchoNet: detected beats, run on the chord features of the single-frame model;
CAMUS: the half-sequence head on the released half-sequence). For every detected ED / ES pair the two 13-frame windows are
stacked and the pair model draws both tracings; EF = formula on the chords, median over beats (EchoNet) or biplane with one
disk height (CAMUS). The annotated-frame EchoNet result is printed as a diagnostic.

Usage: python experiments/eval_pair.py --ckpt checkpoints/scale_space_pair.pt
"""
import argparse
import os
import sys

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import build_context_bank as bank  # noqa: E402
import camus_zeroshot as cz  # noqa: E402
import train_heads as th  # noqa: E402
from geometry import chord_widths, contour_to_mask  # noqa: E402
from PIL import Image  # noqa: E402
from inference import Pipeline  # noqa: E402
from shape_estimator import checkpoint_offsets, load_shape_estimator  # noqa: E402
from temporal_detector import TemporalEDESDetector  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def r2(p, y):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def windows(frames, t, offsets):
    n = len(frames)
    return np.stack([frames[int(np.clip(t + o, 0, n - 1))] for o in offsets]).astype(np.float32) / 255.0


TTA = [(1.0, 0.0)]


def similarity(x, k, phi):
    """Deterministic content map p -> c0 + k R(phi) (p - c0) on the 112 canvas (decoder coordinates), as in training."""
    N, _, H, W = x.shape; dev, dt = x.device, x.dtype
    c0 = torch.tensor([(W + 1) / 2.0, (H + 1) / 2.0], device=dev, dtype=dt)
    cols, rows = torch.arange(W, device=dev, dtype=dt), torch.arange(H, device=dev, dtype=dt)
    P = torch.stack([(cols + 1)[None, :].expand(H, W), (H - rows)[:, None].expand(H, W)], -1).reshape(-1, 2)
    q = P - c0; cs, sn = np.cos(phi), np.sin(phi)
    px = (cs * q[:, 0] + sn * q[:, 1]) / k + c0[0]; py = (-sn * q[:, 0] + cs * q[:, 1]) / k + c0[1]
    grid = torch.stack([(px - 0.5) / W * 2 - 1, ((H - py) + 0.5) / H * 2 - 1], -1).reshape(1, H, W, 2).expand(N, -1, -1, -1)
    return torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="zeros"), c0


def predict(net, x):
    """Points averaged over the TTA transforms, each mapped back to the original frame."""
    acc = 0
    for k, deg in TTA:
        phi = np.deg2rad(deg)
        if k == 1.0 and deg == 0.0:
            acc = acc + net(x); continue
        xt, c0 = similarity(x, k, phi); pt = net(xt)
        cs, sn = np.cos(phi), np.sin(phi); q = (pt - c0) / k
        acc = acc + torch.stack([cs * q[..., 0] + sn * q[..., 1], -sn * q[..., 0] + cs * q[..., 1]], -1) + c0
    return acc / len(TTA)


SIMEX = {"levels": [], "draws": 2, "targets": [0.0, -0.5, -1.0, -1.5, -2.0, -3.0]}


def add_noise(x, level, seed):
    """Known extra ultrasound-like noise at strength `level`: spatially correlated multiplicative speckle + additive Gaussian."""
    if level == 0:
        return x
    g = torch.Generator(device=x.device).manual_seed(seed)
    sp = torch.randn(x.shape, generator=g, device=x.device)
    sp = torch.nn.functional.avg_pool2d(sp.flatten(0, 1)[:, None], 3, 1, 1).view_as(x) * 3.0     # correlated, unit-ish SD
    return (x * (1 + 0.12 * level * sp) + 0.03 * level * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1)


@torch.no_grad()
def pair_points(net, frames, pairs, offsets, dev, level=0, seed=0):
    """frames (n, 112, 112) uint8, pairs [(ed, es)] -> points (k, 2, 42, 2): ED and ES tracings (TTA-averaged)."""
    x = torch.from_numpy(np.stack([np.concatenate([windows(frames, e, offsets), windows(frames, s, offsets)]) for e, s in pairs])).to(dev)
    x = add_noise(x, level, seed)
    P = torch.cat([predict(net, x[i:i + 32]) for i in range(0, len(x), 32)]).cpu().numpy()
    return np.stack([P[:, :42], P[:, 42:]], 1)


@torch.no_grad()
def pair_chords(net, frames, pairs, offsets, dev):
    """frames (n, 112, 112) uint8, pairs [(ed, es)] -> chord widths (k, 2, 20) and axis (k, 2) for ED and ES."""
    x = torch.from_numpy(np.stack([np.concatenate([windows(frames, e, offsets), windows(frames, s, offsets)]) for e, s in pairs])).to(dev)
    P = torch.cat([predict(net, x[i:i + 32]) for i in range(0, len(x), 32)]).cpu().numpy()
    d, a = chord_widths(np.stack([P[:, :42], P[:, 42:]], 1))
    return np.asarray(d), np.asarray(a)


def report(nm, p, y):
    print(f"  {nm:58s} R2 {r2(p, y):.3f}  MAE {np.abs(p - y).mean():.2f}  r {np.corrcoef(p, y)[0, 1]:.3f}  bias {(p - y).mean():+.2f}")


FEATS, DET = ["data_cache/features.npz"], ["checkpoints/ef_head.pt"]
SPLIT, DUMP = ["TEST"], [""]
PTS, DUMP_PTS = [], [""]
TSHIFT = [0]


def echonet(net, offsets, dev):
    z = np.load(os.path.join(ROOT, FEATS[0]), allow_pickle=True); z = {k: z[k] for k in z.files}
    ix = np.where(z["splits"] == SPLIT[0])[0]
    ck = torch.load(os.path.join(ROOT, DET[0]), map_location=dev, weights_only=False)
    det = TemporalEDESDetector(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"]).to(dev)
    det.load_state_dict(ck["state"]); det.eval()
    X, L, ed_a, es_a, y = th.build(z, ix)
    Xn = torch.from_numpy((X - ck["mu"]) / ck["sd"]) * (torch.arange(th.T)[None, :, None] < torch.from_numpy(L)[:, None, None])
    with torch.no_grad():
        heat = np.concatenate([torch.sigmoid(det.forward_all(Xn[i:i + 64].to(dev), torch.from_numpy(L[i:i + 64]))[0]).cpu().numpy() for i in range(0, len(Xn), 64)])
    pk = lambda h: find_peaks(np.concatenate([[0.0], h, [0.0]]), prominence=0.1)[0] - 1
    fin, ann, fin_mean, nbt, bsd, keys = [], [], [], [], [], []
    dice = {"ED": [], "ES": []}
    simex_out, simex_slope = {}, []
    for k in range(len(ix)):
        n = L[k]; s0 = int(z["ed"][ix[k]] - ed_a[k])
        ed, es = pk(heat[k, :n, 0]), pk(heat[k, :n, 1]); beats = []
        for i, e in enumerate(ed):
            nxt = ed[i + 1] if i + 1 < len(ed) else 10 ** 9; c = es[(es > e) & (es < nxt)]
            if len(c):
                beats.append((int(e) + s0, int(c[0]) + s0))
        if not beats:
            beats = [(int(heat[k, :n, 0].argmax()) + s0, int(heat[k, :n, 1].argmax()) + s0)]
        frames = bank.read_video(z["keys"][ix[k]]); nf = len(frames)
        sh = [(i, j) for i in TSHIFT for j in TSHIFT]                         # temporal averaging: every beat traced at (ED+i, ES+j)
        prs = [(int(np.clip(e + i, 0, nf - 1)), int(np.clip(s_ + j, 0, nf - 1))) for e, s_ in beats for i, j in sh]
        if SIMEX["levels"]:                                                   # simulation-extrapolation of the noise-induced shrinkage
            lv = []
            for L_ in SIMEX["levels"]:
                reps = [0] if L_ == 0 else list(range(SIMEX["draws"]))
                e_l = []
                for r_ in reps:
                    Pn = pair_points(net, frames, prs, offsets, dev, level=L_, seed=1000 * k + 10 * r_ + int(10 * L_))
                    dn, an = chord_widths(Pn); vn = (np.asarray(dn) ** 2).sum(-1) * np.asarray(an); e_l.append(100 * (1 - vn[:, 1] / vn[:, 0]))
                lv.append(np.mean(e_l, 0))
            lv = np.array(lv); kk = np.array(SIMEX["levels"], float)
            b_ = ((kk - kk.mean())[:, None] * (lv - lv.mean(0))).sum(0) / ((kk - kk.mean()) ** 2).sum(); a_ = lv.mean(0) - b_ * kk.mean()
            for t_ in SIMEX["targets"]:
                simex_out.setdefault(t_, []).append(np.median((a_ + b_ * t_).reshape(len(beats), len(sh)).mean(1)))
            simex_slope.append(np.median(b_))
        PP = pair_points(net, frames, prs + [(int(z["ed"][ix[k]]), int(z["es"][ix[k]]))], offsets, dev)
        if DUMP_PTS[0]:
            PTS.append((z['keys'][ix[k]], np.asarray(PP[-1]).copy()))
        d, a = chord_widths(PP); d, a = np.asarray(d), np.asarray(a)
        v = (d ** 2).sum(-1) * a; efs = 100 * (1 - v[:, 1] / v[:, 0])
        for ph, pt in (("ED", PP[-1, 0]), ("ES", PP[-1, 1])):                  # Dice at the annotated frames vs the released masks
            mp = os.path.join(ROOT, "data", "annotation", "Masks", ph, f"{z['keys'][ix[k]]}.png")
            if os.path.exists(mp):
                g = np.array(Image.open(mp)) > 0; m = contour_to_mask(pt, 112); dice[ph].append(2 * (m & g).sum() / (m.sum() + g.sum() + 1e-9))
        be = efs[:-1].reshape(len(beats), len(sh)).mean(1)
        fin.append(np.median(be)); fin_mean.append(be.mean()); ann.append(efs[-1])
        nbt.append(len(be)); bsd.append(float(np.median(np.abs(be - np.median(be)))) * 1.4826 if len(be) > 1 else np.nan); keys.append(z["keys"][ix[k]])
    fin, ann = np.array(fin), np.array(ann)
    if DUMP_PTS[0]:                                                          # tracings at the annotated frames (ED, ES)
        np.savez(DUMP_PTS[0], keys=np.array([k for k, _ in PTS]), pts=np.stack([p for _, p in PTS]))
    print(f"EchoNet {SPLIT[0]}, all {len(y)} videos")
    if DUMP[0]:
        import pandas as pd
        pd.DataFrame(dict(key=keys, y=y, ef=fin, ef_annotated=ann, n_beats=nbt, beat_spread=bsd)).to_csv(os.path.join(ROOT, DUMP[0]), index=False)
    report("pair model, detected ED/ES, median over beats (final)", fin, y)
    report("pair model, annotated frames (diagnostic)", ann, y)
    print(f"  Dice at the annotated frames vs released masks: ED {np.mean(dice['ED']):.4f} (n={len(dice['ED'])})  ES {np.mean(dice['ES']):.4f} (n={len(dice['ES'])})")
    report("pair model, detected ED/ES, MEAN over beats", np.array(fin_mean), y)
    if simex_out:
        print(f"  SIMEX: levels {SIMEX['levels']}, {SIMEX['draws']} draws; median dEF/dlevel {np.median(simex_slope):+.2f} (negative = noise lowers EF)")
        for t_, v_ in simex_out.items():
            report(f"SIMEX extrapolated to noise level {t_:+.1f}", np.array(v_), y)
    err = fin - y; se = err ** 2
    print(f"  slope of prediction on label {np.polyfit(y, fin, 1)[0]:.2f}")
    for lo, hi in ((0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)):
        m = (y >= lo) & (y < hi)
        print(f"  EF {lo:3d}-{hi:3d}: n {m.sum():4d}  bias {err[m].mean():+5.2f}  MAE {np.abs(err[m]).mean():5.2f}  share of SSE {100 * se[m].sum() / se.sum():4.1f}%")


def camus(net, offsets, dev):
    pipe = Pipeline(os.path.join(ROOT, "checkpoints/shape_estimator.pt"), os.path.join(ROOT, "checkpoints/ef_head.pt"), None)
    nd = os.path.join(cz.CAMUS, "database_nifti")
    pats = [l.strip() for l in open(os.path.join(cz.CAMUS, "database_split", os.environ.get("CAMUS_SUBSET", "subgroup_testing") + ".txt")) if l.strip()]
    out, y, fb = {"biplane": [], "4CH": [], "2CH": []}, [], 0
    for p in pats:
        dd = {}
        for view in ("4CH", "2CH"):
            im = nib.load(os.path.join(nd, p, f"{p}_{view}_half_sequence.nii.gz")); vol = im.get_fdata().astype(np.float32)
            scale = max(vol.shape[0], vol.shape[1]) / cz.S * float(np.mean(im.header.get_zooms()[:2]))
            frames = np.stack([cz.pad_to_square_resize(vol[:, :, t].T, cz.S, cv2.INTER_LINEAR) for t in range(vol.shape[2])]).clip(0, 255).astype(np.uint8)
            res = pipe.analyse(frames)
            if res["beats"]:
                e, s = res["beats"][0]["ed"], res["beats"][0]["es"]
            else:
                fb += 1; v = np.asarray(res["volume"]); e, s = int(v.argmax()), int(v.argmin())
            d, _ = pair_chords(net, frames, [(e, s)], offsets, dev); dd[view] = d[0] * scale
        (e4, s4), (e2, s2) = dd["4CH"], dd["2CH"]
        out["biplane"].append(100 * (1 - (s4 * s2).sum() / (e4 * e2).sum()))
        out["4CH"].append(100 * (1 - (s4 ** 2).sum() / (e4 ** 2).sum())); out["2CH"].append(100 * (1 - (s2 ** 2).sum() / (e2 ** 2).sum()))
        y.append(float(cz.info(os.path.join(nd, p, "Info_4CH.cfg"))["EF"]))
    y = np.array(y)
    print(f"\nCAMUS zero-shot, all {len(y)} test patients (detector on the released half-sequence; {fb} views fell back to volume extremes)")
    report("pair model, biplane, CAMUS protocol (final)", np.array(out["biplane"]), y)
    report("pair model, 4CH only", np.array(out["4CH"]), y)
    report("pair model, 2CH only", np.array(out["2CH"]), y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/scale_space_pair.pt")
    ap.add_argument("--tta", action="store_true", help="average the tracing over 5 scales x 3 rotations of the input (one model)")
    ap.add_argument("--features", default="data_cache/features.npz", help="per-frame chord cache the frame detector reads")
    ap.add_argument("--det", default="checkpoints/ef_head.pt", help="frame-detector head checkpoint")
    ap.add_argument("--tshift", default="0", help="frame shifts around each detected ED and ES, e.g. -1,0,1 (all combinations are traced and their EFs averaged)")
    ap.add_argument("--tta-set", default="standard", choices=["standard", "large"], help="large: 7 scales x 5 rotations")
    ap.add_argument("--split", default="TEST", choices=["VAL", "TEST"])
    ap.add_argument("--dump", default="", help="write per-video EF, label, number of beats and beat-to-beat spread (robust SD) to this CSV")
    ap.add_argument("--dump-points", default="", help="save the tracings at the annotated ED/ES frames to this .npz")
    ap.add_argument("--simex", default="", help="noise levels for simulation-extrapolation, e.g. 0,1,2 (reports EF extrapolated to levels <= 0)")
    ap.add_argument("--only", choices=["echonet", "camus", "both"], default="both"); a = ap.parse_args()
    FEATS[0], DET[0], SPLIT[0], DUMP[0], DUMP_PTS[0] = a.features, a.det, a.split, a.dump, a.dump_points
    if a.simex:
        SIMEX["levels"] = [float(v) for v in a.simex.split(",")]
    TSHIFT[:] = [int(v) for v in a.tshift.split(",")]
    ap_tta = a.tta_set
    if a.tta:
        TTA[:] = [(k, d) for k in (0.85, 0.92, 1.0, 1.08, 1.17) for d in (-8.0, 0.0, 8.0)]
    if ap_tta == "large":
        TTA[:] = [(k, d) for k in (0.80, 0.87, 0.93, 1.0, 1.07, 1.15, 1.25) for d in (-10.0, -5.0, 0.0, 5.0, 10.0)]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_shape_estimator(os.path.join(ROOT, a.ckpt), dev); offsets = checkpoint_offsets(os.path.join(ROOT, a.ckpt))
    if a.only in ("echonet", "both"):
        echonet(net, offsets, dev)
    if a.only in ("camus", "both"):
        camus(net, offsets, dev)


if __name__ == "__main__":
    main()
