"""Train the temporal head on the per-frame chord widths.

One bidirectional GRU over the clip (input: the 20 chord widths and the axis length of every frame, from cache_features.py)
regresses EF (median and 5% / 95% quantiles) from an attention-pooled summary and produces per-frame ED / ES heatmaps (the
frame detector). ED / ES heatmaps are supervised only inside a window around the single annotated beat. The epoch with the
best VAL EF R2 is kept; TEST is only read with --eval-split TEST.

Usage: python experiments/train_heads.py --features data_cache/features.npz          (train -> checkpoints/ef_head.pt)
       python experiments/train_heads.py --features data_cache/features.npz --test --eval-split VAL
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from temporal_detector import TemporalEDESDetector, beat_window_mask, gaussian_bump_target  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
T = 350


def nearest_peak(sig, annot, prom=0.1):
    """Frame of the heatmap peak nearest to `annot` (arg-max if there is no peak). Edges are zero-padded."""
    p, _ = find_peaks(np.concatenate([[0.0], sig, [0.0]]), prominence=prom)
    p = p - 1
    return int(np.argmax(sig)) if len(p) == 0 else int(p[np.argmin(np.abs(p - annot))])


def r2(p, t):
    return 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def build(z, idx):
    F = z["feats"].shape[1]
    """Padded (n, T, 21) inputs, lengths, annotated ED/ES frames and EF labels for the clips `idx`."""
    off = z["offsets"]
    n = len(idx)
    X = np.zeros((n, T, F), np.float32); L = np.zeros(n, np.int64)
    ed, es = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for b, i in enumerate(idx):
        a, e = off[i], off[i + 1]
        ln = e - a
        s0 = 0
        if ln > T:
            s0 = int(np.clip(min(z["ed"][i], z["es"][i]) - 40, 0, ln - T)); ln = T
        X[b, :ln, :F] = z["feats"][a + s0:a + s0 + ln]
        L[b] = ln; ed[b] = z["ed"][i] - s0; es[b] = z["es"][i] - s0
    return X, L, ed, es, z["ef"][idx].astype(np.float32)


def half_cycles(X, L, ed, es, y):
    """Clips cut exactly to the annotated ED -> ES frames (videos with ED before ES), the annotated frames become the first and last."""
    keep = np.where(ed < es)[0]; Xh = np.zeros_like(X[keep]); Lh = np.zeros(len(keep), np.int64)
    for j, k in enumerate(keep):
        ln = es[k] - ed[k] + 1; Xh[j, :ln] = X[k, ed[k]:es[k] + 1]; Lh[j] = ln
    return Xh, Lh, np.zeros(len(keep), np.int64), Lh - 1, y[keep]


def seed_everything(seed):
    """Fix every random source (Python, NumPy, PyTorch CPU / GPU). GPU kernels stay in their fast (not bit-exact) mode: reruns
    match up to normal run-to-run variation."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def merge(z, z2):
    """Append a second feature cache (same layout) to the first."""
    z2 = {k: z2[k] for k in z2.files}; out = {}
    for k in ("keys", "splits", "ed", "es", "ef", "feats"):
        out[k] = np.concatenate([z[k], z2[k]]) if z[k].dtype.kind == "U" else np.concatenate([z[k], z2[k].astype(z[k].dtype)])
    out["offsets"] = np.concatenate([z["offsets"], z2["offsets"][1:] + z["offsets"][-1]])
    return out


def per_patient(keys, ef, y):
    """Mean of the views' head EF per patient (keys <patient>_<view>) and the patient label."""
    pats = np.array([k.rsplit("_", 1)[0] for k in keys]); u = np.unique(pats)
    return np.array([ef[pats == q].mean() for q in u]), np.array([y[pats == q][0] for q in u])


def targets(L, ed, es, sigma=1.5):
    """Gaussian ED / ES heatmap targets and the supervision window mask, each (n, T)."""
    n = len(L)
    ET, ST, M = (np.zeros((n, T), np.float32) for _ in range(3))
    for b in range(n):
        ET[b] = gaussian_bump_target(T, ed[b], sigma); ST[b] = gaussian_bump_target(T, es[b], sigma)
        M[b] = beat_window_mask(T, ed[b], es[b])
        ET[b, L[b]:] = 0; ST[b, L[b]:] = 0; M[b, L[b]:] = 0
    return ET, ST, M


def crop(x, L, ed, es, et, st, m, noise, half_prob=0.0):
    """Random temporal crop that keeps the annotated beat (+3 frames each side), plus feature noise.

    With probability half_prob (videos with ED before ES only) the clip is cut to the annotated ED -> ES half-cycle, +-1 frame,
    which is the form in which CAMUS releases its sequences."""
    xo, eo, so, mo = torch.zeros_like(x), torch.zeros_like(et), torch.zeros_like(st), torch.zeros_like(m)
    Lo = torch.zeros_like(L)
    for k in range(len(x)):
        n = int(L[k]); lo = max(0, min(int(ed[k]), int(es[k])) - 3); hi = min(n, max(int(ed[k]), int(es[k])) + 4)
        s0 = np.random.randint(0, lo + 1); e0 = np.random.randint(hi, n + 1)
        if half_prob > 0 and int(ed[k]) < int(es[k]) and np.random.rand() < half_prob:
            s0 = max(0, int(ed[k]) - np.random.randint(0, 2)); e0 = min(n, int(es[k]) + 1 + np.random.randint(0, 2))
        ln = e0 - s0
        xo[k, :ln] = x[k, s0:e0]; eo[k, :ln] = et[k, s0:e0]; so[k, :ln] = st[k, s0:e0]; mo[k, :ln] = m[k, s0:e0]; Lo[k] = ln
    valid = (torch.arange(x.shape[1])[None, :] < Lo[:, None]).float().unsqueeze(-1)
    return xo + noise * torch.randn_like(xo) * valid, Lo, eo, so, mo


@torch.no_grad()
def evaluate(net, X, L, ed, es, dev):
    """EF (median output, in EF points) and absolute ED / ES frame errors against the annotations."""
    net.eval(); ef, heat = [], []
    for i in range(0, len(X), 64):
        lg, e = net.forward_all(X[i:i + 64].to(dev), L[i:i + 64])
        e = e if e.ndim == 1 else e[:, 0]
        ef.append(e.cpu().numpy() * 100); heat.append(torch.sigmoid(lg).cpu().numpy())
    ef, heat = np.concatenate(ef), np.concatenate(heat)
    a_ed = np.array([abs(nearest_peak(heat[k, :L[k], 0], int(ed[k])) - ed[k]) for k in range(len(X))])
    a_es = np.array([abs(nearest_peak(heat[k, :L[k], 1], int(es[k])) - es[k]) for k in range(len(X))])
    return ef, a_ed, a_es


def report(name, ef, y, a_ed, a_es):
    rng = np.random.RandomState(0); n = len(y)
    bs = [r2(ef[j], y[j]) for j in (rng.randint(0, n, n) for _ in range(1000))]
    print(f"{name} (n={n}): EF R2={r2(ef, y):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]  "
          f"MAE={np.abs(ef - y).mean():.2f}  r={np.corrcoef(ef, y)[0, 1]:.3f}  EF<=40 acc={100 * np.mean((ef <= 40) == (y <= 40)):.1f}%")
    for nm, a in (("ED", a_ed), ("ES", a_es)):
        print(f"   {nm}: mean aFD={a.mean():.2f} median={np.median(a):.0f}  "
              + "  ".join(f"<={t}f={100 * (a <= t).mean():.0f}%" for t in (1, 2, 3, 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--ef-weight", type=float, default=1.0)
    ap.add_argument("--features", default="data_cache/features.npz")
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--ef-loss", choices=["smoothl1", "laplace", "quantile"], default="quantile",
                    help="smoothl1: point estimate; laplace: mean + per-case scale; quantile: median + 5%%/95%% quantiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-features", default="", help="second feature cache (cache_camus_features.py) merged in: CAMUS_TRAIN joins TRAIN, CAMUS_VAL joins model selection")
    ap.add_argument("--extra-repeat", type=int, default=3, help="times each CAMUS_TRAIN clip is repeated per epoch")
    ap.add_argument("--camus-only", action="store_true", help="train on CAMUS_TRAIN only and select on CAMUS_VAL only (a CAMUS-specific head)")
    ap.add_argument("--init-head", default="", help="start from this head checkpoint (weights and input normalisation)")
    ap.add_argument("--half-prob", type=float, default=0.0, help="share of training clips cut to the annotated ED -> ES half-cycle")
    ap.add_argument("--half-eval", action="store_true", help="with --test: evaluate on the ED -> ES half-cycles (videos with ED before ES)")
    ap.add_argument("--out", default="checkpoints/ef_head.pt")
    ap.add_argument("--test", action="store_true", help="evaluate the saved checkpoint on --eval-split, no training")
    ap.add_argument("--eval-split", choices=["VAL", "TEST", "CAMUS_VAL", "CAMUS_TEST"], default="VAL")
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = torch.device("cuda")
    z = np.load(os.path.join(ROOT, args.features), allow_pickle=True)
    z = {k: z[k] for k in z.files}
    if args.extra_features:
        z = merge(z, np.load(os.path.join(ROOT, args.extra_features), allow_pickle=True))
    ix = {s: np.where(z["splits"] == s)[0] for s in ("TRAIN", "VAL", "TEST", "CAMUS_TRAIN", "CAMUS_VAL", "CAMUS_TEST")}

    if args.test:
        ck = torch.load(os.path.join(ROOT, args.out), map_location=dev, weights_only=False)
        net = TemporalEDESDetector(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0,
                                   ef_outputs=ck["ef_outputs"]).to(dev)
        net.load_state_dict(ck["state"])
        X, L, ed, es, y = build(z, ix[args.eval_split])
        if args.half_eval:
            X, L, ed, es, y = half_cycles(X, L, ed, es, y)
        X = (X - ck["mu"]) / ck["sd"]
        X = torch.from_numpy(X) * (torch.arange(T)[None, :, None] < torch.from_numpy(L)[:, None, None])
        ef, a_ed, a_es = evaluate(net, X, torch.from_numpy(L), ed, es, dev)
        report(args.eval_split + " (per clip)", ef, y, a_ed, a_es)
        if args.eval_split.startswith("CAMUS"):
            pe, py = per_patient(z["keys"][ix[args.eval_split]], ef, y)
            print(f"{args.eval_split} per patient (mean of 4CH and 2CH head EF vs biplane label, n={len(py)}): R2={r2(pe, py):.3f}  "
                  f"MAE={np.abs(pe - py).mean():.2f}  r={np.corrcoef(pe, py)[0, 1]:.3f}  bias={(pe - py).mean():+.2f}")
        return

    tr_ix = ix["CAMUS_TRAIN"] if args.camus_only else np.concatenate([ix["TRAIN"]] + [ix["CAMUS_TRAIN"]] * args.extra_repeat)
    Xtr, Ltr, edtr, estr, ytr = build(z, tr_ix); Xva, Lva, edva, esva, yva = build(z, ix["VAL"])
    cva = len(ix["CAMUS_VAL"]) > 0
    if cva:
        Xcv, Lcv, edcv, escv, ycv = build(z, ix["CAMUS_VAL"])
    valid = (np.arange(T)[None, :] < Ltr[:, None])
    mu = Xtr[valid].mean(0); sd = Xtr[valid].std(0).clip(min=1e-3)
    ck0 = torch.load(os.path.join(ROOT, args.init_head), map_location="cpu", weights_only=False) if args.init_head else None
    if ck0 is not None:
        mu, sd = ck0["mu"], ck0["sd"]
    Xtr = (Xtr - mu) / sd * valid[..., None]; Xva = (Xva - mu) / sd * (np.arange(T)[None, :] < Lva[:, None])[..., None]
    if cva:
        Xcv = torch.from_numpy((Xcv - mu) / sd * (np.arange(T)[None, :] < Lcv[:, None])[..., None]); Lcv_t = torch.from_numpy(Lcv)
    ETtr, STtr, Mtr = targets(Ltr, edtr, estr)
    Xtr, Ltr_t, Xva, Lva_t = (torch.from_numpy(a) for a in (Xtr, Ltr, Xva, Lva))
    ETtr, STtr, Mtr = (torch.from_numpy(a) for a in (ETtr, STtr, Mtr))
    edtr_t, estr_t, ytr_t = torch.from_numpy(edtr), torch.from_numpy(estr), torch.from_numpy(ytr) / 100.0

    net = TemporalEDESDetector(in_dim=Xtr.shape[-1], hidden=args.hidden, n_layers=args.layers, dropout=args.dropout,
                               ef_outputs={"smoothl1": 1, "laplace": 2, "quantile": 3}[args.ef_loss]).to(dev)
    if ck0 is not None:
        net.load_state_dict(ck0["state"])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"head: {sum(p.numel() for p in net.parameters()):,} params; TRAIN {len(Xtr)} VAL {len(Xva)}", flush=True)
    best = -1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(Xtr)); t0 = time.time(); run = nb = 0
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = perm[i:i + args.batch_size]
            x, L, et, st, m = crop(Xtr[b], Ltr_t[b], edtr_t[b], estr_t[b], ETtr[b], STtr[b], Mtr[b], args.noise, args.half_prob)
            lg, ef = net.forward_all(x.to(dev), L)
            m, et, st = m.to(dev), et.to(dev), st.to(dev)
            l_det = (((torch.sigmoid(lg[..., 0]) - et) ** 2 * m).sum() + ((torch.sigmoid(lg[..., 1]) - st) ** 2 * m).sum()) / m.sum()
            yb = ytr_t[b].to(dev)
            if args.ef_loss == "smoothl1":
                l_ef = F.smooth_l1_loss(ef, yb, beta=0.05)
            elif args.ef_loss == "laplace":
                lb = ef[:, 1].clamp(-6.0, 3.0)
                l_ef = ((yb - ef[:, 0]).abs() * torch.exp(-lb) + lb).mean() * 0.1
            else:
                pin = lambda q, t: torch.maximum(q * (yb - t), (q - 1) * (yb - t)).mean()
                l_ef = pin(0.5, ef[:, 0]) + pin(0.05, ef[:, 1]) + pin(0.95, ef[:, 2])
            loss = l_det + args.ef_weight * l_ef
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            run += float(loss); nb += 1
        sched.step()
        ef, a_ed, a_es = evaluate(net, Xva, Lva_t, edva, esva, dev); v = r2(ef, yva); tag = ""
        cmsg = ""
        if cva:
            efc, ced, ces = evaluate(net, Xcv, Lcv_t, edcv, escv, dev); pe, py = per_patient(z["keys"][ix["CAMUS_VAL"]], efc, ycv)
            vc = r2(pe, py); cmsg = f"  CAMUS VAL R2={vc:.3f} aFD ED={ced.mean():.2f} ES={ces.mean():.2f}"; v = vc if args.camus_only else 0.5 * (v + vc)
        if v > best:
            best = v; tag = "  *"
            os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
            torch.save({"state": net.state_dict(), "hidden": args.hidden, "layers": args.layers, "mu": mu, "sd": sd,
                        "ef_outputs": net.ef_outputs, "ef_loss": args.ef_loss, "in_dim": Xtr.shape[-1], "config": vars(args)}, os.path.join(ROOT, args.out))
            res = dict(r2=v, r=float(np.corrcoef(ef, yva)[0, 1]), mae=float(np.abs(ef - yva).mean()), aed=float(a_ed.mean()), aes=float(a_es.mean()),
                       f2ed=float((a_ed <= 2).mean()), f2es=float((a_es <= 2).mean()), ep=ep)
        print(f"[ep {ep:3d}] loss={run / nb:.4f}  VAL EF R2={v:.3f} MAE={np.abs(ef - yva).mean():.2f}  aFD ED={a_ed.mean():.2f} ES={a_es.mean():.2f}  "
              f"<=2f ED={100 * (a_ed <= 2).mean():.0f}% ES={100 * (a_es <= 2).mean():.0f}%{cmsg}  ({time.time() - t0:.0f}s){tag}", flush=True)
    print(f"\nbest VAL EF R2 = {best:.3f}")
    print("RESULT r2=%.4f r=%.4f mae=%.3f aFD_ED=%.2f aFD_ES=%.2f f2_ED=%.3f f2_ES=%.3f epoch=%d" % (res["r2"], res["r"], res["mae"], res["aed"], res["aes"], res["f2ed"], res["f2es"], res["ep"]))


if __name__ == "__main__":
    main()
