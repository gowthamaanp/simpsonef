"""Train BeatFinder on FrameTracer's per-frame chords (data_cache/chords.npz from cache_chords.py).

One bidirectional GRU (2 layers, hidden 128, dropout 0.2) over the clip. Losses:
  ED / ES   squared error between the sigmoid of the per-frame ED and ES logits and Gaussian bumps (sigma 1.5 frames) at the
            annotated frames, only inside a window around the single annotated beat (other beats in the clip are unlabelled)
  EF        pinball loss of the 5%, 50% and 95% EF quantiles, read out from an attention-pooled summary of the clip
Each step sees a random temporal crop that keeps the annotated beat, with Gaussian noise (SD 0.05) on the standardised chords.
AdamW (lr 1e-3, weight decay 1e-3), cosine decay over 16 epochs, batch 32, gradient clipping 5. The epoch with the best VAL R2 of
the median EF output is kept. TEST is never read.

Usage: python scripts/train_beatfinder.py --chords data_cache/chords.npz --out checkpoints/beatfinder.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beatfinder import BeatFinder, beat_window_mask, gaussian_bump_target  # noqa: E402

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
    """Padded (n, T, 21) inputs, lengths, annotated ED/ES frames and EF labels for the clips `idx` (clips longer than T are
    cut to T frames starting 40 frames before the annotated beat; training only, the pipeline runs on the whole clip)."""
    F = z["feats"].shape[1]
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


def seed_everything(seed):
    """Fix every random source (Python, NumPy, PyTorch CPU / GPU). GPU kernels stay in their fast (not bit-exact) mode: reruns
    match up to normal run-to-run variation."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def targets(L, ed, es, sigma=1.5):
    """Gaussian ED / ES heatmap targets and the supervision window mask, each (n, T)."""
    n = len(L)
    ET, ST, M = (np.zeros((n, T), np.float32) for _ in range(3))
    for b in range(n):
        ET[b] = gaussian_bump_target(T, ed[b], sigma); ST[b] = gaussian_bump_target(T, es[b], sigma)
        M[b] = beat_window_mask(T, ed[b], es[b])
        ET[b, L[b]:] = 0; ST[b, L[b]:] = 0; M[b, L[b]:] = 0
    return ET, ST, M


def crop(x, L, ed, es, et, st, m, noise):
    """Random temporal crop that keeps the annotated beat (+3 frames each side), plus Gaussian feature noise."""
    xo, eo, so, mo = torch.zeros_like(x), torch.zeros_like(et), torch.zeros_like(st), torch.zeros_like(m)
    Lo = torch.zeros_like(L)
    for k in range(len(x)):
        n = int(L[k]); lo = max(0, min(int(ed[k]), int(es[k])) - 3); hi = min(n, max(int(ed[k]), int(es[k])) + 4)
        s0 = np.random.randint(0, lo + 1); e0 = np.random.randint(hi, n + 1)
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
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--ef-weight", type=float, default=1.0)
    ap.add_argument("--chords", default="data_cache/chords.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/beatfinder.pt")
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = torch.device("cuda")
    z = np.load(os.path.join(ROOT, args.chords), allow_pickle=True)
    z = {k: z[k] for k in z.files}
    ix = {s: np.where(z["splits"] == s)[0] for s in ("TRAIN", "VAL")}

    Xtr, Ltr, edtr, estr, ytr = build(z, ix["TRAIN"]); Xva, Lva, edva, esva, yva = build(z, ix["VAL"])
    valid = (np.arange(T)[None, :] < Ltr[:, None])
    mu = Xtr[valid].mean(0); sd = Xtr[valid].std(0).clip(min=1e-3)
    Xtr = (Xtr - mu) / sd * valid[..., None]; Xva = (Xva - mu) / sd * (np.arange(T)[None, :] < Lva[:, None])[..., None]
    ETtr, STtr, Mtr = targets(Ltr, edtr, estr)
    Xtr, Ltr_t, Xva, Lva_t = (torch.from_numpy(a) for a in (Xtr, Ltr, Xva, Lva))
    ETtr, STtr, Mtr = (torch.from_numpy(a) for a in (ETtr, STtr, Mtr))
    edtr_t, estr_t, ytr_t = torch.from_numpy(edtr), torch.from_numpy(estr), torch.from_numpy(ytr) / 100.0

    net = BeatFinder(in_dim=Xtr.shape[-1], hidden=args.hidden, n_layers=args.layers, dropout=args.dropout, ef_outputs=3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"BeatFinder: {sum(p.numel() for p in net.parameters()):,} params; TRAIN {len(Xtr)} VAL {len(Xva)}", flush=True)
    best = -1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(Xtr)); t0 = time.time(); run = nb = 0
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = perm[i:i + args.batch_size]
            x, L, et, st, m = crop(Xtr[b], Ltr_t[b], edtr_t[b], estr_t[b], ETtr[b], STtr[b], Mtr[b], args.noise)
            lg, ef = net.forward_all(x.to(dev), L)
            m, et, st = m.to(dev), et.to(dev), st.to(dev)
            l_det = (((torch.sigmoid(lg[..., 0]) - et) ** 2 * m).sum() + ((torch.sigmoid(lg[..., 1]) - st) ** 2 * m).sum()) / m.sum()
            yb = ytr_t[b].to(dev)
            pin = lambda q, t: torch.maximum(q * (yb - t), (q - 1) * (yb - t)).mean()
            l_ef = pin(0.5, ef[:, 0]) + pin(0.05, ef[:, 1]) + pin(0.95, ef[:, 2])
            loss = l_det + args.ef_weight * l_ef
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            run += float(loss); nb += 1
        sched.step()
        ef, a_ed, a_es = evaluate(net, Xva, Lva_t, edva, esva, dev); v = r2(ef, yva); tag = ""
        if v > best:
            best = v; tag = "  *"
            os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
            torch.save({"state": net.state_dict(), "hidden": args.hidden, "layers": args.layers, "mu": mu, "sd": sd,
                        "ef_outputs": 3, "ef_loss": "quantile", "in_dim": Xtr.shape[-1], "config": vars(args)}, os.path.join(ROOT, args.out))
        print(f"[ep {ep:3d}] loss={run / nb:.4f}  VAL EF R2={v:.3f} MAE={np.abs(ef - yva).mean():.2f}  aFD ED={a_ed.mean():.2f} ES={a_es.mean():.2f}  "
              f"<=2f ED={100 * (a_ed <= 2).mean():.0f}% ES={100 * (a_es <= 2).mean():.0f}%  ({time.time() - t0:.0f}s){tag}", flush=True)
    print(f"\nbest VAL EF R2 (BeatFinder's own EF output) = {best:.3f}")


if __name__ == "__main__":
    main()
