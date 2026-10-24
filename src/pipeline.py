"""SimpsonEF: video -> FrameTracer -> BeatFinder -> BeatTracer -> Simpson's rule -> EF with a 90% conformal interval.

  1. FrameTracer traces every frame (13-frame window, offsets -24..24 step 4, clamped at the clip ends) -> per-frame chords.
  2. BeatFinder reads the chord sequence and marks ED and ES; every ED peak is paired with the next ES peak (one beat each).
     If no pair is found, the frames of largest and smallest FrameTracer volume are the one beat.
  3. BeatTracer redraws each beat's ED and ES tracings from the two 13-frame windows stacked, averaged over 15 copies of the
     input (scales 0.85-1.17 x rotations -8/0/8 degrees, mapped back to the original frame).
  4. Single plane (EchoNet-Dynamic convention): Simpson's rule per beat, the median over beats is the EF.
     Interval: EF +- Q * s, s = (q95 - q05) / 2 + |EF - EF_BeatFinder|, Q calibrated on EchoNet VAL (conformal.py).
  4'. Biplane (CAMUS convention): the 4- and 2-chamber views are run separately through steps 1-2, FrameTracer's ED and ES
     tracings of each view give the chord widths, and EF = 1 - sum a_ES b_ES / sum a_ED b_ED (one disk height for both phases).
     FrameTracer, not BeatTracer, is the biplane tracer: the choice was made on the CAMUS training patients. No interval is
     reported (the calibration is for single-plane EchoNet EF).

Frames are grayscale uint8 (T, 112, 112); read_video() zero-pads any other frame size to a square and resizes it.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from beatfinder import load_beatfinder, pair_beats
from frametracer import load_tracer
from simpson import chord_widths, ef_biplane, ef_single_plane, volume

S = 112
COPIES = [(k, d) for k in (0.85, 0.92, 1.0, 1.08, 1.17) for d in (-8.0, 0.0, 8.0)]
CATS = [40.0, 50.0]
CAT_NAMES = ["EF < 40 (reduced)", "EF 40-49 (mildly reduced)", "EF >= 50 (preserved)"]
ROOT = os.path.join(os.path.dirname(__file__), "..")


def read_video(path, size=S):
    """Video file -> (grayscale frames (T, size, size) uint8, frames per second). Non-square frames are zero-padded to a square."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        if g.shape != (size, size):
            h, w = g.shape; side = max(h, w)
            c = np.zeros((side, side), np.uint8); c[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
            g = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        out.append(g)
    cap.release()
    if not out:
        raise ValueError(f"no frames could be read from {path}")
    return np.stack(out), float(fps)


def windows(frames, t, offsets):
    """The frames at t + offsets (clamped to the clip) as one (len(offsets), H, W) float array in [0, 1]."""
    n = len(frames)
    return frames[np.clip(t + np.asarray(offsets), 0, n - 1)].astype(np.float32) / 255.0


def similarity(x, k, phi):
    """Resample frames so that content p moves to c0 + k R(phi) (p - c0) (decoder coordinates, c0 the image centre)."""
    N, _, H, W = x.shape; dev, dt = x.device, x.dtype
    c0 = torch.tensor([(W + 1) / 2.0, (H + 1) / 2.0], device=dev, dtype=dt)
    cols, rows = torch.arange(W, device=dev, dtype=dt), torch.arange(H, device=dev, dtype=dt)
    P = torch.stack([(cols + 1)[None, :].expand(H, W), (H - rows)[:, None].expand(H, W)], -1).reshape(-1, 2)
    q = P - c0; cs, sn = np.cos(phi), np.sin(phi)
    px = (cs * q[:, 0] + sn * q[:, 1]) / k + c0[0]; py = (-sn * q[:, 0] + cs * q[:, 1]) / k + c0[1]
    grid = torch.stack([(px - 0.5) / W * 2 - 1, ((H - py) + 0.5) / H * 2 - 1], -1).reshape(1, H, W, 2).expand(N, -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros"), c0


def trace_copies(net, x, copies):
    """Points averaged over the input copies, each copy's points mapped back to the original frame."""
    acc = 0
    for k, deg in copies:
        if k == 1.0 and deg == 0.0:
            acc = acc + net(x); continue
        phi = np.deg2rad(deg)
        xt, c0 = similarity(x, k, phi); q = (net(xt) - c0) / k
        cs, sn = np.cos(phi), np.sin(phi)
        acc = acc + torch.stack([cs * q[..., 0] + sn * q[..., 1], -sn * q[..., 0] + cs * q[..., 1]], -1) + c0
    return acc / len(copies)


class SimpsonEF:
    def __init__(self, frametracer=os.path.join(ROOT, "checkpoints/frametracer.pt"), beatfinder=os.path.join(ROOT, "checkpoints/beatfinder.pt"),
                 beattracer=os.path.join(ROOT, "checkpoints/beattracer.pt"), conformal=os.path.join(ROOT, "checkpoints/conformal.json"),
                 device=None, copies=True, prominence=0.1):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.frame_net, self.frame_offsets = load_tracer(frametracer, self.dev)
        self.beat_net, self.beat_offsets = load_tracer(beattracer, self.dev)
        self.finder, self.mu, self.sd = load_beatfinder(beatfinder, self.dev)
        self.cal = json.load(open(conformal)) if conformal and os.path.exists(conformal) else None
        self.copies = COPIES if copies else [(1.0, 0.0)]
        self.prominence = prominence

    # ---- steps 1-3 -------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def frame_tracings(self, frames, chunk=64):
        """FrameTracer on every frame: (T, 112, 112) uint8 -> (T, 42, 2)."""
        n = len(frames)
        ctx = np.clip(np.arange(n)[:, None] + np.asarray(self.frame_offsets)[None, :], 0, n - 1)
        x = torch.from_numpy(frames[ctx]).float().div_(255.0)
        return torch.cat([self.frame_net(x[i:i + chunk].to(self.dev)).cpu() for i in range(0, n, chunk)]).numpy()

    @torch.no_grad()
    def find_beats(self, P):
        """BeatFinder on FrameTracer's chords: (T, 42, 2) -> ED / ES heatmaps (T,), beats [(ed, es)], EF quantiles [median, q05, q95]."""
        d, axis = chord_widths(P)
        f = (np.concatenate([d, axis[:, None]], 1).astype(np.float32) - self.mu) / self.sd
        lg, ef = self.finder.forward_all(torch.from_numpy(f.astype(np.float32))[None].to(self.dev), torch.tensor([len(f)]))
        heat = torch.sigmoid(lg[0]).cpu().numpy(); q = ef[0].cpu().numpy() * 100
        beats = pair_beats(heat[:, 0], heat[:, 1], self.prominence)
        if not beats:                                                     # no ED -> ES pair: the extremes of the volume curve
            v = volume(P); beats = [(int(v.argmax()), int(v.argmin()))]
        return heat[:, 0], heat[:, 1], beats, np.array([q[0], min(q[1], q[2]), max(q[1], q[2])])

    @torch.no_grad()
    def beat_tracings(self, frames, beats):
        """BeatTracer: beats [(ed, es)] -> (k, 2, 42, 2) ED and ES tracings, averaged over the input copies."""
        x = torch.from_numpy(np.stack([np.concatenate([windows(frames, e, self.beat_offsets), windows(frames, s, self.beat_offsets)])
                                       for e, s in beats])).to(self.dev)
        P = torch.cat([trace_copies(self.beat_net, x[i:i + 32], self.copies) for i in range(0, len(x), 32)]).cpu().numpy()
        return np.stack([P[:, :42], P[:, 42:]], 1)

    # ---- single plane (EchoNet-Dynamic) ----------------------------------------------------------------------------------
    def analyse(self, frames):
        """One apical 4-chamber clip -> dict with the per-frame tracings, beats, per-beat tracings and EF, and the interval."""
        P = self.frame_tracings(frames)
        h_ed, h_es, beats, q = self.find_beats(P)
        BT = self.beat_tracings(frames, beats)
        beat_ef = np.array([ef_single_plane(b[0], b[1]) for b in BT])
        ef = float(np.median(beat_ef))
        res = dict(mode="single plane", frame_points=P, volume=volume(P), heat_ed=h_ed, heat_es=h_es, beats=beats, beat_points=BT,
                   beat_ef=beat_ef, ef=ef, ef_beatfinder=float(q[0]), q05=float(q[1]), q95=float(q[2]))
        if self.cal:
            s = (q[2] - q[1]) / 2 + abs(ef - q[0]); Q = self.cal["Q90"]
            res.update(lo90=float(ef - Q * s), hi90=float(ef + Q * s))
            straddle = any(res["lo90"] < t <= res["hi90"] for t in CATS)
            wide = res["hi90"] - res["lo90"] > self.cal.get("width90_p80", np.inf)
            res["category"] = CAT_NAMES[int(np.digitize(ef, CATS))]
            res["action"] = ("REFER: unusually wide interval" if wide else "DEFER: interval crosses an EF category threshold" if straddle
                             else "COMMIT")
        return res

    # ---- biplane (CAMUS) -------------------------------------------------------------------------------------------------
    def analyse_biplane(self, frames_4ch, frames_2ch, tracer="frame"):
        """Apical 4- and 2-chamber clips of the same heart -> biplane EF with one disk height (CAMUS label definition).

        Each view: FrameTracer on every frame, BeatFinder beats. The i-th beat of one view is paired with the i-th beat of the
        other (a released CAMUS half-sequence has exactly one) and the median over pairs is the EF. tracer="beat" uses BeatTracer's
        tracings instead of FrameTracer's (not the paper's choice)."""
        views = {}
        for name, fr in (("4CH", frames_4ch), ("2CH", frames_2ch)):
            P = self.frame_tracings(fr)
            h_ed, h_es, beats, _ = self.find_beats(P)
            BT = self.beat_tracings(fr, beats) if tracer == "beat" else np.stack([np.stack([P[e], P[s]]) for e, s in beats])
            views[name] = dict(frame_points=P, volume=volume(P), heat_ed=h_ed, heat_es=h_es, beats=beats, beat_points=BT,
                               beat_ef=np.array([ef_single_plane(b[0], b[1]) for b in BT]))
        n = min(len(views["4CH"]["beats"]), len(views["2CH"]["beats"]))
        a, b = views["4CH"]["beat_points"], views["2CH"]["beat_points"]
        bi = np.array([ef_biplane(a[i, 0], a[i, 1], b[i, 0], b[i, 1]) for i in range(n)])
        return dict(mode="biplane", views=views, beat_ef=bi, ef=float(np.median(bi)))
