"""Run the whole pipeline on one video: frames -> ShapeEstimator -> chord widths -> temporal head -> EF with an interval and ED / ES frames."""
import json
import os

import numpy as np
import torch
from scipy.signal import find_peaks

from geometry import chord_volume, chord_widths
from shape_estimator import checkpoint_offsets, load_shape_estimator
from temporal_detector import TemporalEDESDetector

CATS = [40.0, 50.0]
CAT_NAMES = ["EF < 40 (reduced)", "EF 40-49 (mid-range)", "EF >= 50 (preserved)"]
T_MAX = 350


class Pipeline:
    def __init__(self, ckpt, head, conformal=None, device="cuda"):
        self.dev = torch.device(device)
        self.net = load_shape_estimator(ckpt, self.dev)
        self.offsets = np.array(checkpoint_offsets(ckpt))
        ck = torch.load(head, map_location=self.dev, weights_only=False)
        self.head = TemporalEDESDetector(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"]).to(self.dev)
        self.head.load_state_dict(ck["state"]); self.head.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.cal = json.load(open(conformal)) if conformal else None

    @torch.no_grad()
    def points(self, frames, chunk=64):
        """frames (T, 112, 112) uint8 -> predicted tracing points (T, 42, 2) for every frame."""
        n = len(frames)
        ctx = np.clip(np.arange(n)[:, None] + self.offsets[None, :], 0, n - 1)
        x = torch.from_numpy(frames[ctx]).float().div_(255.0)
        return torch.cat([self.net(x[i:i + chunk].to(self.dev)).cpu() for i in range(0, n, chunk)]).numpy()

    @torch.no_grad()
    def analyse(self, frames):
        """Full analysis of one clip. Returns a dict of numpy arrays and scalars (see keys)."""
        P = self.points(frames)
        d, axis = chord_widths(P)
        feats = np.concatenate([d, axis[:, None]], 1).astype(np.float32)
        n = len(feats)
        s0 = 0
        if n > T_MAX:
            s0 = int(np.clip(n // 2 - T_MAX // 2, 0, n - T_MAX))
        f = feats[s0:s0 + T_MAX]; ln = len(f)
        x = torch.from_numpy((f - self.mu) / self.sd)[None].to(self.dev)
        lg, ef, att = self.head.forward_all(x, torch.tensor([ln]), return_attn=True)
        ef = ef[0].cpu().numpy() * 100 if ef.ndim > 1 else np.array([ef[0].item() * 100])
        q = np.array([ef[0], min(ef[1], ef[2]), max(ef[1], ef[2])]) if len(ef) == 3 else np.array([ef[0], ef[0], ef[0]])
        heat = torch.sigmoid(lg[0]).cpu().numpy()
        res = dict(points=P, chords=feats, volume=chord_volume(P), head_offset=s0, ef_median=float(q[0]), ef_q05=float(q[1]), ef_q95=float(q[2]),
                   heat_ed=heat[:, 0], heat_es=heat[:, 1], attention=att[0].cpu().numpy())
        if self.cal:
            res["ef_lo90"], res["ef_hi90"] = float(q[1] - self.cal["Q90"]), float(q[2] + self.cal["Q90"])
        else:
            res["ef_lo90"], res["ef_hi90"] = float(q[1]), float(q[2])
        res.update(self.beats(res))
        res.update(self.decision(res))
        return res

    @staticmethod
    def beats(res, prom=0.1):
        """ED / ES frames from the heatmap peaks and the formula EF (from the chord volumes) of every beat, paired ED -> next ES."""
        s0 = res["head_offset"]
        def peaks(h):
            p, _ = find_peaks(np.concatenate([[0.0], h, [0.0]]), prominence=prom)
            return (p - 1) + s0
        ed, es = peaks(res["heat_ed"]), peaks(res["heat_es"])
        beats = []
        for i, e in enumerate(ed):
            nxt = ed[i + 1] if i + 1 < len(ed) else 10 ** 9
            c = es[(es > e) & (es < nxt)]
            if len(c):
                v = res["volume"]; s = int(c[0])
                beats.append(dict(ed=int(e), es=s, formula_ef=float(100 * (1 - v[s] / v[e]))))
        return dict(ed_frames=ed.tolist(), es_frames=es.tolist(), beats=beats)

    def decision(self, res):
        """Clinical flags: EF category, whether the 90% interval straddles a category threshold, and whether it is unusually wide."""
        lo, hi, med = res["ef_lo90"], res["ef_hi90"], res["ef_median"]
        straddle = any(lo < t <= hi for t in CATS)
        wide = bool(self.cal and (hi - lo) > self.cal.get("width90_p80", np.inf))
        found = len(res["beats"]) > 0
        if not found:
            action = "REFER: no ED/ES pair detected"
        elif wide:
            action = "REFER: unusually wide uncertainty"
        elif straddle:
            action = "DEFER: interval crosses an EF category threshold"
        else:
            action = "COMMIT"
        return dict(category=CAT_NAMES[int(np.digitize(med, CATS))], straddles_threshold=bool(straddle), wide_interval=wide, action=action)
