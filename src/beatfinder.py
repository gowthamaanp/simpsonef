"""BeatFinder: a bidirectional GRU over FrameTracer's per-frame chords that marks every ED and ES frame of the clip.

Input per frame: the 20 chord widths and the long-axis length (21 values, standardised with the TRAIN statistics stored in the
checkpoint). Outputs: per-frame ED and ES heatmaps (the beats are their peaks) and, from an attention-pooled summary, the 5%, 50%
and 95% EF quantiles. The reported EF never comes from these quantiles; they only set the scale of the conformal interval
(pipeline.py), where the disagreement between this learned EF and Simpson's EF widens the interval.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks


class BeatFinder(nn.Module):
    """x (B, T, 21) -> per-frame [ED, ES] logits (B, T, 2) and EF quantiles (B, 3) = [median, q05, q95] as fractions."""

    def __init__(self, in_dim=21, hidden=128, n_layers=2, dropout=0.0, ef_outputs=3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, 2)
        self.attn = nn.Linear(hidden * 2, 1)
        self.ef_outputs = ef_outputs
        self.ef_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, ef_outputs))

    def forward_all(self, x, lengths, return_attn=False):
        """x (B,T,in_dim), lengths (B,) -> ((B,T,2) ED/ES logits, (B, ef_outputs) EF); with return_attn also the (B,T) attention weights."""
        T = x.shape[1]
        valid = torch.arange(T, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        h = self.drop(self.act(self.proj(x)))
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        out = self.drop(out)
        a = self.attn(out).squeeze(-1).masked_fill(~valid, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        ef = self.ef_head((w * out).sum(1))
        ef = ef[:, 0] if self.ef_outputs == 1 else ef
        return (self.head(out), ef, w.squeeze(-1)) if return_attn else (self.head(out), ef)


def load_beatfinder(path, device="cpu"):
    """Load a checkpoint written by train_beatfinder.py. Returns (network in eval mode, input mean, input SD)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = BeatFinder(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mu"], ck["sd"]


def peaks(h, prominence=0.1):
    """Frames of the peaks of a per-frame heatmap (edges zero-padded, so a peak on the first or last frame counts)."""
    p, _ = find_peaks(np.concatenate([[0.0], h, [0.0]]), prominence=prominence)
    return p - 1


def pair_beats(heat_ed, heat_es, prominence=0.1):
    """ED / ES heatmaps (T,) -> beats [(ed, es)]: every ED peak with the first ES peak before the next ED peak (may be empty)."""
    ed, es = peaks(heat_ed, prominence), peaks(heat_es, prominence)
    beats = []
    for i, e in enumerate(ed):
        nxt = ed[i + 1] if i + 1 < len(ed) else 10 ** 9
        c = es[(es > e) & (es < nxt)]
        if len(c):
            beats.append((int(e), int(c[0])))
    return beats


def gaussian_bump_target(n_frames, center, sigma=1.5):
    t = np.arange(n_frames, dtype=np.float32)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def beat_window_mask(n_frames, ed_i, es_i, margin_frac=0.5):
    """1 inside the labelled beat's window, 0 elsewhere (other beats in the clip are unlabelled)."""
    lo, hi = min(ed_i, es_i), max(ed_i, es_i)
    margin = max(3, int(margin_frac * (hi - lo)))
    mask = np.zeros(n_frames, dtype=np.float32)
    mask[max(0, lo - margin):min(n_frames, hi + margin + 1)] = 1.0
    return mask
