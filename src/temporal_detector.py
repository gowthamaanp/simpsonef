"""Temporal head: BiGRU over the per-frame chord widths -> per-frame ED/ES heatmaps (frame detector) and clip-level EF."""
import numpy as np
import torch
import torch.nn as nn


class TemporalEDESDetector(nn.Module):
    """Per-frame [ED, ES] logits plus an EF read-out pooled with learned attention over the frames.

    ef_outputs: 1 = EF point estimate; 2 = [mean, log Laplace scale]; 3 = [median, 5% quantile, 95% quantile].
    """

    def __init__(self, in_dim=21, hidden=64, n_layers=1, dropout=0.0, ef_outputs=1):
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
        """x (B,T,in_dim), lengths (B,) -> ((B,T,2) ED/ES logits, EF as (B,) or (B, ef_outputs)); with return_attn also the (B,T) attention weights."""
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


def gaussian_bump_target(n_frames, center, sigma=2.0):
    t = np.arange(n_frames, dtype=np.float32)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def beat_window_mask(n_frames, ed_i, es_i, margin_frac=0.5):
    """1 inside the labelled beat's window, 0 elsewhere (other beats in the clip are unlabelled)."""
    lo, hi = min(ed_i, es_i), max(ed_i, es_i)
    margin = max(3, int(margin_frac * (hi - lo)))
    mask = np.zeros(n_frames, dtype=np.float32)
    mask[max(0, lo - margin):min(n_frames, hi + margin + 1)] = 1.0
    return mask
