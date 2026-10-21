"""ShapeEstimator: a fully convolutional heatmap network that finds the 42 tracing points of the left ventricle.

A small encoder-decoder (strides 1-8, dilated bottleneck for context, skip connections) predicts one full-resolution
heatmap per tracing point from a window of neighbouring frames (the frames are stacked as input channels). Each point is
the soft-argmax over a (2r+1)^2 window around its heatmap peak, so every prediction depends only on local evidence: there
is no global pooling, fully connected layer or squeeze-excitation anywhere.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_POINTS = 42


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with instance norm and GELU, and a residual shortcut."""

    def __init__(self, cin, cout, stride=1, dilation=1, drop=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, dilation, dilation=dilation, bias=False)
        self.n1 = nn.InstanceNorm2d(cout, affine=True)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, dilation, dilation=dilation, bias=False)
        self.n2 = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.GELU()
        self.short = (nn.Identity() if cin == cout and stride == 1 else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.InstanceNorm2d(cout, affine=True)))

    def forward(self, x):
        return self.act(self.n2(self.c2(self.drop(self.act(self.n1(self.c1(x)))))) + self.short(x))


class ShapeEstimator(nn.Module):
    """Frames (N, in_ch, 112, 112) -> 42 tracing points (N, 42, 2) in decoder coordinates (x = column + 1, y = 112 - row)."""

    def __init__(self, width=32, in_ch=13, radius=3, dropout=0.0, n_points=N_POINTS):
        super().__init__()
        w = width
        self.width, self.in_ch, self.radius, self.dropout = width, in_ch, radius, dropout
        B = lambda *a, **k: ConvBlock(*a, drop=dropout, **k)
        self.e0 = nn.Sequential(nn.Conv2d(in_ch, w, 3, 1, 1, bias=False), nn.InstanceNorm2d(w, affine=True), nn.GELU(), B(w, w))
        self.e1 = B(w, int(1.5 * w), stride=2)                          # 56 x 56
        self.e2 = B(int(1.5 * w), 3 * w, stride=2)                      # 28 x 28
        self.e3 = B(3 * w, 6 * w, stride=2)                             # 14 x 14
        self.mid = nn.Sequential(B(6 * w, 6 * w, dilation=2), B(6 * w, 6 * w, dilation=4))
        self.d2 = B(9 * w, 3 * w)
        self.d1 = B(int(4.5 * w), int(1.5 * w))
        self.d0 = B(int(2.5 * w), w)
        self.head = nn.Conv2d(w, n_points, 1)                          # 42, or 84 for a joint ED + ES (pair) model
        k = torch.arange(-radius, radius + 1)
        self.register_buffer("win", torch.stack(torch.meshgrid(k, k, indexing="ij"), -1).view(-1, 2))   # (window, 2) = (d row, d col)

    def heatmaps(self, x):
        """(N, in_ch, H, W) -> heatmap logits (N, 42, H, W)."""
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        h0 = self.e0(x); h1 = self.e1(h0); h2 = self.e2(h1); h3 = self.mid(self.e3(h2))
        g2 = self.d2(torch.cat([up(h3, h2), h2], 1))
        g1 = self.d1(torch.cat([up(g2, h1), h1], 1))
        g0 = self.d0(torch.cat([up(g1, h0), h0], 1))
        return self.head(g0)

    def decode(self, logit):
        """Heatmap logits (N, 42, H, W) -> points (N, 42, 2): soft-argmax over a window around each peak."""
        N, K, H, W = logit.shape
        flat = logit.flatten(2)
        pk = flat.argmax(-1)
        r0, c0 = pk // W, pk % W
        rr = (r0[..., None] + self.win[:, 0]).clamp(0, H - 1)
        cc = (c0[..., None] + self.win[:, 1]).clamp(0, W - 1)
        p = torch.softmax(flat.gather(-1, rr * W + cc), -1)
        row = (p * rr.float()).sum(-1)
        col = (p * cc.float()).sum(-1)
        return torch.stack([col + 1.0, H - row], -1)

    def forward(self, x):
        return self.decode(self.heatmaps(x))


class ScaleSpaceEstimator(ShapeEstimator):
    """Scale-space version: ONE shared-weight network is applied to the frame at dyadic scales (1, 1/2, 1/4), each level's heatmap
    logits are resampled to the input grid and the levels are fused per point by log-sum-exp (a soft max over scale).

    Resizing the input by 2 shifts the levels by one, so the fused heatmap moves with the anatomy: the tracing is covariant to
    translation (fully convolutional) and to dyadic scaling (shared weights across levels, strides tied to the scale step), up to
    the end levels. Instance norm keeps the contrast of each level comparable.
    """

    def __init__(self, *a, levels=(1.0, 0.5, 0.25), **k):
        super().__init__(*a, **k)
        self.levels = tuple(levels)

    def heatmaps(self, x):
        H, W = x.shape[-2:]
        acc = None
        for s in self.levels:
            xs = x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="area")
            o = super().heatmaps(xs)
            o = o if s == 1.0 else F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)
            acc = o if acc is None else torch.logaddexp(acc, o)
        return acc


class RegressionEstimator(nn.Module):
    """Ablation baseline: the same encoder, but the 84 point coordinates are regressed directly from a globally pooled feature
    (the 'predict the tracing coordinates' formulation), instead of one heatmap per point."""

    def __init__(self, width=24, in_ch=13, dropout=0.0, hidden=512):
        super().__init__()
        w = width
        self.width, self.in_ch, self.dropout = width, in_ch, dropout
        B = lambda *a, **k: ConvBlock(*a, drop=dropout, **k)
        self.e0 = nn.Sequential(nn.Conv2d(in_ch, w, 3, 1, 1, bias=False), nn.InstanceNorm2d(w, affine=True), nn.GELU(), B(w, w))
        self.e1, self.e2, self.e3 = B(w, int(1.5 * w), stride=2), B(int(1.5 * w), 3 * w, stride=2), B(3 * w, 6 * w, stride=2)
        self.e4 = B(6 * w, 12 * w, stride=2)                                        # 7 x 7
        self.fc = nn.Sequential(nn.Linear(12 * w, hidden), nn.GELU(), nn.Linear(hidden, 2 * N_POINTS))
        nn.init.zeros_(self.fc[-1].weight)
        with torch.no_grad():                                                     # start from a typical LV tracing in the middle of the frame
            ang = torch.linspace(0, 2 * np.pi, N_POINTS)
            self.fc[-1].bias.copy_(torch.stack([56.0 + 18 * torch.cos(ang), 56.0 + 28 * torch.sin(ang)], -1).flatten())

    def forward(self, x):
        h = self.e4(self.e3(self.e2(self.e1(self.e0(x)))))
        return self.fc(h.mean((2, 3))).view(len(x), N_POINTS, 2)


def heat_loss(logit, points, sigma=1.5):
    """Cross-entropy between each heatmap (softmax over the image) and a Gaussian centred on its target point."""
    N, K, H, W = logit.shape
    rows = torch.arange(H, device=logit.device, dtype=logit.dtype)
    cols = torch.arange(W, device=logit.device, dtype=logit.dtype)
    tr, tc = H - points[..., 1], points[..., 0] - 1.0
    g = torch.exp(-((rows.view(1, 1, H, 1) - tr[..., None, None]) ** 2 +
                    (cols.view(1, 1, 1, W) - tc[..., None, None]) ** 2) / (2 * sigma ** 2))
    g = g.flatten(2)
    g = g / g.sum(-1, keepdim=True).clamp(min=1e-12)
    return -(g * torch.log_softmax(logit.flatten(2), -1)).sum(-1).mean(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_shape_estimator(path, device):
    """Load a checkpoint written by train_shape_estimator.py. Returns the network in eval mode."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cls = {"regress": RegressionEstimator, "scalespace": ScaleSpaceEstimator}.get(ck.get("arch"), ShapeEstimator)
    kw = {"n_points": ck.get("n_points", N_POINTS)} if cls is not RegressionEstimator else {}
    net = cls(width=ck["width"], in_ch=ck["in_ch"], dropout=ck.get("dropout", 0.0), **kw).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net


def checkpoint_offsets(path):
    """Frame offsets (relative to the annotated frame) that the checkpoint expects as input channels."""
    return list(torch.load(path, map_location="cpu", weights_only=False)["offsets"])
