"""The tracer network shared by FrameTracer and BeatTracer: a fully convolutional heatmap network that draws the LV volume tracing.

A small encoder-decoder (strides 1-8, dilated bottleneck for context, skip connections, instance norm, GELU) predicts one
full-resolution heatmap per tracing point from a window of neighbouring frames stacked as input channels. The same weights run
at scales 1, 1/2 and 1/4 and the per-point logits are fused by log-sum-exp. Each point is the soft-argmax over the 7x7 window
around its heatmap peak, so every prediction depends only on local evidence: there is no global pooling or fully connected layer.

  FrameTracer  width 24, 13 frames (offsets -24..24, step 4)                 -> 42 points of one frame       (1.42M parameters)
  BeatTracer   width 32, the 13-frame ED window and ES window stacked (26)   -> 84 points: ED 42, then ES 42  (2.53M parameters)

Points are in decoder coordinates (x = column + 1, y = 112 - row), ordered [A1, L1..L20, R20..R1, A2] (see simpson.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_POINTS = 42
OFFSETS = list(range(-24, 25, 4))


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


class Tracer(nn.Module):
    """Frames (N, in_ch, H, W) -> tracing points (N, n_points, 2). FrameTracer: in_ch 13, 42 points; BeatTracer: in_ch 26, 84 points."""

    def __init__(self, width=24, in_ch=13, n_points=N_POINTS, radius=3, dropout=0.0, levels=(1.0, 0.5, 0.25)):
        super().__init__()
        w = width
        self.width, self.in_ch, self.n_points, self.radius, self.dropout, self.levels = width, in_ch, n_points, radius, dropout, tuple(levels)
        B = lambda *a, **k: ConvBlock(*a, drop=dropout, **k)
        self.e0 = nn.Sequential(nn.Conv2d(in_ch, w, 3, 1, 1, bias=False), nn.InstanceNorm2d(w, affine=True), nn.GELU(), B(w, w))
        self.e1 = B(w, int(1.5 * w), stride=2)                          # 1/2
        self.e2 = B(int(1.5 * w), 3 * w, stride=2)                      # 1/4
        self.e3 = B(3 * w, 6 * w, stride=2)                             # 1/8
        self.mid = nn.Sequential(B(6 * w, 6 * w, dilation=2), B(6 * w, 6 * w, dilation=4))
        self.d2 = B(9 * w, 3 * w)
        self.d1 = B(int(4.5 * w), int(1.5 * w))
        self.d0 = B(int(2.5 * w), w)
        self.head = nn.Conv2d(w, n_points, 1)
        k = torch.arange(-radius, radius + 1)
        self.register_buffer("win", torch.stack(torch.meshgrid(k, k, indexing="ij"), -1).view(-1, 2))   # (window, 2) = (d row, d col)

    def single_scale(self, x):
        """(N, in_ch, H, W) -> heatmap logits (N, n_points, H, W) at one scale."""
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        h0 = self.e0(x); h1 = self.e1(h0); h2 = self.e2(h1); h3 = self.mid(self.e3(h2))
        g2 = self.d2(torch.cat([up(h3, h2), h2], 1))
        g1 = self.d1(torch.cat([up(g2, h1), h1], 1))
        g0 = self.d0(torch.cat([up(g1, h0), h0], 1))
        return self.head(g0)

    def heatmaps(self, x):
        """Scale-space heatmap logits: the shared network at every level, resampled to the input grid, fused by log-sum-exp."""
        H, W = x.shape[-2:]
        acc = None
        for s in self.levels:
            o = self.single_scale(x if s == 1.0 else F.interpolate(x, scale_factor=s, mode="area"))
            o = o if s == 1.0 else F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)
            acc = o if acc is None else torch.logaddexp(acc, o)
        return acc

    def decode(self, logit):
        """Heatmap logits (N, K, H, W) -> points (N, K, 2): soft-argmax over the (2r+1)^2 window around each peak."""
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


def heat_loss(logit, points, sigma=1.5):
    """Cross-entropy between each heatmap (softmax over the image) and a Gaussian of width sigma (px) centred on its target point."""
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


def load_tracer(path, device="cpu"):
    """Load a FrameTracer or BeatTracer checkpoint written by train_tracer.py. Returns (network in eval mode, frame offsets)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Tracer(width=ck["width"], in_ch=ck["in_ch"], n_points=ck.get("n_points", N_POINTS), dropout=ck.get("dropout", 0.0)).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, [int(o) for o in ck.get("offsets", OFFSETS)]
