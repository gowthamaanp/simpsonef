"""Export the three SimpsonEF networks to ONNX for the browser demo (docs/), and the constants the demo needs.

  frametracer.onnx  frames (N, 13, 112, 112) in [0, 1]          -> points (N, 42, 2)
  beatfinder.onnx   chords (1, T, 21), raw pixels                -> ED / ES logits (1, T, 2), EF quantiles (1, 3) as fractions
  beattracer.onnx   ED window + ES window (N, 26, 112, 112)       -> points (N, 84, 2)
  config.json       frame offsets of both tracers, conformal Q and how the interval is formed

Points are in the decoder convention of the tracings (x = column + 1, y = 112 - row). BeatFinder takes the raw chord features: the
TRAIN normalisation is folded into the graph, and the GRU runs on the whole (unpadded) sequence, which equals the packed version
used in training for a single video.

Usage: python tools/export_onnx.py --out docs/models
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from beatfinder import BeatFinder  # noqa: E402
from frametracer import Tracer, load_tracer  # noqa: E402


class FixedSizeTracer(nn.Module):
    """Scale-space tracer for a fixed 112 x 112 input: 'area' downsampling by an exact factor 2 / 4 equals average pooling, which
    ONNX supports; the level logits are resized back to 112 and fused by log-sum-exp as in training."""

    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        net, acc = self.net, None
        for s in net.levels:
            xs = x if s == 1.0 else nn.functional.avg_pool2d(x, int(round(1 / s)))
            o = net.single_scale(xs)
            o = o if s == 1.0 else nn.functional.interpolate(o, size=(112, 112), mode="bilinear", align_corners=False)
            acc = o if acc is None else torch.maximum(acc, o) + torch.log1p(torch.exp(-(acc - o).abs()))   # logaddexp
        return net.decode(acc)


class Finder(nn.Module):
    def __init__(self, ck):
        super().__init__()
        self.m = BeatFinder(in_dim=ck["in_dim"], hidden=ck["hidden"], n_layers=ck["layers"], dropout=0.0, ef_outputs=ck["ef_outputs"])
        self.m.load_state_dict(ck["state"]); self.m.eval()
        self.register_buffer("mu", torch.as_tensor(ck["mu"], dtype=torch.float32)); self.register_buffer("sd", torch.as_tensor(ck["sd"], dtype=torch.float32))

    def forward(self, x):
        m = self.m
        h = m.act(m.proj((x - self.mu) / self.sd))
        out, _ = m.gru(h)
        w = torch.softmax(m.attn(out).squeeze(-1), dim=1).unsqueeze(-1)
        return m.head(out), m.ef_head((w * out).sum(1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/models"); a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    cpu = torch.device("cpu")
    offsets = {}
    for name in ("frametracer", "beattracer"):
        net, offsets[name] = load_tracer(os.path.join(ROOT, "checkpoints", f"{name}.pt"), cpu)
        c = net.in_ch
        torch.onnx.export(FixedSizeTracer(net), torch.rand(2, c, 112, 112), os.path.join(out, f"{name}.onnx"), input_names=["frames"],
                          output_names=["points"], dynamic_axes={"frames": {0: "n"}, "points": {0: "n"}}, opset_version=17)
        print(name, "exported, input channels", c)
    ck = torch.load(os.path.join(ROOT, "checkpoints/beatfinder.pt"), map_location="cpu", weights_only=False)
    torch.onnx.export(Finder(ck).eval(), torch.rand(1, 150, 21) * 30, os.path.join(out, "beatfinder.onnx"), input_names=["chords"],
                      output_names=["heat", "ef"], dynamic_axes={"chords": {1: "t"}, "heat": {1: "t"}}, opset_version=17)
    print("beatfinder exported")
    conf = json.load(open(os.path.join(ROOT, "checkpoints/conformal.json")))
    cfg = {"frametracer_offsets": offsets["frametracer"], "beattracer_offsets": offsets["beattracer"],
           "fps": 50, "size": 112, "peak_prominence": 0.1, "conformal_Q90": conf["Q90"],
           "interval": "EF +- Q90 * ((q95 - q05) / 2 + |EF_formula - EF_beatfinder|), EF in points",
           "fallback": "no ED -> ES pair: frames of largest and smallest FrameTracer volume"}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print("config written:", cfg["conformal_Q90"])


if __name__ == "__main__":
    main()
