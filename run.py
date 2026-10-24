"""Run SimpsonEF on your own echo video(s) and write the whole video with the tracings overlaid.

  Single plane (EchoNet-Dynamic convention), one apical 4-chamber clip:
      python run.py a4c.mp4 --out a4c_traced.mp4
  Biplane (CAMUS convention: 4- and 2-chamber views, one disk height for ED and ES):
      python run.py a4c.mp4 --biplane a2c.mp4 --out biplane_traced.mp4

--out may end in .mp4 or .gif. Prints the EF, the beats and (single plane) the 90% interval; --json writes the numbers.
The networks expect a cropped apical view with the sector filling the frame, as in EchoNet-Dynamic (112 x 112 after resizing).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from overlay import render, write  # noqa: E402
from pipeline import SimpsonEF, read_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="apical 4-chamber clip")
    ap.add_argument("--biplane", metavar="A2C", help="apical 2-chamber clip of the same heart: biplane EF with the CAMUS convention")
    ap.add_argument("--out", default="", help="annotated video (.mp4 or .gif)")
    ap.add_argument("--json", default="", help="write EF, interval and beats to this file")
    ap.add_argument("--scale", type=int, default=0, help="upscaling of the 112 x 112 frames in the output video (default 4 for MP4, 3 for GIF)")
    ap.add_argument("--no-copies", action="store_true", help="skip the 15-copy test-time averaging of BeatTracer (faster)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    model = SimpsonEF(device=a.device, copies=not a.no_copies)
    frames, fps = read_video(a.video)
    if a.biplane:
        frames2, _ = read_video(a.biplane)
        res = model.analyse_biplane(frames, frames2)
        print(f"biplane Simpson EF {res['ef']:.1f}%")
        for v in ("4CH", "2CH"):
            print(f"  {v}: beats (ED, ES) {res['views'][v]['beats']}")
        out = dict(mode="biplane", ef=res["ef"], beat_ef=res["beat_ef"].tolist(), beats={v: res["views"][v]["beats"] for v in ("4CH", "2CH")})
    else:
        frames2 = None
        res = model.analyse(frames)
        print(f"Simpson EF {res['ef']:.1f}%" + (f"  (90% interval {res['lo90']:.1f}-{res['hi90']:.1f}; {res['category']}; {res['action']})" if "lo90" in res else ""))
        for (e, s), ef in zip(res["beats"], res["beat_ef"]):
            print(f"  beat ED frame {e + 1:4d}  ES frame {s + 1:4d}  EF {ef:5.1f}%")
        out = {k: res[k] for k in ("mode", "ef", "lo90", "hi90", "category", "action", "ef_beatfinder", "q05", "q95") if k in res}
        out.update(beats=res["beats"], beat_ef=res["beat_ef"].tolist())
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else v)
    if a.out:
        k = a.scale or (3 if a.out.lower().endswith(".gif") else 4)
        write(render(res, frames, k, frames2), a.out, fps)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
