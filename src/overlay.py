"""Render a SimpsonEF result over the whole video: every frame with its tracing (mask, outline, long axis and the 20 chords), the
volume curve with the beats marked, and the EF. On the ED and ES frames of each beat the tracing shown is BeatTracer's (the one
Simpson's rule uses); on the other frames it is FrameTracer's. Writes MP4 (H.264 if available, else MPEG-4) or GIF."""
import cv2
import numpy as np

from simpson import to_pixels

CYAN, YELLOW, RED, GREEN, ORANGE, WHITE = (255, 220, 0), (0, 230, 255), (60, 60, 255), (90, 220, 90), (0, 150, 255), (240, 240, 240)


def draw_tracing(img, P, k, color=CYAN, alpha=0.28):
    """Mask (translucent), outline, chords and long axis of one (42, 2) tracing on a BGR image upscaled by k."""
    q = (to_pixels(P) + 0.5) * k
    poly = np.round(q[1:41]).astype(np.int32)
    fill = img.copy(); cv2.fillPoly(fill, [poly], color)
    img[:] = cv2.addWeighted(fill, alpha, img, 1 - alpha, 0)
    for i in range(1, 21):
        cv2.line(img, tuple(np.round(q[i]).astype(int)), tuple(np.round(q[41 - i]).astype(int)), YELLOW, 1, cv2.LINE_AA)
    cv2.line(img, tuple(np.round(q[0]).astype(int)), tuple(np.round(q[41]).astype(int)), RED, 1, cv2.LINE_AA)
    cv2.polylines(img, [poly], True, color, 2 if k >= 3 else 1, cv2.LINE_AA)


def curve_panel(vol, t, beats, width, height=70):
    """Volume curve strip: FrameTracer's per-frame volume, ED (green) and ES (orange) of every beat, the current frame (white)."""
    pan = np.full((height, width, 3), 18, np.uint8)
    n = len(vol); lo, hi = float(vol.min()), float(vol.max()) + 1e-6
    x = lambda i: int(round(6 + (width - 12) * i / max(1, n - 1)))
    y = lambda v: int(round(height - 8 - (height - 16) * (v - lo) / (hi - lo)))
    pts = np.array([[x(i), y(v)] for i, v in enumerate(vol)], np.int32)
    cv2.polylines(pan, [pts], False, CYAN, 1, cv2.LINE_AA)
    for e, s in beats:
        cv2.circle(pan, (x(e), y(vol[e])), 3, GREEN, -1, cv2.LINE_AA); cv2.circle(pan, (x(s), y(vol[s])), 3, ORANGE, -1, cv2.LINE_AA)
    cv2.line(pan, (x(t), 2), (x(t), height - 2), WHITE, 1)
    return pan


def label(img, text, org, color=WHITE, scale=0.45):
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):                       # dark halo (a thick stroke is spaced differently in some builds)
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def view_frames(frames, r, k, title=""):
    """Annotated BGR frames of one view."""
    ed = {e: i for i, (e, _) in enumerate(r["beats"])}; es = {s: i for i, (_, s) in enumerate(r["beats"])}
    out = []
    for t, f in enumerate(frames):
        img = cv2.cvtColor(cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        if t in ed:
            draw_tracing(img, r["beat_points"][ed[t], 0], k, GREEN); tag, col = f"ED  beat {ed[t] + 1}", GREEN
        elif t in es:
            draw_tracing(img, r["beat_points"][es[t], 1], k, ORANGE); tag, col = f"ES  beat {es[t] + 1}", ORANGE
        else:
            draw_tracing(img, r["frame_points"][t], k); tag, col = "", WHITE
        fs = max(0.33, 0.11 * k + 0.01); dy = int(40 * fs)
        label(img, f"{title}frame {t + 1}/{len(frames)}", (5, dy), scale=fs)
        if tag:
            label(img, tag, (5, 2 * dy + 2), col, fs)
        out.append(np.concatenate([img, curve_panel(r["volume"], t, r["beats"], img.shape[1])], 0))
    return out


def render(res, frames, k=4, frames_2ch=None):
    """Annotated BGR frames of the whole video (two views side by side in biplane mode), with the EF in a footer."""
    if res["mode"] == "biplane":
        a = view_frames(frames, res["views"]["4CH"], k, "4CH  "); b = view_frames(frames_2ch, res["views"]["2CH"], k, "2CH  ")
        n = max(len(a), len(b)); seq = [np.concatenate([a[min(i, len(a) - 1)], b[min(i, len(b) - 1)]], 1) for i in range(n)]
        text = f"Biplane Simpson EF {res['ef']:.1f}%  (CAMUS convention: one disk height)"
    else:
        seq = view_frames(frames, res, k)
        text = f"EF {res['ef']:.1f}%" + (f"  90% {res['lo90']:.1f}-{res['hi90']:.1f}" if "lo90" in res else "") + f"  median of {len(res['beats'])} beats"
    w = seq[0].shape[1]; fs = min(0.45, max(0.33, 0.11 * k + 0.01))
    while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] > w - 10 and fs > 0.25:
        fs -= 0.02
    foot = np.full((int(60 * fs) + 4, w, 3), 18, np.uint8); label(foot, text, (5, int(42 * fs)), WHITE, fs)
    out = [np.concatenate([img, foot], 0) for img in seq]
    return out


def write(seq, path, fps):
    """Write BGR frames to .mp4 or .gif."""
    if path.lower().endswith(".gif"):
        write_gif([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in seq], path, fps)
        return
    h, w = seq[0].shape[:2]
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)          # quiet the probe for an H.264 encoder
    except AttributeError:
        pass
    for cc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if vw.isOpened():
            break
    for f in seq:
        vw.write(f)
    vw.release()


def write_gif(rgb, path, fps, n_grey=160):
    """GIF with one shared palette: n_grey grey levels for the echo, the rest from the coloured (overlay) pixels, no dithering.
    A palette fitted to the whole image spends too few entries on grey and turns speckle into overlay colours."""
    from PIL import Image
    sample = np.concatenate([f.reshape(-1, 3) for f in rgb[::6]])
    col = sample[np.ptp(sample.astype(int), axis=1) > 24]
    g = np.linspace(0, 255, n_grey).round().astype(np.uint8)
    pal = list(np.repeat(g, 3))
    if len(col):
        q = Image.fromarray(col[None]).quantize(colors=256 - n_grey, method=Image.MEDIANCUT).getpalette()[:3 * (256 - n_grey)]
        pal += q
    pim = Image.new("P", (1, 1)); pim.putpalette((pal + [0] * 768)[:768])
    ims = [Image.fromarray(f).quantize(palette=pim, dither=Image.Dither.NONE) for f in rgb]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
