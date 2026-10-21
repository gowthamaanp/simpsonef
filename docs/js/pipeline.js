/* SimpsonEF inference pipeline (browser and Node). Mirrors src/inference.py + experiments/eval_pair.py:
 *   frames (T x 112 x 112, uint8 grey) -> FrameTracer on every frame -> chords -> BeatFinder -> ED / ES of every beat
 *   -> BeatTracer on each beat's ED and ES windows -> Simpson's rule per beat -> median over beats -> conformal 90% interval.
 * Points use the tracing convention of EchoNet-Dynamic: x = column + 1, y = 112 - row.
 * The browser demo does not apply the 15-copy input averaging of the paper (TEST R2 0.801 instead of 0.808). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SimpsonEF = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const S = 112, NP = 42, T_MAX = 350;

  function chords(P, off) {                    // P: Float32Array of 42 x 2 points starting at off -> 20 widths + axis length
    const out = new Float32Array(21);
    for (let i = 0; i < 20; i++) {
      const a = off + 2 * (1 + i), b = off + 2 * (40 - i);
      out[i] = Math.hypot(P[a] - P[b], P[a + 1] - P[b + 1]);
    }
    out[20] = Math.hypot(P[off] - P[off + 82], P[off + 1] - P[off + 83]);
    return out;
  }

  function volume(c) {                          // method of disks, pixel units (pi / 4 / 20 cancels in EF)
    let s = 0;
    for (let i = 0; i < 20; i++) s += c[i] * c[i];
    return (Math.PI / 4) * s * c[20] / 20;
  }

  function windowStack(frames, T, t, offsets, dst, dstOff) {   // frames at t + offset, clipped to the clip, as channels in [0, 1]
    const n = S * S;
    for (let k = 0; k < offsets.length; k++) {
      const f = Math.min(T - 1, Math.max(0, t + offsets[k]));
      const src = f * n, d = dstOff + k * n;
      for (let j = 0; j < n; j++) dst[d + j] = frames[src + j] / 255;
    }
  }

  /* scipy.signal.find_peaks(x, prominence=p): local maxima (middle of plateaus) whose prominence is at least p. */
  function findPeaks(x, prom) {
    const n = x.length, peaks = [];
    let i = 1;
    while (i < n - 1) {
      if (x[i - 1] < x[i]) {
        let ahead = i + 1;
        while (ahead < n - 1 && x[ahead] === x[i]) ahead++;
        if (x[ahead] < x[i]) { peaks.push(Math.floor((i + ahead - 1) / 2)); i = ahead; }
      }
      i++;
    }
    return peaks.filter(function (p) {
      let lmin = x[p], j = p - 1;
      while (j >= 0 && x[j] <= x[p]) { if (x[j] < lmin) lmin = x[j]; j--; }
      let rmin = x[p]; j = p + 1;
      while (j < n && x[j] <= x[p]) { if (x[j] < rmin) rmin = x[j]; j++; }
      return x[p] - Math.max(lmin, rmin) >= prom;
    });
  }

  const sigmoid = function (v) { return 1 / (1 + Math.exp(-v)); };
  function median(a) { const b = a.slice().sort(function (p, q) { return p - q; }); const m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; }

  async function run(session, Tensor, data, dims) {
    const feeds = {}; feeds[session.inputNames[0]] = new Tensor("float32", data, dims);
    return await session.run(feeds);
  }

  /* frames: Uint8Array of T * 112 * 112; models: {frametracer, beatfinder, beattracer} ORT sessions; cfg: docs/models/config.json */
  async function analyse(frames, T, models, Tensor, cfg, progress) {
    progress = progress || function () {};
    const n = S * S, offA = cfg.frametracer_offsets, offB = cfg.beattracer_offsets, chunk = 16;
    // 1. FrameTracer on every frame
    const P = new Float32Array(T * NP * 2);
    for (let s = 0; s < T; s += chunk) {
      const m = Math.min(chunk, T - s), x = new Float32Array(m * offA.length * n);
      for (let b = 0; b < m; b++) windowStack(frames, T, s + b, offA, x, b * offA.length * n);
      const o = await run(models.frametracer, Tensor, x, [m, offA.length, S, S]);
      P.set(o.points.data, s * NP * 2);
      progress("Tracing every frame", (s + m) / T);
    }
    const feats = new Float32Array(T * 21), vol = new Float32Array(T);
    for (let t = 0; t < T; t++) { const c = chords(P, t * NP * 2); feats.set(c, t * 21); vol[t] = volume(c); }
    // 2. BeatFinder on (at most) the central 350 frames
    const s0 = T > T_MAX ? Math.max(0, Math.min(T - T_MAX, Math.floor(T / 2) - T_MAX / 2)) : 0, L = Math.min(T, T_MAX);
    const fo = await run(models.beatfinder, Tensor, feats.slice(s0 * 21, (s0 + L) * 21), [1, L, 21]);
    const heat = fo.heat.data, q = Array.from(fo.ef.data).map(function (v) { return 100 * v; });
    const efHead = q[0], q05 = Math.min(q[1], q[2]), q95 = Math.max(q[1], q[2]);
    const hed = [0], hes = [0];
    for (let t = 0; t < L; t++) { hed.push(sigmoid(heat[2 * t])); hes.push(sigmoid(heat[2 * t + 1])); }
    hed.push(0); hes.push(0);
    const ed = findPeaks(hed, cfg.peak_prominence).map(function (p) { return p - 1 + s0; });
    const es = findPeaks(hes, cfg.peak_prominence).map(function (p) { return p - 1 + s0; });
    let beats = [];
    for (let i = 0; i < ed.length; i++) {
      const nxt = i + 1 < ed.length ? ed[i + 1] : 1e9, c = es.filter(function (v) { return v > ed[i] && v < nxt; });
      if (c.length) beats.push({ ed: ed[i], es: c[0] });
    }
    if (!beats.length) {                        // no ED -> ES pair: use the global extremes of the two detectors
      let a = 0, b = 0;
      for (let t = 1; t < L; t++) { if (hed[t + 1] > hed[a + 1]) a = t; if (hes[t + 1] > hes[b + 1]) b = t; }
      beats = [{ ed: a + s0, es: b + s0, fallback: true }];
    }
    progress("Tracing ED and ES", 0);
    // 3. BeatTracer on each beat, Simpson's rule
    const cB = 2 * offB.length;
    for (let k = 0; k < beats.length; k++) {
      const x = new Float32Array(cB * n);
      windowStack(frames, T, beats[k].ed, offB, x, 0); windowStack(frames, T, beats[k].es, offB, x, offB.length * n);
      const o = await run(models.beattracer, Tensor, x, [1, cB, S, S]), p = o.points.data;
      const cED = chords(p, 0), cES = chords(p, NP * 2);
      beats[k].pointsED = Array.from(p.slice(0, NP * 2)); beats[k].pointsES = Array.from(p.slice(NP * 2, 2 * NP * 2));
      beats[k].ef = 100 * (1 - volume(cES) / volume(cED));
      progress("Tracing ED and ES", (k + 1) / beats.length);
    }
    const ef = median(beats.map(function (b) { return b.ef; }));
    const sc = (q95 - q05) / 2 + Math.abs(ef - efHead), Q = cfg.conformal_Q90;
    return { ef: ef, lo: ef - Q * sc, hi: ef + Q * sc, beats: beats, volume: Array.from(vol), points: P, T: T,
             efBeatFinder: efHead, edFrames: ed, esFrames: es };
  }

  return { analyse: analyse, findPeaks: findPeaks, chords: chords, volume: volume, S: S };
});
