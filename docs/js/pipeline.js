/* SimpsonEF inference pipeline (browser and Node). Mirrors src/pipeline.py (SimpsonEF.analyse):
 *   frames (T x 112 x 112, uint8 grey) -> FrameTracer on every frame -> chords -> BeatFinder -> ED / ES of every beat
 *   -> BeatTracer on each beat's ED and ES windows -> Simpson's rule per beat -> median over beats -> conformal 90% interval;
 * analyseBiplane mirrors SimpsonEF.analyse_biplane (4- and 2-chamber views, FrameTracer tracings, one disk height).
 * Points use the tracing convention of EchoNet-Dynamic: x = column + 1, y = 112 - row.
 * The browser demo skips the 15-copy input averaging of src/pipeline.py (on VAL it changes R2 by less than 0.001). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SimpsonEF = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const S = 112, NP = 42;

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

  /* Steps 1-2 for one view. frames: Uint8Array of T * 112 * 112; models: {frametracer, beatfinder, beattracer} ORT sessions;
   * cfg: docs/models/config.json. Returns FrameTracer's points on every frame, the volume curve, BeatFinder's output and the beats. */
  async function traceAndFindBeats(frames, T, models, Tensor, cfg, progress) {
    const n = S * S, offA = cfg.frametracer_offsets, chunk = 16;
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
    // 2. BeatFinder on the whole clip; every ED peak with the first ES peak before the next ED peak
    const fo = await run(models.beatfinder, Tensor, feats, [1, T, 21]);
    const heat = fo.heat.data, q = Array.from(fo.ef.data).map(function (v) { return 100 * v; });
    const hed = [0], hes = [0];
    for (let t = 0; t < T; t++) { hed.push(sigmoid(heat[2 * t])); hes.push(sigmoid(heat[2 * t + 1])); }
    hed.push(0); hes.push(0);
    const ed = findPeaks(hed, cfg.peak_prominence).map(function (p) { return p - 1; });
    const es = findPeaks(hes, cfg.peak_prominence).map(function (p) { return p - 1; });
    let beats = [];
    for (let i = 0; i < ed.length; i++) {
      const nxt = i + 1 < ed.length ? ed[i + 1] : 1e9, c = es.filter(function (v) { return v > ed[i] && v < nxt; });
      if (c.length) beats.push({ ed: ed[i], es: c[0] });
    }
    if (!beats.length) {                        // no ED -> ES pair: the frames of largest and smallest FrameTracer volume
      let a = 0, b = 0;
      for (let t = 1; t < T; t++) { if (vol[t] > vol[a]) a = t; if (vol[t] < vol[b]) b = t; }
      beats = [{ ed: a, es: b, fallback: true }];
    }
    return { T: T, points: P, volume: Array.from(vol), beats: beats, edFrames: ed, esFrames: es,
             efBeatFinder: q[0], q05: Math.min(q[1], q[2]), q95: Math.max(q[1], q[2]) };
  }

  /* Single plane (EchoNet-Dynamic convention): steps 1-2, BeatTracer on each beat, Simpson per beat, median, conformal interval. */
  async function analyse(frames, T, models, Tensor, cfg, progress) {
    progress = progress || function () {};
    const r = await traceAndFindBeats(frames, T, models, Tensor, cfg, progress), n = S * S, offB = cfg.beattracer_offsets, cB = 2 * offB.length;
    progress("Tracing ED and ES", 0);
    for (let k = 0; k < r.beats.length; k++) {
      const b = r.beats[k], x = new Float32Array(cB * n);
      windowStack(frames, T, b.ed, offB, x, 0); windowStack(frames, T, b.es, offB, x, offB.length * n);
      const p = (await run(models.beattracer, Tensor, x, [1, cB, S, S])).points.data;
      b.pointsED = Array.from(p.slice(0, NP * 2)); b.pointsES = Array.from(p.slice(NP * 2, 2 * NP * 2));
      b.ef = 100 * (1 - volume(chords(p, NP * 2)) / volume(chords(p, 0)));
      progress("Tracing ED and ES", (k + 1) / r.beats.length);
    }
    r.ef = median(r.beats.map(function (b) { return b.ef; }));
    const sc = (r.q95 - r.q05) / 2 + Math.abs(r.ef - r.efBeatFinder), Q = cfg.conformal_Q90;
    r.lo = r.ef - Q * sc; r.hi = r.ef + Q * sc; r.mode = "single";
    return r;
  }

  /* Biplane (CAMUS convention): steps 1-2 on each view, FrameTracer's ED and ES tracings, one disk height for both phases:
   * EF = 1 - sum a_ES b_ES / sum a_ED b_ED. The i-th beat of one view is paired with the i-th of the other; median over pairs. */
  async function analyseBiplane(f4, T4, f2, T2, models, Tensor, cfg, progress) {
    progress = progress || function () {};
    const pick = function (r) {
      r.beats.forEach(function (b) {
        b.pointsED = Array.from(r.points.slice(b.ed * NP * 2, (b.ed + 1) * NP * 2)); b.pointsES = Array.from(r.points.slice(b.es * NP * 2, (b.es + 1) * NP * 2));
        b.ef = 100 * (1 - volume(chords(b.pointsES, 0)) / volume(chords(b.pointsED, 0)));
      });
      return r;
    };
    const a = pick(await traceAndFindBeats(f4, T4, models, Tensor, cfg, function (w, f) { progress(w + " (4-chamber)", f / 2); }));
    const b = pick(await traceAndFindBeats(f2, T2, models, Tensor, cfg, function (w, f) { progress(w + " (2-chamber)", 0.5 + f / 2); }));
    const efs = [];
    for (let i = 0; i < Math.min(a.beats.length, b.beats.length); i++) {
      const aE = chords(a.beats[i].pointsED, 0), aS = chords(a.beats[i].pointsES, 0), bE = chords(b.beats[i].pointsED, 0), bS = chords(b.beats[i].pointsES, 0);
      let num = 0, den = 0;
      for (let j = 0; j < 20; j++) { num += aS[j] * bS[j]; den += aE[j] * bE[j]; }
      efs.push(100 * (1 - num / den));
    }
    return { mode: "biplane", ef: median(efs), beatEF: efs, views: { "4CH": a, "2CH": b } };
  }

  return { analyse: analyse, analyseBiplane: analyseBiplane, findPeaks: findPeaks, chords: chords, volume: volume, S: S };
});
