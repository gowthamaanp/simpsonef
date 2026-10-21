/* Browser front end: video -> 50 fps grey 112 x 112 frames -> SimpsonEF.analyse -> tracings, volume curve, beats, LVEF. */
(function () {
  "use strict";
  const $ = function (id) { return document.getElementById(id); };
  const S = 112;
  let models = null, cfg = null;
  const css = function (v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); };

  function status(text, frac) {
    $("status").textContent = text;
    $("bar").style.width = (frac == null ? 0 : Math.round(100 * frac)) + "%";
  }

  async function loadModels() {
    if (models) return;
    status("Loading models (about 14 MB, once)...", 0.02);
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    cfg = await (await fetch("models/config.json")).json();
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
    const opt = { executionProviders: navigator.gpu ? ["webgpu", "wasm"] : ["wasm"], graphOptimizationLevel: "all" };
    models = {};
    const names = ["frametracer", "beatfinder", "beattracer"];
    for (let i = 0; i < names.length; i++) {
      models[names[i]] = await ort.InferenceSession.create("models/" + names[i] + ".onnx", opt);
      status("Loading models...", (i + 1) / names.length * 0.1);
    }
  }

  function seek(video, t) {
    return new Promise(function (res) {
      const done = function () { video.removeEventListener("seeked", done); res(); };
      video.addEventListener("seeked", done); video.currentTime = t;
    });
  }

  /* Frames at cfg.fps, zero-padded to a square and resized to 112 x 112, grey (mean of R, G, B). */
  async function extractFrames(video, seconds) {
    const c = document.createElement("canvas"); c.width = S; c.height = S;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    const vw = video.videoWidth, vh = video.videoHeight, side = Math.max(vw, vh), k = S / side;
    const dw = vw * k, dh = vh * k, dx = (S - dw) / 2, dy = (S - dh) / 2;
    const dur = Math.min(video.duration, seconds), T = Math.max(1, Math.floor(dur * cfg.fps));
    const out = new Uint8Array(T * S * S);
    for (let t = 0; t < T; t++) {
      await seek(video, Math.min(video.duration - 1e-3, t / cfg.fps));
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S, S); ctx.drawImage(video, dx, dy, dw, dh);
      const px = ctx.getImageData(0, 0, S, S).data;
      for (let j = 0; j < S * S; j++) out[t * S * S + j] = (px[4 * j] + px[4 * j + 1] + px[4 * j + 2]) / 3;
      if (t % 10 === 0) status("Reading frames " + (t + 1) + " / " + T, 0.1 + 0.15 * t / T);
    }
    return { frames: out, T: T };
  }

  function drawFrame(canvas, frames, t, pts) {
    const ctx = canvas.getContext("2d"), z = canvas.width / S, img = ctx.createImageData(S, S);
    for (let j = 0; j < S * S; j++) { const v = frames[t * S * S + j]; img.data[4 * j] = img.data[4 * j + 1] = img.data[4 * j + 2] = v; img.data[4 * j + 3] = 255; }
    const tmp = document.createElement("canvas"); tmp.width = S; tmp.height = S; tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    const X = function (i) { return (pts[2 * i] - 1 + 0.5) * z; }, Y = function (i) { return (S - pts[2 * i + 1] + 0.5) * z; };
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255, 210, 63, 0.9)";
    for (let i = 1; i <= 20; i++) { ctx.beginPath(); ctx.moveTo(X(i), Y(i)); ctx.lineTo(X(41 - i), Y(41 - i)); ctx.stroke(); }
    ctx.lineWidth = 2; ctx.strokeStyle = "#35d6e8"; ctx.beginPath(); ctx.moveTo(X(1), Y(1));
    for (let i = 2; i <= 40; i++) ctx.lineTo(X(i), Y(i));
    ctx.closePath(); ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#ff5a5f"; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(41), Y(41)); ctx.stroke();
  }

  function drawCurve(canvas, vol, beats) {
    const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, pad = 14;
    ctx.clearRect(0, 0, W, H);
    const lo = Math.min.apply(null, vol), hi = Math.max.apply(null, vol), n = vol.length;
    const x = function (t) { return pad + (W - 2 * pad) * t / Math.max(1, n - 1); }, y = function (v) { return H - pad - (H - 2 * pad) * (v - lo) / (hi - lo || 1); };
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.beginPath();
    vol.forEach(function (v, t) { if (t) ctx.lineTo(x(t), y(v)); else ctx.moveTo(x(t), y(v)); }); ctx.stroke();
    beats.forEach(function (b) {
      [[b.ed, css("--ed")], [b.es, css("--es")]].forEach(function (p) {
        ctx.fillStyle = p[1]; ctx.beginPath(); ctx.arc(x(p[0]), y(vol[p[0]]), 4, 0, 2 * Math.PI); ctx.fill();
      });
    });
    ctx.fillStyle = css("--mute"); ctx.font = "12px IBM Plex Sans, sans-serif";
    ctx.fillText("LV volume over time (FrameTracer)   ● ED   ● ES", pad, 12);
  }

  async function analyseFile(file) {
    try {
      $("result").hidden = true;
      await loadModels();
      const video = $("video"); video.src = URL.createObjectURL(file);
      await new Promise(function (res, rej) { video.onloadeddata = res; video.onerror = function () { rej(new Error("This browser cannot decode the video; try an MP4 (H.264).")); }; });
      const secs = Math.max(1, Math.min(20, +$("secs").value || 8));
      const fr = await extractFrames(video, secs);
      if (fr.T < 30) throw new Error("The clip is too short; at least about one heartbeat is needed.");
      const t0 = performance.now();
      const r = await SimpsonEF.analyse(fr.frames, fr.T, models, ort.Tensor, cfg, function (what, f) { status(what + "...", 0.25 + 0.75 * f * (what.indexOf("ED") >= 0 ? 1 : 0.8)); });
      const efs = r.beats.map(function (b) { return b.ef; }), sorted = efs.slice().sort(function (a, b) { return a - b; });
      const mid = r.beats[efs.indexOf(sorted[(sorted.length - 1) >> 1])];
      $("ef").textContent = r.ef.toFixed(0) + "%";
      $("efint").textContent = "90% interval " + Math.max(0, r.lo).toFixed(0) + "–" + Math.min(100, r.hi).toFixed(0) + "%";
      drawFrame($("cED"), fr.frames, mid.ed, mid.pointsED); drawFrame($("cES"), fr.frames, mid.es, mid.pointsES);
      $("fED").textContent = "(frame " + mid.ed + ")"; $("fES").textContent = "(frame " + mid.es + ")";
      drawCurve($("curve"), r.volume, r.beats);
      $("beatrows").innerHTML = r.beats.map(function (b, i) {
        return "<tr><td>" + (i + 1) + "</td><td>" + b.ed + "</td><td>" + b.es + "</td><td>" + b.ef.toFixed(1) + "%</td></tr>";
      }).join("");
      $("how").textContent = "LVEF is the median of the " + r.beats.length + " beat EF" + (r.beats.length > 1 ? "s" : "") +
        " above, each from Simpson's rule on BeatTracer's ED and ES tracings (shown for the median beat). " + fr.T + " frames at " + cfg.fps +
        " fps analysed in " + ((performance.now() - t0) / 1000).toFixed(1) + " s." + (r.beats[0].fallback ? " No complete beat was found; the strongest ED and ES frames were used." : "");
      $("result").hidden = false; status("Done.", 1);
    } catch (e) { status("Error: " + e.message, 0); }
  }

  $("file").addEventListener("change", function (e) { if (e.target.files[0]) analyseFile(e.target.files[0]); });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) { if (e.dataTransfer.files[0]) analyseFile(e.dataTransfer.files[0]); });
})();
