/* Browser front end: MP4 -> 50 fps grey 112 x 112 frames -> SimpsonEF.analyse (single plane) or SimpsonEF.analyseBiplane ->
 * the whole clip replayed with the tracing on every frame (mask, outline, chords, long axis), the volume curve with the beats, LVEF. */
(function () {
  "use strict";
  const $ = function (id) { return document.getElementById(id); };
  const S = 112, Z = 4, ED = "#5ad85a", ES = "#ff9600", FRAME = "#35d6e8";    // as in run.py's videos
  let models = null, cfg = null, players = [];
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

  async function openVideo(file) {
    if (!/\.mp4$/i.test(file.name) && file.type !== "video/mp4") throw new Error("Please choose MP4 files (H.264); other formats are not supported.");
    const video = document.createElement("video"); video.muted = true; video.playsInline = true; video.preload = "auto";
    video.src = URL.createObjectURL(file);
    await new Promise(function (res, rej) { video.onloadeddata = res; video.onerror = function () { rej(new Error("This browser cannot decode " + file.name + "; try an MP4 (H.264).")); }; });
    return video;
  }

  /* Frames at cfg.fps, zero-padded to a square and resized to 112 x 112, grey (mean of R, G, B). */
  async function extractFrames(video, seconds, label, f0, f1) {
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
      if (t % 10 === 0) status("Reading frames" + label + " " + (t + 1) + " / " + T, f0 + (f1 - f0) * t / T);
    }
    if (T < 30) throw new Error("The clip" + label + " is too short; at least about one heartbeat is needed.");
    return { frames: out, T: T };
  }

  /* One frame with its tracing: translucent mask, outline, the 20 chords and the long axis. */
  function drawFrame(ctx, frames, t, pts, color) {
    const img = ctx.createImageData(S, S);
    for (let j = 0; j < S * S; j++) { const v = frames[t * S * S + j]; img.data[4 * j] = img.data[4 * j + 1] = img.data[4 * j + 2] = v; img.data[4 * j + 3] = 255; }
    const tmp = document.createElement("canvas"); tmp.width = S; tmp.height = S; tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(tmp, 0, 0, S * Z, S * Z);
    const X = function (i) { return (pts[2 * i] - 1 + 0.5) * Z; }, Y = function (i) { return (S - pts[2 * i + 1] + 0.5) * Z; };
    ctx.beginPath(); ctx.moveTo(X(1), Y(1));
    for (let i = 2; i <= 40; i++) ctx.lineTo(X(i), Y(i));
    ctx.closePath(); ctx.globalAlpha = 0.22; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
    ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255, 210, 63, 0.9)";
    for (let i = 1; i <= 20; i++) { ctx.beginPath(); ctx.moveTo(X(i), Y(i)); ctx.lineTo(X(41 - i), Y(41 - i)); ctx.stroke(); }
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#ff5a5f"; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(41), Y(41)); ctx.stroke();
  }

  function drawCurve(canvas, vol, beats, t) {
    const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, pad = 14;
    ctx.clearRect(0, 0, W, H);
    const lo = Math.min.apply(null, vol), hi = Math.max.apply(null, vol), n = vol.length;
    const x = function (i) { return pad + (W - 2 * pad) * i / Math.max(1, n - 1); }, y = function (v) { return H - pad - (H - 2 * pad) * (v - lo) / (hi - lo || 1); };
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.6; ctx.beginPath();
    vol.forEach(function (v, i) { if (i) ctx.lineTo(x(i), y(v)); else ctx.moveTo(x(i), y(v)); }); ctx.stroke();
    beats.forEach(function (b) {
      [[b.ed, ED], [b.es, ES]].forEach(function (p) {
        ctx.fillStyle = p[1]; ctx.beginPath(); ctx.arc(x(p[0]), y(vol[p[0]]), 4, 0, 2 * Math.PI); ctx.fill();
      });
    });
    if (t != null) { ctx.strokeStyle = css("--mute"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(t), pad); ctx.lineTo(x(t), H - pad); ctx.stroke(); }
    ctx.fillStyle = css("--mute"); ctx.font = "12px IBM Plex Sans, sans-serif";
    ctx.fillText("LV volume over time (FrameTracer)   ● ED   ● ES", pad, 12);
  }

  /* Replays the analysed clip: FrameTracer's tracing on every frame; on each beat's ED / ES frame the tracing Simpson's rule used. */
  function makePlayer(host, title, frames, r) {
    const fig = document.createElement("figure"); fig.className = "player";
    fig.innerHTML = '<div class="player-head"><span class="ptitle"></span><span class="pframe"></span></div>' +
      '<canvas class="pview" width="' + S * Z + '" height="' + S * Z + '"></canvas>' +
      '<canvas class="pcurve" width="' + S * Z + '" height="120" aria-label="Left-ventricular volume over time with the detected beats"></canvas>' +
      '<div class="pctl"><button type="button" class="btn small pplay">Pause</button><input type="range" class="pseek" min="0" value="0" aria-label="Frame">' +
      '<button type="button" class="btn small psave">Save video</button></div>';
    host.appendChild(fig);
    const q = function (c) { return fig.querySelector(c); };
    const view = q(".pview"), ctx = view.getContext("2d"), curve = q(".pcurve"), seekbar = q(".pseek");
    q(".ptitle").textContent = title; seekbar.max = r.T - 1;
    const ed = {}, es = {};
    r.beats.forEach(function (b, i) { ed[b.ed] = i; es[b.es] = i; });
    let t = 0, playing = true, last = 0, timer = null;
    function show(i) {
      t = i; seekbar.value = i;
      let pts = r.points.subarray(i * 84, (i + 1) * 84), col = FRAME, tag = "";
      if (i in ed) { pts = r.beats[ed[i]].pointsED; col = ED; tag = "  ·  ED of beat " + (ed[i] + 1); }
      else if (i in es) { pts = r.beats[es[i]].pointsES; col = ES; tag = "  ·  ES of beat " + (es[i] + 1); }
      drawFrame(ctx, frames, i, pts, col);
      q(".pframe").textContent = "frame " + (i + 1) + " / " + r.T + tag;
      drawCurve(curve, r.volume, r.beats, i);
    }
    function tick(now) {
      if (!fig.isConnected) return;
      if (playing && now - last >= 1000 / cfg.fps * (((t in ed) || (t in es)) ? 12 : 1)) { last = now; show((t + 1) % r.T); }
      timer = requestAnimationFrame(tick);
    }
    q(".pplay").onclick = function () { playing = !playing; this.textContent = playing ? "Pause" : "Play"; };
    seekbar.oninput = function () { playing = false; q(".pplay").textContent = "Play"; show(+seekbar.value); };
    curve.onclick = function (e) {
      const b = curve.getBoundingClientRect(), f = ((e.clientX - b.left) / b.width * curve.width - 14) / (curve.width - 28);
      playing = false; q(".pplay").textContent = "Play"; show(Math.max(0, Math.min(r.T - 1, Math.round(f * (r.T - 1)))));
    };
    q(".psave").onclick = function () { saveVideo(view, curve, frames, r, show, title); };
    show(0); timer = requestAnimationFrame(tick);
    return { stop: function () { cancelAnimationFrame(timer); } };
  }

  /* Records one pass over the clip (frame + curve) with MediaRecorder; browsers record WebM. */
  function saveVideo(view, curve, frames, r, show, title) {
    if (!window.MediaRecorder) { status("This browser cannot record video.", 1); return; }
    const c = document.createElement("canvas"); c.width = view.width; c.height = view.height + curve.height;
    const g = c.getContext("2d"), rec = new MediaRecorder(c.captureStream(cfg.fps), { mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm" });
    const chunks = []; rec.ondataavailable = function (e) { chunks.push(e.data); };
    rec.onstop = function () {
      const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      a.download = "simpsonef_" + title.replace(/\W+/g, "_").toLowerCase() + ".webm"; a.click(); status("Saved.", 1);
    };
    let i = 0; rec.start(); status("Recording the annotated video...", 0);
    const step = function () {
      show(i); g.fillStyle = "#111"; g.fillRect(0, 0, c.width, c.height); g.drawImage(view, 0, 0); g.drawImage(curve, 0, view.height);
      status("Recording the annotated video...", i / r.T);
      if (++i < r.T) setTimeout(step, 1000 / cfg.fps); else setTimeout(function () { rec.stop(); }, 200);
    };
    step();
  }

  function beatTable(beats) {
    return beats.map(function (b, i) {
      return "<tr><td>" + (i + 1) + "</td><td>" + (b.ed + 1) + "</td><td>" + (b.es + 1) + "</td><td>" + b.ef.toFixed(1) + "%</td></tr>";
    }).join("");
  }

  async function analyse() {
    const biplane = $("mode-bi").checked, f4 = $("file").files[0], f2 = $("file2").files[0];
    if (!f4 || (biplane && !f2)) { status(biplane ? "Choose a 4-chamber and a 2-chamber MP4." : "Choose an MP4 video.", 0); return; }
    try {
      $("result").hidden = true; players.forEach(function (p) { p.stop(); }); players = []; $("players").innerHTML = "";
      await loadModels();
      const secs = Math.max(1, Math.min(20, +$("secs").value || 8)), t0 = performance.now();
      const prog = function (a, b) { return function (what, f) { status(what + "...", a + (b - a) * f); }; };
      let r, fr4, fr2;
      if (biplane) {
        fr4 = await extractFrames(await openVideo(f4), secs, " (4-chamber)", 0.1, 0.2);
        fr2 = await extractFrames(await openVideo(f2), secs, " (2-chamber)", 0.2, 0.3);
        r = await SimpsonEF.analyseBiplane(fr4.frames, fr4.T, fr2.frames, fr2.T, models, ort.Tensor, cfg, prog(0.3, 1));
        $("efint").textContent = "biplane Simpson, one disk height (CAMUS convention); no interval";
        players.push(makePlayer($("players"), "Apical 4-chamber", fr4.frames, r.views["4CH"]));
        players.push(makePlayer($("players"), "Apical 2-chamber", fr2.frames, r.views["2CH"]));
        $("beatrows").innerHTML = beatTable(r.views["4CH"].beats.map(function (b, i) { return { ed: b.ed, es: b.es, ef: r.beatEF[i] != null ? r.beatEF[i] : b.ef }; }));
        $("how").textContent = "LVEF is biplane Simpson's rule on FrameTracer's 4- and 2-chamber tracings at ED and ES, with one disk height for both phases, " +
          "as CAMUS defines its label; the i-th beat of each view is paired and the median over pairs is reported (beat table: 4-chamber frames, biplane EF). " +
          "Analysed in " + ((performance.now() - t0) / 1000).toFixed(1) + " s.";
      } else {
        fr4 = await extractFrames(await openVideo(f4), secs, "", 0.1, 0.25);
        r = await SimpsonEF.analyse(fr4.frames, fr4.T, models, ort.Tensor, cfg, prog(0.25, 1));
        $("efint").textContent = "90% interval " + Math.max(0, r.lo).toFixed(0) + "–" + Math.min(100, r.hi).toFixed(0) + "%";
        players.push(makePlayer($("players"), "Apical 4-chamber", fr4.frames, r));
        $("beatrows").innerHTML = beatTable(r.beats);
        $("how").textContent = "LVEF is the median of the " + r.beats.length + " beat EF" + (r.beats.length > 1 ? "s" : "") +
          " above, each from Simpson's rule on BeatTracer's ED (green) and ES (orange) tracings; the other frames show FrameTracer's tracing. " +
          fr4.T + " frames at " + cfg.fps + " fps analysed in " + ((performance.now() - t0) / 1000).toFixed(1) + " s." +
          (r.beats[0].fallback ? " No complete beat was found; the frames of largest and smallest volume were used." : "");
      }
      $("ef").textContent = r.ef.toFixed(0) + "%";
      $("result").hidden = false; status("Done.", 1);
    } catch (e) { status("Error: " + e.message, 0); }
  }

  function syncMode() {
    const bi = $("mode-bi").checked;
    $("drop2").hidden = !bi;
    $("drop-title").textContent = bi ? "Apical 4-chamber MP4" : "Choose or drop an MP4 video";
  }
  ["mode-single", "mode-bi"].forEach(function (id) { $(id).addEventListener("change", syncMode); });
  $("run").addEventListener("click", analyse);
  [["drop", "file"], ["drop2", "file2"]].forEach(function (p) {
    const drop = $(p[0]), input = $(p[1]);
    input.addEventListener("change", function () { drop.querySelector(".drop-sub").textContent = input.files[0] ? input.files[0].name : ""; });
    ["dragenter", "dragover"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
    drop.addEventListener("drop", function (e) { if (e.dataTransfer.files[0]) { input.files = e.dataTransfer.files; input.dispatchEvent(new Event("change")); } });
  });
  syncMode();
})();
