/**
 * FootballVision — Tactical View
 *
 * Connects to /ws/tactical and renders the live analysis: the annotated video
 * feed, a top-down pitch in real metres, team/role counts, structured football
 * concepts, and similar-situation retrieval.
 */

(function () {
  "use strict";

  const el = (id) => document.getElementById(id);

  // ── State ────────────────────────────────────────────────────────────
  let ws = null;
  let running = false;
  let paused = false;
  let PITCH_L = 105, PITCH_W = 68;
  let showControl = true;
  let showHull = true;
  let showTrails = false;
  let currentSessionId = null;
  let currentPitch = null;
  let editingEvent = null;
  let editingSessionId = null;
  let compareBaseline = null;
  let playerLabels = {};
  let calibrationPoints = [];
  let calibrating = false;
  const sessionCache = new Map();
  let timelineSignature = "";

  const TEAM_COLORS = { "0": "#e5484d", "1": "#3b82f6", "-1": "#9aa0a6" };
  const TEAM_RGB = { "0": [229, 72, 77], "1": [59, 130, 246] };
  const ROLE_STYLE = { referee: "#f5c542", goalkeeper: "#22c55e" };

  // ── Pitch canvas ─────────────────────────────────────────────────────
  const canvas = el("pitch-canvas");
  const ctx = canvas.getContext("2d");
  const MARGIN = 26;

  function toCanvas(x, y) {
    const w = canvas.width - 2 * MARGIN;
    const h = canvas.height - 2 * MARGIN;
    return [MARGIN + (x / PITCH_L) * w, MARGIN + (1 - y / PITCH_W) * h];
  }

  function drawPitch(concepts) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0e2a19";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (showControl && concepts && concepts.control_grid) drawControl(concepts.control_grid);

    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1.4;
    const line = (ax, ay, bx, by) => {
      const [x0, y0] = toCanvas(ax, ay), [x1, y1] = toCanvas(bx, by);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    };
    const [bx0, by0] = toCanvas(0, 0), [bx1, by1] = toCanvas(PITCH_L, PITCH_W);
    ctx.strokeRect(bx0, by1, bx1 - bx0, by0 - by1);
    line(PITCH_L / 2, 0, PITCH_L / 2, PITCH_W);

    const [ccx, ccy] = toCanvas(PITCH_L / 2, PITCH_W / 2);
    const r = (9.15 / PITCH_L) * (canvas.width - 2 * MARGIN);
    ctx.beginPath(); ctx.arc(ccx, ccy, r, 0, Math.PI * 2); ctx.stroke();
    dot(ccx, ccy, 2, "rgba(255,255,255,0.7)");

    const bt = PITCH_W / 2 + 20.16, bb = PITCH_W / 2 - 20.16;
    const st = PITCH_W / 2 + 9.16, sb = PITCH_W / 2 - 9.16;
    line(0, bb, 16.5, bb); line(16.5, bb, 16.5, bt); line(16.5, bt, 0, bt);
    line(0, sb, 5.5, sb); line(5.5, sb, 5.5, st); line(5.5, st, 0, st);
    line(PITCH_L, bb, PITCH_L - 16.5, bb); line(PITCH_L - 16.5, bb, PITCH_L - 16.5, bt); line(PITCH_L - 16.5, bt, PITCH_L, bt);
    line(PITCH_L, sb, PITCH_L - 5.5, sb); line(PITCH_L - 5.5, sb, PITCH_L - 5.5, st); line(PITCH_L - 5.5, st, PITCH_L, st);
    const [p1x, p1y] = toCanvas(11, PITCH_W / 2); dot(p1x, p1y, 2, "rgba(255,255,255,0.7)");
    const [p2x, p2y] = toCanvas(PITCH_L - 11, PITCH_W / 2); dot(p2x, p2y, 2, "rgba(255,255,255,0.7)");
  }

  function dot(cx, cy, r, color) {
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
  }

  /**
   * Pitch-control shading.
   *
   * The server sends a coarse 32x20 ownership grid. Painting it as raw
   * rectangles looks like a QR code, so it is drawn into a small offscreen
   * canvas at grid resolution and then scaled up with image smoothing on,
   * which the browser interpolates into soft territory boundaries. That reads
   * as "space owned" rather than as blocky cells, and it costs nothing.
   */
  function drawControl(grid) {
    const { cols, rows, cells } = grid;
    if (!cells || !cells.length) return;

    const off = document.createElement("canvas");
    off.width = cols; off.height = rows;
    const octx = off.getContext("2d");
    const img = octx.createImageData(cols, rows);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Grid row 0 is pitch y=0 (bottom), so flip vertically.
        const src = (rows - 1 - r) * cols + c;
        const rgb = TEAM_RGB[String(cells[src])];
        const i = (r * cols + c) * 4;
        if (rgb) {
          img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2];
          img.data[i + 3] = 64;
        } else {
          img.data[i + 3] = 0;
        }
      }
    }
    octx.putImageData(img, 0, 0);

    const w = canvas.width - 2 * MARGIN, h = canvas.height - 2 * MARGIN;
    ctx.save();
    ctx.beginPath(); ctx.rect(MARGIN, MARGIN, w, h); ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, MARGIN, MARGIN, w, h);
    ctx.restore();
  }

  function drawShapes(concepts) {
    if (!concepts || !concepts.teams) return;
    for (const k of ["0", "1"]) {
      const shape = concepts.teams[k];
      if (!shape) continue;
      const color = TEAM_COLORS[k];
      if (showHull && shape.hull && shape.hull.length >= 3) {
        ctx.beginPath();
        shape.hull.forEach((pt, i) => {
          const [x, y] = toCanvas(pt[0], pt[1]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.strokeStyle = color; ctx.lineWidth = 1.4;
        ctx.globalAlpha = 0.65; ctx.stroke(); ctx.globalAlpha = 1;
      }
      const [cx, cy] = toCanvas(shape.centroid[0], shape.centroid[1]);
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy);
      ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
    }
  }

  function drawPeople(pitch) {
    const players = pitch.players || [];
    for (const p of players) {
      const [cx, cy] = toCanvas(p.x, p.y);
      const role = p.role || "player";
      let color = TEAM_COLORS[String(p.team)] || TEAM_COLORS["-1"];
      if (ROLE_STYLE[role]) color = ROLE_STYLE[role];

      dot(cx, cy, role === "player" ? 7 : 6, color);
      ctx.lineWidth = 1.4; ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath(); ctx.arc(cx, cy, role === "player" ? 7 : 6, 0, Math.PI * 2); ctx.stroke();

      if (role === "player") {
        ctx.fillStyle = "#fff"; ctx.font = "9px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const label = playerLabels[String(p.id)];
        ctx.fillText(String(label && label.shirt_number != null ? label.shirt_number : p.id), cx, cy);
      } else {
        ctx.fillStyle = "#000"; ctx.font = "bold 8px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(role === "referee" ? "R" : "GK", cx, cy);
      }
    }
    if (pitch.ball) {
      const [bx, by] = toCanvas(pitch.ball.x, pitch.ball.y);
      ctx.save();
      ctx.globalAlpha = pitch.ball.stale ? 0.45 : 1;
      dot(bx, by, 5, "#ffffff");
      ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
    }
  }

  // ── Source picker ────────────────────────────────────────────────────
  el("btn-start").addEventListener("click", openPicker);
  el("btn-stop").addEventListener("click", stopSession);
  el("btn-pause").addEventListener("click", togglePause);
  el("btn-calibrate").addEventListener("click", toggleCalibration);
  el("video-frame").addEventListener("click", calibrationClick);
  el("modal-cancel").addEventListener("click", closePicker);
  el("btn-upload").addEventListener("click", () => el("video-upload").click());
  el("video-upload").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) uploadVideo(e.target.files[0]);
    e.target.value = "";
  });
  el("source-modal").addEventListener("click", (e) => {
    if (e.target === el("source-modal")) closePicker();
  });
  el("event-edit-cancel").addEventListener("click", closeEventEditor);
  el("event-edit-save").addEventListener("click", saveEventEditor);
  el("event-modal").addEventListener("click", (e) => {
    if (e.target === el("event-modal")) closeEventEditor();
  });
  el("comparison-clear").addEventListener("click", () => {
    compareBaseline = null;
    el("comparison-card").style.display = "none";
    el("comparison-body").innerHTML = "";
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePicker();
  });

  function openPicker() {
    const grid = el("source-grid"), state = el("source-state");
    grid.innerHTML = ""; grid.style.display = "none";
    state.style.display = "block";
    state.innerHTML = '<div class="spinner"></div>Finding sources…';
    el("source-modal").classList.add("open");

    Promise.all([
      fetch("/tactical/videos").then((r) => r.json()).catch(() => ({ videos: [] })),
      fetch("/sources").then((r) => r.json()).catch(() => ({ windows: [], displays: [] })),
      fetch("/tactical/health").then((r) => r.json()).catch(() => ({ ready:false })),
    ]).then(([vids, src, health]) => {
      state.style.display = "none";
      grid.style.display = "grid";

      const videos = vids.videos || [];
      if (health.ready) {
        grid.appendChild(section(
          "Local system ready",
          `Models installed · ${health.free_gb} GB free · no account required`
        ));
      } else {
        grid.appendChild(warn(
          "<b>Setup needs attention.</b> Run <code>./scripts/diagnose.py</code> for exact local checks."
        ));
      }
      grid.appendChild(section("Your video", "Copied into the app's local uploads folder; nothing is sent to a cloud service."));
      const upload = card(
        '<div class="src-icon film"></div><div class="src-name">Upload from this computer</div><div class="src-meta">MP4, MOV, MKV, AVI, or WebM · up to 5 GB</div>',
        () => el("video-upload").click()
      );
      grid.appendChild(upload);
      if (videos.length) {
        grid.appendChild(section("Video files", "Most reliable. Runs start to finish without needing anything visible on screen."));
        videos.forEach((v) => grid.appendChild(videoCard(v)));
      }

      if (src.permission === false) {
        grid.appendChild(warn(
          "<b>Screen Recording permission is off.</b> macOS will hand back your desktop wallpaper with all windows removed, so nothing will be detected. " +
          "Grant it in System Settings → Privacy &amp; Security → Screen &amp; System Audio Recording, then restart the app running the server."
        ));
      }

      const wins = src.windows || [], disps = src.displays || [];
      if (wins.length) {
        grid.appendChild(section("Windows", "Includes windows on other Spaces. The chosen window is raised on start and must stay visible while analysing."));
        wins.forEach((w) => grid.appendChild(winCard(w)));
      }
      if (disps.length) {
        grid.appendChild(section("Entire screen", ""));
        disps.forEach((d) => grid.appendChild(dispCard(d)));
      }
      if (!videos.length && !wins.length && !disps.length) {
        grid.style.display = "none";
        state.style.display = "block";
        state.innerHTML = "No sources found. Put a match video in the project's <code>data/</code> folder.";
      }
    });
  }

  function closePicker() { el("source-modal").classList.remove("open"); }

  function section(title, sub) {
    const d = document.createElement("div");
    d.className = "section-label";
    d.innerHTML = title + (sub ? `<span class="section-sub">${sub}</span>` : "");
    return d;
  }
  function warn(html) {
    const d = document.createElement("div");
    d.className = "perm-warn"; d.innerHTML = html; return d;
  }
  function card(inner, onClick) {
    const b = document.createElement("button");
    b.className = "src-card"; b.innerHTML = inner;
    b.addEventListener("click", onClick);
    return b;
  }
  function videoCard(v) {
    return card(
      `<div class="src-icon film"></div>
       <div class="src-name">${esc(v.name)}</div>
       <div class="src-meta">${v.size_mb} MB</div>`,
      () => begin({ video_path: v.path })
    );
  }

  function uploadVideo(file) {
    const grid = el("source-grid"), state = el("source-state");
    grid.style.display = "none"; state.style.display = "block";
    state.textContent = `Uploading ${file.name}… 0%`;
    const form = new FormData(); form.append("video", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/tactical/upload");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) state.textContent = `Uploading ${file.name}… ${Math.round(100 * e.loaded / e.total)}%`;
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300 && data.ok) begin({ video_path: data.video.path });
      else { state.textContent = data.error || "Upload failed."; }
    };
    xhr.onerror = () => { state.textContent = "Upload failed. Check that the server is still running."; };
    xhr.send(form);
  }
  function winCard(w) {
    // `on_screen` is false for a minimised window or one on another Space.
    // macOS draws no pixels for those, so they cannot be captured until
    // raised — starting analysis brings the window forward automatically.
    const hidden = w.on_screen === false;
    const note = hidden
      ? '<div class="src-meta warn-text">Not on screen — will be brought to the front</div>'
      : "";
    return card(
      `<div class="src-icon mon"></div>
       <div class="src-name">${esc(w.app)}</div>
       <div class="src-meta">${esc(trunc(w.title || w.app, 34))}</div>
       <div class="src-meta">${w.width} × ${w.height}</div>
       ${note}`,
      () => begin({ window_id: w.id })
    );
  }
  function dispCard(d) {
    return card(
      `<div class="src-icon mon"></div>
       <div class="src-name">Display ${d.index}${d.primary ? ' <span class="badge">Primary</span>' : ""}</div>
       <div class="src-meta">${d.width} × ${d.height}</div>`,
      () => begin({ display_index: d.index })
    );
  }

  // ── Session ──────────────────────────────────────────────────────────
  async function begin(source) {
    closePicker();
    const btn = el("btn-start");
    btn.disabled = true; btn.textContent = "Starting…";
    setNotice("Loading models. First run downloads them, which can take a minute.");
    try {
      const res = await fetch("/tactical/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign({ target_fps: 6 }, source)),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Failed to start");
      running = true;
      paused = false;
      currentSessionId = data.session_id || null;
      playerLabels = {};
      el("analytics-card").style.display = "none";
      el("timeline-review").style.display = "none";
      btn.style.display = "none";
      el("btn-stop").style.display = "";
      el("btn-pause").style.display = "";
      el("btn-calibrate").style.display = "";
      el("btn-pause").textContent = "Pause";
      if (data.duration_s) el("job-progress").style.display = "block";
      el("source-label").textContent = data.source || "";
      setStatus(true);
      setNotice("");
      connect();
    } catch (err) {
      btn.disabled = false; btn.textContent = "Start Analysis";
      setNotice("Could not start: " + err.message, true);
    }
  }

  async function stopSession() {
    el("btn-stop").disabled = true;
    try {
      const res = await fetch("/tactical/stop", { method: "POST" });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Failed to stop");
      currentSessionId = data.session_id || currentSessionId;
    } catch (err) {
      el("btn-stop").disabled = false;
      setNotice("Could not stop analysis: " + err.message, true);
      return;
    }
    setIdleUi();
    loadHistory();
    if (currentSessionId) loadSessionTimeline(currentSessionId);
  }

  async function togglePause() {
    if (!running) return;
    const button = el("btn-pause");
    button.disabled = true;
    try {
      const endpoint = paused ? "/tactical/resume" : "/tactical/pause";
      const res = await fetch(endpoint, { method:"POST" });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Could not change job state");
      paused = !!data.paused;
      button.textContent = paused ? "Resume" : "Pause";
      el("status-text").textContent = paused ? "Paused" : "Analysing";
    } catch (err) {
      setNotice(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function setIdleUi() {
    running = false;
    paused = false;
    if (ws) { ws.close(); ws = null; }
    el("btn-stop").disabled = false;
    el("btn-stop").style.display = "none";
    el("btn-pause").style.display = "none";
    el("btn-calibrate").style.display = "none";
    clearCalibration();
    el("job-progress").style.display = "none";
    const b = el("btn-start");
    b.style.display = ""; b.disabled = false; b.textContent = "Start Analysis";
    setStatus(false);
    el("source-label").textContent = "";
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/tactical`);
    ws.onopen = () => { const t = setInterval(() => { if (ws && ws.readyState === 1) ws.send("p"); else clearInterval(t); }, 1000); };
    ws.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        if (payload.type === "error") {
          setIdleUi();
          setNotice(payload.error || "Analysis stopped unexpectedly.", true);
          return;
        }
        if (payload.type === "complete") {
          setIdleUi();
          setNotice("Video analysis complete.");
          loadHistory();
          if (currentSessionId) loadSessionTimeline(currentSessionId);
          return;
        }
        render(payload);
      } catch (_) {}
    };
    ws.onclose = () => { if (running) setTimeout(connect, 1000); };
  }

  // ── Render ───────────────────────────────────────────────────────────
  function render(p) {
    const pitch = p.pitch || {};
    currentPitch = pitch;
    PITCH_L = pitch.pitch_length || PITCH_L;
    PITCH_W = pitch.pitch_width || PITCH_W;

    if (p.frame_b64) {
      const img = el("video-frame");
      img.src = "data:image/jpeg;base64," + p.frame_b64;
      // Must be an explicit value: setting "" removes the inline style and
      // lets the stylesheet's `#video-frame { display: none }` win again.
      img.style.display = "block";
      el("video-empty").style.display = "none";
    }

    el("frame-counter").textContent = `Frame ${p.frame_id}`;
    el("fps-chip").textContent = `${p.fps ?? 0} fps`;
    el("elapsed").textContent = fmtTime(p.elapsed_s || 0);
    if (p.progress_pct != null) {
      const progress = Math.max(0, Math.min(100, p.progress_pct));
      el("job-progress").style.display = "block";
      el("job-progress-text").textContent = `${progress.toFixed(1)}%`;
      el("job-progress-fill").style.width = `${progress}%`;
    }

    if (p.status && p.status !== "ok") {
      setNotice(p.status, true);
    } else {
      setNotice("");
    }

    const cal = el("cal-chip");
    if (pitch.saved) {
      cal.textContent = "Saved pitch state";
      cal.className = "chip ok";
    } else if (pitch.calibrated) {
      cal.textContent = pitch.calibration_mode === "manual"
        ? "Manual calibration" : `Calibrated · ${pitch.n_keypoints} landmarks`;
      cal.className = "chip ok";
    } else {
      cal.textContent = "Locating pitch…";
      cal.className = "chip warn";
    }

    drawPitch(pitch.concepts);
    drawShapes(pitch.concepts);
    drawPeople(pitch);

    const c = pitch.counts || {};
    el("n-players").textContent = c.player ?? 0;
    el("n-gk").textContent = c.goalkeeper ?? 0;
    el("n-ref").textContent = c.referee ?? 0;
    el("n-ball").textContent = c.ball ?? 0;

    const tc = pitch.team_counts || {};
    el("count-a").textContent = tc["0"] ?? 0;
    el("count-b").textContent = tc["1"] ?? 0;
    const un = tc["-1"] ?? 0;
    el("count-u").textContent = un;
    el("row-unassigned").style.display = un > 0 ? "" : "none";
    const teamPending = el("team-pending");
    teamPending.style.display = pitch.team_ready ? "none" : "";
    teamPending.textContent = pitch.team_status === "unavailable"
      ? "Team classification is unavailable; player tracking continues."
      : (String(pitch.team_status || "").includes("siglip")
          ? "Learning teams with the optional SigLIP model…"
          : "Learning kit colours locally from the opening frames…");

    el("sit-count").textContent = pitch.situations_stored ?? 0;
    renderConcepts(pitch.concepts);
    renderIntelligence(pitch.intelligence || {});

    const t = p.timings_ms || {};
    el("timings").textContent = `detect ${t.detect ?? "–"} · track ${t.track ?? "–"} · team ${t.team ?? "–"} · pitch ${t.pitch ?? "–"} ms`;
  }

  function toggleCalibration() {
    if (!running) return;
    if (calibrating) { clearCalibration(); setNotice(""); return; }
    calibrating = true; calibrationPoints = [];
    el("btn-calibrate").textContent = "Cancel correction";
    setNotice("Click four visible pitch corners: near-left, near-right, far-right, then far-left.");
  }

  function clearCalibration() {
    calibrating = false; calibrationPoints = [];
    el("btn-calibrate").textContent = "Correct pitch";
    document.querySelectorAll(".calibration-marker").forEach((marker) => marker.remove());
  }

  async function calibrationClick(event) {
    if (!calibrating) return;
    const img = el("video-frame"), rect = img.getBoundingClientRect();
    const naturalW = img.naturalWidth || rect.width, naturalH = img.naturalHeight || rect.height;
    const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
    const shownW = naturalW * scale, shownH = naturalH * scale;
    const offsetX = (rect.width - shownW) / 2, offsetY = (rect.height - shownH) / 2;
    const localX = event.clientX - rect.left - offsetX;
    const localY = event.clientY - rect.top - offsetY;
    if (localX < 0 || localY < 0 || localX > shownW || localY > shownH) return;
    calibrationPoints.push({x:localX / shownW, y:localY / shownH});
    const marker = document.createElement("span"); marker.className = "calibration-marker";
    marker.textContent = String(calibrationPoints.length);
    marker.style.left = `${event.clientX - img.parentElement.getBoundingClientRect().left}px`;
    marker.style.top = `${event.clientY - img.parentElement.getBoundingClientRect().top}px`;
    img.parentElement.appendChild(marker);
    if (calibrationPoints.length < 4) return;
    const points = calibrationPoints.slice();
    clearCalibration();
    try {
      const res = await fetch("/tactical/calibrate", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({points}),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Calibration failed");
      setNotice(`Manual pitch correction applied · ${data.reprojection_error_m} m corner error.`);
    } catch (err) { setNotice(err.message, true); }
  }

  function renderConcepts(c) {
    const box = el("concepts-body"), none = el("concepts-none");
    if (!c || !c.teams || (!c.teams["0"] && !c.teams["1"])) {
      box.style.display = "none"; none.style.display = ""; return;
    }
    box.style.display = ""; none.style.display = "none";

    const ctrl = c.control || {};
    const a = ctrl["0"] ?? 50, b = ctrl["1"] ?? 50;
    el("ctrl-a").style.width = a + "%";
    el("ctrl-b").style.width = b + "%";
    el("ctrl-a-pct").textContent = a + "%";
    el("ctrl-b-pct").textContent = b + "%";

    const ta = c.teams["0"], tb = c.teams["1"];
    const m = (v) => (v == null ? "–" : v + " m");
    el("cmp-a").textContent = m(ta && ta.compactness_m);
    el("cmp-b").textContent = m(tb && tb.compactness_m);
    el("wid-ab").textContent = `${ta ? ta.width_m : "–"} / ${tb ? tb.width_m : "–"}`;
    el("lin-ab").textContent = `${ta ? ta.line_height_m : "–"} / ${tb ? tb.line_height_m : "–"}`;
    el("press").textContent = m(c.pressing_m);
  }

  // ── Similar situations ───────────────────────────────────────────────
  el("btn-similar").addEventListener("click", async () => {
    const box = el("similar-results"), btn = el("btn-similar");
    btn.disabled = true; btn.textContent = "Searching…";
    box.innerHTML = "";
    try {
      const data = await (await fetch("/tactical/similar?k=4")).json();
      if (!data.ok) {
        box.innerHTML = `<div class="muted">${esc(data.error || "No match")}</div>`;
      } else if (!data.results.length) {
        box.innerHTML = `<div class="muted">No comparable earlier moments yet. Let it run a little longer.</div>`;
      } else {
        data.results.forEach((r) => box.appendChild(simRow(r)));
      }
    } catch (err) {
      box.innerHTML = `<div class="muted">Request failed.</div>`;
    } finally {
      btn.disabled = false; btn.textContent = "Find similar to now";
    }
  });

  function simRow(r) {
    const row = document.createElement("div");
    row.className = "sim-row";
    const cv = document.createElement("canvas");
    cv.width = 92; cv.height = 60; cv.className = "sim-pitch";
    miniPitch(cv, (r.meta && r.meta.players) || []);
    const meta = document.createElement("div");
    meta.innerHTML = `<div class="sim-pct">${Math.round(r.similarity * 100)}% match</div>
                      <div class="muted">at ${fmtTime(r.meta.t)} · frame ${r.meta.frame_id}</div>`;
    row.appendChild(cv); row.appendChild(meta);
    return row;
  }

  function miniPitch(cv, players) {
    const c = cv.getContext("2d");
    c.fillStyle = "#0e2a19"; c.fillRect(0, 0, cv.width, cv.height);
    const m = 3;
    c.strokeStyle = "rgba(255,255,255,0.3)"; c.lineWidth = 1;
    c.strokeRect(m, m, cv.width - 2 * m, cv.height - 2 * m);
    c.beginPath(); c.moveTo(cv.width / 2, m); c.lineTo(cv.width / 2, cv.height - m); c.stroke();
    for (const p of players) {
      const x = m + (p.x / PITCH_L) * (cv.width - 2 * m);
      const y = m + (1 - p.y / PITCH_W) * (cv.height - 2 * m);
      c.beginPath(); c.arc(x, y, 2.4, 0, Math.PI * 2);
      c.fillStyle = TEAM_COLORS[String(p.team)] || TEAM_COLORS["-1"]; c.fill();
    }
  }

  // ── Persistent session history + exports ─────────────────────────────
  async function loadHistory() {
    const box = el("history-list");
    try {
      const data = await (await fetch("/tactical/sessions?limit=12")).json();
      const sessions = data.sessions || [];
      box.innerHTML = "";
      if (!sessions.length) {
        box.innerHTML = '<div class="muted">No sessions yet.</div>';
        return;
      }
      sessions.forEach((s) => box.appendChild(historyRow(s)));
    } catch (_) {
      box.innerHTML = '<div class="muted">Could not load local history.</div>';
    }
  }

  function historyRow(s) {
    const row = document.createElement("div"); row.className = "history-row";
    const date = new Date((s.started_at || 0) * 1000).toLocaleString([], { dateStyle:"short", timeStyle:"short" });
    row.innerHTML =
      `<div class="history-head"><div class="history-name">${esc(s.source_name)}</div><span class="chip ${s.status === "complete" ? "ok" : "warn"}">${esc(s.status)}</span></div>` +
      `<div class="muted">${date} · ${s.frames || 0} frames · ${fmtTime(s.elapsed_s)} · ${s.snapshots || 0} snapshots · ${s.events || 0} events</div>` +
      '<div class="history-actions"><button class="btn view-session">View last state</button><button class="btn build-report">Build exports</button><button class="btn build-video">Overlay video</button></div>' +
      '<div class="download-links"></div>';
    row.querySelector(".view-session").addEventListener("click", () => viewSession(s.id));
    row.querySelector(".build-report").addEventListener("click", (e) => buildExports(s.id, e.target, row));
    const videoButton = row.querySelector(".build-video");
    videoButton.textContent = s.export_status === "running"
      ? "Exporting…" : s.annotated_path ? "Rebuild video" : "Overlay video";
    videoButton.addEventListener("click", () => buildAnnotatedVideo(s.id, videoButton, row));
    const compare = document.createElement("button");
    compare.className = "btn compare-session"; compare.textContent = "Compare";
    compare.addEventListener("click", () => compareSession(s.id, s.source_name));
    row.querySelector(".history-actions").appendChild(compare);
    const links = row.querySelector(".download-links");
    links.insertAdjacentHTML("beforeend", `<a href="/tactical/sessions/${s.id}/download/dataset">dataset JSON</a>`);
    for (const kind of ["report", "pdf", "tracking", "metrics", "events", "annotated"]) {
      if (s[`${kind}_path`]) links.insertAdjacentHTML("beforeend", `<a href="/tactical/sessions/${s.id}/download/${kind}">${kind}</a>`);
    }
    if (s.export_status === "running") pollAnnotatedVideo(s.id, videoButton, row);
    return row;
  }

  async function viewSession(id) {
    try {
      const data = await getSessionData(id, true);
      const snaps = data.snapshots || [], snap = snaps[snaps.length - 1];
      if (!snap) throw new Error("This session has no saved pitch states.");
      currentSessionId = id;
      renderAnalytics(data.analytics || {}, data.player_labels || [], snaps, id);
      renderSavedMoment(snap, snaps.length);
      renderTimeline(data.events || [], data.session);
      configureScrubber(data, snaps.length - 1);
      setNotice(`Reviewing saved session ${id}, frame ${snap.frame_id}.`);
    } catch (err) { setNotice(err.message, true); }
  }

  async function getSessionData(id, refresh) {
    if (!refresh && sessionCache.has(id)) return sessionCache.get(id);
    const res = await fetch(`/tactical/sessions/${id}`);
    if (!res.ok) throw new Error("Could not load this session.");
    const data = await res.json(); sessionCache.set(id, data); return data;
  }

  function renderSavedMoment(snap, total) {
    const pitch = {
      pitch_length:PITCH_L, pitch_width:PITCH_W, calibrated:true, saved:true,
      n_keypoints:0, players:snap.players, ball:snap.ball,
      counts:snap.counts, concepts:snap.concepts, team_counts:{"0":0,"1":0,"-1":0},
      team_ready:true, situations_stored:total, intelligence:snap.intelligence || {},
    };
    (snap.players || []).filter((p) => p.role === "player").forEach((p) => {
      const k = String(p.team); pitch.team_counts[k] = (pitch.team_counts[k] || 0) + 1;
    });
    render({ pitch, frame_id:snap.frame_id,
      elapsed_s:snap.source_time_s ?? snap.time_s, fps:0, timings_ms:{} });
  }

  async function viewSessionMoment(id, frameId) {
    if (running) {
      setNotice("Stop analysis before reviewing an earlier event.");
      return;
    }
    try {
      const data = await getSessionData(id, false), snaps = data.snapshots || [];
      if (!snaps.length) throw new Error("No saved pitch state is available for this event.");
      const snap = snaps.reduce((best, item) =>
        Math.abs(item.frame_id - frameId) < Math.abs(best.frame_id - frameId) ? item : best
      );
      renderSavedMoment(snap, snaps.length);
      setNotice(`Event at frame ${frameId}; showing nearest saved state at frame ${snap.frame_id}.`);
    } catch (err) { setNotice(err.message, true); }
  }

  async function loadSessionTimeline(id) {
    try {
      const data = await getSessionData(id, true);
      renderAnalytics(
        data.analytics || {}, data.player_labels || [], data.snapshots || [], id
      );
      renderTimeline(data.events || [], data.session);
      configureScrubber(data, Math.max(0, (data.snapshots || []).length - 1));
    } catch (_) {}
  }

  function renderAnalytics(data, labels, snapshots, sessionId) {
    const teams = data.teams || {}, a = teams["0"] || {}, b = teams["1"] || {};
    playerLabels = Object.fromEntries((labels || []).map((label) => [String(label.track_id), label]));
    const value = (v, suffix="") => v == null ? "–" : `${v}${suffix}`;
    const formation = (team) => (team.top_formations || []).map((f) => `${esc(f.name)} (${f.samples})`).join(", ") || "–";
    el("analytics-card").style.display = "";
    el("analytics-body").innerHTML =
      '<div class="analytics-teams">' +
      teamAnalytics("Team A", "var(--teamA)", a, value, formation(a)) +
      teamAnalytics("Team B", "var(--teamB)", b, value, formation(b)) +
      '</div>' +
      '<div class="analytics-canvases"><canvas id="heatmap-a" width="150" height="76"></canvas><canvas id="heatmap-b" width="150" height="76"></canvas></div>' +
      '<canvas class="momentum-canvas" id="momentum-chart" width="320" height="70"></canvas>' +
      '<div class="muted" style="margin-top:4px">Local dominance index: control and possession, Team A above the line.</div>' +
      `<div class="analytics-events">${eventSummary(data.event_counts || {})}</div>` +
      '<div class="player-labels" id="player-labels"></div>';
    drawAnalyticsHeatmap(el("heatmap-a"), a.heatmap || [], TEAM_RGB["0"]);
    drawAnalyticsHeatmap(el("heatmap-b"), b.heatmap || [], TEAM_RGB["1"]);
    drawMomentum(el("momentum-chart"), data.momentum || []);
    renderPlayerLabels(labels || [], snapshots || [], sessionId);
  }

  function renderPlayerLabels(labels, snapshots, sessionId) {
    const box = el("player-labels");
    const latest = snapshots.length ? snapshots[snapshots.length - 1] : {};
    const players = (latest.players || []).filter((player) => player.role === "player");
    const unique = [...new Map(players.map((player) => [player.id, player])).values()]
      .sort((a, b) => a.team - b.team || a.id - b.id);
    if (!unique.length) { box.innerHTML = ""; return; }
    const byId = Object.fromEntries(labels.map((label) => [String(label.track_id), label]));
    box.innerHTML = `<details><summary>Player labels (${unique.length})</summary><div id="player-label-rows"></div></details>`;
    const rows = box.querySelector("#player-label-rows");
    unique.forEach((player) => {
      const saved = byId[String(player.id)] || {};
      const row = document.createElement("div"); row.className = "player-label-row";
      row.innerHTML = `<span style="color:${TEAM_COLORS[String(player.team)]}">ID ${player.id}</span>` +
        `<input class="player-name" aria-label="Name for player ${player.id}" placeholder="Name" value="${esc(saved.display_name || "")}">` +
        `<input class="player-number" aria-label="Shirt number for player ${player.id}" type="number" min="0" max="99" placeholder="#" value="${saved.shirt_number == null ? "" : saved.shirt_number}">` +
        '<button class="btn">Save</button>';
      row.querySelector("button").addEventListener("click", async () => {
        const name = row.querySelector(".player-name").value.trim();
        const number = row.querySelector(".player-number").value;
        try {
          const res = await fetch(`/tactical/sessions/${sessionId}/players/${player.id}`, {
            method:"PUT", headers:{"Content-Type":"application/json"},
            body:JSON.stringify({
              display_name:name || null,
              shirt_number:number === "" ? null : Number(number),
              team:player.team,
            }),
          });
          const data = await res.json();
          if (!res.ok || !data.ok) throw new Error(data.error || "Could not save player label");
          sessionCache.delete(sessionId);
          timelineSignature = "";
          await loadSessionTimeline(sessionId);
        } catch (err) { setNotice(err.message, true); }
      });
      rows.appendChild(row);
    });
  }

  function teamAnalytics(label, color, team, value, formations) {
    return `<div class="analytics-team"><h4 style="color:${color}">${label}</h4><div class="kv">` +
      `<span class="k">Possession samples</span><span class="v">${value(team.possession_sample_pct, "%")}</span>` +
      `<span class="k">Average control</span><span class="v">${value(team.avg_control_pct, "%")}</span>` +
      `<span class="k">Final-third presence</span><span class="v">${value(team.final_third_presence_pct, "%")}</span>` +
      `<span class="k">Compactness</span><span class="v">${value(team.avg_compactness_m, " m")}</span>` +
      `<span class="k">Passes / recoveries</span><span class="v">${value(team.passes)} / ${value(team.turnovers_won)}</span>` +
      `<span class="k">Common shapes</span><span class="v">${formations}</span></div></div>`;
  }

  function eventSummary(counts) {
    const entries = Object.entries(counts);
    return entries.length
      ? "Reviewed timeline: " + entries.map(([kind, n]) => `${n} ${esc(kind.replaceAll("_", " "))}`).join(" · ")
      : "No accepted events yet.";
  }

  function drawAnalyticsHeatmap(canvas, grid, rgb) {
    const c = canvas.getContext("2d"), rows = grid.length, cols = rows ? grid[0].length : 0;
    c.fillStyle = "#0e2a19"; c.fillRect(0, 0, canvas.width, canvas.height);
    if (!rows || !cols) return;
    const max = Math.max(1, ...grid.flat());
    for (let r = 0; r < rows; r++) for (let col = 0; col < cols; col++) {
      const alpha = .08 + .82 * grid[r][col] / max;
      c.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
      c.fillRect(col * canvas.width / cols, canvas.height - (r + 1) * canvas.height / rows,
        canvas.width / cols + .5, canvas.height / rows + .5);
    }
    c.strokeStyle = "rgba(255,255,255,.35)"; c.strokeRect(.5, .5, canvas.width - 1, canvas.height - 1);
    c.beginPath(); c.moveTo(canvas.width / 2, 0); c.lineTo(canvas.width / 2, canvas.height); c.stroke();
  }

  function drawMomentum(canvas, points) {
    const c = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
    c.clearRect(0, 0, w, h); c.strokeStyle = "rgba(255,255,255,.2)";
    c.beginPath(); c.moveTo(0, h / 2); c.lineTo(w, h / 2); c.stroke();
    if (points.length < 2) return;
    const maxTime = Math.max(1, ...points.map((p) => p.time_s || 0));
    c.beginPath();
    points.forEach((point, i) => {
      const x = w * (point.time_s || 0) / maxTime;
      const y = h / 2 - (point.team_a_dominance || 0) * (h * .45 / 100);
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.strokeStyle = "#ff6f3c"; c.lineWidth = 2; c.stroke();
  }

  async function compareSession(id, name) {
    if (!compareBaseline) {
      compareBaseline = {id, name};
      el("comparison-card").style.display = "";
      el("comparison-body").innerHTML = `<div class="muted">Baseline: ${esc(name)}. Choose Compare on another session.</div>`;
      return;
    }
    if (compareBaseline.id === id) {
      setNotice("Choose a different session to compare.", true);
      return;
    }
    try {
      const res = await fetch(`/tactical/compare?session_a=${compareBaseline.id}&session_b=${id}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not compare sessions");
      renderComparison(data, compareBaseline.name, name);
      compareBaseline = null;
    } catch (err) { setNotice(err.message, true); }
  }

  function renderComparison(data, firstName, secondName) {
    const deltas = (data.comparison || {}).teams || {};
    const line = (label, value, unit="", lowerIsBetter=false) => {
      const direction = lowerIsBetter && value != null ? -value : value;
      const cls = direction > 0 ? "delta-up" : direction < 0 ? "delta-down" : "";
      const shown = value == null ? "–" : `${value > 0 ? "+" : ""}${value}${unit}`;
      return `<span class="k">${label}</span><span class="v ${cls}">${shown}</span>`;
    };
    const team = (label, values) => `<div class="analytics-team"><h4>${label}</h4><div class="kv">` +
      line("Possession", values.possession_sample_pct, "%") +
      line("Control", values.avg_control_pct, "%") +
      line("Final-third presence", values.final_third_presence_pct, "%") +
      line("Compactness", values.avg_compactness_m, " m", true) +
      line("Passes", values.passes) + '</div></div>';
    el("comparison-card").style.display = "";
    el("comparison-body").innerHTML =
      `<div class="muted" style="margin-bottom:7px">${esc(secondName)} minus ${esc(firstName)}</div><div class="analytics-teams">` +
      team("Team A", deltas["0"] || {}) + team("Team B", deltas["1"] || {}) + '</div>';
  }

  function configureScrubber(data, selected) {
    const snaps = data.snapshots || [], wrap = el("timeline-review"), input = el("timeline-scrubber");
    if (!snaps.length || running) { wrap.style.display = "none"; return; }
    wrap.style.display = "block";
    input.min = "0"; input.max = String(snaps.length - 1); input.value = String(selected);
    const show = () => {
      const snap = snaps[Number(input.value)] || snaps[0];
      renderSavedMoment(snap, snaps.length);
      const shownTime = snap.source_time_s ?? snap.time_s;
      el("timeline-position").textContent = `${fmtTime(shownTime)} · frame ${snap.frame_id} · saved state ${Number(input.value) + 1}/${snaps.length}`;
    };
    input.oninput = show;
    const snap = snaps[selected] || snaps[0];
    const shownTime = snap.source_time_s ?? snap.time_s;
    el("timeline-position").textContent = `${fmtTime(shownTime)} · frame ${snap.frame_id} · saved state ${selected + 1}/${snaps.length}`;
  }

  function renderIntelligence(intel) {
    const possession = intel.possession;
    if (possession) {
      const team = possession.team === 0 ? "Team A" : "Team B";
      el("possession").textContent = `${team} · player ${possession.player_id}`;
      el("possession").style.color = TEAM_COLORS[String(possession.team)];
      el("possession-confidence").style.width = `${Math.round(100 * (possession.confidence || 0))}%`;
      el("possession-confidence").style.background = TEAM_COLORS[String(possession.team)];
    } else {
      el("possession").textContent = "Possession unresolved";
      el("possession").style.color = "";
      el("possession-confidence").style.width = "0%";
    }
    const phase = intel.phase || {};
    el("phase").textContent = phase.label && phase.label !== "transition"
      ? `${phase.label} · ${phase.zone || ""}` : "Transition / waiting for ball evidence";
    const forms = intel.formations || {};
    for (const [team, id] of [["0", "formation-a"], ["1", "formation-b"]]) {
      const f = forms[team];
      el(id).textContent = !f || f.name === "insufficient data"
        ? "–" : `${f.name} · ${Math.round(100 * (f.confidence || 0))}%`;
    }
    if (intel.events) renderTimeline(intel.events, { id:currentSessionId, source_type:"live" });
  }

  function renderTimeline(events, session) {
    const signature = `${session && session.id}:${session && session.source_type}:` +
      events.map((e) => `${e.id || e.event_seq}:${e.review_status || "inferred"}:${e.label}:${e.source_time_s}:${e.clip_url || ""}`).join(",");
    if (signature === timelineSignature) return;
    timelineSignature = signature;
    const box = el("event-list"); box.innerHTML = "";
    if (!events.length) {
      box.innerHTML = '<div class="muted">Confirmed possession changes and event candidates will appear here.</div>';
      return;
    }
    events.slice(-10).reverse().forEach((event) => {
      const row = document.createElement("div");
      row.className = `event-row ${event.review_status || "inferred"}`;
      const teamColor = TEAM_COLORS[String(event.team)] || "#6b7280";
      row.innerHTML = `<span class="event-team" style="background:${teamColor}"></span>` +
        `<span class="event-time">${fmtTime(event.source_time_s ?? event.time_s)}</span>` +
        `<span class="event-body"><div class="event-label">${esc(event.label)}</div>` +
        `<div class="event-detail">${esc(eventDetail(event))} · ${Math.round(100 * (event.confidence || 0))}% confidence</div></span>`;
      row.addEventListener("click", () => viewSessionMoment((session && session.id) || currentSessionId, event.frame_id));
      if (event.clip_url) {
        const link = document.createElement("a"); link.className = "clip-action";
        link.href = event.clip_url; link.target = "_blank"; link.textContent = "play clip";
        link.addEventListener("click", (e) => e.stopPropagation()); row.appendChild(link);
      } else if (event.id && session && session.source_type === "video") {
        const clip = document.createElement("button"); clip.className = "clip-action";
        clip.textContent = "make clip";
        clip.addEventListener("click", (e) => { e.stopPropagation(); buildEventClip(session.id, event.id, clip); });
        row.appendChild(clip);
      }
      if (event.id && session && session.id && session.source_type !== "live") {
        const actions = document.createElement("span"); actions.className = "review-actions";
        const confirm = document.createElement("button"); confirm.className = "review-action";
        confirm.textContent = "✓"; confirm.title = "Confirm event";
        confirm.addEventListener("click", (e) => {
          e.stopPropagation(); reviewEvent(session.id, event.id, {review_status:"confirmed"});
        });
        const reject = document.createElement("button"); reject.className = "review-action";
        reject.textContent = "×"; reject.title = "Reject event";
        reject.addEventListener("click", (e) => {
          e.stopPropagation(); reviewEvent(session.id, event.id, {review_status:"rejected"});
        });
        const edit = document.createElement("button"); edit.className = "review-action";
        edit.textContent = "edit"; edit.title = "Edit event";
        edit.addEventListener("click", (e) => { e.stopPropagation(); openEventEditor(session.id, event); });
        actions.append(confirm, reject, edit); row.appendChild(actions);
      }
      box.appendChild(row);
    });
  }

  async function reviewEvent(sessionId, eventId, changes) {
    try {
      const res = await fetch(`/tactical/sessions/${sessionId}/events/${eventId}`, {
        method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify(changes),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Could not update event");
      sessionCache.delete(sessionId); timelineSignature = "";
      await loadSessionTimeline(sessionId);
    } catch (err) { setNotice(err.message, true); }
  }

  function openEventEditor(sessionId, event) {
    editingSessionId = sessionId; editingEvent = event;
    el("event-edit-label").value = event.label || "";
    el("event-edit-type").value = event.type || "possession_start";
    el("event-edit-status").value = event.review_status || "inferred";
    el("event-edit-team").value = event.team == null ? "" : String(event.team);
    el("event-edit-player").value = event.player_id == null ? "" : String(event.player_id);
    el("event-edit-time").value = event.source_time_s == null ? "" : String(event.source_time_s);
    el("event-edit-notes").value = event.notes || "";
    el("event-modal").classList.add("open");
  }

  function closeEventEditor() {
    el("event-modal").classList.remove("open");
    editingEvent = null; editingSessionId = null;
  }

  async function saveEventEditor() {
    if (!editingEvent || !editingSessionId) return;
    const label = el("event-edit-label").value.trim();
    if (!label) { setNotice("Event label cannot be empty.", true); return; }
    const team = el("event-edit-team").value;
    const player = el("event-edit-player").value;
    const sourceTime = el("event-edit-time").value;
    const changes = {
      label,
      type:el("event-edit-type").value,
      review_status:el("event-edit-status").value,
      team:team === "" ? null : Number(team),
      player_id:player === "" ? null : Number(player),
      source_time_s:sourceTime === "" ? null : Number(sourceTime),
      notes:el("event-edit-notes").value.trim(),
    };
    const sessionId = editingSessionId, eventId = editingEvent.id;
    closeEventEditor();
    await reviewEvent(sessionId, eventId, changes);
  }

  function eventDetail(event) {
    const d = event.detail || {};
    if (event.type === "pass") return `${playerName(d.from_player_id)} → ${playerName(d.to_player_id)}`;
    if (event.type === "turnover") return `won by ${playerName(event.player_id)}`;
    if (event.type === "carry") return `${d.distance_m || "?"} m by ${playerName(event.player_id)}`;
    if (event.type === "shot_candidate") return `${d.ball_speed_toward_goal_mps || "?"} m/s toward goal`;
    return event.player_id == null ? "ball evidence" : playerName(event.player_id);
  }

  function playerName(id) {
    if (id == null) return "unassigned player";
    const label = playerLabels[String(id)] || {};
    if (label.display_name) return label.shirt_number == null
      ? label.display_name : `${label.display_name} (#${label.shirt_number})`;
    return label.shirt_number == null ? `player ${id}` : `#${label.shirt_number}`;
  }

  async function buildEventClip(sessionId, eventId, button) {
    button.disabled = true; button.textContent = "building…";
    try {
      const res = await fetch(`/tactical/sessions/${sessionId}/events/${eventId}/clip`, { method:"POST" });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Clip export failed");
      sessionCache.delete(sessionId); await loadSessionTimeline(sessionId);
    } catch (err) { setNotice(err.message, true); }
    finally { button.disabled = false; }
  }

  async function buildExports(id, btn, row) {
    btn.disabled = true; btn.textContent = "Building…";
    try {
      const res = await fetch(`/tactical/sessions/${id}/artifacts`, { method:"POST" });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Could not build exports");
      const links = row.querySelector(".download-links"); links.innerHTML = "";
      Object.entries(data.downloads || {}).forEach(([kind, url]) => {
        links.insertAdjacentHTML("beforeend", `<a href="${url}">${kind}</a>`);
      });
      if (!data.report_available) setNotice("CSV exports are ready. A visual report needs resolved team data from a longer session.");
    } catch (err) { setNotice(err.message, true); }
    finally { btn.disabled = false; btn.textContent = "Rebuild exports"; }
  }

  async function buildAnnotatedVideo(id, button, row) {
    button.disabled = true; button.textContent = "Starting…";
    try {
      const res = await fetch(`/tactical/sessions/${id}/annotated-video`, {method:"POST"});
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Could not start video export");
      await pollAnnotatedVideo(id, button, row);
    } catch (err) {
      button.disabled = false; button.textContent = "Overlay video";
      setNotice(err.message, true);
    }
  }

  async function pollAnnotatedVideo(id, button, row) {
    button.disabled = true;
    while (document.body.contains(button)) {
      try {
        const res = await fetch(`/tactical/sessions/${id}/annotated-video`);
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || "Could not read export status");
        if (data.status === "complete") {
          button.disabled = false; button.textContent = "Rebuild video";
          const links = row.querySelector(".download-links");
          if (!links.querySelector('a[href$="/annotated"]')) {
            links.insertAdjacentHTML("beforeend", `<a href="${data.download}">annotated</a>`);
          }
          return;
        }
        if (data.status === "failed" || data.status === "interrupted") {
          throw new Error(data.error || "Video export did not complete");
        }
        button.textContent = `Exporting ${Math.round(data.progress || 0)}%`;
      } catch (err) {
        button.disabled = false; button.textContent = "Overlay video";
        setNotice(err.message, true); return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  // ── Toggles + helpers ────────────────────────────────────────────────
  el("toggle-control").addEventListener("change", (e) => { showControl = e.target.checked; });
  el("toggle-hull").addEventListener("change", (e) => { showHull = e.target.checked; });

  function setStatus(live) {
    el("status-dot").className = live ? "live" : "";
    el("status-text").textContent = live ? "Analysing" : "Idle";
  }
  function setNotice(msg, isError) {
    const n = el("notice");
    if (!msg) { n.style.display = "none"; return; }
    n.style.display = "";
    n.className = "notice" + (isError ? " err" : "");
    n.textContent = msg;
  }
  function fmtTime(s) {
    s = Math.round(s || 0);
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }
  function trunc(s, n) { return s && s.length > n ? s.slice(0, n - 1) + "…" : (s || ""); }
  function esc(s) { const d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }

  // ── Ask the match ────────────────────────────────────────────────────
  el("btn-ask").addEventListener("click", ask);
  el("ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  document.querySelectorAll(".ask-chip").forEach((c) =>
    c.addEventListener("click", () => { el("ask-input").value = c.dataset.q; ask(); }));

  async function ask() {
    const q = el("ask-input").value.trim();
    const box = el("ask-answer"), btn = el("btn-ask");
    if (!q) return;
    btn.disabled = true; btn.textContent = "\u2026";
    box.innerHTML = '<div class="ask-ans muted">Thinking\u2026</div>';
    try {
      const res = await fetch("/tactical/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, pitch: currentPitch }),
      });
      const d = await res.json();
      if (d.answer) {
        const nFacts = (d.facts || "").split("\n").length;
        box.innerHTML =
          '<div class="ask-ans"></div>' +
          '<div class="ask-meta">grounded in ' + nFacts + ' measured facts \u00b7 ' +
          esc(d.backend || "?") + ' \u00b7 <a id="facts-toggle">show the facts it was given</a></div>' +
          '<div class="facts-box" id="facts-box"></div>';
        box.querySelector(".ask-ans").textContent = d.answer;
        el("facts-box").textContent = d.facts || "";
        el("facts-toggle").addEventListener("click", () => {
          const fb = el("facts-box");
          const open = fb.style.display === "block";
          fb.style.display = open ? "none" : "block";
          el("facts-toggle").textContent = open ? "show the facts it was given" : "hide facts";
        });
      } else {
        box.innerHTML = '<div class="ask-ans" style="color:var(--warn)"></div>';
        box.querySelector(".ask-ans").textContent = d.error || "No answer.";
      }
    } catch (e) {
      box.innerHTML = '<div class="ask-ans" style="color:var(--err)">Request failed.</div>';
    } finally {
      btn.disabled = false; btn.textContent = "Ask";
    }
  }

  drawPitch(null);
  loadHistory();
  window.FootballVisionTactical = { render };
})();
