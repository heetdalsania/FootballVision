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
  let PITCH_L = 105, PITCH_W = 68;
  let showControl = true;
  let showHull = true;
  let showTrails = false;

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
        ctx.fillText(String(p.id), cx, cy);
      } else {
        ctx.fillStyle = "#000"; ctx.font = "bold 8px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(role === "referee" ? "R" : "GK", cx, cy);
      }
    }
    if (pitch.ball) {
      const [bx, by] = toCanvas(pitch.ball.x, pitch.ball.y);
      dot(bx, by, 5, "#ffffff");
      ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // ── Source picker ────────────────────────────────────────────────────
  el("btn-start").addEventListener("click", openPicker);
  el("btn-stop").addEventListener("click", stopSession);
  el("modal-cancel").addEventListener("click", closePicker);
  el("source-modal").addEventListener("click", (e) => {
    if (e.target === el("source-modal")) closePicker();
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
    ]).then(([vids, src]) => {
      state.style.display = "none";
      grid.style.display = "grid";

      const videos = vids.videos || [];
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
      btn.style.display = "none";
      el("btn-stop").style.display = "";
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
    await fetch("/tactical/stop", { method: "POST" });
    running = false;
    if (ws) { ws.close(); ws = null; }
    el("btn-stop").disabled = false;
    el("btn-stop").style.display = "none";
    const b = el("btn-start");
    b.style.display = ""; b.disabled = false; b.textContent = "Start Analysis";
    setStatus(false);
    el("source-label").textContent = "";
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/tactical`);
    ws.onopen = () => { const t = setInterval(() => { if (ws && ws.readyState === 1) ws.send("p"); else clearInterval(t); }, 1000); };
    ws.onmessage = (e) => { try { render(JSON.parse(e.data)); } catch (_) {} };
    ws.onclose = () => { if (running) setTimeout(connect, 1000); };
  }

  // ── Render ───────────────────────────────────────────────────────────
  function render(p) {
    const pitch = p.pitch || {};
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

    if (p.status && p.status !== "ok") {
      setNotice(p.status, true);
    } else {
      setNotice("");
    }

    const cal = el("cal-chip");
    if (pitch.calibrated) {
      cal.textContent = `Calibrated · ${pitch.n_keypoints} landmarks`;
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
    el("team-pending").style.display = pitch.team_ready ? "none" : "";

    el("sit-count").textContent = pitch.situations_stored ?? 0;
    renderConcepts(pitch.concepts);

    const t = p.timings_ms || {};
    el("timings").textContent = `detect ${t.detect ?? "–"} · track ${t.track ?? "–"} · team ${t.team ?? "–"} · pitch ${t.pitch ?? "–"} ms`;
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
        body: JSON.stringify({ question: q }),
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
  window.FootballVisionTactical = { render };
})();
