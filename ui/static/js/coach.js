/**
 * FootballVision — Coach Mode WebSocket Client
 * Connects to /ws/coach and renders live analysis data.
 */

(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────────────
  let ws = null;
  let sessionActive = false;
  let lastFrameTime = 0;
  let frameCount = 0;
  let fpsInterval = null;
  const playLog = [];

  // ── DOM refs ───────────────────────────────────────────────────────────
  const el = (id) => document.getElementById(id);
  const videoCanvas   = el("video-canvas");
  const noSignal      = el("no-signal");
  const statusDot     = el("status-dot");
  const statusText    = el("status-text");
  const frameCounter  = el("frame-counter");
  const fpsChip       = el("fps-chip");
  const playChip      = el("play-chip");
  const formationName = el("formation-name");
  const formationConf = el("formation-conf");
  const formationAlts = el("formation-alts");
  const barRun        = el("bar-run");
  const barPass       = el("bar-pass");
  const pctRun        = el("pct-run");
  const pctPass       = el("pct-pass");
  const alertList     = el("alert-list");
  const pcountTotal   = el("pcount-total");
  const pcountPlays   = el("pcount-plays");
  const tendRun       = el("tend-run");
  const tendPass      = el("tend-pass");
  const tendRunPct    = el("tend-run-pct");
  const tendPassPct   = el("tend-pass-pct");
  const playLogEl     = el("play-log");
  const btnStart      = el("btn-start");
  const btnStop       = el("btn-stop");

  // ── Session control ────────────────────────────────────────────────────
  btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    btnStart.textContent = "Starting…";

    const demo = new URLSearchParams(location.search).has("demo");
    try {
      const res = await fetch("/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_demo: demo, coach_fps: 10, fan_fps: 2 }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Failed");
      startSession();
    } catch (err) {
      btnStart.disabled = false;
      btnStart.textContent = "Start Session";
      alert("Could not start session: " + err.message);
    }
  });

  btnStop.addEventListener("click", async () => {
    btnStop.disabled = true;
    await fetch("/session/stop", { method: "POST" });
    stopSession();
    btnStop.disabled = false;
  });

  function startSession() {
    sessionActive = true;
    btnStart.style.display = "none";
    btnStop.style.display = "";
    setStatus(true);
    connectWS();
    startFpsCounter();
  }

  function stopSession() {
    sessionActive = false;
    if (ws) { ws.close(); ws = null; }
    stopFpsCounter();
    setStatus(false);
    btnStart.style.display = "";
    btnStart.disabled = false;
    btnStart.textContent = "Start Session";
    btnStop.style.display = "none";
    videoCanvas.style.display = "none";
    noSignal.style.display = "";
  }

  // ── WebSocket ──────────────────────────────────────────────────────────
  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/coach`);

    ws.onopen = () => {
      console.log("[coach] WebSocket connected");
    };

    ws.onmessage = (evt) => {
      let payload;
      try { payload = JSON.parse(evt.data); } catch { return; }
      renderFrame(payload);
    };

    ws.onclose = () => {
      if (sessionActive) {
        // Auto-reconnect after 1s
        setTimeout(connectWS, 1000);
      }
    };

    ws.onerror = (e) => console.error("[coach] WS error", e);
  }

  // ── Render ─────────────────────────────────────────────────────────────
  function renderFrame(p) {
    frameCount++;

    // Video frame
    if (p.frame_b64) {
      videoCanvas.src = "data:image/jpeg;base64," + p.frame_b64;
      if (videoCanvas.style.display === "none") {
        videoCanvas.style.display = "";
        noSignal.style.display = "none";
      }
    }

    // Frame counter
    frameCounter.textContent = `Frame ${p.frame_id ?? "—"}`;

    // Formation
    if (p.formation) {
      formationName.textContent = p.formation.name || "—";
      formationConf.textContent = `Confidence: ${pct(p.formation.confidence)}`;
      renderAltFormations(p.formation.top_k || []);
    }

    // Play prediction
    const runProb  = p.play_prediction?.run  ?? 0.5;
    const passProb = p.play_prediction?.pass ?? 0.5;
    barRun.style.width  = pct(runProb);
    barPass.style.width = pct(passProb);
    pctRun.textContent  = pct(runProb);
    pctPass.textContent = pct(passProb);

    // Alerts
    renderAlerts(p.alerts || []);

    // Player count
    pcountTotal.textContent = p.player_count ?? "—";
    pcountPlays.textContent = p.play_count ?? "—";
    playChip.textContent    = `Play #${p.play_count ?? "—"}`;

    // Tendency
    const t = p.tendency || {};
    const runT  = t.run  ?? 0;
    const passT = t.pass ?? 0;
    tendRun.style.width  = pct(runT);
    tendPass.style.width = pct(passT);
    tendRunPct.textContent  = pct(runT);
    tendPassPct.textContent = pct(passT);

    // Play log entry on snap
    if (p.snap_detected && p.formation) {
      addPlayLogEntry(p.play_count, p.formation.name, p.play_prediction);
    }
  }

  function renderAltFormations(topK) {
    formationAlts.innerHTML = topK.slice(1, 3).map(f => `
      <div class="formation-alt-row">
        <span style="min-width:110px">${f.name}</span>
        <div class="formation-alt-bar">
          <div class="formation-alt-fill" style="width:${pct(f.confidence)}"></div>
        </div>
        <span style="min-width:36px;text-align:right">${pct(f.confidence)}</span>
      </div>
    `).join("");
  }

  function renderAlerts(alerts) {
    if (!alerts.length) {
      alertList.innerHTML = '<div class="no-alerts">No alerts</div>';
      return;
    }
    alertList.innerHTML = alerts.map(a => `
      <div class="alert-item">
        <div class="alert-dot"></div>
        <span>${escHtml(a)}</span>
      </div>
    `).join("");
  }

  function addPlayLogEntry(playNum, formation, prediction) {
    const type = (prediction?.pass ?? 0) > (prediction?.run ?? 0) ? "PASS" : "RUN";
    playLog.unshift({ playNum, type, formation });
    if (playLog.length > 30) playLog.pop();

    if (playLogEl.querySelector("[data-empty]")) playLogEl.innerHTML = "";

    const entry = document.createElement("div");
    entry.className = "play-log-entry";
    entry.innerHTML = `
      <span class="play-log-num">#${playNum}</span>
      <span class="play-log-type">${type}</span>
      <span class="play-log-form">${escHtml(formation || "—")}</span>
    `;
    playLogEl.insertBefore(entry, playLogEl.firstChild);

    // Keep last 15 visible
    while (playLogEl.children.length > 15) {
      playLogEl.removeChild(playLogEl.lastChild);
    }
  }

  // ── FPS counter ────────────────────────────────────────────────────────
  function startFpsCounter() {
    let last = frameCount;
    fpsInterval = setInterval(() => {
      const fps = frameCount - last;
      last = frameCount;
      fpsChip.textContent = `${fps} fps`;
    }, 1000);
  }

  function stopFpsCounter() {
    clearInterval(fpsInterval);
    fpsChip.textContent = "— fps";
  }

  // ── Status indicator ───────────────────────────────────────────────────
  function setStatus(live) {
    statusDot.className = live ? "live" : "";
    statusText.textContent = live ? "LIVE" : "Offline";
  }

  // ── Helpers ────────────────────────────────────────────────────────────
  function pct(v) {
    return (Math.round((v ?? 0) * 1000) / 10).toFixed(1) + "%";
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Auto-start if ?demo in URL
  if (new URLSearchParams(location.search).has("demo")) {
    btnStart.click();
  }
})();
