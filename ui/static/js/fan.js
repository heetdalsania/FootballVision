/**
 * FootballVision — Fan Mode WebSocket Client
 * Handles live stats, formation diagram, Predict the Play, and highlight gallery.
 */

(function () {
  "use strict";

  // ── Session ID (persisted per browser tab) ─────────────────────────────
  const SESSION_ID = (() => {
    let id = sessionStorage.getItem("fv_session_id");
    if (!id) {
      id = "fan_" + Math.random().toString(36).slice(2, 10);
      sessionStorage.setItem("fv_session_id", id);
    }
    return id;
  })();

  // ── State ──────────────────────────────────────────────────────────────
  let ws = null;
  let score = parseInt(sessionStorage.getItem("fv_score") || "0", 10);
  let correctPicks = parseInt(sessionStorage.getItem("fv_correct") || "0", 10);
  let totalPicks   = parseInt(sessionStorage.getItem("fv_total")   || "0", 10);
  let predictState = "idle";   // idle | open | submitted | revealed
  let countdownTimer = null;
  let currentPrediction = null;
  let lastPositions = [];

  // ── DOM refs ───────────────────────────────────────────────────────────
  const el = (id) => document.getElementById(id);
  const liveDot         = el("live-dot");
  const statusText      = el("status-text");
  const formationLabel  = el("formation-label");
  const formationSub    = el("formation-sublabel");
  const formationCanvas = el("formation-canvas");
  const statPlays       = el("stat-plays");
  const statPlayers     = el("stat-players");
  const statRunBar      = el("stat-run-bar");
  const statPassBar     = el("stat-pass-bar");
  const statRunPct      = el("stat-run-pct");
  const statPassPct     = el("stat-pass-pct");
  const freqList        = el("freq-list");
  const fanAlerts       = el("fan-alerts");
  const clipGrid        = el("clip-grid");
  const playerScore     = el("player-score");
  const predictStatus   = el("predict-status");
  const predictCountdown= el("predict-countdown");
  const predictButtons  = el("predict-buttons");
  const predictResult   = el("predict-result");
  const predictCorrect  = el("predict-plays-correct");

  // ── Init ───────────────────────────────────────────────────────────────
  updateScoreUI();
  connectWS();
  setInterval(refreshClips, 5000);
  refreshClips();

  // ── WebSocket ──────────────────────────────────────────────────────────
  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/fan`);

    ws.onopen = () => {
      liveDot.className = "live";
      statusText.textContent = "LIVE";
    };

    ws.onmessage = (evt) => {
      let p;
      try { p = JSON.parse(evt.data); } catch { return; }
      renderPayload(p);
    };

    ws.onclose = () => {
      liveDot.className = "";
      statusText.textContent = "Reconnecting…";
      setTimeout(connectWS, 1500);
    };

    ws.onerror = () => {};
  }

  // ── Render ─────────────────────────────────────────────────────────────
  function renderPayload(p) {
    // Formation
    const fname = p.formation_name || "—";
    formationLabel.textContent = fname;
    formationSub.textContent   = `Detected formation`;

    if (p.formation_positions?.length) {
      lastPositions = p.formation_positions;
      drawFormationDiagram(p.formation_positions);
    }

    // Stats
    const stats = p.stats || {};
    statPlays.textContent   = stats.plays_analyzed ?? "0";
    statPlayers.textContent = p.formation_positions?.length ?? "0";

    const runT  = stats.run_tendency  ?? 0.5;
    const passT = stats.pass_tendency ?? 0.5;
    statRunBar.style.width  = pct(runT);
    statPassBar.style.width = pct(passT);
    statRunPct.textContent  = pct(runT);
    statPassPct.textContent = pct(passT);

    // Formation frequency
    renderFrequency(stats.formation_freq || {});

    // Alerts
    renderAlerts(p.alerts || []);

    // Predict window
    const pw = p.predict_window || {};
    handlePredictWindow(pw.open, pw.seconds_remaining, fname);

    // New highlight
    if (p.last_highlight) {
      addClip(p.last_highlight);
    }
  }

  // ── Formation diagram ──────────────────────────────────────────────────
  function drawFormationDiagram(positions) {
    const wrap = document.getElementById("formation-diagram-wrap");
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    formationCanvas.width  = cw;
    formationCanvas.height = ch;

    const ctx = formationCanvas.getContext("2d");
    ctx.clearRect(0, 0, cw, ch);

    // Field lines
    ctx.strokeStyle = "rgba(0,180,100,0.15)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (ch / 5) * i);
      ctx.lineTo(cw, (ch / 5) * i);
      ctx.stroke();
    }
    // LOS
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, ch * 0.5);
    ctx.lineTo(cw, ch * 0.5);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "10px sans-serif";
    ctx.fillText("Line of Scrimmage", 6, ch * 0.5 - 4);

    // Split players by y position into two groups (offense / defense)
    const midY = 0.5;
    const offPlayers = positions.filter(p => p.y > midY);
    const defPlayers = positions.filter(p => p.y <= midY);

    // Draw defense (top half)
    defPlayers.forEach(p => {
      const px = p.x * cw;
      const py = p.y * ch;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fillStyle   = "rgba(255,71,87,0.85)";
      ctx.strokeStyle = "#ff4757";
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();
    });

    // Draw offense (bottom half)
    offPlayers.forEach(p => {
      const px = p.x * cw;
      const py = p.y * ch;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fillStyle   = "rgba(0,212,170,0.85)";
      ctx.strokeStyle = "#00d4aa";
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();
    });

    // Legend
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#00d4aa";
    ctx.fillText("● Offense", 8, ch - 18);
    ctx.fillStyle = "#ff4757";
    ctx.fillText("● Defense", 8, ch - 6);
  }

  // ── Formation frequency ────────────────────────────────────────────────
  function renderFrequency(freq) {
    const entries = Object.entries(freq);
    if (!entries.length) return;
    freqList.innerHTML = entries.map(([name, val]) => `
      <div class="freq-row">
        <span class="freq-name">${escHtml(name)}</span>
        <div class="freq-bar-wrap">
          <div class="freq-bar" style="width:${pct(val)}"></div>
        </div>
        <span class="freq-pct">${pct(val)}</span>
      </div>
    `).join("");
  }

  // ── Alerts ─────────────────────────────────────────────────────────────
  function renderAlerts(alerts) {
    if (!alerts.length) {
      fanAlerts.innerHTML = '<div style="font-size:0.75rem;color:var(--text-muted)">No alerts</div>';
      return;
    }
    fanAlerts.innerHTML = alerts.map(a => `
      <div class="alert-item">⚡ ${escHtml(a)}</div>
    `).join("");
  }

  // ── Predict the Play ───────────────────────────────────────────────────
  function handlePredictWindow(open, secsRemaining, formationName) {
    if (open && predictState === "idle") {
      openPredictWindow(secsRemaining, formationName);
    }
    if (!open && predictState === "open") {
      // Window closed without prediction — reveal result
      closePredictWindow(false);
    }
    if (open && predictState === "open") {
      predictCountdown.textContent = secsRemaining ?? "…";
    }
  }

  function openPredictWindow(secs, formationName) {
    predictState = "open";
    currentPrediction = null;

    predictStatus.style.display    = "none";
    predictCountdown.style.display = "";
    predictButtons.style.display   = "";
    predictResult.style.display    = "none";
    predictCountdown.textContent   = secs ?? 5;

    // Update formation button label
    const formBtn = predictButtons.querySelector("[data-val='formation']");
    if (formBtn) formBtn.textContent = `📐 ${formationName || "Formation"}`;

    // Remove selected state
    predictButtons.querySelectorAll(".predict-btn").forEach(b => b.classList.remove("selected"));
  }

  window.submitPrediction = async function (val, btnEl) {
    if (predictState !== "open") return;
    predictState      = "submitted";
    currentPrediction = val;

    predictButtons.querySelectorAll(".predict-btn").forEach(b => b.classList.remove("selected"));
    btnEl.classList.add("selected");

    try {
      const res = await fetch("/fan/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, prediction: val, play_id: 0 }),
      });
      const data = await res.json();
      revealResult(data.correct, data.points, data.actual);
    } catch {
      revealResult(false, 0, "—");
    }
  };

  function revealResult(correct, points, actual) {
    predictState = "revealed";
    predictCountdown.style.display = "none";
    predictButtons.style.display   = "none";
    predictResult.style.display    = "";
    predictResult.className        = "predict-result " + (correct ? "correct" : "wrong");
    predictResult.textContent      = correct
      ? `✅ Correct! +${points} pts`
      : `❌ Nope — it was ${actual}. Better luck next play!`;

    if (correct) {
      score += points;
      correctPicks++;
    }
    totalPicks++;

    sessionStorage.setItem("fv_score", score);
    sessionStorage.setItem("fv_correct", correctPicks);
    sessionStorage.setItem("fv_total", totalPicks);
    updateScoreUI();

    setTimeout(resetPredictUI, 3000);
  }

  function closePredictWindow(submitted) {
    if (!submitted) {
      predictState = "idle";
      resetPredictUI();
    }
  }

  function resetPredictUI() {
    predictState = "idle";
    currentPrediction = null;
    predictStatus.style.display    = "";
    predictCountdown.style.display = "none";
    predictButtons.style.display   = "none";
    predictResult.style.display    = "none";
    predictStatus.textContent      = "Waiting for next snap…";
  }

  function updateScoreUI() {
    playerScore.textContent = score;
    if (totalPicks > 0) {
      predictCorrect.textContent = `${correctPicks}/${totalPicks} correct`;
    }
  }

  // ── Highlight clips ────────────────────────────────────────────────────
  const shownClips = new Set();

  async function refreshClips() {
    try {
      const res  = await fetch("/clips");
      const data = await res.json();
      (data.clips || []).forEach(addClip);
    } catch {}
  }

  function addClip(filename) {
    if (shownClips.has(filename)) return;
    shownClips.add(filename);

    const noClipsEl = clipGrid.querySelector(".no-clips");
    if (noClipsEl) noClipsEl.remove();

    const item = document.createElement("div");
    item.className = "clip-item";
    item.innerHTML = `🎬 ${escHtml(filename)}`;
    item.onclick   = () => window.open(`/clips/${encodeURIComponent(filename)}`, "_blank");
    clipGrid.insertBefore(item, clipGrid.firstChild);
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
})();
