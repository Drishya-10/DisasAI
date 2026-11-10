/* ============================================================
   Emergency AI Dispatch System — Frontend JavaScript
   Handles: Scenario loading, Recording, Upload, Pipeline API
   calls, Waveform rendering, LLM & Dispatch display, Operator
   Confirmation
   ============================================================ */

const API = "http://127.0.0.1:8000";

let selectedScenarioId = null;
let uploadedFile = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let currentCallId = null;
let currentData = null;

// ── Live Clock ──────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById("hud-time").textContent =
    now.toLocaleTimeString("en-IN", { hour12: false, timeZone: "Asia/Kolkata" });
}
setInterval(updateClock, 1000);
updateClock();

// ── Scenario Loading ────────────────────────────────────────
const EMERGENCY_ICONS = {
  FLOOD: "🌊", FIRE: "🔥", ROAD_ACCIDENT: "🚗",
  BUILDING_COLLAPSE: "🏚️", GAS_LEAK: "☣️", MEDICAL: "🏥", OTHER: "❓"
};
const SEV_COLORS = {
  CRITICAL: "#ff2d55", HIGH: "#ff8c00", MEDIUM: "#ffb800", LOW: "#00e676"
};

async function loadScenarios() {
  try {
    const res = await fetch(`${API}/api/scenarios`);
    const scenarios = await res.json();
    renderScenarios(scenarios);
  } catch (e) {
    document.getElementById("scenario-grid").innerHTML =
      '<div style="color:var(--red-emergency);font-size:12px;grid-column:span 2;text-align:center;padding:16px;">⚠️ Backend not reachable. Start server first: <code>python backend/app.py</code></div>';
  }
}

function renderScenarios(scenarios) {
  const grid = document.getElementById("scenario-grid");
  if (!scenarios.length) { grid.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No scenarios loaded.</div>'; return; }
  grid.innerHTML = "";
  for (const sc of scenarios) {
    const icon = EMERGENCY_ICONS[sc.emergency_type] || "🚨";
    const typeLower = (sc.emergency_type || "other").toLowerCase().replace("_", "-");
    const sevColor = SEV_COLORS[sc.severity] || "#aaa";
    const btn = document.createElement("button");
    btn.className = "scenario-btn";
    btn.dataset.id = sc.id;
    btn.dataset.lang = sc.language_code;
    btn.innerHTML = `
      <div class="scenario-btn-title">${icon} ${sc.title}</div>
      <div class="scenario-btn-meta">
        <span class="type-pill ${typeLower}">${sc.emergency_type}</span>
        <span style="color:${sevColor};font-size:10px;font-weight:700;">${sc.severity}</span>
        <span style="color:var(--text-muted);font-size:10px;">${sc.language_name}</span>
      </div>`;
    btn.onclick = () => selectScenario(sc.id, sc.language_code, btn);
    grid.appendChild(btn);
  }
}

function selectScenario(id, lang, btn) {
  selectedScenarioId = id;
  uploadedFile = null;
  document.getElementById("upload-status").style.display = "none";
  document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("selected"));
  btn.classList.add("selected");
  document.getElementById("lang-select").value = lang || "auto";
  document.getElementById("btn-process").disabled = false;
  updatePipelineStep(null);
}

// ── File Upload ─────────────────────────────────────────────
function onFileUpload(input) {
  if (!input.files.length) return;
  uploadedFile = input.files[0];
  selectedScenarioId = null;
  document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("selected"));
  const statusEl = document.getElementById("upload-status");
  statusEl.textContent = `📂 Loaded: ${uploadedFile.name} (${(uploadedFile.size / 1024).toFixed(1)} KB)`;
  statusEl.style.display = "block";
  document.getElementById("btn-process").disabled = false;
}

// ── Live Recording ──────────────────────────────────────────
async function toggleRecording() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      recordedChunks = [];
      mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: "audio/webm" });
        uploadedFile = new File([blob], "recorded_emergency.webm", { type: "audio/webm" });
        selectedScenarioId = null;
        document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("selected"));
        const statusEl = document.getElementById("upload-status");
        statusEl.textContent = `🎙️ Recorded: ${(blob.size / 1024).toFixed(1)} KB — Ready to process`;
        statusEl.style.display = "block";
        document.getElementById("btn-process").disabled = false;
      };
      mediaRecorder.start();
      isRecording = true;
      const btn = document.getElementById("btn-record");
      btn.classList.add("recording");
      btn.innerHTML = "⏹ Stop Recording";
    } catch (e) {
      alert("Microphone access denied. Please allow microphone access and try again.");
    }
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    isRecording = false;
    const btn = document.getElementById("btn-record");
    btn.classList.remove("recording");
    btn.innerHTML = "🎤 Record Live Call";
  }
}

// ── Pipeline Stepper Control ─────────────────────────────────
function updatePipelineStep(activeStep) {
  for (let i = 1; i <= 7; i++) {
    const row = document.getElementById(`ps-${i}`);
    const badge = document.getElementById(`pss-${i}`);
    const pstep = document.getElementById(`pstep-${i - 1}`);
    if (activeStep === null) {
      row.className = "pipeline-stage-row";
      badge.className = "pipeline-stage-status waiting";
      badge.textContent = "waiting";
      if (pstep) { pstep.className = "pipe-step"; }
    } else if (i < activeStep) {
      row.className = "pipeline-stage-row done";
      badge.className = "pipeline-stage-status done";
      badge.textContent = "✓ done";
      if (pstep) { pstep.className = "pipe-step completed"; }
    } else if (i === activeStep) {
      row.className = "pipeline-stage-row active";
      badge.className = "pipeline-stage-status running";
      badge.textContent = "● running";
      if (pstep) { pstep.className = "pipe-step active"; }
    } else {
      row.className = "pipeline-stage-row";
      badge.className = "pipeline-stage-status waiting";
      badge.textContent = "waiting";
      if (pstep) { pstep.className = "pipe-step"; }
    }
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Main: Process Emergency Call ────────────────────────────
async function processCall() {
  const btn = document.getElementById("btn-process");
  btn.disabled = true;
  btn.innerHTML = "⏳ Processing…";
  document.getElementById("dispatch-success-banner").classList.remove("active");

  // Reset result cards
  for (const id of ["card-transcripts", "card-llm", "card-dispatch", "card-operator"]) {
    document.getElementById(id).style.display = "none";
  }
  document.getElementById("audio-placeholder").style.display = "block";
  document.getElementById("audio-results").style.display = "none";

  const form = new FormData();
  form.append("enable_butterworth", document.getElementById("toggle-butterworth").checked);
  form.append("enable_spectral_gating", document.getElementById("toggle-spectral").checked);
  form.append("enable_gru", document.getElementById("toggle-gru").checked);
  form.append("language_hint", document.getElementById("lang-select").value);

  // Optional API key overrides from modal
  const gnaniKeyOverride = document.getElementById("modal-gnani-key")?.value.trim();
  const geminiKeyOverride = document.getElementById("modal-gemini-key")?.value.trim();
  if (gnaniKeyOverride) form.append("gnani_api_key", gnaniKeyOverride);
  if (geminiKeyOverride) form.append("gemini_api_key", geminiKeyOverride);

  if (selectedScenarioId) {
    form.append("scenario_id", selectedScenarioId);
  } else if (uploadedFile) {
    form.append("audio_file", uploadedFile);
  } else {
    alert("Please select a scenario or upload/record audio first.");
    btn.disabled = false;
    btn.innerHTML = "⚡ Process Emergency Call";
    return;
  }

  // Animate stages 1 & 2 immediately (ingestion + denoising happens server-side)
  updatePipelineStep(1);
  await sleep(300);
  updatePipelineStep(2);

  let data;
  try {
    const res = await fetch(`${API}/api/process-audio`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    data = await res.json();
  } catch (e) {
    alert(`❌ Pipeline Error: ${e.message}`);
    btn.disabled = false;
    btn.innerHTML = "⚡ Process Emergency Call";
    updatePipelineStep(null);
    return;
  }

  currentCallId = data.call_id;
  currentData = data;

  // ─ Stage 2 Done: Audio & Waveforms ──────────────────────
  updatePipelineStep(2);
  renderAudioResults(data);
  await sleep(400);

  // ─ Stage 3: ASR ─────────────────────────────────────────
  updatePipelineStep(3);
  await sleep(300);

  // ─ Stage 4: Translation ──────────────────────────────────
  updatePipelineStep(4);
  renderTranscripts(data);
  await sleep(400);

  // ─ Stage 5: LLM ─────────────────────────────────────────
  updatePipelineStep(5);
  await sleep(300);
  renderLLMResults(data);
  await sleep(400);

  // ─ Stage 6: Dispatch ────────────────────────────────────
  updatePipelineStep(6);
  await sleep(300);
  renderDispatchPlan(data);
  await sleep(400);

  // ─ Stage 7: Operator Console ────────────────────────────
  updatePipelineStep(7);
  renderOperatorConsole(data);

  btn.disabled = false;
  btn.innerHTML = "⚡ Process Emergency Call";
}

// ── Render: Audio Waveforms & Player ────────────────────────
function renderAudioResults(data) {
  document.getElementById("audio-placeholder").style.display = "none";
  document.getElementById("audio-results").style.display = "block";

  document.getElementById("audio-raw").src = `${API}${data.original_audio_url}`;
  document.getElementById("audio-denoised").src = `${API}${data.denoised_audio_url}`;

  if (data.waveforms?.raw) drawWaveform("waveform-raw", data.waveforms.raw, "#ff2d55");
  if (data.waveforms?.denoised) drawWaveform("waveform-denoised", data.waveforms.denoised, "#00f0ff");

  const m = data.audio_metrics;
  document.getElementById("met-raw-snr").textContent = `${m.raw_snr_db} dB`;
  document.getElementById("met-enh-snr").textContent = `${m.enhanced_snr_db} dB`;
  document.getElementById("met-snr-imp").textContent = `+${m.snr_improvement_db} dB`;
  document.getElementById("met-dur").textContent = `${m.duration_seconds}s`;
}

function drawWaveform(canvasId, peaks, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.offsetWidth;
  canvas.height = 64;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const midY = canvas.height / 2;
  const barW = Math.max(1, (canvas.width / peaks.length) - 0.5);

  ctx.fillStyle = color + "80";
  for (let i = 0; i < peaks.length; i++) {
    const h = Math.max(2, peaks[i] * canvas.height * 0.9);
    const x = (i / peaks.length) * canvas.width;
    ctx.fillRect(x, midY - h / 2, barW, h);
  }

  // Bright top line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < peaks.length; i++) {
    const h = Math.max(2, peaks[i] * canvas.height * 0.9);
    const x = (i / peaks.length) * canvas.width + barW / 2;
    if (i === 0) ctx.moveTo(x, midY - h / 2);
    else ctx.lineTo(x, midY - h / 2);
  }
  ctx.stroke();
}

// ── Render: Transcripts ──────────────────────────────────────
function renderTranscripts(data) {
  document.getElementById("card-transcripts").style.display = "block";
  document.getElementById("lang-detected-chip").textContent = data.detected_language;
  document.getElementById("asr-provider-badge").textContent = data.asr_provider || "Gnani ASR";
  document.getElementById("trans-provider-badge").textContent = data.translation_provider || "";
  document.getElementById("transcript-native").textContent = data.native_transcript || "—";
  document.getElementById("transcript-english").textContent = data.english_translation || "—";
}

// ── Render: LLM Intelligence ─────────────────────────────────
function renderLLMResults(data) {
  document.getElementById("card-llm").style.display = "block";
  const llm = data.llm_analysis;

  const icon = EMERGENCY_ICONS[llm.emergency_type] || "🚨";
  document.getElementById("llm-type-badge").textContent = `${icon} ${llm.emergency_type}`;

  const score = llm.severity_score;
  document.getElementById("llm-score").textContent = score;
  document.getElementById("llm-severity-fill").style.width = `${(score / 5) * 100}%`;
  document.getElementById("llm-severity-label").textContent = llm.severity;
  document.getElementById("llm-confidence").textContent = `${(llm.confidence * 100).toFixed(1)}%`;

  document.getElementById("llm-loc").textContent = llm.location || "—";
  document.getElementById("llm-city").textContent = llm.city || "—";
  document.getElementById("llm-landmark").textContent = llm.landmark || "—";
  document.getElementById("llm-trapped").textContent = llm.people_trapped;
  document.getElementById("llm-injured").textContent = llm.people_injured;
  document.getElementById("llm-at-risk").textContent = llm.people_at_risk;
  document.getElementById("llm-summary").textContent = llm.summary || "—";

  const hz = document.getElementById("hazards-container");
  hz.innerHTML = "";
  for (const h of (llm.hazards || [])) {
    const chip = document.createElement("div");
    chip.className = "hazard-chip";
    chip.textContent = `⚠️ ${h}`;
    hz.appendChild(chip);
  }
}

// ── Render: Dispatch Plan ────────────────────────────────────
function renderDispatchPlan(data) {
  document.getElementById("card-dispatch").style.display = "block";
  const dp = data.dispatch_plan;

  document.getElementById("disp-agency").textContent = `🏛️ ${dp.primary_agency}`;
  document.getElementById("disp-eta").textContent = `ETA ${dp.estimated_eta_minutes} min`;
  document.getElementById("disp-radio").textContent = `📡 ${dp.radio_channel}`;
  document.getElementById("disp-plan-id").textContent = `Plan: ${dp.plan_id} | Depot: ${dp.depot_assigned}`;
  document.getElementById("disp-priority").textContent = `🚨 ${dp.priority_level}`;

  const tbody = document.getElementById("units-tbody");
  tbody.innerHTML = "";
  for (const u of (dp.recommended_units || [])) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="unit-callsign">${u.call_sign}</td>
      <td>${u.unit_type}</td>
      <td style="font-weight:700;color:var(--cyan-primary)">${u.quantity}</td>
      <td style="color:var(--text-secondary);font-size:11px">${u.station}</td>`;
    tbody.appendChild(tr);
  }

  const eqTags = document.getElementById("equipment-tags");
  eqTags.innerHTML = "";
  for (const eq of (dp.equipment_checklist || [])) {
    const tag = document.createElement("span");
    tag.className = "equipment-tag";
    tag.textContent = eq;
    eqTags.appendChild(tag);
  }

  const prList = document.getElementById("protocols-list");
  prList.innerHTML = "";
  for (const pr of (dp.critical_protocols || [])) {
    const li = document.createElement("li");
    li.textContent = pr;
    prList.appendChild(li);
  }
}

// ── Render: Operator Console ─────────────────────────────────
function renderOperatorConsole(data) {
  document.getElementById("card-operator").style.display = "block";
  const llm = data.llm_analysis;

  document.getElementById("op-call-id-display").textContent = data.call_id;
  document.getElementById("op-type").value = llm.emergency_type;
  document.getElementById("op-severity").value = llm.severity;
  document.getElementById("op-people").value = llm.people_trapped;
  document.getElementById("op-location").value = llm.location || "";
}

// ── Operator: Confirm Dispatch ───────────────────────────────
async function confirmDispatch(action) {
  if (!currentCallId) { alert("No active call to confirm."); return; }

  const payload = {
    call_id: currentCallId,
    operator_id: `OP-${Math.floor(Math.random() * 900) + 100}`,
    confirmed_emergency_type: document.getElementById("op-type").value,
    confirmed_severity: document.getElementById("op-severity").value,
    confirmed_location: document.getElementById("op-location").value,
    confirmed_people_count: parseInt(document.getElementById("op-people").value) || 0,
    operator_notes: document.getElementById("op-notes").value || "Operator authorized dispatch.",
    action: action
  };

  try {
    const res = await fetch(`${API}/api/operator/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    // Show success banner
    const banner = document.getElementById("dispatch-success-banner");
    banner.classList.add("active");
    document.getElementById("dispatch-success-detail").textContent =
      `${action} authorized by ${payload.operator_id} at ${new Date().toLocaleTimeString()}`;
    document.getElementById("dispatch-broadcast-code").textContent =
      result.operator_action?.broadcast_code || "";

    // Mark pipeline complete
    updatePipelineStep(8);

  } catch (e) {
    alert(`Error confirming dispatch: ${e.message}`);
  }
}

// ── Incident Log ─────────────────────────────────────────────
async function loadIncidentLog() {
  try {
    const res = await fetch(`${API}/api/incidents`);
    const incidents = await res.json();
    const tbody = document.getElementById("audit-tbody");
    tbody.innerHTML = "";
    if (!incidents.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted);padding:20px;">No incidents yet.</td></tr>';
    } else {
      for (const inc of incidents.reverse()) {
        const llm = inc.llm_analysis || {};
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="font-family:var(--font-mono);font-size:11px;color:var(--cyan-primary)">${inc.call_id}</td>
          <td style="font-size:11px">${inc.timestamp}</td>
          <td>${inc.detected_language}</td>
          <td>${llm.emergency_type || "—"}</td>
          <td style="color:${SEV_COLORS[llm.severity] || '#aaa'}">${llm.severity || "—"}</td>
          <td style="font-size:11px">${llm.location || "—"}</td>
          <td><span class="status-badge ${(inc.status || '').toLowerCase()}">${inc.status}</span></td>`;
        tbody.appendChild(tr);
      }
    }
    document.getElementById("log-modal").classList.add("active");
  } catch (e) {
    alert("Could not load incident log — backend not reachable.");
  }
}

// Close modals on overlay click
document.querySelectorAll(".modal-overlay").forEach(el => {
  el.addEventListener("click", e => { if (e.target === el) el.classList.remove("active"); });
});

// ── Init ─────────────────────────────────────────────────────
loadScenarios();
