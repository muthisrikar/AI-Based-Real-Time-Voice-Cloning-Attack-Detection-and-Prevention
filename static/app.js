/**
 * app.js - Holographic Fluid Voice Animation & Real-Time Detection
 */

// DOM Elements
const waveCanvas = document.getElementById("waveCanvas");
const waveCtx = waveCanvas.getContext("2d");
const artifactsCanvas = document.getElementById("artifactsCanvas");
const artifactsCtx = artifactsCanvas.getContext("2d");
const biologicalCanvas = document.getElementById("biologicalCanvas");
const biologicalCtx = biologicalCanvas.getContext("2d");
const fakeWaveformCanvas = document.getElementById("fakeWaveformCanvas");
const fakeWaveCtx = fakeWaveformCanvas.getContext("2d");
const realWaveformCanvas = document.getElementById("realWaveformCanvas");
const realWaveCtx = realWaveformCanvas.getContext("2d");

const recordBtn = document.getElementById("recordBtn");
const recordBtnLabel = document.getElementById("recordBtnLabel");
const audioFileInput = document.getElementById("audioFileInput");
const statusText = document.getElementById("statusText");
const idleLineHint = document.getElementById("idleLineHint");
const liveLevelBar = document.getElementById("liveLevelBar");
const dotIdle = document.getElementById("dotIdle");
const dotRec = document.getElementById("dotRec");

const fakeCard = document.getElementById("fakeCard");
const realCard = document.getElementById("realCard");

// Audio Context & State
let audioCtx = null;
let analyser = null;
let micStream = null;
let isRecording = false;
let audioFrequencyData = null;
let currentVolume = 0;
let waveTransitionProgress = 0; // 0 = flat line, 1 = full fluid wave graph

// High DPI Canvas Resizing
function resizeAllCanvases() {
  [
    { c: waveCanvas, ctx: waveCtx },
    { c: artifactsCanvas, ctx: artifactsCtx },
    { c: biologicalCanvas, ctx: biologicalCtx },
    { c: fakeWaveformCanvas, ctx: fakeWaveCtx },
    { c: realWaveformCanvas, ctx: realWaveCtx },
  ].forEach(({ c, ctx }) => {
    const rect = c.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      c.width = rect.width * window.devicePixelRatio;
      c.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
  });
}
window.addEventListener("resize", resizeAllCanvases);
window.addEventListener("DOMContentLoaded", () => {
  resizeAllCanvases();
  startAnimationLoops();
});

// Particles for Holographic Stage (shown while recording)
const particles = [];
for (let i = 0; i < 40; i++) {
  particles.push({
    x: Math.random() * 1000,
    y: Math.random() * 220,
    radius: Math.random() * 2.2 + 0.6,
    speedX: (Math.random() - 0.5) * 0.8,
    speedY: (Math.random() - 0.5) * 0.5,
    alpha: Math.random() * 0.7 + 0.3,
  });
}

// ==============================================================================
// 1. Holographic Waveform: Flat Line when Idle -> Fluid Graph when Recording
// ==============================================================================
let wavePhase = 0;

function drawHolographicWave() {
  const w = waveCanvas.width / window.devicePixelRatio;
  const h = waveCanvas.height / window.devicePixelRatio;
  waveCtx.clearRect(0, 0, w, h);

  const midY = h * 0.52;

  // Smooth transition interpolation
  const targetTransition = isRecording ? 1.0 : 0.0;
  waveTransitionProgress += (targetTransition - waveTransitionProgress) * 0.12;

  // Compute live audio volume modulation
  if (analyser && isRecording) {
    analyser.getByteFrequencyData(audioFrequencyData);
    let sum = 0;
    for (let i = 0; i < 32; i++) {
      sum += audioFrequencyData[i];
    }
    const targetVol = sum / 32 / 255;
    currentVolume += (targetVol - currentVolume) * 0.25;
  } else {
    currentVolume += (0.0 - currentVolume) * 0.1;
  }

  // Update level bar
  if (liveLevelBar) {
    const levelWidth = isRecording ? Math.max(10, currentVolume * 100) : 10;
    liveLevelBar.style.width = `${levelWidth}px`;
  }

  // --- STATE A: WHEN NOT RECORDING (FLAT STRAIGHT LINE) ---
  if (waveTransitionProgress < 0.02) {
    if (idleLineHint) idleLineHint.style.opacity = "1";

    // Clean horizontal laser line across the center
    waveCtx.save();
    waveCtx.beginPath();
    waveCtx.moveTo(0, midY);
    waveCtx.lineTo(w, midY);

    const flatGrad = waveCtx.createLinearGradient(0, 0, w, 0);
    flatGrad.addColorStop(0, "rgba(0, 242, 254, 0.05)");
    flatGrad.addColorStop(0.2, "rgba(0, 242, 254, 0.45)");
    flatGrad.addColorStop(0.5, "rgba(255, 255, 255, 0.95)");
    flatGrad.addColorStop(0.8, "rgba(0, 242, 254, 0.45)");
    flatGrad.addColorStop(1, "rgba(0, 242, 254, 0.05)");

    waveCtx.strokeStyle = flatGrad;
    waveCtx.lineWidth = 2.2;
    waveCtx.shadowBlur = 12;
    waveCtx.shadowColor = "#00f2fe";
    waveCtx.stroke();

    // Center laser core highlight
    waveCtx.beginPath();
    waveCtx.moveTo(w * 0.35, midY);
    waveCtx.lineTo(w * 0.65, midY);
    waveCtx.strokeStyle = "#ffffff";
    waveCtx.lineWidth = 2.8;
    waveCtx.shadowBlur = 16;
    waveCtx.shadowColor = "#ffffff";
    waveCtx.stroke();

    waveCtx.restore();

    requestAnimationFrame(drawHolographicWave);
    return;
  }

  // --- STATE B: WHILE RECORDING (DYNAMIC FLUID WAVE GRAPH) ---
  if (idleLineHint) idleLineHint.style.opacity = "0";

  const baseAmp = (35 + currentVolume * 75) * waveTransitionProgress;
  const speed = 0.045 + currentVolume * 0.06;
  wavePhase += speed;

  // Draw floating sparkles / particles
  particles.forEach((p) => {
    p.x += p.speedX;
    p.y += p.speedY;
    if (p.x < 0) p.x = w;
    if (p.x > w) p.x = 0;
    if (p.y < 0) p.y = h;
    if (p.y > h) p.y = 0;

    waveCtx.beginPath();
    waveCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
    waveCtx.fillStyle = `rgba(0, 242, 254, ${p.alpha * waveTransitionProgress * 0.9})`;
    waveCtx.shadowBlur = 8;
    waveCtx.shadowColor = "#00f2fe";
    waveCtx.fill();
    waveCtx.shadowBlur = 0;
  });

  // Layered Holographic Fluid Ribbons (Cyan, Magenta, Coral, Gold)
  const ribbonGradients = [
    {
      c1: "rgba(0, 242, 254, 0.8)",
      c2: "rgba(114, 9, 183, 0.45)",
      freq: 0.009,
      phaseOffset: 0,
      ampMult: 1.0,
      thickness: 3.5,
    },
    {
      c1: "rgba(247, 37, 133, 0.85)",
      c2: "rgba(255, 117, 140, 0.5)",
      freq: 0.012,
      phaseOffset: 1.8,
      ampMult: 0.85,
      thickness: 3.0,
    },
    {
      c1: "rgba(255, 183, 3, 0.85)",
      c2: "rgba(247, 37, 133, 0.35)",
      freq: 0.015,
      phaseOffset: 3.4,
      ampMult: 0.7,
      thickness: 2.5,
    },
    {
      c1: "rgba(76, 201, 240, 0.9)",
      c2: "rgba(0, 242, 254, 0.6)",
      freq: 0.007,
      phaseOffset: 5.1,
      ampMult: 1.15,
      thickness: 4.0,
    },
  ];

  ribbonGradients.forEach((ribbon) => {
    waveCtx.save();
    waveCtx.beginPath();
    waveCtx.moveTo(0, midY);

    for (let x = 0; x <= w; x += 3) {
      const normX = (x / w) * 2 - 1;
      const envelope = Math.exp(-Math.pow(normX * 1.5, 2));

      const y =
        midY +
        Math.sin(x * ribbon.freq + wavePhase + ribbon.phaseOffset) *
          baseAmp *
          ribbon.ampMult *
          envelope +
        Math.cos(x * ribbon.freq * 1.6 - wavePhase * 0.7) *
          (baseAmp * 0.35) *
          envelope;

      waveCtx.lineTo(x, y);
    }

    const grad = waveCtx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0.1, "rgba(0, 242, 254, 0.3)");
    grad.addColorStop(0.35, ribbon.c1);
    grad.addColorStop(0.7, ribbon.c2);
    grad.addColorStop(0.95, "rgba(247, 37, 133, 0.3)");

    waveCtx.strokeStyle = grad;
    waveCtx.lineWidth = ribbon.thickness;
    waveCtx.shadowBlur = 18 * waveTransitionProgress;
    waveCtx.shadowColor = ribbon.c1;
    waveCtx.stroke();
    waveCtx.restore();
  });

  requestAnimationFrame(drawHolographicWave);
}

// ==============================================================================
// 2. Mini Dual-Analysis Waveforms
// ==============================================================================
let miniPhase = 0;

function drawMiniAnalysisWaves() {
  miniPhase += 0.06;

  // 1. Digital Artifacts Canvas
  const aw = artifactsCanvas.width / window.devicePixelRatio;
  const ah = artifactsCanvas.height / window.devicePixelRatio;
  artifactsCtx.clearRect(0, 0, aw, ah);

  const numBars = 36;
  const barWidth = aw / numBars - 2;

  for (let i = 0; i < numBars; i++) {
    let barHeight = 2; // Flat idle baseline
    if (isRecording && audioFrequencyData) {
      const idx = Math.floor((i / numBars) * 40);
      barHeight = Math.max(3, (audioFrequencyData[idx] / 255) * (ah * 0.85));
    } else if (isRecording) {
      barHeight = (Math.sin(i * 0.35 + miniPhase) * 0.5 + 0.5) * (ah * 0.65) + 3;
    }

    const x = i * (barWidth + 2);
    const y = ah - barHeight;

    const grad = artifactsCtx.createLinearGradient(0, y, 0, ah);
    grad.addColorStop(0, "#ff758c");
    grad.addColorStop(1, "rgba(255, 42, 68, 0.3)");

    artifactsCtx.fillStyle = grad;
    artifactsCtx.fillRect(x, y, barWidth, barHeight);
  }

  // 2. Biological Speech Patterns Canvas
  const bw = biologicalCanvas.width / window.devicePixelRatio;
  const bh = biologicalCanvas.height / window.devicePixelRatio;
  biologicalCtx.clearRect(0, 0, bw, bh);

  biologicalCtx.beginPath();
  biologicalCtx.moveTo(0, bh / 2);
  for (let x = 0; x <= bw; x += 2) {
    const waveAmp = isRecording ? 14 + currentVolume * 22 : 0; // Flat when not recording!
    const y = bh / 2 + Math.sin(x * 0.05 + miniPhase * 1.2) * waveAmp;
    biologicalCtx.lineTo(x, y);
  }
  biologicalCtx.strokeStyle = "#00f2fe";
  biologicalCtx.lineWidth = 2.0;
  biologicalCtx.shadowBlur = isRecording ? 8 : 0;
  biologicalCtx.shadowColor = "rgba(0, 242, 254, 0.8)";
  biologicalCtx.stroke();

  requestAnimationFrame(drawMiniAnalysisWaves);
}

// ==============================================================================
// 3. Verdict Cards Static & Animated Waveforms
// ==============================================================================
let verdictPhase = 0;

function drawVerdictWaves() {
  verdictPhase += 0.04;

  // Fake Waveform: Erratic, jagged, glitching red frequency wave
  const fw = fakeWaveformCanvas.width / window.devicePixelRatio;
  const fh = fakeWaveformCanvas.height / window.devicePixelRatio;
  fakeWaveCtx.clearRect(0, 0, fw, fh);

  fakeWaveCtx.beginPath();
  fakeWaveCtx.moveTo(0, fh / 2);
  const fakePoints = [
    0, 2, -6, 12, -18, 14, -22, 24, -14, 8, -4, 18, -20, 26, -16, 6, -2, 14, -8, 0,
  ];
  for (let i = 0; i < fakePoints.length; i++) {
    const x = (i / (fakePoints.length - 1)) * fw;
    const jitter = Math.sin(verdictPhase * 3 + i) * 3;
    const y = fh / 2 + fakePoints[i] + jitter;
    fakeWaveCtx.lineTo(x, y);
  }
  fakeWaveCtx.strokeStyle = "#ff4d6d";
  fakeWaveCtx.lineWidth = 2.4;
  fakeWaveCtx.shadowBlur = 10;
  fakeWaveCtx.shadowColor = "rgba(255, 42, 68, 0.8)";
  fakeWaveCtx.stroke();

  // Real Waveform: Smooth, calm, natural green sinusoidal wave
  const rw = realWaveformCanvas.width / window.devicePixelRatio;
  const rh = realWaveformCanvas.height / window.devicePixelRatio;
  realWaveCtx.clearRect(0, 0, rw, rh);

  realWaveCtx.beginPath();
  realWaveCtx.moveTo(0, rh / 2);
  for (let x = 0; x <= rw; x += 2) {
    const y = rh / 2 + Math.sin(x * 0.04 + verdictPhase) * 12 * Math.sin(x * 0.015);
    realWaveCtx.lineTo(x, y);
  }
  realWaveCtx.strokeStyle = "#10b981";
  realWaveCtx.lineWidth = 2.8;
  realWaveCtx.shadowBlur = 10;
  realWaveCtx.shadowColor = "rgba(16, 185, 129, 0.8)";
  realWaveCtx.stroke();

  requestAnimationFrame(drawVerdictWaves);
}

function startAnimationLoops() {
  requestAnimationFrame(drawHolographicWave);
  requestAnimationFrame(drawMiniAnalysisWaves);
  requestAnimationFrame(drawVerdictWaves);
}

// ==============================================================================
// 4. Live Audio Recording & Unified API Detection
// ==============================================================================
let mediaRecorder = null;
let recordedAudioChunks = [];
let isAnalyzing = false;

async function toggleRecording() {
  if (isAnalyzing) {
    return;
  }
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  if (isAnalyzing || isRecording) {
    return;
  }
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioCtx.createMediaStreamSource(micStream);

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    audioFrequencyData = new Uint8Array(analyser.frequencyBinCount);
    isRecording = true;

    // Reset visible verdict cards while new recording is underway
    clearVerdict();

    // Set up MediaRecorder to capture real microphone audio
    recordedAudioChunks = [];
    let mimeType = "";
    if (typeof MediaRecorder !== "undefined") {
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
        mimeType = "audio/ogg;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/wav")) {
        mimeType = "audio/wav";
      }
    }

    try {
      mediaRecorder = mimeType ? new MediaRecorder(micStream, { mimeType }) : new MediaRecorder(micStream);
    } catch (e) {
      console.warn("Fallback to default MediaRecorder options", e);
      mediaRecorder = new MediaRecorder(micStream);
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedAudioChunks.push(event.data);
      }
    };

    mediaRecorder.start(100);

    // UI Updates
    recordBtn.classList.add("active-recording");
    recordBtnLabel.textContent = "STOP RECORDING";
    statusText.textContent = "RECORDING LIVE AUDIO • SPEAK CLEARLY...";
    statusText.style.color = "#00f2fe";

    if (dotIdle && dotRec) {
      dotIdle.className = "toggle-dot dot-active";
      dotRec.className = "toggle-dot dot-inactive";
    }
  } catch (err) {
    console.error("Mic error:", err);
    alert("Microphone permission denied or unavailable: " + err.message);
  }
}

function stopRecording() {
  if (!isRecording) {
    return;
  }
  isRecording = false;

  recordBtn.classList.remove("active-recording");
  recordBtnLabel.textContent = "RECORD LIVE AUDIO";
  statusText.textContent = "PROCESSING RECORDED AUDIO VIA LOCAL AI DETECTOR...";
  statusText.style.color = "#00f2fe";

  if (dotIdle && dotRec) {
    dotIdle.className = "toggle-dot dot-inactive";
    dotRec.className = "toggle-dot dot-active";
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    const activeRecorder = mediaRecorder;
    mediaRecorder = null;
    activeRecorder.onstop = async () => {
      activeRecorder.onstop = null;
      const mime = activeRecorder.mimeType || "audio/webm";
      const audioBlob = new Blob(recordedAudioChunks, { type: mime });
      recordedAudioChunks = [];

      // Clean up mic stream
      if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
        micStream = null;
      }
      if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
      }

      let ext = "webm";
      if (mime.includes("ogg")) {
        ext = "ogg";
      } else if (mime.includes("wav")) {
        ext = "wav";
      } else if (mime.includes("mp4") || mime.includes("aac")) {
        ext = "mp4";
      }

      await sendAudioForDetection(audioBlob, `recorded_voice.${ext}`);
    };

    activeRecorder.stop();
  } else {
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
  }
}

// Unified Audio Detection Submission (1 user action = 1 detection request)
async function sendAudioForDetection(audioFileOrBlob, filename) {
  if (isAnalyzing) {
    return;
  }
  isAnalyzing = true;
  if (recordBtn) recordBtn.disabled = true;
  if (audioFileInput) audioFileInput.disabled = true;

  clearVerdict();
  statusText.textContent = `ANALYZING ${filename.toUpperCase()} VIA PRETRAINED LOCAL MODEL...`;
  statusText.style.color = "#00f2fe";

  const formData = new FormData();
  formData.append("file", audioFileOrBlob, filename);

  try {
    const response = await fetch("/api/detect", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Audio detection failed on server.");
    }

    // Pass label ("HUMAN" or "AI") and confidence
    triggerVerdict(data.label, data.confidence);
  } catch (err) {
    console.error("Audio detection error:", err);
    clearVerdict();
    statusText.textContent = `DETECTION ERROR: ${err.message.toUpperCase()}`;
    statusText.style.color = "#ff2a44";
  } finally {
    isAnalyzing = false;
    if (recordBtn) recordBtn.disabled = false;
    if (audioFileInput) audioFileInput.disabled = false;
  }
}

// File Upload Handler
audioFileInput.addEventListener("change", async (e) => {
  if (e.target.files && e.target.files[0]) {
    const file = e.target.files[0];
    // Reset input value immediately so duplicate change events cannot fire
    audioFileInput.value = "";
    await sendAudioForDetection(file, file.name);
  }
});

// ==============================================================================
// 5. Verdict Display: Shows EITHER AI OR HUMAN
// ==============================================================================
function triggerVerdict(verdict, confidence = null) {
  // Hide both cards first
  fakeCard.style.display = "none";
  realCard.style.display = "none";

  const confPercent = confidence !== null ? `${(confidence * 100).toFixed(1)}%` : null;

  if (verdict === "AI" || verdict === "FAKE") {
    // Show AI card
    fakeCard.style.display = "flex";
    const confText = confPercent ? ` (CONFIDENCE: ${confPercent})` : "";
    statusText.textContent = `VERDICT: AI-GENERATED SYNTHETIC VOICE DETECTED${confText}`;
    statusText.style.color = "#ff2a44";

    const fakeConfBanner = document.getElementById("fakeConfidenceBanner");
    if (fakeConfBanner && confPercent) {
      fakeConfBanner.textContent = `AI VOICE DETECTED • MODEL CONFIDENCE: ${confPercent} • STREAM MUTED`;
    }
  } else if (verdict === "HUMAN" || verdict === "REAL") {
    // Show HUMAN card
    realCard.style.display = "flex";
    const confText = confPercent ? ` (CONFIDENCE: ${confPercent})` : "";
    statusText.textContent = `VERDICT: GENUINE HUMAN VOICE CONFIRMED${confText}`;
    statusText.style.color = "#10b981";

    const realConfBanner = document.getElementById("realConfidenceBanner");
    if (realConfBanner && confPercent) {
      realConfBanner.textContent = `GENUINE HUMAN VOICE CONFIRMED • MODEL CONFIDENCE: ${confPercent} • CALL PROCEEDING`;
    }
  }

  // Ensure canvases inside newly shown card resize accurately
  setTimeout(resizeAllCanvases, 50);
}

function clearVerdict() {
  fakeCard.style.display = "none";
  realCard.style.display = "none";
  statusText.textContent = "STANDBY • AWAITING AUDIO (LOCAL AI DETECTION ENGINE)";
  statusText.style.color = "rgba(255, 255, 255, 0.85)";
}

// Event Listeners
recordBtn.addEventListener("click", toggleRecording);
setTimeout(resizeAllCanvases, 100);

// ==============================================================================
// 6. User Authentication State & Logout
// ==============================================================================
async function checkAuthSession() {
  const token = localStorage.getItem("aegis_token");
  try {
    const res = await fetch("/api/auth/me", {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) {
      window.location.href = "/login";
      return;
    }
    const data = await res.json();
    const navUserName = document.getElementById("navUserName");
    if (navUserName && data.user) {
      navUserName.textContent = data.user.username.toUpperCase();
    }
  } catch (err) {
    // Offline or network error
  }
}

async function handleSignOut() {
  try {
    const token = localStorage.getItem("aegis_token");
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  } catch (e) {}
  localStorage.removeItem("aegis_token");
  localStorage.removeItem("aegis_user");
  window.location.href = "/login";
}

document.addEventListener("DOMContentLoaded", checkAuthSession);

