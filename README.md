# Real-Time AI Voice Cloning & Deepfake Detection Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-CUDA%20Accelerated-2496ed.svg)](https://www.docker.com/)
[![Model](https://img.shields.io/badge/Model-XLS--R--2B--AntiDeepfake-blueviolet.svg)](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, real-time AI-generated voice and audio deepfake detection system. Combines state-of-the-art foundation models (**Yamagishi Lab XLS-R-2B AntiDeepfake**, 2.16B parameters), dual-signal digital signal processing (DSP), forensic acoustic telemetry, and modular SynthID watermark provenance verification into a unified full-duplex dashboard and API.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │    Client Dashboard / Microphone /     │
                      │        Audio File Ingestion            │
                      └──────────────────┬─────────────────────┘
                                         │
                   ┌─────────────────────┴───────────────────────┐
                   │                                             │
                   ▼ (HTTP Multipart Upload / WebM Opus)         ▼ (WebSocket 150ms PCM Stream)
        ┌────────────────────────────┐               ┌────────────────────────────┐
        │   FastAPI Detection API    │               │  WebSocket Stream Router   │
        │      (/api/detect)         │               │        (/ws/audio)         │
        └──────────────┬─────────────┘               └─────────────┬──────────────┘
                       │                                           │
                       ▼                                           ▼
        ┌────────────────────────────┐               ┌────────────────────────────┐
        │ FFmpeg Multi-Format Engine │               │   DSP Dual-Signal Engine   │
        │ (WebM/MP3/M4A/WAV -> 16kHz)│               │        (detector.py)       │
        └──────────────┬─────────────┘               └─────────────┬──────────────┘
                       │                                           │
                       ▼                                           ▼
        ┌────────────────────────────┐               ┌────────────────────────────┐
        │  XLS-R-2B Anti-Deepfake    │               │  Branch A: LFCC & Phase    │
        │  (2.16B Params, FP16 CUDA) │               │  Branch B: Jitter/Shimmer  │
        │  Batched 2.0s Time Windows │               │  Sub-150ms Loop / Mute     │
        └──────────────┬─────────────┘               └────────────────────────────┘
                       │
        ┌──────────────┴─────────────┐
        │ Forensic & Provenance      │
        │ - 12 Acoustic Diagnostics  │
        │ - SynthID Watermark Layer  │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │ Binary Verdict: HUMAN / AI │
        │ Sub-second Response (<0.4s)│
        └────────────────────────────┘
```

---

## Key Features

### 1. State-of-the-Art Deepfake Foundation Model
- **Model**: `nii-yamagishilab/xls-r-2b-anti-deepfake` (2.16 billion parameters).
- **Architecture**: 48-layer transformer Wav2Vec2 encoder with `AdaptiveAvgPool1d` and binary classification projection.
- **Hardware Acceleration**: Automatic NVIDIA CUDA discovery (FP16 precision) with physical GPU prioritization (e.g., NVIDIA GeForce RTX 4050/4090) and graceful CPU fallback.
- **Batched Sliding Windows**: 2.0-second multi-scale temporal window scanning with mini-batch inference, yielding **sub-second inference (~300 ms on a 20-second clip)**.

### 2. Real-Time Dual-Signal DSP Analyzer (`detector.py`)
- Engineered for sub-150ms live VoIP/telephony stream monitoring.
- **Branch A (Digital Synthesis Artifacts)**:
  - Linear Frequency Cepstral Coefficients (LFCC) filterbank distribution to detect vocoder high-band spectral ripple.
  - Spectral phase continuity and Instantaneous Frequency Deviation (IFD).
  - High-frequency brick-wall truncation detection (>7.2 kHz).
- **Branch B (Biological Speech Dynamics)**:
  - Micro-tremor perturbation tracking: Pitch Jitter and Amplitude Shimmer.
  - Normalized autocorrelation fundamental frequency ($F_0$) trajectory.
  - Physical implausibility detection: mathematical zero-variance silence during micro-pauses.

### 3. Forensic Acoustic Diagnostics & SynthID Provenance
- Computes **12 non-overriding forensic telemetry metrics**:
  - Spectral: Centroid, Bandwidth, Flatness, Rolloff, Flux.
  - Dynamics: RMS Energy, Dynamic Range ($dB$), Silence Ratio, Clipping Ratio.
  - Voice: Pitch ($F_0$), Voicing Confidence, Frame-to-frame Variation.
- **SynthID Audio Watermark Layer**: Modular architecture reporting `DETECTED`, `NOT_DETECTED`, or `UNAVAILABLE` without falsified heuristics, ready for official verifier plugins.

### 4. Robust Audio Ingestion Pipeline
- Native in-memory decoding combined with FFmpeg conversion.
- Directly supports:
  - **Live Microphone**: Browser `MediaRecorder` WebM / Opus format.
  - **Uploaded Files**: WAV, MP3, AAC, M4A, FLAC, OGG, and MP4 containers.
  - Validates and handles corrupt, empty, and silent audio with structured error codes.

### 5. Holographic Cyberpunk User Interface
- Real-time HTML5 canvas audio visualizers:
  - Holographic fluid wave ribbons reflecting live recording volume.
  - Mini spectrum displays for digital artifacts and biological speech patterns.
  - Interactive recording toggle with single-flight request guards.
- Clear binary verdicts: **`GENUINE HUMAN VOICE`** or **`AI VOICE DETECTED`** with model confidence and threat mitigation alerts.
- Built-in user authentication with PBKDF2-HMAC-SHA256 salted password hashing and SQLite session management.

---

## Directory Structure

```text
├── ai_detector.py        # Core XLS-R-2B GPU/CPU inference engine & SynthID layer
├── detector.py           # Real-time dual-signal DSP analyzer (LFCC, Jitter, Shimmer)
├── main.py               # FastAPI ASGI server, WebSocket router, and REST API
├── database.py           # SQLite authentication and session management
├── index.html            # Holographic voice detection dashboard
├── app.js                # Frontend audio recording, visualizers, and API dispatcher
├── login.html            # Authentication and registration portal
├── login.js              # Authentication frontend script
├── style.css             # Cyberpunk holographic dashboard styling
├── static/               # Static assets mirrored for production deployment
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition with FFmpeg and CUDA support
├── docker-compose.yml    # Docker Compose configuration with GPU passthrough
├── DEPLOYMENT.md         # GCP Shielded VM enterprise deployment guide
├── test_ai_detector.py   # Full test & benchmark suite (GPU, accuracy, formats, 20s latency)
├── test_live_http.py     # Live HTTP server integration tests
├── human_sample.wav      # Sample real speech audio for validation
├── tts_sample.wav        # Sample synthetic TTS audio for validation
└── WhatsApp.mp4          # Sample compressed speech for regression validation
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- FFmpeg installed and available on system `PATH`
- *(Optional)* NVIDIA GPU with CUDA drivers for acceleration

### 1. Local Run

```bash
# 1. Clone repository
git clone https://github.com/maithreshkumar/Real-Time-Voice-Cloning-Detection.git
cd Real-Time-Voice-Cloning-Detection

# 2. Create and activate virtual environment
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate

# 3. Install PyTorch with CUDA (or CPU fallback)
# For CUDA 12.4 / 12.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# For CPU only:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Start the server
python main.py
```

Access the dashboard at **`http://localhost:8000`**.

---

### 2. Docker Deployment (with NVIDIA GPU Support)

The included `Dockerfile` and `docker-compose.yml` configure a self-contained container with FFmpeg, CUDA PyTorch, and persistent volume caching.

```bash
# Build and start container in detached mode
docker compose up -d --build

# Inspect logs
docker compose logs -f

# Stop container
docker compose down
```

---

## API Reference

### Health & Hardware Telemetry
`GET /api/health`

Returns engine status, active device, VRAM allocation, and SynthID provider availability.

```json
{
  "status": "HEALTHY",
  "service": "VoiceCloningDetectorEngine",
  "model_ready": true,
  "model_info": "nii-yamagishilab/xls-r-2b-anti-deepfake on cuda:0 (NVIDIA GeForce RTX 4050 Laptop GPU, torch.float16)",
  "hardware": {
    "cuda_available": true,
    "cuda_device_index": 0,
    "cuda_device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "vram_total_mb": 6140,
    "vram_allocated_mb": 4126.8,
    "vram_reserved_mb": 4150.0,
    "model_device": "cuda:0",
    "model_dtype": "torch.float16",
    "ffmpeg_available": true,
    "synthid_available": false
  }
}
```

### Voice Deepfake Detection
`POST /api/detect` (Multipart Form with `file: UploadFile`)

Analyzes an uploaded audio file or recorded microphone blob.

**Response**:
```json
{
  "status": "SUCCESS",
  "label": "AI",
  "confidence": 0.9997,
  "probabilities": {
    "HUMAN": 0.0003,
    "AI": 0.9997
  },
  "duration_seconds": 5.94,
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "dtype": "torch.float16",
  "model_id": "nii-yamagishilab/xls-r-2b-anti-deepfake",
  "synthid": {
    "status": "UNAVAILABLE",
    "available": false,
    "provider": "Google SynthID Audio Provenance Layer",
    "reason": "No official offline Google SynthID Audio watermark detector SDK is publicly available."
  },
  "timings_ms": {
    "upload_read_ms": 1.2,
    "decode_ms": 0.6,
    "ffmpeg_ms": 0.0,
    "resample_ms": 0.0,
    "preprocess_ms": 28.4,
    "gpu_inference_ms": 110.5,
    "postprocess_ms": 0.1,
    "total_ms": 140.8
  },
  "forensic_diagnostics": {
    "spectral_centroid_hz": 1845.2,
    "spectral_bandwidth_hz": 1620.4,
    "spectral_flatness": 0.00412,
    "spectral_rolloff_hz": 3450.0,
    "spectral_flux": 0.321,
    "rms_energy": 0.0821,
    "dynamic_range_db": 42.5,
    "silence_ratio": 0.081,
    "clipping_ratio": 0.0,
    "pitch_f0_hz": 142.5,
    "voicing_confidence": 0.78,
    "temporal_frame_variation": 0.412
  }
}
```

---

## Validation & Benchmarks

Run the complete automated test suite:

```bash
# Run unit and benchmark tests
python test_ai_detector.py

# Run live HTTP integration tests
python test_live_http.py
```

### Benchmark Summary (20-Second Speech Sample on RTX 4050 FP16)
- **Cold Run**: ~350 ms
- **Warm Average**: **~320 ms** (GPU Inference: ~285 ms, Preprocessing: ~30 ms, Decode: ~1 ms)
- **Speedup vs. CPU Sequential**: **>1,800x faster**
- **Classification Accuracy**:
  - Known Real Speech (`human_sample.wav`): `HUMAN` (99.97% confidence)
  - Known Synthetic Speech (`tts_sample.wav`): `AI` (99.97% confidence)
  - Known Regression Target (`WhatsApp.mp4`): `AI` (77.0% confidence)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
