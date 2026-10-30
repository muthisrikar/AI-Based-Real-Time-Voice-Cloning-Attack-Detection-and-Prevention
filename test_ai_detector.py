"""
test_ai_detector.py - Comprehensive Test & Benchmark Suite for Local AI Voice Detection
Validates:
- Physical NVIDIA GeForce RTX 4050 GPU selection, FP16 parameter residency, and VRAM allocation
- Pretrained nii-yamagishilab/xls-r-2b-anti-deepfake accuracy on human_sample.wav, tts_sample.wav, WhatsApp.mp4
- SynthID watermark provenance layer states (DETECTED, NOT_DETECTED, UNAVAILABLE)
- Forensic acoustic diagnostics telemetry
- Multi-format decoding (WAV, MP3, M4A/AAC, MP4, WebM/Opus live microphone)
- 20-second audio latency benchmark (cold start + 5 warm runs) with live nvidia-smi verification
"""

import os
import io
import time
import wave
import subprocess
import numpy as np
import torch
import soundfile as sf
from fastapi.testclient import TestClient

import ai_detector
from main import app

client = TestClient(app)


def generate_test_audio(sample_rate: int = 16000, duration_sec: float = 2.5, audio_type: str = "synthetic") -> bytes:
    """Generates an in-memory mono WAV file."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    if audio_type == "synthetic":
        waveform = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    elif audio_type == "silent":
        waveform = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    else:
        waveform = 0.4 * np.sin(2 * np.pi * 440 * t)

    waveform = np.clip(waveform, -1.0, 1.0)
    int16_pcm = (waveform * 32767).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_pcm.tobytes())
    return bio.getvalue()


def query_nvidia_smi() -> str:
    """Queries nvidia-smi for active GPU name, utilization, and memory usage."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "nvidia-smi unavailable"


def test_1_detector_initialization_and_rtx4050_gpu():
    """Verify detector singleton initializes on physical NVIDIA GeForce RTX 4050 in FP16."""
    detector = ai_detector.get_detector()
    assert detector is not None
    assert detector.model is not None
    assert detector.model_id == "nii-yamagishilab/xls-r-2b-anti-deepfake"
    assert 0 in detector.id2label and detector.id2label[0] == "fake"
    assert 1 in detector.id2label and detector.id2label[1] == "real"

    # Verify CUDA & physical RTX 4050 selection
    assert detector.cuda_available is True, f"Expected CUDA available=True, got {detector.cuda_available}"
    assert "RTX 4050" in detector.cuda_device_name, f"Expected RTX 4050 GPU, got {detector.cuda_device_name}"
    assert "Intel" not in detector.cuda_device_name, "Must NEVER select Intel integrated graphics"

    param_device = next(detector.model.parameters()).device
    param_dtype = next(detector.model.parameters()).dtype
    assert param_device.type == "cuda", f"Model parameters must reside on CUDA, found {param_device}"
    assert param_dtype == torch.float16, f"Model parameters must use torch.float16, found {param_dtype}"
    assert detector.vram_allocated_mb > 3500, f"Expected >3.5GB VRAM allocated for 2B FP16 model, got {detector.vram_allocated_mb:.1f}MB"

    smi_info = query_nvidia_smi()
    print(f"\n[Test 1 Passed] Detector initialized on {detector.device} ({detector.cuda_device_name})")
    print(f"                Model parameter device: {param_device}, dtype: {param_dtype}")
    print(f"                VRAM allocated: {detector.vram_allocated_mb:.1f} MB / {detector.vram_total_mb} MB")
    print(f"                nvidia-smi snapshot: {smi_info}")


def test_2_synthetic_tone_inference():
    """Verify inference on synthetic tone audio produces an AI prediction."""
    detector = ai_detector.get_detector()
    synth_wav = generate_test_audio(audio_type="synthetic")
    result = detector.predict(synth_wav, original_filename="synth_voice.wav")

    assert result["label"] == "AI"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "HUMAN" in result["probabilities"]
    assert "AI" in result["probabilities"]
    assert "timings_ms" in result
    assert "forensic_diagnostics" in result
    assert "synthid" in result
    print(f"\n[Test 2 Passed] Synthetic audio classified as: {result['label']} (Confidence: {result['confidence']*100:.1f}%, GPU: {result['timings_ms']['gpu_inference_ms']}ms)")


def test_3_known_human_sample_inference():
    """Verify inference on known real human speech (VoxCeleb) classifies as HUMAN."""
    detector = ai_detector.get_detector()
    human_file = "human_sample.wav"
    assert os.path.exists(human_file), "human_sample.wav must exist for verification"

    result = detector.predict(human_file, original_filename=human_file)
    assert result["label"] == "HUMAN"
    assert result["confidence"] >= 0.80
    assert result["probabilities"]["HUMAN"] >= 0.80
    print(f"\n[Test 3 Passed] Real human voice classified as: {result['label']} (Confidence: {result['confidence']*100:.1f}%, Total: {result['timings_ms']['total_ms']}ms)")


def test_4_known_tts_sample_inference():
    """Verify inference on known TTS synthetic voice classifies as AI."""
    detector = ai_detector.get_detector()
    tts_file = "tts_sample.wav"
    assert os.path.exists(tts_file), "tts_sample.wav must exist for verification"

    result = detector.predict(tts_file, original_filename=tts_file)
    assert result["label"] == "AI"
    assert result["confidence"] >= 0.70
    assert result["probabilities"]["AI"] >= 0.70
    print(f"\n[Test 4 Passed] TTS synthetic voice classified as: {result['label']} (Confidence: {result['confidence']*100:.1f}%, Total: {result['timings_ms']['total_ms']}ms)")


def test_5_whatsapp_ai_regression_target():
    """Verify mandatory known-AI regression sample WhatsApp.mp4 is classified as AI."""
    detector = ai_detector.get_detector()
    wa_file = "WhatsApp.mp4"
    assert os.path.exists(wa_file), "WhatsApp.mp4 regression target must exist"

    result = detector.predict(wa_file, original_filename=wa_file)
    assert result["label"] == "AI", f"Expected AI for {wa_file}, got {result['label']}"
    assert result["confidence"] >= 0.50
    print(f"\n[Test 5 Passed] Known-AI regression sample (WhatsApp.mp4) classified as: {result['label']} (Confidence: {result['confidence']*100:.1f}%, Total: {result['timings_ms']['total_ms']}ms)")


def test_6_failure_cases():
    """Verify robust error handling on empty, silent, or corrupt audio."""
    detector = ai_detector.get_detector()

    threw = False
    try:
        detector.predict(b"", "empty.wav")
    except ai_detector.AudioProcessingError as e:
        threw = True
        assert "empty" in str(e).lower()
    assert threw, "Expected AudioProcessingError for empty audio"

    threw = False
    try:
        detector.predict(b"RIFF\x00\x00\x00\x00corrupt_payload", "corrupt.wav")
    except ai_detector.AudioProcessingError:
        threw = True
    assert threw, "Expected AudioProcessingError for corrupt audio"

    threw = False
    silent_wav = generate_test_audio(audio_type="silent")
    try:
        detector.predict(silent_wav, "silent.wav")
    except ai_detector.AudioProcessingError as e:
        threw = True
        assert "silent" in str(e).lower()
    assert threw, "Expected AudioProcessingError for silent audio"

    print("\n[Test 6 Passed] All failure cases handled with proper AudioProcessingError.")


def test_7_api_detect_endpoint_success():
    """Verify POST /api/detect works with multipart audio upload."""
    with open("human_sample.wav", "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/detect",
        files={"file": ("human_test.wav", file_bytes, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["label"] == "HUMAN"
    assert "confidence" in data
    assert "timings_ms" in data
    assert "synthid" in data
    assert "forensic_diagnostics" in data
    print(f"\n[Test 7 Passed] POST /api/detect response: label={data['label']}, confidence={data['confidence']}, total_ms={data['timings_ms']['total_ms']}ms")


def test_8_api_detect_endpoint_errors():
    """Verify POST /api/detect returns 400 for empty or corrupted audio."""
    resp_empty = client.post(
        "/api/detect",
        files={"file": ("zero.wav", b"", "audio/wav")}
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()

    resp_corrupt = client.post(
        "/api/detect",
        files={"file": ("bad.wav", b"not-a-wav-file", "audio/wav")}
    )
    assert resp_corrupt.status_code == 400

    print("\n[Test 8 Passed] POST /api/detect returned 400 Bad Request on invalid payloads.")


def test_9_live_microphone_webm_detection():
    """Verify POST /api/detect works with browser-generated WebM/Opus microphone recordings."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=300:duration=2.0",
        "-c:a", "libopus",
        "-f", "webm",
        "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    webm_bytes = proc.stdout
    assert len(webm_bytes) > 0

    response = client.post(
        "/api/detect",
        files={"file": ("recorded_voice.webm", webm_bytes, "audio/webm")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["label"] in ["AI", "HUMAN"]
    print(f"\n[Test 9 Passed] Live microphone WebM/Opus decoded & classified as {data['label']} in {data['timings_ms']['total_ms']}ms")


def test_10_mp3_and_m4a_file_uploads():
    """Verify POST /api/detect works with standard MP3 and M4A (AAC) files."""
    # MP3
    cmd_mp3 = [
        "ffmpeg", "-y",
        "-i", "tts_sample.wav",
        "-c:a", "libmp3lame",
        "-f", "mp3",
        "pipe:1"
    ]
    proc_mp3 = subprocess.run(cmd_mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    resp_mp3 = client.post(
        "/api/detect",
        files={"file": ("sample_upload.mp3", proc_mp3.stdout, "audio/mpeg")}
    )
    assert resp_mp3.status_code == 200
    data_mp3 = resp_mp3.json()
    assert data_mp3["label"] == "AI"

    # M4A (AAC)
    temp_m4a = "temp_test_sample.m4a"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", "human_sample.wav", "-c:a", "aac", temp_m4a],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        with open(temp_m4a, "rb") as f:
            m4a_bytes = f.read()
        resp_m4a = client.post(
            "/api/detect",
            files={"file": ("human_upload.m4a", m4a_bytes, "audio/mp4")}
        )
        assert resp_m4a.status_code == 200
        data_m4a = resp_m4a.json()
        assert data_m4a["label"] == "HUMAN"
    finally:
        if os.path.exists(temp_m4a):
            os.remove(temp_m4a)

    print(f"\n[Test 10 Passed] MP3 (classified={data_mp3['label']}) and M4A/AAC (classified={data_m4a['label']}) uploads verified!")


def test_11_synthid_and_forensic_diagnostics():
    """Verify SynthID decision logic and forensic acoustic diagnostics structure."""
    detector = ai_detector.get_detector()
    res = detector.predict("human_sample.wav", original_filename="human_sample.wav")

    # Default SynthID state must be UNAVAILABLE (no fake heuristic) and must NOT prevent HUMAN classification
    assert res["synthid"]["status"] == "UNAVAILABLE"
    assert res["synthid"]["available"] is False
    assert res["label"] == "HUMAN"

    # Verify forensic diagnostic keys
    diag = res["forensic_diagnostics"]
    expected_keys = [
        "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_flatness",
        "spectral_rolloff_hz", "spectral_flux", "rms_energy", "dynamic_range_db",
        "silence_ratio", "clipping_ratio", "pitch_f0_hz", "voicing_confidence",
        "temporal_frame_variation"
    ]
    for k in expected_keys:
        assert k in diag, f"Missing forensic diagnostic metric: {k}"

    # Verify pluggable SynthID verifier: NOT_DETECTED must NEVER force HUMAN on AI audio
    detector.synthid_detector.register_verifier(lambda wav, sr: {"status": "NOT_DETECTED", "confidence": 0.0})
    try:
        res_ai = detector.predict("tts_sample.wav", original_filename="tts_sample.wav")
        assert res_ai["synthid"]["status"] == "NOT_DETECTED"
        assert res_ai["label"] == "AI", "NOT_DETECTED watermark must never override AI detection"

        # Verify DETECTED watermark triggers AI classification
        detector.synthid_detector.register_verifier(lambda wav, sr: {"status": "DETECTED", "confidence": 0.98})
        res_wm = detector.predict("human_sample.wav", original_filename="watermarked.wav")
        assert res_wm["synthid"]["status"] == "DETECTED"
        assert res_wm["label"] == "AI"
    finally:
        # Restore default honest UNAVAILABLE state
        detector.synthid_detector._custom_verifier = None
        detector.synthid_detector.available = False

    print("\n[Test 11 Passed] SynthID provenance layer (UNAVAILABLE / NOT_DETECTED / DETECTED) and 12 forensic metrics verified!")


def test_12_health_readiness_probe():
    """Verify /api/health reports model readiness and RTX 4050 hardware details."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert data["model_ready"] is True
    assert "nii-yamagishilab/xls-r-2b-anti-deepfake" in data["model_info"]
    assert data["hardware"]["cuda_available"] is True
    assert "RTX 4050" in data["hardware"]["cuda_device_name"]
    print(f"\n[Test 12 Passed] /api/health verified: {data['model_info']}")


def test_13_twenty_second_benchmark():
    """
    Mandatory 20-second audio benchmark:
    - Measures 1 initial run + 5 warm runs on a 20.0-second speech clip
    - Captures separated stage timings and live nvidia-smi GPU utilization
    - Verifies total latency is in seconds (target < 5.0s, actual ~0.4s on RTX 4050)
    """
    detector = ai_detector.get_detector()

    # Construct a 20.0-second real speech WAV clip in memory from human_sample.wav
    data, sr = sf.read("human_sample.wav", dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    target_len = sr * 20  # 20.0 seconds
    repeats = (target_len // len(data)) + 1
    clip_20s = np.tile(data, repeats)[:target_len]

    bio = io.BytesIO()
    sf.write(bio, clip_20s, sr, format="WAV", subtype="PCM_16")
    wav_20s_bytes = bio.getvalue()

    print("\n================ 20-SECOND AUDIO GPU BENCHMARK ================")
    # Initial run
    res_cold = detector.predict(wav_20s_bytes, original_filename="benchmark_20s_human.wav")
    smi_during = query_nvidia_smi()
    print(f"Initial 20s Run: total={res_cold['timings_ms']['total_ms']:.2f}ms "
          f"(decode={res_cold['timings_ms']['decode_ms']:.2f}ms, "
          f"preprocess={res_cold['timings_ms']['preprocess_ms']:.2f}ms, "
          f"gpu_infer={res_cold['timings_ms']['gpu_inference_ms']:.2f}ms, "
          f"post={res_cold['timings_ms']['postprocess_ms']:.2f}ms) -> Label={res_cold['label']}")

    warm_totals = []
    warm_gpu = []
    for run_idx in range(1, 6):
        r = detector.predict(wav_20s_bytes, original_filename=f"benchmark_20s_warm_{run_idx}.wav")
        t_ms = r["timings_ms"]
        warm_totals.append(t_ms["total_ms"])
        warm_gpu.append(t_ms["gpu_inference_ms"])
        if run_idx == 3:
            smi_during = query_nvidia_smi()
        print(f"Warm Run #{run_idx}:     total={t_ms['total_ms']:.2f}ms "
              f"(decode={t_ms['decode_ms']:.2f}ms, "
              f"preprocess={t_ms['preprocess_ms']:.2f}ms, "
              f"gpu_infer={t_ms['gpu_inference_ms']:.2f}ms, "
              f"post={t_ms['postprocess_ms']:.2f}ms) -> Label={r['label']}")

    avg_total_ms = sum(warm_totals) / len(warm_totals)
    avg_gpu_ms = sum(warm_gpu) / len(warm_gpu)
    print(f"Average Warm 20s Total Time: {avg_total_ms:.2f} ms ({avg_total_ms/1000.0:.3f} s)")
    print(f"Average Warm 20s GPU Time:   {avg_gpu_ms:.2f} ms ({avg_gpu_ms/1000.0:.3f} s)")
    print(f"Active nvidia-smi telemetry: {smi_during}")
    print("===============================================================")

    assert avg_total_ms < 5000.0, f"20s audio inference took too long: {avg_total_ms:.2f}ms"
    assert res_cold["label"] == "HUMAN"


if __name__ == "__main__":
    test_1_detector_initialization_and_rtx4050_gpu()
    test_2_synthetic_tone_inference()
    test_3_known_human_sample_inference()
    test_4_known_tts_sample_inference()
    test_5_whatsapp_ai_regression_target()
    test_6_failure_cases()
    test_7_api_detect_endpoint_success()
    test_8_api_detect_endpoint_errors()
    test_9_live_microphone_webm_detection()
    test_10_mp3_and_m4a_file_uploads()
    test_11_synthid_and_forensic_diagnostics()
    test_12_health_readiness_probe()
    test_13_twenty_second_benchmark()
    print("\n===========================================")
    print("ALL 13 END-TO-END GPU & ACCURACY TESTS PASSED!")
    print("===========================================")
