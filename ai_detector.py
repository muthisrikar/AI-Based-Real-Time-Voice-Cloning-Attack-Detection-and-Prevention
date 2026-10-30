"""
ai_detector.py - High-Performance GPU-Accelerated XLS-R-2B AntiDeepfake Detection Engine
Uses the pretrained nii-yamagishilab/xls-r-2b-anti-deepfake foundation model (2.16B parameters)
bound explicitly to the physical NVIDIA GeForce RTX 4050 GPU (CUDA FP16) with batched window
inference, SynthID provenance layer integration, and acoustic forensic telemetry.
"""

import io
import os
import sys
import time
import tempfile
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Union, BinaryIO, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import torchaudio.models.wav2vec2.utils.import_fairseq as ifairseq

logger = logging.getLogger("ai_detector")
logging.basicConfig(level=logging.INFO)


class AudioProcessingError(Exception):
    """Raised when audio decoding, validation, or preprocessing fails."""
    pass


class SynthIDWatermarkDetector:
    """
    SynthID Audio-Watermark Provenance Layer.

    Technical Grounding:
    - Google DeepMind's open-source SynthID release (`synthid-text`) only supports LLM text
      token watermarking.
    - Google's SynthID for Audio (used for Lyria / Gemini / NotebookLM audio) relies on
      proprietary spectrogram watermark detectors hosted inside Google's internal SynthID
      verification infrastructure and has no publicly released offline weights or Python SDK.
    - Per strict forensic integrity rules, this layer NEVER fabricates a SynthID result
      using spectral heuristics or filename inspection. Unless an official local SynthID
      audio verifier callable is explicitly registered, it reports `UNAVAILABLE` with the
      exact technical reason, and the classifier defers 100% to `xls-r-2b-anti-deepfake`.
    """

    def __init__(self):
        self._custom_verifier = None
        self.available = False
        self.provider_name = "Google SynthID Audio Provenance Layer"
        self.unavailable_reason = (
            "No official offline Google SynthID Audio watermark detector SDK or model weights "
            "are publicly available (public synthid-text SDK supports text tokens only)."
        )

    def register_verifier(self, verifier_fn):
        """Allows plugging in an official local SynthID audio verifier if released."""
        if callable(verifier_fn):
            self._custom_verifier = verifier_fn
            self.available = True

    def detect(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Evaluates audio for a SynthID watermark.
        Returns status in {"DETECTED", "NOT_DETECTED", "UNAVAILABLE"}.
        """
        if not self.available or self._custom_verifier is None:
            return {
                "status": "UNAVAILABLE",
                "available": False,
                "confidence": None,
                "provider": self.provider_name,
                "reason": self.unavailable_reason,
            }

        try:
            res = self._custom_verifier(waveform, sample_rate)
            status = res.get("status", "NOT_DETECTED")
            if status not in ("DETECTED", "NOT_DETECTED", "UNAVAILABLE"):
                status = "UNAVAILABLE"
            return {
                "status": status,
                "available": True,
                "confidence": res.get("confidence"),
                "provider": self.provider_name,
                "reason": res.get("reason", "Official SynthID verifier executed."),
            }
        except Exception as exc:
            logger.warning(f"SynthID verifier execution error: {exc}")
            return {
                "status": "UNAVAILABLE",
                "available": False,
                "confidence": None,
                "provider": self.provider_name,
                "reason": f"SynthID verifier error: {str(exc)}",
            }


class PretrainedXLSR2BAntiDeepfake(nn.Module):
    """
    Native PyTorch / TorchAudio implementation of the Yamagishi Lab
    XLS-R-2B-AntiDeepfake model architecture (2.16B parameters).
    Combines Wav2Vec2 48-layer transformer backbone with AdaptiveAvgPool1d
    and Linear(1920, 2) binary classification head.
    """
    def __init__(self):
        super().__init__()
        self.ssl_model = torchaudio.models.wav2vec2_model(
            extractor_mode="layer_norm",
            extractor_conv_layer_config=None,
            extractor_conv_bias=True,
            encoder_embed_dim=1920,
            encoder_projection_dropout=0.0,
            encoder_pos_conv_kernel=128,
            encoder_pos_conv_groups=16,
            encoder_num_layers=48,
            encoder_num_heads=16,
            encoder_attention_dropout=0.0,
            encoder_ff_interm_features=7680,
            encoder_ff_interm_dropout=0.0,
            encoder_dropout=0.0,
            encoder_layer_norm_first=True,
            encoder_layer_drop=0.0,
            aux_num_out=None
        )
        self.adap_pool1d = nn.AdaptiveAvgPool1d(1)
        self.proj_fc = nn.Linear(1920, 2)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [batch, time]
        Returns: logits [batch, 2] where index 0 is FAKE (AI) and index 1 is REAL (HUMAN).
        """
        out, _ = self.ssl_model(wav)               # [B, T, 1920]
        emb = out.transpose(1, 2)                  # [B, 1920, T]
        pooled = self.adap_pool1d(emb).squeeze(-1) # [B, 1920]
        logits = self.proj_fc(pooled)              # [B, 2]
        return logits


def extract_forensic_diagnostics(waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
    """
    Extracts comprehensive acoustic and spectral forensic telemetry from the 16 kHz mono
    waveform. These metrics are logged and returned strictly as diagnostic telemetry
    and NEVER override the neural anti-deepfake model's classification.
    """
    n_samples = len(waveform)
    if n_samples == 0:
        return {}

    eps = 1e-10
    rms_energy = float(np.sqrt(np.mean(waveform ** 2) + eps))
    peak_amp = float(np.max(np.abs(waveform)))

    # Clipping ratio: fraction of samples at or near digital full-scale (>= 0.995)
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.995))

    # Frame-level analysis (25ms window = 400 samples, 10ms hop = 160 samples)
    n_fft = 512
    win_length = 400
    hop_length = 160

    if n_samples >= win_length:
        num_frames = 1 + (n_samples - win_length) // hop_length
        # Strided framing for fast vectorized STFT
        shape = (num_frames, win_length)
        strides = (waveform.strides[0] * hop_length, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
        window = np.hanning(win_length).astype(np.float32)
        windowed = frames * window
    else:
        windowed = (waveform * np.hanning(n_samples).astype(np.float32))[None, :]

    # Frame RMS & dynamic range / silence ratio
    frame_rms = np.sqrt(np.mean(windowed ** 2, axis=1) + eps)
    frame_db = 20.0 * np.log10(frame_rms + eps)
    p95_db = float(np.percentile(frame_db, 95))
    p05_db = float(np.percentile(frame_db, 5))
    dynamic_range_db = float(max(0.0, p95_db - p05_db))

    # Silence ratio: frames more than 40 dB below peak or below absolute floor (1e-3 RMS)
    silence_threshold = max(1e-3, peak_amp * 0.01)
    silence_ratio = float(np.mean(frame_rms < silence_threshold))

    # Temporal frame-to-frame RMS variation
    if len(frame_rms) > 1:
        temporal_frame_variation = float(np.std(np.diff(frame_rms)) / (np.mean(frame_rms) + eps))
    else:
        temporal_frame_variation = 0.0

    # Magnitude spectrum
    mag_spec = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)) + eps
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # Spectral centroid & bandwidth
    spec_sum = np.sum(mag_spec, axis=1, keepdims=True)
    spec_norm = mag_spec / spec_sum
    centroids = np.sum(spec_norm * freqs[None, :], axis=1)
    spectral_centroid_hz = float(np.mean(centroids))

    bandwidths = np.sqrt(np.sum(spec_norm * ((freqs[None, :] - centroids[:, None]) ** 2), axis=1))
    spectral_bandwidth_hz = float(np.mean(bandwidths))

    # Spectral flatness (geometric mean / arithmetic mean)
    geom_mean = np.exp(np.mean(np.log(mag_spec), axis=1))
    arith_mean = np.mean(mag_spec, axis=1)
    spectral_flatness = float(np.mean(geom_mean / (arith_mean + eps)))

    # Spectral rolloff (85% cumulative energy)
    cumsum_energy = np.cumsum(mag_spec ** 2, axis=1)
    total_frame_energy = cumsum_energy[:, -1:]
    rolloff_mask = cumsum_energy >= (0.85 * total_frame_energy)
    rolloff_indices = np.argmax(rolloff_mask, axis=1)
    spectral_rolloff_hz = float(np.mean(freqs[rolloff_indices]))

    # Spectral flux (frame-to-frame normalized magnitude difference)
    if mag_spec.shape[0] > 1:
        diff_spec = np.diff(mag_spec / (np.max(mag_spec, axis=1, keepdims=True) + eps), axis=0)
        spectral_flux = float(np.mean(np.sqrt(np.sum(diff_spec ** 2, axis=1))))
    else:
        spectral_flux = 0.0

    # Pitch (F0) & Voicing confidence via autocorrelation on active voiced frames
    min_lag = int(sample_rate / 400)  # 400 Hz max F0
    max_lag = int(sample_rate / 70)   # 70 Hz min F0
    f0_estimates = []
    voicing_peaks = []

    # Sample up to 25 highest-energy frames for fast autocorrelation F0 estimation
    active_indices = np.argsort(frame_rms)[-min(25, len(frame_rms)):]
    for idx in active_indices:
        frm = frames[idx] if n_samples >= win_length else waveform
        if len(frm) > max_lag:
            frm_zm = frm - np.mean(frm)
            energy0 = float(np.sum(frm_zm ** 2))
            if energy0 > 1e-6:
                corr = np.correlate(frm_zm, frm_zm, mode="full")[len(frm_zm) - 1:]
                lag_win = corr[min_lag:max_lag]
                if len(lag_win) > 0:
                    pk_idx = int(np.argmax(lag_win)) + min_lag
                    norm_pk = float(corr[pk_idx] / (energy0 + eps))
                    voicing_peaks.append(np.clip(norm_pk, 0.0, 1.0))
                    if norm_pk >= 0.30:
                        f0_estimates.append(float(sample_rate / pk_idx))

    pitch_f0_hz = round(float(np.median(f0_estimates)), 2) if f0_estimates else 0.0
    voicing_confidence = round(float(np.mean(voicing_peaks)), 4) if voicing_peaks else 0.0

    return {
        "spectral_centroid_hz": round(spectral_centroid_hz, 2),
        "spectral_bandwidth_hz": round(spectral_bandwidth_hz, 2),
        "spectral_flatness": round(spectral_flatness, 5),
        "spectral_rolloff_hz": round(spectral_rolloff_hz, 2),
        "spectral_flux": round(spectral_flux, 4),
        "rms_energy": round(rms_energy, 5),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 5),
        "pitch_f0_hz": pitch_f0_hz,
        "voicing_confidence": voicing_confidence,
        "temporal_frame_variation": round(temporal_frame_variation, 4),
    }


class DeepfakeVoiceDetector:
    """
    Singleton AI Voice Detector that loads the local pretrained
    nii-yamagishilab/xls-r-2b-anti-deepfake model once onto the physical
    NVIDIA GeForce RTX 4050 GPU (CUDA FP16) and executes batched window inference.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DeepfakeVoiceDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "nii-yamagishilab/xls-r-2b-anti-deepfake"):
        if getattr(self, "_initialized", False):
            return

        self.model_id = model_id
        self.model_variant = "2B (2.16B parameters)"
        self.sample_rate = 16000
        self.max_duration_seconds = 30.0   # Limit to 30s to prevent unbounded memory usage
        self.min_duration_seconds = 0.25   # Minimum required audio duration
        self.window_duration_seconds = 2.0 # 2.0s sliding analysis window
        self.hop_duration_seconds = 2.0    # 2.0s hop step
        self.max_batch_windows = 8         # Mini-batch size on 6GB RTX 4050 to avoid VRAM OOM

        self.cuda_available = False
        self.cuda_device_index: Optional[int] = None
        self.cuda_device_name: str = "None"
        self.vram_total_mb: int = 0
        self.vram_allocated_mb: float = 0.0
        self.device = None
        self.dtype = None
        self.model = None
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.synthid_detector = SynthIDWatermarkDetector()
        self.id2label = {0: "fake", 1: "real"}

        self._load_model()
        self._initialized = True

    def _select_nvidia_rtx_device(self) -> Tuple[torch.device, torch.dtype]:
        """
        Inspects all available PyTorch CUDA devices and explicitly selects the
        physical NVIDIA GeForce RTX 4050 GPU. Never selects Intel integrated graphics.
        """
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning(
                "WARNING: torch.cuda.is_available() is False! "
                "Ensure CUDA-enabled PyTorch (+cu124/+cu128) and NVIDIA drivers are active."
            )
            self.cuda_device_index = None
            self.cuda_device_name = "CPU (CUDA unavailable)"
            self.vram_total_mb = 0
            return torch.device("cpu"), torch.bfloat16

        device_count = torch.cuda.device_count()
        devices_info = []
        rtx_4050_idx = None
        nvidia_idx = None

        for idx in range(device_count):
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            vram_mb = int(props.total_memory // (1024 * 1024))
            devices_info.append((idx, name, vram_mb))
            logger.info(f"Discovered CUDA device [{idx}]: {name} ({vram_mb} MB VRAM)")

            name_lower = name.lower()
            if "intel" in name_lower or "uhd" in name_lower or "iris" in name_lower:
                logger.info(f"Skipping integrated non-NVIDIA adapter [{idx}]: {name}")
                continue

            if "rtx 4050" in name_lower:
                rtx_4050_idx = idx
            elif "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower:
                if nvidia_idx is None:
                    nvidia_idx = idx

        selected_idx = rtx_4050_idx if rtx_4050_idx is not None else (nvidia_idx if nvidia_idx is not None else 0)
        torch.cuda.set_device(selected_idx)
        torch.backends.cudnn.benchmark = False

        self.cuda_device_index = selected_idx
        self.cuda_device_name = torch.cuda.get_device_name(selected_idx)
        self.vram_total_mb = int(torch.cuda.get_device_properties(selected_idx).total_memory // (1024 * 1024))

        return torch.device(f"cuda:{selected_idx}"), torch.float16

    def _log_startup_banner(self):
        """Logs the required production startup diagnostic summary."""
        param_device = str(next(self.model.parameters()).device) if self.model is not None else "unloaded"
        param_dtype = str(next(self.model.parameters()).dtype) if self.model is not None else "unloaded"
        ffmpeg_status = f"available ({self.ffmpeg_path})" if self.ffmpeg_path else "unavailable"
        synthid_status = (
            "AVAILABLE" if self.synthid_detector.available
            else f"UNAVAILABLE ({self.synthid_detector.unavailable_reason})"
        )

        banner = (
            "\n=== VOICE DETECTOR ===\n"
            f"Model: {self.model_id}\n"
            f"Variant: {self.model_variant}\n"
            f"CUDA available: {self.cuda_available}\n"
            f"CUDA device index: {self.cuda_device_index}\n"
            f"CUDA device name: {self.cuda_device_name}\n"
            f"VRAM: {self.vram_total_mb} MB total ({self.vram_allocated_mb:.1f} MB allocated by model)\n"
            f"Model device: {param_device}\n"
            f"Model dtype: {param_dtype}\n"
            f"FFmpeg: {ffmpeg_status}\n"
            f"SynthID detector: {synthid_status}\n"
            "======================"
        )
        logger.info(banner)
        print(banner, flush=True)

    def _load_model(self):
        """
        Loads the pretrained model weights from safetensors once onto the physical
        NVIDIA GeForce RTX 4050 GPU in FP16 and runs a warmup pass to pre-compile CUDA kernels.
        """
        t0 = time.perf_counter()
        self.device, self.dtype = self._select_nvidia_rtx_device()

        logger.info(f"Loading pretrained anti-deepfake model: {self.model_id} onto {self.device} ({self.dtype})")
        try:
            fast_cache_dir = Path("/fast_cache")
            fast_weights = fast_cache_dir / "model.safetensors"
            if fast_cache_dir.is_dir() and fast_weights.exists() and fast_weights.stat().st_size > 8_000_000_000:
                weights_path = str(fast_weights)
                logger.info(f"Using fast native Linux Docker volume weights: {weights_path}")
            else:
                try:
                    weights_path = hf_hub_download(self.model_id, "model.safetensors", local_files_only=True)
                except Exception:
                    weights_path = hf_hub_download(self.model_id, "model.safetensors")

                if fast_cache_dir.is_dir():
                    logger.info("Staging model.safetensors into persistent native Linux Docker volume (/fast_cache) for fast restarts...")
                    tmp_fast = fast_cache_dir / "model.safetensors.tmp"
                    with open(weights_path, "rb") as f_src, open(tmp_fast, "wb") as f_dst:
                        shutil.copyfileobj(f_src, f_dst, length=16 * 1024 * 1024)
                    tmp_fast.replace(fast_weights)
                    weights_path = str(fast_weights)
                    logger.info("Staging to /fast_cache complete.")

            import gc
            prev_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(self.dtype)
                self.model = PretrainedXLSR2BAntiDeepfake()
            finally:
                torch.set_default_dtype(prev_dtype)

            # Move empty FP16 skeleton directly to RTX 4050 VRAM before streaming weights
            # so WSL2 / Docker CPU RAM never holds duplicate 8.65GB / 4.32GB weight dicts
            self.model.to(device=self.device, dtype=self.dtype)
            gc.collect()

            ssl_params = dict(self.model.ssl_model.named_parameters())
            ssl_buffers = dict(self.model.ssl_model.named_buffers())
            pos_conv_dict = {}
            head_dict = {}

            with torch.no_grad(), safe_open(weights_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("m_ssl.model."):
                        fk = k.replace("m_ssl.model.", "")
                        mapped_k = ifairseq._map_key(fk)
                        if mapped_k is not None:
                            if "pos_conv_embed" in mapped_k:
                                pos_conv_dict[mapped_k] = f.get_tensor(k).to(dtype=self.dtype)
                            elif mapped_k in ssl_params:
                                ssl_params[mapped_k].copy_(f.get_tensor(k), non_blocking=True)
                            elif mapped_k in ssl_buffers:
                                ssl_buffers[mapped_k].copy_(f.get_tensor(k), non_blocking=True)
                    elif k.startswith("proj_fc."):
                        head_dict[k.replace("proj_fc.", "")] = f.get_tensor(k).to(dtype=self.dtype)

            if pos_conv_dict:
                self.model.ssl_model.load_state_dict(pos_conv_dict, strict=False)
                del pos_conv_dict
            self.model.proj_fc.load_state_dict(head_dict, strict=True)
            del head_dict
            gc.collect()
            self.model.eval()

            # Verify parameter residency on target GPU
            actual_param_device = next(self.model.parameters()).device
            if self.cuda_available and actual_param_device.type != "cuda":
                raise RuntimeError(
                    f"Model parameters failed to move to CUDA GPU! Found on {actual_param_device}"
                )

            # Warmup GPU kernels with a single 2.0s dummy pass
            if self.cuda_available:
                with torch.inference_mode():
                    dummy = torch.zeros((1, int(self.window_duration_seconds * self.sample_rate)),
                                        device=self.device, dtype=self.dtype)
                    _ = self.model(dummy)
                    torch.cuda.synchronize(self.device)
                self.vram_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

            load_sec = time.perf_counter() - t0
            logger.info(f"Model {self.model_id} loaded and warmed up in {load_sec:.2f}s.")
            self._log_startup_banner()
        except Exception as e:
            logger.error(f"Failed to load anti-deepfake model {self.model_id}: {e}", exc_info=True)
            raise RuntimeError(f"Unable to initialize local anti-deepfake model: {e}")

    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns live GPU / device telemetry for health checks and diagnostics."""
        param_device = str(next(self.model.parameters()).device) if self.model is not None else "unloaded"
        param_dtype = str(next(self.model.parameters()).dtype) if self.model is not None else "unloaded"
        vram_alloc = (
            round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2)
            if self.cuda_available and self.device is not None else 0.0
        )
        vram_reserved = (
            round(torch.cuda.memory_reserved(self.device) / (1024 * 1024), 2)
            if self.cuda_available and self.device is not None else 0.0
        )
        return {
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "cuda_available": self.cuda_available,
            "cuda_device_index": self.cuda_device_index,
            "cuda_device_name": self.cuda_device_name,
            "vram_total_mb": self.vram_total_mb,
            "vram_allocated_mb": vram_alloc,
            "vram_reserved_mb": vram_reserved,
            "model_device": param_device,
            "model_dtype": param_dtype,
            "ffmpeg_available": self.ffmpeg_path is not None,
            "ffmpeg_path": self.ffmpeg_path,
            "synthid_available": self.synthid_detector.available,
            "synthid_reason": self.synthid_detector.unavailable_reason,
        }

    def _convert_with_ffmpeg(self, audio_bytes: bytes, original_filename: str = "") -> Tuple[np.ndarray, float]:
        """
        Uses FFmpeg via subprocess to decode and convert WebM/Opus, MP3, AAC, MP4, M4A,
        or any browser-recorded format directly to 16kHz mono float32 PCM WAV in one pass.
        Returns (waveform_np, ffmpeg_elapsed_ms).
        """
        import soundfile as sf

        ffmpeg_bin = self.ffmpeg_path or shutil.which("ffmpeg")
        if not ffmpeg_bin:
            logger.error("FFmpeg executable not found on system PATH.")
            raise AudioProcessingError(
                "Required audio decoding runtime is unavailable. "
                "Please start the application using the project's configured environment or ensure FFmpeg is on PATH."
            )

        ext = ".webm"
        if original_filename:
            _, file_ext = os.path.splitext(original_filename.lower())
            if file_ext in [".webm", ".opus", ".ogg", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".mp4"]:
                ext = file_ext

        temp_in = None
        temp_out = None
        t_ff0 = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f_in:
                f_in.write(audio_bytes)
                temp_in = Path(f_in.name)

            temp_out = temp_in.with_suffix(".pcm16k.wav")
            if temp_out.exists():
                try:
                    temp_out.unlink()
                except OSError:
                    pass

            cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel", "error",
                "-i", str(temp_in),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                str(temp_out)
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=25,
                check=False
            )
            ffmpeg_ms = (time.perf_counter() - t_ff0) * 1000.0

            if proc.returncode != 0:
                err_msg = proc.stderr.decode("utf-8", errors="ignore").strip()
                logger.error(f"FFmpeg decoding failed (exit code {proc.returncode}): {err_msg}")
                raise AudioProcessingError("Unable to decode audio recording. Audio format may be corrupted or unsupported.")

            if not temp_out.exists() or temp_out.stat().st_size == 0:
                raise AudioProcessingError("The audio file or microphone recording is empty or invalid.")

            data, sr = sf.read(str(temp_out), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data, ffmpeg_ms
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg conversion timed out after 25 seconds.")
            raise AudioProcessingError("Audio decoding timed out.")
        finally:
            if temp_in and temp_in.exists():
                try:
                    temp_in.unlink()
                except OSError:
                    pass
            if temp_out and temp_out.exists():
                try:
                    temp_out.unlink()
                except OSError:
                    pass

    def preprocess_audio_with_timings(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO, np.ndarray],
        original_filename: str = ""
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Decodes and normalizes incoming audio into a 16kHz mono float32 waveform array
        while recording granular timings for decode, FFmpeg conversion, resampling, and validation.
        """
        import soundfile as sf

        timings = {
            "decode_ms": 0.0,
            "ffmpeg_ms": 0.0,
            "resample_ms": 0.0,
            "preprocess_ms": 0.0,
        }

        t_dec_start = time.perf_counter()
        waveform = None

        if isinstance(audio_input, np.ndarray):
            waveform = audio_input.astype(np.float32)
            timings["decode_ms"] = (time.perf_counter() - t_dec_start) * 1000.0
        elif isinstance(audio_input, (bytes, bytearray)):
            if len(audio_input) == 0:
                raise AudioProcessingError("The audio file or microphone recording is empty or invalid.")

            is_complex_container = False
            if original_filename:
                lower_name = original_filename.lower()
                if lower_name.endswith((".webm", ".opus", ".ogg", ".mp4", ".m4a", ".aac")):
                    is_complex_container = True

            # Attempt fast in-memory decode with soundfile for standard WAV/FLAC/MP3
            if not is_complex_container:
                try:
                    t_sf = time.perf_counter()
                    data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
                    timings["decode_ms"] = (time.perf_counter() - t_sf) * 1000.0
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != self.sample_rate:
                        t_res = time.perf_counter()
                        # Fast polyphase / linear tensor resample using torchaudio on CPU
                        wav_t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
                        resampled_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=self.sample_rate)
                        data = resampled_t.squeeze(0).numpy()
                        timings["resample_ms"] = (time.perf_counter() - t_res) * 1000.0
                    waveform = data
                except Exception:
                    waveform, ff_ms = self._convert_with_ffmpeg(audio_input, original_filename)
                    timings["ffmpeg_ms"] = ff_ms
                    timings["decode_ms"] = (time.perf_counter() - t_dec_start) * 1000.0
            else:
                waveform, ff_ms = self._convert_with_ffmpeg(audio_input, original_filename)
                timings["ffmpeg_ms"] = ff_ms
                timings["decode_ms"] = (time.perf_counter() - t_dec_start) * 1000.0
        elif hasattr(audio_input, "read"):
            raw_bytes = audio_input.read()
            return self.preprocess_audio_with_timings(raw_bytes, original_filename)
        elif isinstance(audio_input, (str, Path)):
            path_obj = Path(audio_input)
            if not path_obj.exists():
                raise AudioProcessingError(f"File not found: {audio_input}")
            file_bytes = path_obj.read_bytes()
            return self.preprocess_audio_with_timings(file_bytes, original_filename or path_obj.name)
        else:
            raise AudioProcessingError(f"Unsupported audio input type: {type(audio_input)}")

        t_prep_start = time.perf_counter()
        if waveform is None or len(waveform) == 0:
            raise AudioProcessingError("Decoded audio waveform is empty.")

        waveform = np.ascontiguousarray(waveform, dtype=np.float32).flatten()

        min_samples = int(self.min_duration_seconds * self.sample_rate)
        if len(waveform) < min_samples:
            raise AudioProcessingError(
                f"Audio is too short ({len(waveform)/self.sample_rate:.2f}s). "
                f"Minimum required is {self.min_duration_seconds}s."
            )

        max_samples = int(self.max_duration_seconds * self.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        if not np.all(np.isfinite(waveform)):
            raise AudioProcessingError("Audio signal contains invalid numerical values (NaN or Inf).")

        max_val = float(np.max(np.abs(waveform)))
        if max_val < 1e-4:
            raise AudioProcessingError("Audio signal is completely silent.")

        if max_val > 1.0:
            waveform = waveform / max_val

        timings["preprocess_ms"] = (time.perf_counter() - t_prep_start) * 1000.0
        return waveform, timings

    def preprocess_audio(self, audio_input: Union[str, Path, bytes, BinaryIO, np.ndarray], original_filename: str = "") -> np.ndarray:
        """Backward-compatible wrapper returning the normalized 16kHz float32 waveform."""
        waveform, _ = self.preprocess_audio_with_timings(audio_input, original_filename)
        return waveform

    def _infer_windows_batched(self, windows_list: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Executes high-throughput batched forward passes on the RTX 4050 GPU under
        torch.inference_mode() with automatic FP16 precision.
        Each window in windows_list has identical sample length.
        Returns list of (fake_prob, real_prob) tuples.
        """
        if not windows_list:
            return []

        results: List[Tuple[float, float]] = []
        batch_np = np.stack(windows_list, axis=0)  # [N, L]
        win_len = batch_np.shape[1]

        with torch.inference_mode():
            for start_idx in range(0, len(windows_list), self.max_batch_windows):
                sub_np = batch_np[start_idx:start_idx + self.max_batch_windows]
                wav_tensor = torch.from_numpy(sub_np).to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=self.cuda_available
                )
                # Normalize each window across its time axis (exact match to 1D layer_norm)
                wav_norm = torch.nn.functional.layer_norm(wav_tensor, (win_len,))
                logits = self.model(wav_norm)
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                for row in probs:
                    results.append((float(row[0]), float(row[1])))

        return results

    def predict(
        self,
        audio_input: Union[str, Path, bytes, BinaryIO, np.ndarray],
        original_filename: str = "",
        upload_read_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes local GPU model inference using nii-yamagishilab/xls-r-2b-anti-deepfake
        with batched 2.0s temporal window scanning, SynthID watermark check, and
        forensic acoustic diagnostics.
        """
        t_total_start = time.perf_counter()

        waveform, stage_timings = self.preprocess_audio_with_timings(audio_input, original_filename)
        duration_sec = round(len(waveform) / self.sample_rate, 2)

        # Extract forensic acoustic diagnostics (fast vectorized STFT on CPU)
        t_forensic_start = time.perf_counter()
        forensic_metrics = extract_forensic_diagnostics(waveform, self.sample_rate)
        forensic_ms = (time.perf_counter() - t_forensic_start) * 1000.0
        stage_timings["preprocess_ms"] += forensic_ms

        # Build 2.0s analysis windows
        win_samples = int(self.window_duration_seconds * self.sample_rate)
        step_samples = int(self.hop_duration_seconds * self.sample_rate)

        windows_to_eval: List[np.ndarray] = []
        if len(waveform) <= win_samples:
            windows_to_eval.append(waveform)
        else:
            for i in range(0, len(waveform) - win_samples + 1, step_samples):
                windows_to_eval.append(waveform[i:i + win_samples])

            if (len(waveform) - win_samples) % step_samples != 0:
                windows_to_eval.append(waveform[-win_samples:])

        # GPU Inference with accurate CUDA event/barrier timing
        try:
            if self.cuda_available:
                torch.cuda.synchronize(self.device)
            t_gpu_start = time.perf_counter()

            window_scores = self._infer_windows_batched(windows_to_eval)

            if self.cuda_available:
                torch.cuda.synchronize(self.device)
            gpu_inference_ms = (time.perf_counter() - t_gpu_start) * 1000.0
        except Exception as e:
            logger.error(f"Neural model forward pass failed: {e}", exc_info=True)
            raise RuntimeError(f"Local AI detection model inference failed: {str(e)}")

        # Postprocessing + SynthID provenance evaluation
        t_post_start = time.perf_counter()

        synthid_result = self.synthid_detector.detect(waveform, self.sample_rate)

        fake_probs = [s[0] for s in window_scores]
        real_probs = [s[1] for s in window_scores]

        max_fake_prob = max(fake_probs) if fake_probs else 0.0
        mean_fake_prob = sum(fake_probs) / len(fake_probs) if fake_probs else 0.0
        min_real_prob = min(real_probs) if real_probs else 1.0
        mean_real_prob = sum(real_probs) / len(real_probs) if real_probs else 1.0

        # Decision Architecture:
        # 1. If SynthID watermark is explicitly DETECTED -> classify as AI
        # 2. If SynthID is NOT_DETECTED or UNAVAILABLE -> defer 100% to xls-r-2b-anti-deepfake
        #    (Never conclude HUMAN from absence of SynthID watermark)
        if synthid_result["status"] == "DETECTED" and (synthid_result.get("confidence") or 1.0) >= 0.80:
            final_label = "AI"
            confidence = float(max(max_fake_prob, synthid_result.get("confidence") or 0.99))
            decision_source = "SYNTHID_WATERMARK_AND_XLSR2B"
        elif max_fake_prob >= 0.50:
            final_label = "AI"
            confidence = max_fake_prob
            decision_source = "XLSR_2B_ANTI_DEEPFAKE"
        else:
            final_label = "HUMAN"
            confidence = mean_real_prob
            decision_source = "XLSR_2B_ANTI_DEEPFAKE"

        postprocess_ms = (time.perf_counter() - t_post_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0 + upload_read_ms

        timings_ms = {
            "upload_read_ms": round(float(upload_read_ms), 2),
            "decode_ms": round(float(stage_timings["decode_ms"]), 2),
            "ffmpeg_ms": round(float(stage_timings["ffmpeg_ms"]), 2),
            "resample_ms": round(float(stage_timings["resample_ms"]), 2),
            "preprocess_ms": round(float(stage_timings["preprocess_ms"]), 2),
            "gpu_inference_ms": round(float(gpu_inference_ms), 2),
            "postprocess_ms": round(float(postprocess_ms), 2),
            "total_ms": round(float(total_ms), 2),
        }

        logger.info(
            f"[Detection Complete] file={original_filename or 'buffer'} duration={duration_sec}s "
            f"label={final_label} conf={confidence:.4f} device={self.device} ({self.cuda_device_name}) "
            f"gpu_ms={timings_ms['gpu_inference_ms']}ms total_ms={timings_ms['total_ms']}ms "
            f"synthid={synthid_result['status']}"
        )

        return {
            "label": final_label,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "HUMAN": round(float(mean_real_prob), 4),
                "AI": round(float(max_fake_prob), 4),
            },
            "duration_seconds": duration_sec,
            "device": str(self.device),
            "gpu_name": self.cuda_device_name,
            "dtype": str(self.dtype),
            "model_id": self.model_id,
            "synthid": synthid_result,
            "timings_ms": timings_ms,
            "forensic_diagnostics": forensic_metrics,
            "analysis_details": {
                "decision_source": decision_source,
                "max_fake_prob": round(float(max_fake_prob), 4),
                "mean_fake_prob": round(float(mean_fake_prob), 4),
                "min_real_prob": round(float(min_real_prob), 4),
                "mean_real_prob": round(float(mean_real_prob), 4),
                "windows_evaluated": len(window_scores),
                "window_scores": [
                    {"window_index": idx, "fake_prob": round(fp, 4), "real_prob": round(rp, 4)}
                    for idx, (fp, rp) in enumerate(window_scores)
                ],
            }
        }


# Lazy loaded singleton getter
_global_detector = None


def get_detector() -> DeepfakeVoiceDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = DeepfakeVoiceDetector()
    return _global_detector
