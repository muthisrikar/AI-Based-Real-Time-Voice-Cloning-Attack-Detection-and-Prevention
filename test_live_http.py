import io
import json
import subprocess
import urllib.request
import urllib.error
import numpy as np
import soundfile as sf


def make_multipart(filename, content, mime_type):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.extend(content)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def send_request(filename, content, mime_type):
    data, content_type = make_multipart(filename, content, mime_type)
    req = urllib.request.Request("http://localhost:8000/api/detect", data=data, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def run_tests():
    print("Testing against live HTTP server on http://localhost:8000 ...\n")

    # 0. Wait for server startup & verify RTX 4050 GPU readiness
    import time
    health = None
    for _ in range(90):
        try:
            with urllib.request.urlopen("http://localhost:8000/api/health", timeout=10) as resp:
                health = json.loads(resp.read().decode())
                break
        except Exception:
            time.sleep(2)
    assert health is not None, "Server on http://localhost:8000 did not become ready in time"
    print(f"0. /api/health: {health['model_info']}")
    assert health["status"] == "HEALTHY"
    assert health["hardware"]["cuda_available"] is True
    assert "RTX 4050" in health["hardware"]["cuda_device_name"]

    # 1. Live microphone WebM recording
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=1.5", "-c:a", "libopus", "-f", "webm", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    webm_bytes = proc.stdout
    status, res = send_request("recorded_voice.webm", webm_bytes, "audio/webm")
    print(f"\n1. Live mic WebM response: HTTP {status} -> label={res['label']}, total_ms={res['timings_ms']['total_ms']}ms")
    assert status == 200 and res["status"] == "SUCCESS"
    assert res["label"] in ["HUMAN", "AI"]

    # 2. Uploaded WAV file
    t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
    wav_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, wav_data, 16000, format="WAV")
    status, res = send_request("voice.wav", buf.getvalue(), "audio/wav")
    print(f"2. WAV upload response: HTTP {status} -> label={res['label']}, total_ms={res['timings_ms']['total_ms']}ms")
    assert status == 200 and res["status"] == "SUCCESS"

    # 3. Uploaded MP3 file
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:a", "libmp3lame", "-f", "mp3", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    mp3_bytes = proc.stdout
    status, res = send_request("sample.mp3", mp3_bytes, "audio/mpeg")
    print(f"3. MP3 upload response: HTTP {status} -> label={res['label']}, total_ms={res['timings_ms']['total_ms']}ms")
    assert status == 200 and res["status"] == "SUCCESS"

    # 4. Empty audio file (failure handling)
    status, res = send_request("empty.wav", b"", "audio/wav")
    print(f"4. Empty audio response: HTTP {status} -> {res}")
    assert status == 400

    # 5. Corrupt audio file (failure handling)
    status, res = send_request("corrupt.wav", b"corrupted_garbage_bytes", "audio/wav")
    print(f"5. Corrupt audio response: HTTP {status} -> {res}")
    assert status == 400

    # 6. Real Human Speech (VoxCeleb)
    with open("human_sample.wav", "rb") as f:
        h_bytes = f.read()
    status, res = send_request("human_sample.wav", h_bytes, "audio/wav")
    print(f"6. Real Human speech response: HTTP {status} -> label={res['label']}, conf={res['confidence']}, total_ms={res['timings_ms']['total_ms']}ms")
    assert status == 200 and res["status"] == "SUCCESS"
    assert res["label"] == "HUMAN"

    # 7. Known AI Regression Target (WhatsApp.mp4)
    with open("WhatsApp.mp4", "rb") as f:
        wa_bytes = f.read()
    status, res = send_request("WhatsApp.mp4", wa_bytes, "video/mp4")
    print(f"7. Known-AI regression (WhatsApp.mp4) response: HTTP {status} -> label={res['label']}, conf={res['confidence']}, total_ms={res['timings_ms']['total_ms']}ms")
    assert status == 200 and res["status"] == "SUCCESS"
    assert res["label"] == "AI"

    print("\n=======================================================")
    print("ALL REAL HTTP LIVE SERVER GPU TESTS PASSED PERFECTLY!")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()
