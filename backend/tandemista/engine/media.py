from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .signals import Sample, SignalSeries


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found in PATH; install ffmpeg to use tandemista")


def probe_duration(path: Path) -> float:
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def has_audio(path: Path) -> bool:
    """Check if a media file has an audio stream."""
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    streams = json.loads(out).get("streams", [])
    return any(s.get("codec_type") == "audio" for s in streams)


def probe_frame_rate(path: Path) -> float | None:
    """Probe the frame rate of a video file. Returns None if not determinable."""
    require_ffmpeg()
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", str(path)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        if data.get("streams"):
            r_frame_rate = data["streams"][0].get("r_frame_rate")
            if r_frame_rate:
                # r_frame_rate is a string like "30/1" or "60000/1001"
                num, denom = map(int, r_frame_rate.split("/"))
                return num / denom
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, KeyError):
        pass
    return None


def probe_audio_properties(path: Path) -> dict | None:
    """Probe audio sample rate and channel layout. Returns None if no audio."""
    require_ffmpeg()
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", str(path)],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        data = json.loads(out)
        if data.get("streams"):
            stream = data["streams"][0]
            return {
                "sample_rate": int(stream.get("sample_rate", 0)),
                "channels": int(stream.get("channels", 0)),
            }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, KeyError):
        pass
    return None


def extract_audio_rms(path: Path, step: float = 1.0) -> SignalSeries:
    require_ffmpeg()
    rate = 8000
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    win = int(rate * step)
    samples: list[Sample] = []
    for i in range(0, len(pcm) - win + 1, win):
        rms = float(np.sqrt(np.mean(pcm[i : i + win] ** 2)))
        samples.append(Sample(i / rate, rms))
    peak = max((s.value for s in samples), default=1.0) or 1.0
    return SignalSeries("audio_rms", [Sample(s.t, s.value / peak) for s in samples])


def extract_frames(path: Path, out_dir: Path, fps: float = 1.0) -> list[Path]:
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={fps}", "-q:v", "4", str(pattern)],
        check=True, capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))
