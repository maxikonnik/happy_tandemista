from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

from .media import require_ffmpeg
from .signals import Sample, SignalSeries

_INT_TYPES = {b"l": (">l", 4), b"L": (">L", 4), b"s": (">h", 2), b"S": (">H", 2)}


def parse_gpmf(data: bytes) -> dict[bytes, list]:
    """Parse GPMF KLV into {key: [values...]}; nested containers are flattened."""
    out: dict[bytes, list] = {}
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos : pos + 4]
        type_ = data[pos + 4 : pos + 5]
        size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        payload_len = size * repeat
        payload = data[pos + 8 : pos + 8 + payload_len]
        if type_ == b"\x00":
            nested = parse_gpmf(payload)
            for k, v in nested.items():
                out.setdefault(k, []).extend(v)
        elif type_ in _INT_TYPES:
            fmt, width = _INT_TYPES[type_]
            n = size // width
            for r in range(repeat):
                chunk = payload[r * size : (r + 1) * size]
                values = [struct.unpack(fmt, chunk[i * width : (i + 1) * width])[0] for i in range(n)]
                out.setdefault(key, []).append(values if n > 1 else values[0])
        # other types (strings, floats) are irrelevant for altitude and skipped
        pos += 8 + payload_len + ((-payload_len) % 4)
    return out


def gps5_series(tree: dict[bytes, list], packet_rate_hz: float = 18.0) -> dict[str, SignalSeries]:
    gps = tree.get(b"GPS5", [])
    scal = tree.get(b"SCAL", [])
    if not gps or len(scal) < 5:
        return {}
    alt_div = float(scal[2] if not isinstance(scal[2], list) else scal[2][0])
    step = 1.0 / packet_rate_hz
    alt = [Sample(i * step, row[2] / alt_div) for i, row in enumerate(gps)]
    alt_series = SignalSeries("altitude_m", alt).resample(1.0)
    vs: list[Sample] = []
    window = 3
    pts = alt_series.samples
    for i in range(1, len(pts)):
        lo = max(0, i - window)
        dt = pts[i].t - pts[lo].t
        vs.append(Sample(pts[i].t, (pts[i].value - pts[lo].value) / dt if dt else 0.0))
    return {"altitude_m": alt_series, "vspeed_ms": SignalSeries("vspeed_ms", vs)}


def telemetry_from_gopro(path: Path) -> dict[str, SignalSeries]:
    require_ffmpeg()
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    idx = next(
        (s["index"] for s in probe["streams"] if s.get("codec_tag_string") == "gpmd"), None
    )
    if idx is None:
        return {}
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-codec", "copy",
         "-map", f"0:{idx}", "-f", "data", "-"],
        check=True, capture_output=True,
    ).stdout
    return gps5_series(parse_gpmf(raw))
