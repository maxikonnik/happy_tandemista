from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

from .media import require_ffmpeg
from .signals import Sample, SignalSeries

_INT_TYPES = {b"l": (">l", 4), b"L": (">L", 4), b"s": (">h", 2), b"S": (">H", 2)}


class _GPMFTree(dict):
    """GPMF parse tree preserving per-stream boundaries to prevent SCAL contamination."""
    def __init__(self):
        super().__init__()
        self.streams: list[dict[bytes, list]] = []


def _parse_klv_container(data: bytes) -> dict[bytes, list]:
    """Parse KLV data within a single container (non-recursive for streams)."""
    out: dict[bytes, list] = {}
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos : pos + 4]
        type_ = data[pos + 4 : pos + 5]
        size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        payload_len = size * repeat
        payload = data[pos + 8 : pos + 8 + payload_len]
        if type_ in _INT_TYPES:
            fmt, width = _INT_TYPES[type_]
            n = size // width
            for r in range(repeat):
                chunk = payload[r * size : (r + 1) * size]
                values = [struct.unpack(fmt, chunk[i * width : (i + 1) * width])[0] for i in range(n)]
                out.setdefault(key, []).append(values if n > 1 else values[0])
        pos += 8 + payload_len + ((-payload_len) % 4)
    return out


def _parse_gpmf_streams(data: bytes) -> list[dict[bytes, list]]:
    """Parse GPMF and yield separate dict for each STRM, preserving stream boundaries."""
    streams = []
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos : pos + 4]
        type_ = data[pos + 4 : pos + 5]
        size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        payload_len = size * repeat
        payload = data[pos + 8 : pos + 8 + payload_len]
        if key == b"STRM" and type_ == b"\x00":
            # Parse this stream's contents (SCAL + data) as a unit
            stream_dict = _parse_klv_container(payload)
            streams.append(stream_dict)
        elif type_ == b"\x00":
            # Other nested containers: recurse and flatten (for compatibility)
            nested_streams = _parse_gpmf_streams(payload)
            streams.extend(nested_streams)
        pos += 8 + payload_len + ((-payload_len) % 4)
    return streams


def parse_gpmf(data: bytes) -> dict[bytes, list]:
    """Parse GPMF KLV into {key: [values...]}; preserves stream boundaries internally."""
    tree = _GPMFTree()
    tree.streams = _parse_gpmf_streams(data)

    # Also populate flattened data for backward compatibility with tests
    flat: dict[bytes, list] = {}
    for stream in tree.streams:
        for k, v in stream.items():
            flat.setdefault(k, []).extend(v)
    tree.update(flat)
    return tree


def gps5_series(tree: dict[bytes, list], packet_rate_hz: float = 18.0) -> dict[str, SignalSeries]:
    # Try to use per-stream data if available (prevents SCAL contamination)
    if isinstance(tree, _GPMFTree) and tree.streams:
        gps = None
        scal = None
        for stream in tree.streams:
            if b"GPS5" in stream:
                gps = stream.get(b"GPS5", [])
                scal = stream.get(b"SCAL", [])
                break
    else:
        # Fall back to flattened data (old behavior for backward compatibility)
        gps = tree.get(b"GPS5", [])
        scal = tree.get(b"SCAL", [])

    if not gps or not scal or len(scal) < 5:
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
