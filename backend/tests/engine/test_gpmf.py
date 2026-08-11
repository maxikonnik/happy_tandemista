import struct
from pathlib import Path

import pytest

from tandemista.engine.gpmf import gps5_series, parse_gpmf


def klv(key: bytes, type_: bytes, size: int, repeat: int, payload: bytes) -> bytes:
    pad = (-len(payload)) % 4
    return key + type_ + bytes([size]) + struct.pack(">H", repeat) + payload + b"\x00" * pad


def make_stream(alts_m: list[float]) -> bytes:
    # SCAL for GPS5: divisors [1e7, 1e7, 1000, 1000, 100]
    scal = klv(b"SCAL", b"l", 4, 5, struct.pack(">5l", 10000000, 10000000, 1000, 1000, 100))
    gps_payload = b"".join(
        struct.pack(">5l", 0, 0, int(alt * 1000), 0, 0) for alt in alts_m
    )
    gps5 = klv(b"GPS5", b"l", 20, len(alts_m), gps_payload)
    strm = klv(b"STRM", b"\x00", 1, len(scal) + len(gps5), scal + gps5)
    return klv(b"DEVC", b"\x00", 1, len(strm), strm)


def test_parse_nested_and_scaled():
    data = make_stream([4000.0, 3950.0, 3900.0])
    tree = parse_gpmf(data)
    series = gps5_series(tree, packet_rate_hz=1.0)
    alt = series["altitude_m"]
    assert [round(s.value) for s in alt.samples] == [4000, 3950, 3900]


def test_vspeed_derivative():
    data = make_stream([4000.0, 3950.0, 3900.0, 3850.0])
    series = gps5_series(parse_gpmf(data), packet_rate_hz=1.0)
    v = series["vspeed_ms"]
    assert v.value_at(1.5) == pytest.approx(-50.0, abs=5.0)


def make_multi_stream_devc(alts_m: list[float]) -> bytes:
    """Build realistic DEVC with multiple STRM blocks (ACCL + GPS5).

    ACCL has SCAL [418, 418, 418] (accelerometer divisor).
    GPS5 has SCAL [1e7, 1e7, 1000, 1000, 100] (GPS5 divisor).

    Tests that GPS5's altitude uses its own SCAL, not ACCL's.
    """
    # ACCL stream with its own SCAL (3 values, not 5)
    accl_scal = klv(b"SCAL", b"l", 4, 3, struct.pack(">3l", 418, 418, 418))
    accl_payload = b"".join(
        struct.pack(">3l", 0, 0, 0) for _ in range(2)  # 2 samples
    )
    accl = klv(b"ACCL", b"l", 12, 2, accl_payload)
    accl_strm = klv(b"STRM", b"\x00", 1, len(accl_scal) + len(accl), accl_scal + accl)

    # GPS5 stream with its own SCAL (5 values)
    gps5_scal = klv(b"SCAL", b"l", 4, 5, struct.pack(">5l", 10000000, 10000000, 1000, 1000, 100))
    gps5_payload = b"".join(
        struct.pack(">5l", 0, 0, int(alt * 1000), 0, 0) for alt in alts_m
    )
    gps5 = klv(b"GPS5", b"l", 20, len(alts_m), gps5_payload)
    gps5_strm = klv(b"STRM", b"\x00", 1, len(gps5_scal) + len(gps5), gps5_scal + gps5)

    # Combine into DEVC
    devc_payload = accl_strm + gps5_strm
    return klv(b"DEVC", b"\x00", 1, len(devc_payload), devc_payload)


def test_scal_not_contaminated_across_streams():
    """Regression test: GPS5 altitude must use GPS5's SCAL, not ACCL's.

    If SCAL values are mixed up, altitude will be scaled by 418 instead of 1000,
    resulting in values ~418x larger than actual.
    """
    data = make_multi_stream_devc([5000.0, 4500.0])
    tree = parse_gpmf(data)
    series = gps5_series(tree, packet_rate_hz=1.0)
    alt = series["altitude_m"]
    # Correct altitude should be ~5000 and ~4500 meters
    # If ACCL's SCAL (418) was used, we'd get ~5000*1000/418 ≈ 11961 (garbage)
    values = [round(s.value) for s in alt.samples]
    assert values == [5000, 4500], f"Got {values}; contamination likely (expected [5000, 4500])"


SAMPLE = Path(__file__).parent.parent / "samples" / "gopro.mp4"


@pytest.mark.skipif(not SAMPLE.exists(), reason="no real GoPro sample provided")
def test_real_gopro_file():
    from tandemista.engine.gpmf import telemetry_from_gopro

    series = telemetry_from_gopro(SAMPLE)
    assert "altitude_m" in series
