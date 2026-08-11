import pytest

from tandemista.engine.phases import PhaseName, detect_phases_from_telemetry
from tandemista.engine.synthetic import make_jump_signals


def by_name(phases, name):
    return next(p for p in phases if p.name == name)


def test_clean_jump_phases():
    sig = make_jump_signals(exit_t=300, freefall_s=60, canopy_s=240)
    phases = detect_phases_from_telemetry(sig["vspeed_ms"], sig["accel_g"])
    exit_p = by_name(phases, PhaseName.EXIT)
    assert exit_p.start == pytest.approx(300, abs=6)
    ff = by_name(phases, PhaseName.FREEFALL)
    assert ff.start == pytest.approx(300, abs=6)
    assert ff.end == pytest.approx(360, abs=6)
    dep = by_name(phases, PhaseName.DEPLOYMENT)
    assert dep.start == pytest.approx(360, abs=6)
    canopy = by_name(phases, PhaseName.CANOPY)
    assert canopy.end == pytest.approx(600, abs=10)
    landing = by_name(phases, PhaseName.LANDING)
    assert landing.start == pytest.approx(600, abs=10)
    assert ff.confidence >= 0.9  # accel spike confirms deployment


def test_noisy_jump_still_detected():
    sig = make_jump_signals(noise=3.0, seed=42)
    phases = detect_phases_from_telemetry(sig["vspeed_ms"], sig["accel_g"])
    assert {p.name for p in phases} >= {PhaseName.FREEFALL, PhaseName.CANOPY, PhaseName.LANDING}


def test_no_freefall_returns_empty():
    sig = make_jump_signals()
    ground_only = sig["vspeed_ms"].__class__(
        "vspeed_ms", [s for s in sig["vspeed_ms"].samples if s.t < 200]
    )
    assert detect_phases_from_telemetry(ground_only) == []
