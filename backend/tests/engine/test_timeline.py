from pathlib import Path

from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.timeline import SourceFile, build_timeline


def sf(role, offset, phases, duration=600.0):
    return SourceFile(Path(f"/{role}.mp4"), role, duration, offset, phases, [])


def test_telemetry_beats_cv_and_offsets_applied():
    handcam = sf(
        "handcam", 100.0,
        [Phase(PhaseName.FREEFALL, 200.0, 260.0, 0.9, "telemetry")],
    )
    outside = sf(
        "outside", 0.0,
        [Phase(PhaseName.FREEFALL, 310.0, 365.0, 0.7, "cv")],
    )
    tl = build_timeline([handcam, outside])
    ff = next(p for p in tl.phases if p.name == PhaseName.FREEFALL)
    assert ff.source == "telemetry"
    assert ff.start == 300.0  # 200 local + 100 offset


def test_ground_roles_fill_missing_phases():
    interview = sf("ground_interview", -1800.0, [], duration=120.0)
    handcam = sf("handcam", 0.0, [Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry")])
    tl = build_timeline([interview, handcam])
    iv = next(p for p in tl.phases if p.name == PhaseName.INTERVIEW)
    assert iv.source == "role"
    assert iv.start == -1800.0 and iv.end == -1680.0
