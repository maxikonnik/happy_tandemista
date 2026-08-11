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


def test_role_phase_outranks_a_spurious_cv_phase_of_the_same_name():
    """Regression: a rank-0 CV guess must not lock out the rank-2 role phase.

    A hand-held interview clip can trip the CV freefall/landing heuristics and emit
    a LANDING phase an hour before the jump. Offering the role phase only when the
    name is absent let that guess win permanently, so the timeline reported LANDING
    before the interview even finished.
    """
    interview = sf(
        "ground_interview", -3600.0,
        [Phase(PhaseName.LANDING, 10.0, 20.0, 0.5, "cv")],
        duration=120.0,
    )
    handcam = sf("handcam", 0.0, [Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry")])
    landing = sf("ground_landing", 3600.0, [], duration=60.0)

    tl = build_timeline([interview, handcam, landing])
    land = next(p for p in tl.phases if p.name == PhaseName.LANDING)
    assert land.source == "role", f"role (rank 2) must beat cv (rank 0), got {land.source}"
    assert land.start == 3600.0 and land.end == 3660.0

    # And the timeline must be chronologically sane: landing after the interview.
    iv = next(p for p in tl.phases if p.name == PhaseName.INTERVIEW)
    ff = next(p for p in tl.phases if p.name == PhaseName.FREEFALL)
    assert iv.end <= ff.start <= land.start, (
        f"phases out of order: interview ends {iv.end}, freefall {ff.start}, landing {land.start}"
    )
    assert [p.name for p in tl.phases] == [PhaseName.INTERVIEW, PhaseName.FREEFALL,
                                           PhaseName.LANDING]


def test_role_phase_does_not_override_telemetry():
    """The role fallback is ranked, not preferred: telemetry (rank 3) still wins."""
    landing = sf(
        "ground_landing", 100.0,
        [Phase(PhaseName.LANDING, 5.0, 15.0, 0.9, "telemetry")],
        duration=60.0,
    )
    tl = build_timeline([landing])
    land = next(p for p in tl.phases if p.name == PhaseName.LANDING)
    assert land.source == "telemetry"
    assert land.start == 105.0


def test_ground_roles_fill_missing_phases():
    interview = sf("ground_interview", -1800.0, [], duration=120.0)
    handcam = sf("handcam", 0.0, [Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry")])
    tl = build_timeline([interview, handcam])
    iv = next(p for p in tl.phases if p.name == PhaseName.INTERVIEW)
    assert iv.source == "role"
    assert iv.start == -1800.0 and iv.end == -1680.0
