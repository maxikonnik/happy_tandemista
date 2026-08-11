from pathlib import Path

import pytest

from tandemista.engine.cv import Moment
from tandemista.engine.edl import SlotUnfillableError, generate_edl
from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.templates import TEMPLATES
from tandemista.engine.timeline import SourceFile, build_timeline


def full_jump_files():
    handcam_phases = [
        Phase(PhaseName.CLIMB, 0.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.EXIT, 300.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, 360.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.CANOPY, 360.0, 600.0, 0.9, "telemetry"),
        Phase(PhaseName.LANDING, 600.0, 615.0, 0.9, "telemetry"),
    ]
    moments = [Moment(320.0, 0.95, "emotion")]
    return [
        SourceFile(Path("/interview.mp4"), "ground_interview", 90.0, -2000.0, [], []),
        SourceFile(Path("/handcam.mp4"), "handcam", 640.0, 0.0, handcam_phases, moments),
        SourceFile(Path("/landing.mp4"), "ground_landing", 60.0, 590.0, [], []),
    ]


def test_full_template_covers_slots_in_order():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["full_16x9"])
    assert edl.aspect == "16:9"
    assert edl.clips[0].source == Path("/interview.mp4")
    assert any(c.source == Path("/handcam.mp4") for c in edl.clips)
    for c in edl.clips:
        assert c.src_out > c.src_in >= 0.0


def test_freefall_clip_centered_on_best_moment():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["emotions_16x9"])
    ff = [c for c in edl.clips if c.source == Path("/handcam.mp4")]
    assert any(c.src_in <= 320.0 <= c.src_out for c in ff)


def test_required_slot_without_footage_raises():
    files = [SourceFile(Path("/interview.mp4"), "ground_interview", 90.0, 0.0, [], [])]
    with pytest.raises(SlotUnfillableError):
        generate_edl(build_timeline(files), TEMPLATES["full_16x9"])
