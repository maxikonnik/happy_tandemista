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


def test_no_overlap_on_same_source():
    """Regression: adjacent slots from same source must not overlap in time."""
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["full_16x9"])
    # Group clips by source
    by_source = {}
    for clip in edl.clips:
        if clip.source not in by_source:
            by_source[clip.source] = []
        by_source[clip.source].append(clip)

    # For each source, verify no two clips overlap
    for source, clips in by_source.items():
        for i, c1 in enumerate(clips):
            for c2 in clips[i+1:]:
                # Either c1 ends before c2 starts, or c2 ends before c1 starts
                assert c1.src_out <= c2.src_in or c2.src_out <= c1.src_in, \
                    f"Clips from {source} overlap: {c1} and {c2}"


def test_moment_with_nonzero_offset():
    """Verify moment centering with non-zero clock_offset."""
    # Create a handcam with non-zero offset and a moment that tests sign correctness.
    # File-local phases: [10, 40] for FREEFALL
    # clock_offset = 120.0 means phases in jump-time become [130, 160]
    # Moment at file-local t=30.0 converts to jump-time: 30 + 120 = 150
    handcam_phases = [
        Phase(PhaseName.EXIT, 10.0, 10.0, 0.9, "telemetry"),
        Phase(PhaseName.FREEFALL, 10.0, 40.0, 0.9, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, 40.0, 40.0, 0.9, "telemetry"),
        Phase(PhaseName.LANDING, 40.0, 50.0, 0.9, "telemetry"),
    ]
    moments = [Moment(30.0, 0.95, "emotion")]  # file-local t=30.0
    files = [
        SourceFile(Path("/handcam.mp4"), "handcam", 100.0, 120.0, handcam_phases, moments),
    ]

    edl = generate_edl(build_timeline(files), TEMPLATES["emotions_16x9"])

    # After build_timeline, phases are adjusted to jump-time:
    # FREEFALL jump-time: [10 + 120, 40 + 120] = [130, 160]
    # Moment: file-local t=30 → jump-time t = 30 + 120 = 150
    # emotions_16x9 FREEFALL slot: min_s=15, max_s=25
    # Window: [130, 160], length = clamp(30, [15, 25]) = 25
    # Moment at 150 is within [130, 160], so center on it
    # Centered window: [150 - 12.5, 150 + 12.5] = [137.5, 162.5]
    # Convert to file-local: [137.5 - 120, 162.5 - 120] = [17.5, 42.5]
    # Clamp to file bounds [0, 100]: [17.5, 42.5]

    # Template has EXIT then FREEFALL; find the clip that contains the moment
    hc_clips = [c for c in edl.clips if c.source == Path("/handcam.mp4")]
    assert len(hc_clips) >= 2, f"Should have at least 2 clips from handcam, got {len(hc_clips)}"

    # Find which clip contains the moment at t=30
    ff_clip = [c for c in hc_clips if 30.0 >= c.src_in and 30.0 <= c.src_out]
    assert len(ff_clip) == 1, f"Should have exactly 1 clip containing moment at t=30, got {ff_clip}"
    ff_clip = ff_clip[0]

    # Verify the moment is within the clip (which it is)
    # The exact bounds depend on whether overlap prevention trimmed the clip
    # With no overlap, centered on t=30 (jump-t=150) with length 25:
    # Jump-time window: [150 - 12.5, 150 + 12.5] = [137.5, 162.5]
    # File-local: [17.5, 42.5]
    # EXIT clip would be [7.0, 20.0], which overlaps [17.5, 42.5]
    # So FREEFALL is trimmed to start at 20.0: [20.0, 45.0]
    assert abs(ff_clip.src_in - 20.0) < 0.01, f"src_in should be ~20.0 after overlap trim, got {ff_clip.src_in}"
    assert abs(ff_clip.src_out - 45.0) < 0.01, f"src_out should be ~45.0, got {ff_clip.src_out}"
