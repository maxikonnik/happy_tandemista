from pathlib import Path

import pytest

from tandemista.engine.cv import Moment
from tandemista.engine.edl import SlotUnfillableError, generate_edl
from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.templates import TEMPLATES, Slot, Template
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


def multi_camera_files(moments):
    """The normal multi-camera case: EXIT comes from the outside operator,
    FREEFALL from the instructor's handcam, so no overlap trim hides a bad window."""
    handcam_phases = [
        Phase(PhaseName.EXIT, 300.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, 360.0, 360.0, 0.9, "telemetry"),
    ]
    return [
        SourceFile(Path("/handcam.mp4"), "handcam", 400.0, 0.0, handcam_phases, moments),
        SourceFile(Path("/outside.mp4"), "outside", 400.0, 0.0, [], []),
    ]


def test_freefall_clip_not_centred_before_the_phase_starts():
    """Regression: a boundary moment must not drag the clip in front of its phase.

    The CV annotator emits an "exit" moment with a fixed score of 1.0 at the very
    first frame of freefall, so it always wins the centring contest. Centring on it
    made the FREEFALL clip [ff_start - length/2, ff_start + length/2], i.e. half the
    reel was in-plane footage labelled freefall.
    """
    files = multi_camera_files([Moment(300.0, 1.0, "exit"), Moment(330.0, 0.9, "scenic")])
    edl = generate_edl(build_timeline(files), TEMPLATES["highlights_9x16"])

    ff_clips = [c for c in edl.clips if c.source == Path("/handcam.mp4")]
    assert len(ff_clips) == 1, f"expected one handcam clip (the freefall), got {ff_clips}"
    ff = ff_clips[0]
    assert ff.src_in >= 300.0, (
        f"freefall clip starts at {ff.src_in}, before the freefall phase begins at 300.0"
    )
    assert ff.src_out <= 360.0, (
        f"freefall clip ends at {ff.src_out}, past the freefall phase end 360.0"
    )
    # The exit itself must still come from the outside camera, as the template asks.
    assert any(c.source == Path("/outside.mp4") for c in edl.clips)


def test_interior_moment_still_recentres_the_clip():
    """Guard: the fix must not disable moment centring for moments inside the phase."""
    files = multi_camera_files([Moment(330.0, 1.0, "scenic")])
    edl = generate_edl(build_timeline(files), TEMPLATES["highlights_9x16"])

    ff = next(c for c in edl.clips if c.source == Path("/handcam.mp4"))
    # highlights_9x16 FREEFALL slot: min 6, max 8 -> length 8, centred on t=330
    assert ff.src_in == pytest.approx(326.0)
    assert ff.src_out == pytest.approx(334.0)


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


def test_overlap_with_gaps_in_consumed():
    """Regression: trimming must handle non-adjacent consumed intervals with gaps.

    Without recalculating local_out in the trim loop, clips can still overlap consumed intervals.
    Example: consumed [(5, 10), (12, 25)] with gap [10, 12]. A clip with local_in=0, length=6
    (local_out=6) gets trimmed past [5,10] to local_in=10, but the stale local_out=6 means
    the loop doesn't detect overlap with [12, 25]. Final clip becomes [10, 16], which overlaps [12, 25].
    """
    # Create phases that will produce consumed intervals with a gap
    camera_phases = [
        Phase(PhaseName.INTERVIEW, 5.0, 10.0, 0.9, "telemetry"),
        Phase(PhaseName.CLIMB, 12.0, 25.0, 0.9, "telemetry"),
        Phase(PhaseName.EXIT, 0.0, 20.0, 0.9, "telemetry"),
    ]
    files = [
        SourceFile(Path("/camera.mp4"), "camera", 100.0, 0.0, camera_phases, []),
    ]

    # Custom template that produces consumed intervals with a gap
    test_template = Template(
        "gap_test", "16:9",
        (
            # Slot A: INTERVIEW [5, 10] with min=5, max=5 → [5, 10]
            Slot(PhaseName.INTERVIEW, 5, 5, prefer_roles=("camera",)),
            # Slot B: CLIMB [12, 25] with min=13, max=13 → [12, 25]
            Slot(PhaseName.CLIMB, 13, 13, prefer_roles=("camera",)),
            # Slot C: EXIT [0, 20] with min=6, max=6
            # Initially: start=0, end=20, length=clamp(20, [6,6])=6
            # local_in=0, local_out=6
            # Trim past [5, 10]: local_in=10, but local_out still stale at 6
            # Check vs [12, 25]: is 10 < 25 and 6 > 12? NO (6 not > 12)
            # Loop exits without advancing further
            # After loop: local_out = min(100, 10 + 6) = 16
            # Result: [10, 16] which overlaps [12, 25]!
            Slot(PhaseName.EXIT, 6, 6, prefer_roles=("camera",)),
        ),
    )

    edl = generate_edl(build_timeline(files), test_template)

    # Extract all clips from this camera
    camera_clips = [c for c in edl.clips if c.source == Path("/camera.mp4")]

    # Verify no two clips overlap (the bug would produce overlapping clips here)
    for i, c1 in enumerate(camera_clips):
        for c2 in camera_clips[i+1:]:
            assert c1.src_out <= c2.src_in or c2.src_out <= c1.src_in, \
                f"Clips from /camera.mp4 overlap: {c1} and {c2}"


def test_emotions_ships_in_both_shapes():
    """The emotions cut is delivered horizontally and vertically from one analysis."""
    horizontal = TEMPLATES["emotions_16x9"]
    vertical = TEMPLATES["emotions_9x16"]
    assert horizontal.aspect == "16:9"
    assert vertical.aspect == "9:16"
    # Same story beats in both shapes, so the two variants stay recognisably one edit.
    assert [s.phase for s in vertical.slots] == [s.phase for s in horizontal.slots]
    assert [(s.min_s, s.max_s) for s in vertical.slots] == [
        (s.min_s, s.max_s) for s in horizontal.slots
    ]


def test_vertical_emotions_prefers_the_face_camera_for_the_exit():
    """A 9:16 crop of a wide outside shot loses the faces the emotions cut is about,
    so the vertical variant reaches for the handcam first."""
    vertical = TEMPLATES["emotions_9x16"]
    exit_slot = next(s for s in vertical.slots if s.phase == PhaseName.EXIT)
    assert exit_slot.prefer_roles[0] == "handcam"
    assert exit_slot.required is True


def test_vertical_emotions_generates_a_usable_edl():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["emotions_9x16"])
    assert edl.variant == "emotions_9x16"
    assert edl.aspect == "9:16"
    assert edl.clips
    for c in edl.clips:
        assert c.src_out > c.src_in >= 0.0
