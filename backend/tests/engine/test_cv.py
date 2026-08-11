from pathlib import Path

from tandemista.engine.cv import (
    CVAnnotation,
    LocalCVAnnotator,
    Moment,
    moments_from_features,
    phases_from_features,
)
from tandemista.engine.frames import FrameFeatures
from tandemista.engine.phases import PhaseName


def jump_features() -> list[FrameFeatures]:
    """Ground interview, then boarding, then a violent freefall, canopy, landing."""
    feats: list[FrameFeatures] = []

    def add(n: int, motion: float, sky: float, sharp: float, bright: float = 0.5) -> None:
        for _ in range(n):
            t = len(feats) * 2.0
            feats.append(FrameFeatures(t, sharp, motion, bright, sky))

    add(10, motion=0.3, sky=0.05, sharp=60.0)    # interview on the ground
    add(10, motion=0.6, sky=0.10, sharp=40.0)    # inside the plane
    add(20, motion=9.0, sky=0.75, sharp=90.0)    # freefall: violent flow, lots of sky
    add(20, motion=1.5, sky=0.60, sharp=80.0)    # under canopy: calm, still sky
    add(10, motion=0.8, sky=0.10, sharp=70.0)    # landed
    return feats


def test_freefall_detected_from_flow_and_sky():
    phases = phases_from_features(jump_features())
    ff = next(p for p in phases if p.name == PhaseName.FREEFALL)
    assert ff.source == "cv"
    assert ff.confidence == 0.5
    assert ff.start == 40.0            # frame 20 * 2s
    assert ff.end == 80.0              # frame 40 * 2s
    assert any(p.name == PhaseName.CANOPY for p in phases)


def test_no_freefall_when_nothing_moves():
    flat = [FrameFeatures(i * 2.0, 50.0, 0.2, 0.5, 0.1) for i in range(30)]
    assert phases_from_features(flat) == []


def test_exit_and_deployment_moments_bracket_the_freefall():
    moments = moments_from_features(jump_features())
    exit_m = next(m for m in moments if m.kind == "exit")
    dep_m = next(m for m in moments if m.kind == "deployment")
    assert exit_m.t == 40.0            # flow jumps here
    assert dep_m.t == 80.0             # flow collapses here
    assert dep_m.t > exit_m.t
    assert any(m.kind == "scenic" for m in moments)


def test_annotator_returns_annotation_for_a_real_clip(tmp_path):
    import subprocess

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=s=320x180:r=10:d=6", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )
    ann = LocalCVAnnotator(fps=1.0).annotate(clip)
    assert isinstance(ann, CVAnnotation)
    assert all(isinstance(m, Moment) for m in ann.moments)


def test_annotator_satisfies_the_protocol():
    from tandemista.engine.cv import CVAnnotator

    annotator: CVAnnotator = LocalCVAnnotator()
    assert callable(annotator.annotate)
