import subprocess
from pathlib import Path

import pytest

from tandemista.engine.frames import extract_frame_features


def lavfi_clip(path: Path, source: str, seconds: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"{source}:d={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("frames")
    return {
        "sky": lavfi_clip(d / "sky.mp4", "color=c=0x3399FF:s=640x360:r=10", 8),
        "busy": lavfi_clip(d / "busy.mp4", "testsrc=s=640x360:r=10", 8),
    }


def test_static_sky_clip_is_calm_flat_and_skylike(clips):
    feats = extract_frame_features(clips["sky"], fps=1.0)
    assert len(feats) >= 6
    assert all(f.motion < 0.5 for f in feats[1:])       # nothing moves
    assert all(f.sharpness < 5.0 for f in feats)        # flat colour has no edges
    assert all(f.sky_ratio > 0.8 for f in feats)        # blue and bright
    assert feats[0].motion == 0.0                       # no previous frame


def test_moving_pattern_has_motion_and_detail(clips):
    feats = extract_frame_features(clips["busy"], fps=1.0)
    assert max(f.motion for f in feats) > 1.0
    assert max(f.sharpness for f in feats) > 50.0
    assert max(f.sky_ratio for f in feats) < 0.8        # test pattern is not sky


def test_timestamps_follow_sampling_rate(clips):
    feats = extract_frame_features(clips["sky"], fps=2.0)
    assert feats[0].t == pytest.approx(0.0, abs=0.1)
    assert feats[1].t == pytest.approx(0.5, abs=0.1)
    assert feats[2].t == pytest.approx(1.0, abs=0.1)


def test_missing_file_returns_empty(tmp_path):
    assert extract_frame_features(tmp_path / "nope.mp4") == []
