import subprocess
from pathlib import Path

import pytest

from tandemista.engine.edl import EDL, Clip
from tandemista.engine.media import probe_duration
from tandemista.engine.render import render_edl


def make_clip(path: Path, color: str, seconds: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


def make_clip_no_audio(path: Path, color: str, seconds: int) -> Path:
    """Create a video clip without an audio track."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d={seconds}",
         "-c:v", "libx264", str(path)],
        check=True, capture_output=True,
    )
    return path


def make_clip_custom(path: Path, color: str, seconds: int, size: str, fps: str) -> Path:
    """Create a video clip with custom resolution and frame rate."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c={color}:s={size}:r={fps}:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("render")
    return [
        make_clip(d / "a.mp4", "red", 5),
        make_clip(d / "b.mp4", "green", 5),
    ]


def test_render_16x9_duration(sources, tmp_path):
    edl = EDL("full_16x9", "16:9", [
        Clip(sources[0], 0.0, 3.0),
        Clip(sources[1], 1.0, 4.0),
    ])
    out = render_edl(edl, tmp_path / "out.mp4", height=360)
    assert probe_duration(out) == pytest.approx(6.0, abs=0.5)


def test_render_9x16_aspect(sources, tmp_path):
    edl = EDL("highlights_9x16", "9:16", [Clip(sources[0], 0.0, 2.0)])
    out = render_edl(edl, tmp_path / "v.mp4", height=640)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    assert abs(w / h - 9 / 16) < 0.02


def test_render_mixed_audio_presence(tmp_path):
    """Test render with one clip having audio and one without."""
    d = tmp_path / "mixed_audio"
    d.mkdir()
    with_audio = make_clip(d / "with.mp4", "red", 3)
    no_audio = make_clip_no_audio(d / "no.mp4", "blue", 2)

    edl = EDL("mixed", "16:9", [
        Clip(with_audio, 0.0, 3.0),
        Clip(no_audio, 0.0, 2.0),
    ])
    out = render_edl(edl, tmp_path / "mixed_out.mp4", height=360)
    # Should complete successfully with total duration ~5 seconds
    assert out.exists()
    assert probe_duration(out) == pytest.approx(5.0, abs=0.5)


def test_render_mixed_resolution_framerate(tmp_path):
    """Test render with clips of different resolution and frame rate."""
    d = tmp_path / "mixed_specs"
    d.mkdir()
    # 640x360 at 10 fps
    clip1 = make_clip_custom(d / "clip1.mp4", "red", 2, "640x360", "10")
    # 320x240 at 25 fps
    clip2 = make_clip_custom(d / "clip2.mp4", "green", 2, "320x240", "25")

    edl = EDL("mixed_specs", "16:9", [
        Clip(clip1, 0.0, 2.0),
        Clip(clip2, 0.0, 2.0),
    ])
    out = render_edl(edl, tmp_path / "mixed_specs_out.mp4", height=360)
    # Should complete successfully
    assert out.exists()
    assert probe_duration(out) == pytest.approx(4.0, abs=0.5)
    # Verify output has expected aspect ratio
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    # 16:9 aspect ratio
    assert abs(w / h - 16 / 9) < 0.02


def test_render_odd_height(tmp_path):
    """Test render with odd height parameter."""
    d = tmp_path / "odd_height"
    d.mkdir()
    clip = make_clip(d / "clip.mp4", "red", 2)

    edl = EDL("odd", "16:9", [Clip(clip, 0.0, 2.0)])
    # Pass odd height (361 instead of 360)
    out = render_edl(edl, tmp_path / "odd_height_out.mp4", height=361)
    # Should complete successfully despite odd height
    assert out.exists()
    # Verify output height is even
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    # Height should be rounded to even
    assert h % 2 == 0
    # Width should also be even
    assert w % 2 == 0
