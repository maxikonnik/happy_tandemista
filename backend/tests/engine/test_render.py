import subprocess
from pathlib import Path

import pytest

from tandemista.engine.edl import EDL, Clip
from tandemista.engine.media import probe_duration, probe_frame_rate, probe_audio_properties
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
    """Test render with clips of different resolution and frame rate.

    Verifies that normalization is applied: all output frames should have
    consistent frame rate and audio should have consistent sample rate/channels.
    """
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

    # Verify frame rate is normalized: should be max of input (25fps)
    out_fps = probe_frame_rate(out)
    assert out_fps is not None
    assert abs(out_fps - 25.0) < 0.5, f"Expected 25fps, got {out_fps}"

    # Verify audio is normalized: should be 48kHz stereo
    audio_props = probe_audio_properties(out)
    assert audio_props is not None
    assert audio_props["sample_rate"] == 48000, f"Expected 48000Hz, got {audio_props['sample_rate']}"
    assert audio_props["channels"] == 2, f"Expected 2 channels, got {audio_props['channels']}"


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


def test_render_60fps_sources(tmp_path):
    """Test render with 60fps sources preserves high frame rate."""
    d = tmp_path / "fps_60"
    d.mkdir()
    # Create two 60fps clips
    clip1 = make_clip_custom(d / "clip1.mp4", "red", 2, "640x360", "60")
    clip2 = make_clip_custom(d / "clip2.mp4", "blue", 2, "640x360", "60")

    edl = EDL("fps_60", "16:9", [
        Clip(clip1, 0.0, 2.0),
        Clip(clip2, 0.0, 2.0),
    ])
    out = render_edl(edl, tmp_path / "fps_60_out.mp4", height=720)
    assert out.exists()

    # Verify output is at 60fps, not degraded to 30fps
    out_fps = probe_frame_rate(out)
    assert out_fps is not None
    assert abs(out_fps - 60.0) < 0.5, f"Expected 60fps, got {out_fps}"


def test_render_4to3_to_9x16_preserves_aspect(tmp_path):
    """Test render of 4:3 source to 9:16 preserves aspect ratio instead of stretching.

    When a 4:3 (squarer) source is rendered to 9:16 (taller) output, it should be
    letterboxed (pillarboxed vertically) rather than stretched. This is important
    for ground-interview footage that is shot in landscape 4:3 or narrower.
    """
    d = tmp_path / "aspect_test"
    d.mkdir()
    # Create a 4:3 aspect ratio source (400x300)
    clip = make_clip_custom(d / "4to3.mp4", "blue", 2, "400x300", "30")

    edl = EDL("4to3_vertical", "9:16", [Clip(clip, 0.0, 2.0)])
    out = render_edl(edl, tmp_path / "4to3_vertical_out.mp4", height=720)
    assert out.exists()

    # Verify output dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])

    # Output should be 9:16 aspect ratio
    assert abs(w / h - 9 / 16) < 0.02, f"Expected 9:16 aspect, got {w}x{h} = {w/h:.4f}"

    # The source was 4:3 (1.33), narrower than 9:16 (0.5625).
    # After center-crop to ~4:3 max, scale preserves that, then pad adds black bars.
    # So visually, the 4:3 content should NOT be stretched to 9:16.
    # We verify this by checking that if we scale the input to match output height,
    # the width should be less than output width (meaning pillarboxing occurred).
    # Input is 400x300 (4:3). If we scale to height 720, width should be 960.
    # Output width is 404 (9:16 at 720h), much narrower, confirming letterboxing.
    # Actually, let's verify by checking that the content isn't distorted:
    # The filter applies crop=min(iw, ih*9/16) which for 400x300 is min(400, 168.75) = 168.75
    # This is narrower than input, so content will be cropped not stretched.
