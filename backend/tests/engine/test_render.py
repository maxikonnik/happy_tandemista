import subprocess
from pathlib import Path

import numpy as np
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


def make_clip_bands(path: Path, seconds: int, size: str, fps: str) -> Path:
    """Create a clip with a wide green centre band flanked by red and blue edges.

    Lets a test tell a centre crop (only green survives) apart from a stretch
    (the red and blue edges survive, squashed).
    """
    w, h = (int(x) for x in size.split("x"))
    edge = w // 5
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=green:s={size}:r={fps}:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-vf", f"drawbox=x=0:y=0:w={edge}:h={h}:color=red:t=fill,"
                f"drawbox=x={w - edge}:y=0:w={edge}:h={h}:color=blue:t=fill",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


def probe_dimensions(path: Path) -> tuple[int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    return int(probe[0]), int(probe[1])


def first_frame_rgb(path: Path) -> np.ndarray:
    """Decode the first frame as an (h, w, 3) uint8 RGB array."""
    w, h = probe_dimensions(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3)


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
    """Render of a 4:3 source to 9:16 must centre-crop, never squash the full frame.

    A 4:3 (1.33) source is much wider than a 9:16 (0.5625) frame. The vertical
    pipeline centre-crops to 9:16 first, so only the middle of the source survives.
    A stretching implementation would instead squeeze the whole frame in, keeping
    the left and right edges of the source visible - which is what this test rules
    out by painting those edges red and blue and asserting they are gone.
    """
    d = tmp_path / "aspect_test"
    d.mkdir()
    # 400x300 (4:3): red on the left fifth, blue on the right fifth, green between.
    clip = make_clip_bands(d / "4to3.mp4", 2, "400x300", "30")

    edl = EDL("4to3_vertical", "9:16", [Clip(clip, 0.0, 2.0)])
    out = render_edl(edl, tmp_path / "4to3_vertical_out.mp4", height=720)
    assert out.exists()

    w, h = probe_dimensions(out)
    assert w % 2 == 0 and h % 2 == 0, f"dimensions must stay even, got {w}x{h}"
    assert abs(w / h - 9 / 16) < 0.02, f"Expected 9:16 aspect, got {w}x{h} = {w/h:.4f}"

    frame = first_frame_rgb(out).astype(int)
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    total = frame.shape[0] * frame.shape[1]
    red_frac = float((r > g + 40).sum()) / total
    blue_frac = float((b > g + 40).sum()) / total
    # A stretch would keep ~20% red and ~20% blue; the centre crop keeps neither.
    assert red_frac < 0.02, f"source left edge survived, {red_frac:.1%} red - frame was squashed"
    assert blue_frac < 0.02, f"source right edge survived, {blue_frac:.1%} blue - frame was squashed"


def test_render_portrait_to_16x9_preserves_aspect(tmp_path):
    """Regression: the 16:9 path must pillarbox a portrait source, not stretch it.

    A phone-shot portrait interview rendered into full_16x9 must keep its
    proportions, with black padding on either side.
    """
    d = tmp_path / "portrait_test"
    d.mkdir()
    # 300x400 portrait source, solid blue so padding is distinguishable
    clip = make_clip_custom(d / "portrait.mp4", "blue", 2, "300x400", "30")

    edl = EDL("portrait_16x9", "16:9", [Clip(clip, 0.0, 2.0)])
    out = render_edl(edl, tmp_path / "portrait_16x9_out.mp4", height=360)
    assert out.exists()

    w, h = probe_dimensions(out)
    assert (w, h) == (640, 360), f"expected exactly 640x360, got {w}x{h}"

    frame = first_frame_rgb(out).astype(int)
    lit = frame.sum(axis=2).max(axis=0) > 60  # per column: does any pixel carry image?
    assert not lit[0], "left edge should be black padding, not stretched image"
    assert not lit[-1], "right edge should be black padding, not stretched image"

    columns = np.flatnonzero(lit)
    content_w = int(columns[-1] - columns[0] + 1)
    # 300x400 scaled to height 360 must be ~270 wide; a stretch would fill all 640.
    assert abs(content_w / h - 300 / 400) < 0.05, (
        f"content is {content_w}x{h} (ratio {content_w / h:.3f}), "
        f"expected source ratio 0.75 - image was stretched"
    )
