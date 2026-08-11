import subprocess
from pathlib import Path

import pytest

from tandemista.cli import main
from tandemista.engine.media import probe_duration


def lavfi_clip(path: Path, seconds: int, audio_expr: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=640x360:rate=10",
         "-f", "lavfi", "-i", f"aevalsrc={audio_expr}:d={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


def lavfi_clip_no_audio(path: Path, seconds: int) -> Path:
    """Create a clip with a video stream only - some outside cameras record mute."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=640x360:rate=10",
         "-c:v", "libx264", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def jump_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("jump")
    lavfi_clip(d / "interview_01.mp4", 30, "0.2*sin(2*PI*300*t)")
    # 120s handcam: quiet 0..40, wind roar 40..100 (freefall), quiet after
    roar = "if(between(t\\,40\\,100)\\,0.9*random(0)\\,0.05*sin(2*PI*200*t))"
    lavfi_clip(d / "handcam_01.mp4", 120, roar)
    lavfi_clip(d / "landing_01.mp4", 20, "0.2*sin(2*PI*300*t)")
    return d


def test_cli_renders_variants(jump_dir, tmp_path):
    code = main([str(jump_dir), "--out", str(tmp_path), "--height", "240"])
    assert code == 0
    rendered = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert "full_16x9.mp4" in rendered
    assert "emotions_16x9.mp4" in rendered
    assert "highlights_9x16.mp4" in rendered
    # Verify each rendered file has non-trivial duration
    for name in rendered:
        duration = probe_duration(tmp_path / name)
        assert duration > 1.0, f"{name} has duration {duration}s, expected > 1.0s"


def test_cli_handles_uppercase_extensions(tmp_path_factory, tmp_path):
    """Regression test: GoPro cameras write uppercase extensions (e.g., .MP4)."""
    d = tmp_path_factory.mktemp("jump_uppercase")
    lavfi_clip(d / "interview_01.MP4", 30, "0.2*sin(2*PI*300*t)")
    roar = "if(between(t\\,40\\,100)\\,0.9*random(0)\\,0.05*sin(2*PI*200*t))"
    lavfi_clip(d / "handcam_01.MP4", 120, roar)
    lavfi_clip(d / "landing_01.MP4", 20, "0.2*sin(2*PI*300*t)")

    code = main([str(d), "--out", str(tmp_path), "--height", "240"])
    assert code == 0, "CLI should succeed with uppercase .MP4 extensions"
    rendered = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert "full_16x9.mp4" in rendered, "Should render full_16x9 from uppercase .MP4 files"
    assert "highlights_9x16.mp4" in rendered, "Should render highlights_9x16 from uppercase .MP4 files"


def test_cli_skips_corrupt_file(tmp_path_factory, tmp_path):
    """Regression test: corrupt/unreadable files should be skipped, not crash CLI."""
    d = tmp_path_factory.mktemp("jump_with_corrupt")
    lavfi_clip(d / "interview_01.mp4", 30, "0.2*sin(2*PI*300*t)")
    roar = "if(between(t\\,40\\,100)\\,0.9*random(0)\\,0.05*sin(2*PI*200*t))"
    lavfi_clip(d / "handcam_01.mp4", 120, roar)
    lavfi_clip(d / "landing_01.mp4", 20, "0.2*sin(2*PI*300*t)")

    # Create a corrupt/unreadable file
    corrupt = d / "handcam_02.mp4"
    corrupt.write_bytes(b"this is not a valid mp4 file")

    code = main([str(d), "--out", str(tmp_path), "--height", "240"])
    # Should return 0 if at least one variant rendered from good files
    assert code == 0, "CLI should succeed despite corrupt file, using good files"
    rendered = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert "full_16x9.mp4" in rendered, "Should render full_16x9 from good files"
    assert "highlights_9x16.mp4" in rendered, "Should render highlights_9x16 from good files"


def test_cli_keeps_camera_without_audio_track(tmp_path_factory, tmp_path, capsys):
    """Regression: a camera with no audio track must fall through to CV, not be discarded.

    Audio analysis is only a fallback when telemetry yields nothing; a file with no
    audio stream is perfectly usable footage and must reach the CV annotator instead
    of being reported as corrupt and dropped.
    """
    d = tmp_path_factory.mktemp("jump_audioless_outside")
    lavfi_clip(d / "interview_01.mp4", 30, "0.2*sin(2*PI*300*t)")
    roar = "if(between(t\\,40\\,100)\\,0.9*random(0)\\,0.05*sin(2*PI*200*t))"
    lavfi_clip(d / "handcam_01.mp4", 120, roar)
    lavfi_clip_no_audio(d / "outside_01.mp4", 60)
    lavfi_clip(d / "landing_01.mp4", 20, "0.2*sin(2*PI*300*t)")

    code = main([str(d), "--out", str(tmp_path), "--height", "240"])
    err = capsys.readouterr().err

    assert "outside_01.mp4" not in err, (
        f"audio-less outside camera must not be skipped, stderr was:\n{err}"
    )
    assert code == 0, "CLI should succeed with an audio-less camera present"
    rendered = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert "full_16x9.mp4" in rendered
    assert "emotions_16x9.mp4" in rendered
    assert "highlights_9x16.mp4" in rendered


def test_analyze_file_without_audio_falls_through_to_cv(tmp_path):
    """The audio-less file must yield a usable SourceFile rather than raising."""
    from tandemista.cli import analyze_file

    clip = lavfi_clip_no_audio(tmp_path / "outside_01.mp4", 20)
    sf = analyze_file(clip, "outside")
    assert sf.duration > 1.0
    assert sf.role == "outside"
