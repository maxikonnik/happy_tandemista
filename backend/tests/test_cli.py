import subprocess
from pathlib import Path

import pytest

from tandemista.cli import main


def lavfi_clip(path: Path, seconds: int, audio_expr: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=640x360:rate=10",
         "-f", "lavfi", "-i", f"aevalsrc={audio_expr}:d={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
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
    assert "highlights_9x16.mp4" in rendered
