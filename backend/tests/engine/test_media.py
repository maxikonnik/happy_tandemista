import subprocess
from pathlib import Path

import pytest

from tandemista.engine.media import extract_audio_rms, extract_frames, probe_duration


@pytest.fixture(scope="module")
def tone_then_silence(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=10",
            "-f", "lavfi", "-i",
            "aevalsrc=if(lt(t\\,3)\\,sin(2*PI*440*t)\\,0):d=6",
            "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


def test_probe_duration(tone_then_silence):
    assert probe_duration(tone_then_silence) == pytest.approx(6.0, abs=0.5)


def test_audio_rms_loud_then_quiet(tone_then_silence):
    rms = extract_audio_rms(tone_then_silence, step=1.0)
    assert rms.value_at(1.0) > 0.5
    assert rms.value_at(5.0) < 0.1


def test_extract_frames(tone_then_silence, tmp_path):
    frames = extract_frames(tone_then_silence, tmp_path, fps=1.0)
    assert len(frames) == pytest.approx(6, abs=1)
    assert frames[0].exists()
