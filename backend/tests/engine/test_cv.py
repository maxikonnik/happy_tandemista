import json
import subprocess
from pathlib import Path

import pytest

from tandemista.engine.cv import ClaudeVisionAnnotator
from tandemista.engine.phases import PhaseName


class FakeMessages:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.payloads.pop(0))

        class Block:
            def __init__(self, t):
                self.text = t

        class Resp:
            content = [Block(text)]

        return Resp()


class FakeClient:
    def __init__(self, payloads):
        self.messages = FakeMessages(payloads)


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("cv") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10",
         "-c:v", "libx264", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_annotate_merges_phases_and_moments(tiny_clip):
    payload = {
        "frames": [
            {"t": 0, "phase": "climb", "emotion": 0.2, "notable": None},
            {"t": 2, "phase": "freefall", "emotion": 0.9, "notable": "exit"},
        ]
    }
    ann = ClaudeVisionAnnotator(client=FakeClient([payload]), fps=0.5).annotate(tiny_clip)
    assert any(p.name == PhaseName.FREEFALL and p.source == "cv" for p in ann.phases)
    assert any(m.kind == "exit" for m in ann.moments)
    assert any(m.kind == "emotion" and m.score == 0.9 for m in ann.moments)
