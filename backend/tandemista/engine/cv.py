from __future__ import annotations

import base64
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .media import extract_frames
from .phases import Phase, PhaseName

PROMPT = (
    "You are analyzing frames from a tandem skydive video. Frames are 1 per N seconds, "
    "in order; the i-th image corresponds to t seconds given below. Return STRICT JSON "
    '{"frames": [{"t": <sec>, "phase": "<interview|boarding|climb|exit|freefall|'
    'deployment|canopy|landing|after>", "emotion": <0..1 how emotional/joyful the '
    'passenger looks>, "notable": <null|"exit"|"deployment"|"scenic">}]}. No prose.'
)

BATCH = 20
CV_CONFIDENCE = 0.7
EMOTION_MIN = 0.6


@dataclass(frozen=True)
class Moment:
    t: float
    score: float
    kind: str


@dataclass(frozen=True)
class CVAnnotation:
    phases: list[Phase]
    moments: list[Moment]


class CVAnnotator(Protocol):
    def annotate(self, video: Path) -> CVAnnotation: ...


class ClaudeVisionAnnotator:
    def __init__(self, client: object | None = None, model: str = "claude-opus-5", fps: float = 0.5):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.fps = fps

    def annotate(self, video: Path) -> CVAnnotation:
        with tempfile.TemporaryDirectory() as td:
            frames = extract_frames(video, Path(td), fps=self.fps)
            step = 1.0 / self.fps
            rows: list[dict] = []
            for i in range(0, len(frames), BATCH):
                batch = frames[i : i + BATCH]
                times = [round((i + j) * step, 1) for j in range(len(batch))]
                content: list[dict] = [
                    {"type": "text", "text": f"{PROMPT}\nFrame times (s): {times}"}
                ]
                for f in batch:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(f.read_bytes()).decode(),
                            },
                        }
                    )
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": content}],
                )
                rows.extend(json.loads(resp.content[0].text)["frames"])
        return CVAnnotation(self._merge_phases(rows), self._moments(rows))

    def _merge_phases(self, rows: list[dict]) -> list[Phase]:
        phases: list[Phase] = []
        cur_name, cur_start, cur_end = None, 0.0, 0.0
        step = 1.0 / self.fps
        for r in rows:
            name = r.get("phase")
            if name == cur_name:
                cur_end = r["t"] + step
                continue
            if cur_name is not None:
                phases.append(Phase(PhaseName(cur_name), cur_start, cur_end, CV_CONFIDENCE, "cv"))
            cur_name, cur_start, cur_end = name, r["t"], r["t"] + step
        if cur_name is not None:
            phases.append(Phase(PhaseName(cur_name), cur_start, cur_end, CV_CONFIDENCE, "cv"))
        return phases

    def _moments(self, rows: list[dict]) -> list[Moment]:
        moments: list[Moment] = []
        for r in rows:
            if r.get("notable"):
                moments.append(Moment(r["t"], 1.0, r["notable"]))
            if (r.get("emotion") or 0) >= EMOTION_MIN:
                moments.append(Moment(r["t"], r["emotion"], "emotion"))
        return moments
