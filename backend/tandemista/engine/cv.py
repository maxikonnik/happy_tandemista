from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .frames import FrameFeatures, extract_frame_features
from .phases import Phase, PhaseName

CV_CONFIDENCE = 0.5          # below telemetry (0.8-0.95) and audio (0.6): this is a guess from pixels
FREEFALL_FLOW_PERCENTILE = 0.75
FREEFALL_MIN_SKY = 0.35
MIN_FREEFALL_FRAMES = 3
CANOPY_MIN_SKY = 0.25
MAX_SCENIC = 3


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


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _step(feats: list[FrameFeatures]) -> float:
    return feats[1].t - feats[0].t if len(feats) > 1 else 1.0


def phases_from_features(feats: list[FrameFeatures]) -> list[Phase]:
    """Guess phases from motion and sky alone. Last-resort fallback: no telemetry, no audio."""
    if len(feats) < MIN_FREEFALL_FRAMES + 1:
        return []
    flows = [f.motion for f in feats]
    threshold = _percentile(flows, FREEFALL_FLOW_PERCENTILE)
    if threshold <= 0.0:
        return []
    step = _step(feats)

    # freefall: the longest run of high flow over open sky
    best: tuple[int, int] | None = None
    start: int | None = None
    for i, f in enumerate(feats):
        if f.motion >= threshold and f.sky_ratio >= FREEFALL_MIN_SKY:
            start = i if start is None else start
        elif start is not None:
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    if start is not None and (best is None or (len(feats) - start) > (best[1] - best[0])):
        best = (start, len(feats))
    if best is None or (best[1] - best[0]) < MIN_FREEFALL_FRAMES:
        return []

    ff_start, ff_end = feats[best[0]].t, feats[best[1] - 1].t + step
    phases = [
        Phase(PhaseName.EXIT, ff_start, ff_start, CV_CONFIDENCE, "cv"),
        Phase(PhaseName.FREEFALL, ff_start, ff_end, CV_CONFIDENCE, "cv"),
        Phase(PhaseName.DEPLOYMENT, ff_end, ff_end, CV_CONFIDENCE, "cv"),
    ]
    if best[0] > 0:
        phases.insert(0, Phase(PhaseName.CLIMB, feats[0].t, ff_start, CV_CONFIDENCE, "cv"))

    # canopy: frames after freefall that still show sky
    tail = [f for f in feats[best[1]:] if f.sky_ratio >= CANOPY_MIN_SKY]
    if tail:
        phases.append(
            Phase(PhaseName.CANOPY, ff_end, tail[-1].t + step, CV_CONFIDENCE, "cv")
        )
        ground = [f for f in feats if f.t > tail[-1].t and f.sky_ratio < CANOPY_MIN_SKY]
        if ground:
            phases.append(
                Phase(PhaseName.LANDING, ground[0].t, ground[-1].t + step, CV_CONFIDENCE, "cv")
            )
    return phases


def moments_from_features(feats: list[FrameFeatures]) -> list[Moment]:
    """Highlights that need no face model: the flow spike, the flow collapse, the pretty frames."""
    if len(feats) < 3:
        return []
    moments: list[Moment] = []
    deltas = [feats[i].motion - feats[i - 1].motion for i in range(1, len(feats))]
    rise = max(range(len(deltas)), key=lambda i: deltas[i])
    if deltas[rise] > 0:
        moments.append(Moment(feats[rise + 1].t, 1.0, "exit"))
        after = deltas[rise + 1:]
        if after:
            fall = rise + 1 + min(range(len(after)), key=lambda i: after[i])
            if deltas[fall] < 0:
                moments.append(Moment(feats[fall + 1].t, 1.0, "deployment"))

    peak_flow = _percentile([f.motion for f in feats], FREEFALL_FLOW_PERCENTILE)
    calm_and_pretty = [
        f for f in feats if f.motion < peak_flow and f.sky_ratio >= CANOPY_MIN_SKY
    ]
    candidates_sorted = sorted(calm_and_pretty, key=lambda f: -f.sharpness)[:MAX_SCENIC]
    max_sharpness = max([f.sharpness for f in candidates_sorted]) if candidates_sorted else 0.0
    for f in candidates_sorted:
        score = f.sharpness / max_sharpness if max_sharpness > 0 else 0.0
        moments.append(Moment(f.t, score, "scenic"))
    return sorted(moments, key=lambda m: m.t)


class LocalCVAnnotator:
    """Own CV pipeline: OpenCV only, no model weights, no network, no LLM."""

    def __init__(self, fps: float = 0.5):
        self.fps = fps

    def annotate(self, video: Path) -> CVAnnotation:
        feats = extract_frame_features(video, fps=self.fps)
        return CVAnnotation(phases_from_features(feats), moments_from_features(feats))
