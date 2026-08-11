from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cv import Moment
from .phases import Phase, PhaseName

SOURCE_RANK = {"telemetry": 3, "role": 2, "audio": 1, "cv": 0}
ROLE_PHASE = {
    "ground_interview": PhaseName.INTERVIEW,
    "ground_landing": PhaseName.LANDING,
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    role: str
    duration: float
    clock_offset: float
    phases: list[Phase]
    moments: list[Moment]


@dataclass(frozen=True)
class JumpTimeline:
    files: list[SourceFile]
    phases: list[Phase]


def build_timeline(files: list[SourceFile]) -> JumpTimeline:
    candidates: dict[PhaseName, Phase] = {}

    def offer(phase: Phase) -> None:
        cur = candidates.get(phase.name)
        if cur is None or (SOURCE_RANK[phase.source], phase.confidence) > (
            SOURCE_RANK[cur.source], cur.confidence
        ):
            candidates[phase.name] = phase

    for f in files:
        for p in f.phases:
            offer(
                Phase(p.name, p.start + f.clock_offset, p.end + f.clock_offset,
                      p.confidence, p.source)
            )
    for f in files:
        name = ROLE_PHASE.get(f.role)
        if name is not None:
            # Offer it like any other phase: SOURCE_RANK already encodes the policy,
            # so a role phase beats a cv guess but yields to audio or telemetry.
            offer(
                Phase(name, f.clock_offset, f.clock_offset + f.duration, 0.9, "role")
            )
    phases = sorted(candidates.values(), key=lambda p: p.start)
    return JumpTimeline(files, phases)
