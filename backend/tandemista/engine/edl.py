from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .phases import Phase
from .templates import Slot, Template
from .timeline import JumpTimeline, ROLE_PHASE, SourceFile


class SlotUnfillableError(Exception):
    def __init__(self, slot: Slot):
        super().__init__(f"no footage for required slot {slot.phase}")
        self.slot = slot


@dataclass(frozen=True)
class Clip:
    source: Path
    src_in: float
    src_out: float


@dataclass(frozen=True)
class EDL:
    variant: str
    aspect: str
    clips: list[Clip]


def _covers(f: SourceFile, start: float, end: float) -> bool:
    return f.clock_offset <= start and end <= f.clock_offset + f.duration


def _pick_file(timeline: JumpTimeline, slot: Slot, phase: Phase) -> SourceFile | None:
    def candidates():
        for role in slot.prefer_roles:
            yield from (f for f in timeline.files if f.role == role)
        yield from (f for f in timeline.files if f.role not in slot.prefer_roles)

    for f in candidates():
        if ROLE_PHASE.get(f.role) == phase.name or _covers(f, phase.start, phase.end):
            return f
    return None


def generate_edl(timeline: JumpTimeline, template: Template) -> EDL:
    by_name = {p.name: p for p in timeline.phases}
    clips: list[Clip] = []
    # Track consumed intervals per source file to avoid overlaps
    consumed: dict[Path, list[tuple[float, float]]] = {}

    for slot in template.slots:
        phase = by_name.get(slot.phase)
        f = _pick_file(timeline, slot, phase) if phase else None
        if f is None:
            if slot.required:
                raise SlotUnfillableError(slot)
            continue
        window_start = phase.start - slot.lead_in
        window_end = phase.end + slot.lead_out
        start = window_start
        length = min(max(window_end - window_start, slot.min_s), slot.max_s)
        best = max(
            (m for m in f.moments
             if phase.start <= m.t + f.clock_offset <= phase.end),
            key=lambda m: m.score, default=None,
        )
        if best is not None:
            start = best.t + f.clock_offset - length / 2
            # Centring looks for the best part INSIDE the slot's window, so keep the
            # window there. Moments sitting on a phase boundary (the CV annotator
            # scores "exit" 1.0 at the very first freefall frame) would otherwise
            # drag the clip in front of the phase it is meant to show.
            latest_start = max(window_start, window_end - length)
            start = min(max(start, window_start), latest_start)
        # convert to file-local time and clamp to the file bounds
        local_in = max(0.0, start - f.clock_offset)
        local_out = min(f.duration, local_in + length)

        # Trim to avoid overlaps with already-consumed intervals from the same file
        # Resolve iteratively: each time local_in advances, recalculate local_out
        if f.path in consumed:
            for prev_in, prev_out in sorted(consumed[f.path]):
                if local_in < prev_out and local_out > prev_in:
                    # There's an overlap, move start forward past the consumed interval
                    local_in = max(local_in, prev_out)
                    # Recalculate end to maintain the desired length
                    local_out = min(f.duration, local_in + length)

        if local_out - local_in < 1.0:
            if slot.required:
                raise SlotUnfillableError(slot)
            continue

        clips.append(Clip(f.path, round(local_in, 3), round(local_out, 3)))
        # Track the consumed interval
        if f.path not in consumed:
            consumed[f.path] = []
        consumed[f.path].append((round(local_in, 3), round(local_out, 3)))

    return EDL(template.variant, template.aspect, clips)
