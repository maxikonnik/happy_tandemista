from __future__ import annotations

import subprocess
from pathlib import Path

from .edl import EDL
from .media import require_ffmpeg, has_audio, probe_frame_rate, probe_audio_properties


def render_edl(edl: EDL, out_path: Path, height: int = 720, fps: float | None = None) -> Path:
    require_ffmpeg()
    if not edl.clips:
        raise ValueError("EDL has no clips")

    # Round height to even number
    height = int(height / 2) * 2

    # Determine target frame rate: use the highest frame rate from all clips,
    # or fall back to 30fps if rates cannot be determined
    if fps is None:
        frame_rates = []
        for c in edl.clips:
            rate = probe_frame_rate(c.source)
            if rate is not None:
                frame_rates.append(rate)
        fps = max(frame_rates) if frame_rates else 30.0

    # Calculate width based on aspect ratio, rounded to even number
    if edl.aspect == "9:16":
        width = int(height * 9 / 16 / 2) * 2
    else:
        width = int(height * 16 / 9 / 2) * 2

    inputs: list[str] = []
    filters: list[str] = []

    # Build filter for each clip
    for i, c in enumerate(edl.clips):
        inputs += ["-i", str(c.source)]

        # Video processing: trim, reset timestamps, scale/crop, normalize
        if edl.aspect == "9:16":
            # For 9:16 (vertical) output:
            # 1. Crop to preserve 9:16 aspect (limited by input dimensions)
            # 2. Scale to target height while preserving aspect
            # 3. Pad with black bars to reach target width
            # This preserves the source's proportions instead of stretching
            vscale = (
                f"crop=min(iw\\,ih*9/16):ih,"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
            )
        else:
            # For 16:9 (landscape) output:
            # 1. Scale to fit inside the target frame while preserving aspect
            # 2. Pad with black bars to reach the exact target size
            # Portrait phone interviews and 4:3 ground cameras get pillarboxed
            # instead of stretched.
            vscale = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
            )

        # Build video filter chain: trim + setpts + scale/crop + setsar + fps normalization
        has_audio_stream = has_audio(c.source)

        v_filter = (
            f"[{i}:v]trim=start={c.src_in}:end={c.src_out},setpts=PTS-STARTPTS,"
            f"{vscale},setsar=1,fps={fps}[v{i}]"
        )
        filters.append(v_filter)

        # Audio processing: trim and reset timestamps
        if has_audio_stream:
            # Audio from source
            a_filter = (
                f"[{i}:a]atrim=start={c.src_in}:end={c.src_out},asetpts=PTS-STARTPTS,"
                f"aformat=sample_rates=48000:channel_layouts=stereo[a{i}]"
            )
        else:
            # Synthesize silence: generate silence for the duration of the trimmed clip
            duration = c.src_out - c.src_in
            a_filter = (
                f"anullsrc=r=48000:cl=stereo,atrim=0:{duration},asetpts=PTS-STARTPTS[a{i}]"
            )
        filters.append(a_filter)

    # Build concat filter
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(edl.clips)))
    concat_filter = f"{pairs}concat=n={len(edl.clips)}:v=1:a=1[vo][ao]"
    filters.append(concat_filter)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", *inputs,
             "-filter_complex", ";".join(filters),
             "-map", "[vo]", "-map", "[ao]",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg render failed: {e.stderr if e.stderr else e}"
        ) from e

    return out_path
