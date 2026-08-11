from __future__ import annotations

import subprocess
from pathlib import Path

from .edl import EDL, Clip
from .media import require_ffmpeg, has_audio, probe_duration


def render_edl(edl: EDL, out_path: Path, height: int = 720) -> Path:
    require_ffmpeg()
    if not edl.clips:
        raise ValueError("EDL has no clips")

    # Round height to even number
    height = int(height / 2) * 2

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
            # Center-crop to 9:16, but ensure crop width doesn't exceed input width
            # crop=min(width, ih):ih would clip the crop width, so we use:
            # crop=iw*9/16:ih (if iw is wider than ih*9/16)
            # or crop=iw:iw*16/9 (if ih is taller than iw*16/9)
            # Simplest: crop=min(iw, ih*9/16):min(ih, iw*16/9), but that's complex
            # Better: crop=min(iw, ih*(9/16)):ih with center position
            vscale = f"crop=min(iw\\,ih*9/16):ih,scale={width}:{height}"
        else:
            # 16:9: just scale to target dimensions
            vscale = f"scale={width}:{height}"

        # Build video filter chain: trim + setpts + scale/crop + setsar + fps normalization
        has_audio_stream = has_audio(c.source)

        v_filter = (
            f"[{i}:v]trim=start={c.src_in}:end={c.src_out},setpts=PTS-STARTPTS,"
            f"{vscale},setsar=1,fps=30[v{i}]"
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
