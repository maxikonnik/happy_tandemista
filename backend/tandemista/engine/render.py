from __future__ import annotations

import subprocess
from pathlib import Path

from .edl import EDL
from .media import require_ffmpeg


def render_edl(edl: EDL, out_path: Path, height: int = 720) -> Path:
    require_ffmpeg()
    if not edl.clips:
        raise ValueError("EDL has no clips")
    inputs: list[str] = []
    filters: list[str] = []
    if edl.aspect == "9:16":
        width = int(height * 9 / 16 / 2) * 2
        vscale = f"crop=ih*9/16:ih,scale={width}:{height}"
    else:
        width = int(height * 16 / 9 / 2) * 2
        vscale = f"scale={width}:{height}"
    for i, c in enumerate(edl.clips):
        inputs += ["-i", str(c.source)]
        filters.append(
            f"[{i}:v]trim=start={c.src_in}:end={c.src_out},setpts=PTS-STARTPTS,"
            f"{vscale},setsar=1[v{i}];"
            f"[{i}:a]atrim=start={c.src_in}:end={c.src_out},asetpts=PTS-STARTPTS[a{i}]"
        )
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(edl.clips)))
    filters.append(f"{pairs}concat=n={len(edl.clips)}:v=1:a=1[vo][ao]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filters),
         "-map", "[vo]", "-map", "[ao]",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)],
        check=True, capture_output=True,
    )
    return out_path
