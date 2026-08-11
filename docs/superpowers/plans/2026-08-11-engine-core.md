# Движок анализа и автомонтажа (headless) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Headless-ядро happy_tandemista: из папки с файлами прыжка (handcam/outside/интервью/приземление) построить таймлайн фаз по телеметрии+аудио+CV и отрендерить три варианта монтажа (full 16:9, emotions 16:9, highlights 9:16) через CLI.

**Architecture:** Чистый Python-пакет `tandemista.engine` без БД и сети (кроме CV-адаптера к Claude API). Конвейер: извлечение сигналов (GPMF-телеметрия, аудио-RMS, CV-разметка) → детект фаз → фьюжн в общий таймлайн → EDL по декларативным шаблонам → ffmpeg-рендер. CLI связывает всё. Облачный каркас (FastAPI/Celery) подключит этот пакет в следующем плане.

**Tech Stack:** Python 3.12+, numpy, anthropic (CV-адаптер), pytest; системные ffmpeg/ffprobe. Полная спека: `docs/superpowers/specs/2026-08-11-happy-tandemista-design.md`.

## Global Constraints

- Python ≥ 3.12; зависимости backend только: `numpy`, `anthropic`, dev: `pytest` (новые — только через обсуждение).
- `ffmpeg` и `ffprobe` должны быть в PATH; проверять через `shutil.which` с понятной ошибкой.
- Код, имена и комментарии — на английском; тексты для пользователей — позже, не в этом плане.
- Все публичные функции — с type hints; dataclasses — `frozen=True` где нет мутаций.
- Коммиты — conventional commits (`feat:`, `test:`, `chore:`).
- Времена в сигналах и фазах — секунды `float` от начала файла; общий таймлайн — секунды от общего нуля прыжка.
- CV-модель: `claude-sonnet-5`; при реализации Task 8 свериться со скиллом `claude-api` (актуальные параметры vision-запросов).
- Тесты, требующие реальных съёмок GoPro, ищут файлы в `backend/tests/samples/` и скипаются, если их нет (`pytest.mark.skipif`).

---

## File Structure

```
backend/
  pyproject.toml
  tandemista/
    __init__.py
    engine/
      __init__.py
      signals.py     # Sample, SignalSeries, resample
      phases.py      # PhaseName, Phase, детекторы (телеметрия, аудио)
      synthetic.py   # генератор синтетических сигналов прыжка (для тестов и демо)
      media.py       # обёртки ffmpeg/ffprobe: длительность, извлечение аудио, кадров, gpmd-потока
      gpmf.py        # минимальный KLV-парсер GPMF, GPS5 → SignalSeries
      cv.py          # CVAnnotator (Protocol), ClaudeVisionAnnotator, типы CVAnnotation/Moment
      timeline.py    # SourceFile, JumpTimeline, build_timeline (фьюжн)
      templates.py   # Slot, Template, три встроенных шаблона
      edl.py         # Clip, EDL, generate_edl
      render.py      # render_edl через ffmpeg
    cli.py           # tandemista analyze-and-cut
  tests/
    engine/ (зеркалит модули)
    samples/         # (в .gitignore, кладёт пользователь) реальные GoPro-файлы
```

Каждый модуль — одна ответственность; порядок задач следует зависимостям.

---

### Task 1: Скелет репозитория и backend-пакета

**Files:**
- Create: `backend/pyproject.toml`, `backend/tandemista/__init__.py`, `backend/tandemista/engine/__init__.py`, `backend/tests/engine/test_sanity.py`, `.gitignore`, `README.md`

**Interfaces:**
- Produces: устанавливаемый пакет `tandemista`, работающий `pytest`.

- [ ] **Step 1: Создать файлы**

`backend/pyproject.toml`:
```toml
[project]
name = "tandemista"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26", "anthropic>=0.40"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["tandemista*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
__pycache__/
*.egg-info/
.venv/
backend/tests/samples/
.pytest_cache/
node_modules/
.next/
dist/
```

`README.md`:
```markdown
# happy_tandemista

Automatic collection, matching, editing and delivery of tandem skydive videos.

- Spec: docs/superpowers/specs/2026-08-11-happy-tandemista-design.md
- Engine (this stage): backend/tandemista/engine — headless analyze+cut pipeline.

## Dev setup
    cd backend
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
Requires ffmpeg/ffprobe in PATH.
```

`backend/tests/engine/test_sanity.py`:
```python
import tandemista


def test_package_imports():
    assert tandemista is not None
```

Оба `__init__.py` — пустые.

- [ ] **Step 2: Установить и прогнать тесты**

Run: `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest -q`
Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "chore: repo skeleton, backend package, spec and plan docs"
```

---

### Task 2: Модель сигналов (SignalSeries)

**Files:**
- Create: `backend/tandemista/engine/signals.py`
- Test: `backend/tests/engine/test_signals.py`

**Interfaces:**
- Produces: `Sample(t: float, value: float)`; `SignalSeries(name: str, samples: list[Sample])` с методами `resample(step: float) -> SignalSeries`, `value_at(t: float) -> float | None`, свойствами `t0`/`t1`. Все детекторы дальше принимают `SignalSeries`.

- [ ] **Step 1: Написать падающий тест**

```python
# backend/tests/engine/test_signals.py
from tandemista.engine.signals import Sample, SignalSeries


def make(name, pairs):
    return SignalSeries(name, [Sample(t, v) for t, v in pairs])


def test_value_at_interpolates_and_none_outside():
    s = make("v", [(0.0, 0.0), (10.0, 10.0)])
    assert s.value_at(5.0) == 5.0
    assert s.value_at(-1.0) is None
    assert s.value_at(11.0) is None


def test_resample_uniform_grid():
    s = make("v", [(0.0, 0.0), (4.0, 8.0)])
    r = s.resample(2.0)
    assert [(p.t, p.value) for p in r.samples] == [(0.0, 0.0), (2.0, 4.0), (4.0, 8.0)]


def test_bounds():
    s = make("v", [(1.5, 0.0), (9.0, 1.0)])
    assert (s.t0, s.t1) == (1.5, 9.0)
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `cd backend && pytest tests/engine/test_signals.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/signals.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    t: float
    value: float


@dataclass(frozen=True)
class SignalSeries:
    """Time series of one scalar signal; t is seconds from file start."""

    name: str
    samples: list[Sample]

    @property
    def t0(self) -> float:
        return self.samples[0].t

    @property
    def t1(self) -> float:
        return self.samples[-1].t

    def value_at(self, t: float) -> float | None:
        if not self.samples or t < self.t0 or t > self.t1:
            return None
        lo, hi = 0, len(self.samples) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.samples[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[hi]
        if b.t == a.t:
            return a.value
        k = (t - a.t) / (b.t - a.t)
        return a.value + k * (b.value - a.value)

    def resample(self, step: float) -> SignalSeries:
        out: list[Sample] = []
        t = self.t0
        while t <= self.t1 + 1e-9:
            v = self.value_at(min(t, self.t1))
            assert v is not None
            out.append(Sample(round(t, 6), v))
            t += step
        return SignalSeries(self.name, out)
```

- [ ] **Step 4: Тесты зелёные**

Run: `cd backend && pytest tests/engine/test_signals.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/signals.py backend/tests/engine/test_signals.py
git commit -m "feat: SignalSeries time-series model"
```

---

### Task 3: Синтетический генератор сигналов прыжка

**Files:**
- Create: `backend/tandemista/engine/synthetic.py`
- Test: `backend/tests/engine/test_synthetic.py`

**Interfaces:**
- Consumes: `SignalSeries`, `Sample` из Task 2.
- Produces: `make_jump_signals(exit_t: float = 300.0, freefall_s: float = 60.0, canopy_s: float = 240.0, noise: float = 0.0, seed: int = 0) -> dict[str, SignalSeries]` с ключами `"vspeed_ms"`, `"altitude_m"`, `"accel_g"` (шаг 1 с). Используется тестами детектора и демо.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_synthetic.py
from tandemista.engine.synthetic import make_jump_signals


def test_shape_and_phases_present():
    sig = make_jump_signals(exit_t=300, freefall_s=60, canopy_s=240)
    v = sig["vspeed_ms"]
    assert v.value_at(100.0) > -5.0            # climb: почти нет вертикальной скорости вниз
    assert v.value_at(330.0) < -45.0           # terminal freefall
    assert -15.0 < v.value_at(450.0) < -3.0    # canopy
    assert abs(v.value_at(620.0)) < 1.0        # landed
    alt = sig["altitude_m"]
    assert alt.value_at(299.0) > 3500.0
    assert alt.value_at(620.0) < 50.0
    acc = sig["accel_g"]
    assert max(p.value for p in acc.samples if 360 <= p.t <= 366) > 2.5  # deployment spike
```

- [ ] **Step 2: Убедиться, что падает**

Run: `cd backend && pytest tests/engine/test_synthetic.py -q` — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/synthetic.py
from __future__ import annotations

import random

from .signals import Sample, SignalSeries

CLIMB_RATE = 8.0        # m/s up
TERMINAL = -50.0        # m/s freefall
CANOPY_RATE = -6.0      # m/s under canopy
EXIT_ALT = 4000.0


def make_jump_signals(
    exit_t: float = 300.0,
    freefall_s: float = 60.0,
    canopy_s: float = 240.0,
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, SignalSeries]:
    rng = random.Random(seed)
    deploy_t = exit_t + freefall_s
    land_t = deploy_t + canopy_s
    total = land_t + 30.0

    vs: list[Sample] = []
    acc: list[Sample] = []
    alt: list[Sample] = []
    altitude = EXIT_ALT - CLIMB_RATE * exit_t
    t = 0.0
    while t <= total:
        if t < exit_t:
            v = CLIMB_RATE
            a = 1.0
        elif t < exit_t + 5.0:            # accelerating after exit
            v = TERMINAL * (t - exit_t) / 5.0
            a = 0.3
        elif t < deploy_t:
            v = TERMINAL
            a = 1.0                        # drag at terminal reads ~1g
        elif t < deploy_t + 3.0:           # opening shock and deceleration
            v = TERMINAL + (CANOPY_RATE - TERMINAL) * (t - deploy_t) / 3.0
            a = 3.5
        elif t < land_t:
            v = CANOPY_RATE
            a = 1.0
        else:
            v = 0.0
            a = 1.0
        v += rng.gauss(0.0, noise)
        altitude = max(0.0, altitude + v)
        vs.append(Sample(t, v))
        acc.append(Sample(t, a))
        alt.append(Sample(t, altitude))
        t += 1.0

    return {
        "vspeed_ms": SignalSeries("vspeed_ms", vs),
        "altitude_m": SignalSeries("altitude_m", alt),
        "accel_g": SignalSeries("accel_g", acc),
    }
```

- [ ] **Step 4: Тесты зелёные**

Run: `cd backend && pytest tests/engine/test_synthetic.py -q` — `1 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/synthetic.py backend/tests/engine/test_synthetic.py
git commit -m "feat: synthetic jump signal generator for tests and demos"
```

---

### Task 4: Детектор фаз по телеметрии

**Files:**
- Create: `backend/tandemista/engine/phases.py`
- Test: `backend/tests/engine/test_phases_telemetry.py`

**Interfaces:**
- Consumes: `SignalSeries` (Task 2), `make_jump_signals` (Task 3).
- Produces: `PhaseName` (StrEnum: `INTERVIEW, BOARDING, CLIMB, EXIT, FREEFALL, DEPLOYMENT, CANOPY, LANDING, AFTER`), `Phase(name: PhaseName, start: float, end: float, confidence: float, source: str)`, `detect_phases_from_telemetry(vspeed: SignalSeries, accel: SignalSeries | None = None) -> list[Phase]`. EXIT — точечная фаза (start==end), DEPLOYMENT — тоже.

- [ ] **Step 1: Падающие тесты**

```python
# backend/tests/engine/test_phases_telemetry.py
import pytest

from tandemista.engine.phases import PhaseName, detect_phases_from_telemetry
from tandemista.engine.synthetic import make_jump_signals


def by_name(phases, name):
    return next(p for p in phases if p.name == name)


def test_clean_jump_phases():
    sig = make_jump_signals(exit_t=300, freefall_s=60, canopy_s=240)
    phases = detect_phases_from_telemetry(sig["vspeed_ms"], sig["accel_g"])
    exit_p = by_name(phases, PhaseName.EXIT)
    assert exit_p.start == pytest.approx(300, abs=6)
    ff = by_name(phases, PhaseName.FREEFALL)
    assert ff.start == pytest.approx(300, abs=6)
    assert ff.end == pytest.approx(360, abs=6)
    dep = by_name(phases, PhaseName.DEPLOYMENT)
    assert dep.start == pytest.approx(360, abs=6)
    canopy = by_name(phases, PhaseName.CANOPY)
    assert canopy.end == pytest.approx(600, abs=10)
    landing = by_name(phases, PhaseName.LANDING)
    assert landing.start == pytest.approx(600, abs=10)
    assert ff.confidence >= 0.9  # accel spike confirms deployment


def test_noisy_jump_still_detected():
    sig = make_jump_signals(noise=3.0, seed=42)
    phases = detect_phases_from_telemetry(sig["vspeed_ms"], sig["accel_g"])
    assert {p.name for p in phases} >= {PhaseName.FREEFALL, PhaseName.CANOPY, PhaseName.LANDING}


def test_no_freefall_returns_empty():
    sig = make_jump_signals()
    ground_only = sig["vspeed_ms"].__class__(
        "vspeed_ms", [s for s in sig["vspeed_ms"].samples if s.t < 200]
    )
    assert detect_phases_from_telemetry(ground_only) == []
```

- [ ] **Step 2: Убедиться, что падает** — `pytest tests/engine/test_phases_telemetry.py -q` → FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/phases.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .signals import SignalSeries

FREEFALL_VSPEED = -35.0   # m/s
CANOPY_VSPEED = -15.0     # deployment complete when slower than this
LANDED_VSPEED = 2.5       # |v| below this means on the ground
MIN_FREEFALL_S = 5.0
DEPLOY_SPIKE_G = 2.5
SMOOTH_WINDOW = 5         # seconds, centered moving average


class PhaseName(StrEnum):
    INTERVIEW = "interview"
    BOARDING = "boarding"
    CLIMB = "climb"
    EXIT = "exit"
    FREEFALL = "freefall"
    DEPLOYMENT = "deployment"
    CANOPY = "canopy"
    LANDING = "landing"
    AFTER = "after"


@dataclass(frozen=True)
class Phase:
    name: PhaseName
    start: float
    end: float
    confidence: float
    source: str  # "telemetry" | "audio" | "cv" | "role"


def _smooth(samples: list, window: int = SMOOTH_WINDOW) -> list:
    """Centered moving average; keeps Sample type and timestamps."""
    from .signals import Sample

    half = window // 2
    out = []
    for i in range(len(samples)):
        lo, hi = max(0, i - half), min(len(samples), i + half + 1)
        mean = sum(p.value for p in samples[lo:hi]) / (hi - lo)
        out.append(Sample(samples[i].t, mean))
    return out


def detect_phases_from_telemetry(
    vspeed: SignalSeries, accel: SignalSeries | None = None
) -> list[Phase]:
    s = _smooth(vspeed.resample(1.0).samples)
    # freefall: first sustained run of v <= FREEFALL_VSPEED
    ff_start = ff_end = None
    run_start = None
    for p in s:
        if p.value <= FREEFALL_VSPEED:
            run_start = p.t if run_start is None else run_start
            if p.t - run_start >= MIN_FREEFALL_S:
                ff_start = run_start
        elif run_start is not None and ff_start is not None:
            ff_end = p.t
            break
        else:
            run_start = None
    if ff_start is None:
        return []
    if ff_end is None:
        ff_end = s[-1].t

    # deployment confirmed by accel spike near freefall end
    conf = 0.8
    if accel is not None:
        window = [
            p.value for p in accel.resample(1.0).samples if ff_end - 3 <= p.t <= ff_end + 5
        ]
        if window and max(window) >= DEPLOY_SPIKE_G:
            conf = 0.95

    # landing: first sustained |v| < LANDED_VSPEED after deployment
    land_t = None
    calm_start = None
    for p in s:
        if p.t <= ff_end + 3:
            continue
        if abs(p.value) < LANDED_VSPEED:
            calm_start = p.t if calm_start is None else calm_start
            if p.t - calm_start >= 5.0:
                land_t = calm_start
                break
        else:
            calm_start = None
    canopy_end = land_t if land_t is not None else s[-1].t

    phases = [
        Phase(PhaseName.CLIMB, s[0].t, ff_start, conf, "telemetry"),
        Phase(PhaseName.EXIT, ff_start, ff_start, conf, "telemetry"),
        Phase(PhaseName.FREEFALL, ff_start, ff_end, conf, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, ff_end, ff_end, conf, "telemetry"),
        Phase(PhaseName.CANOPY, ff_end, canopy_end, conf, "telemetry"),
    ]
    if land_t is not None:
        phases.append(Phase(PhaseName.LANDING, land_t, min(land_t + 15.0, s[-1].t), conf, "telemetry"))
    return phases
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_phases_telemetry.py -q` → `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/phases.py backend/tests/engine/test_phases_telemetry.py
git commit -m "feat: telemetry-based jump phase detector"
```

---

### Task 5: Обёртки ffmpeg и аудио-профиль файла

**Files:**
- Create: `backend/tandemista/engine/media.py`
- Test: `backend/tests/engine/test_media.py`

**Interfaces:**
- Produces: `require_ffmpeg() -> None` (RuntimeError с внятным текстом, если нет ffmpeg/ffprobe); `probe_duration(path: Path) -> float`; `extract_audio_rms(path: Path, step: float = 1.0) -> SignalSeries` (имя `"audio_rms"`, значения 0..1 — RMS, нормированный на пик файла); `extract_frames(path: Path, out_dir: Path, fps: float = 1.0) -> list[Path]` (jpg-кадры, имена `frame_%06d.jpg`, индекс = секунда). Внутри — `subprocess.run` ffmpeg с decode в `s16le` 8kHz mono через pipe + numpy.

- [ ] **Step 1: Падающий тест** (тестовое медиа генерим самим ffmpeg — тон 440 Гц в первой половине, тишина во второй)

```python
# backend/tests/engine/test_media.py
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
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/media.py
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .signals import Sample, SignalSeries


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found in PATH; install ffmpeg to use tandemista")


def probe_duration(path: Path) -> float:
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_audio_rms(path: Path, step: float = 1.0) -> SignalSeries:
    require_ffmpeg()
    rate = 8000
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    win = int(rate * step)
    samples: list[Sample] = []
    for i in range(0, len(pcm) - win + 1, win):
        rms = float(np.sqrt(np.mean(pcm[i : i + win] ** 2)))
        samples.append(Sample(i / rate, rms))
    peak = max((s.value for s in samples), default=1.0) or 1.0
    return SignalSeries("audio_rms", [Sample(s.t, s.value / peak) for s in samples])


def extract_frames(path: Path, out_dir: Path, fps: float = 1.0) -> list[Path]:
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={fps}", "-q:v", "4", str(pattern)],
        check=True, capture_output=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_media.py -q` → `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/media.py backend/tests/engine/test_media.py
git commit -m "feat: ffmpeg wrappers - duration, audio RMS profile, frame extraction"
```

---

### Task 6: Аудио-детектор фаз (fallback без телеметрии)

**Files:**
- Modify: `backend/tandemista/engine/phases.py` (добавить функцию в конец файла)
- Test: `backend/tests/engine/test_phases_audio.py`

**Interfaces:**
- Consumes: `SignalSeries` c именем `"audio_rms"` (Task 5).
- Produces: `detect_phases_from_audio(rms: SignalSeries) -> list[Phase]` — находит FREEFALL как самый длинный непрерывный интервал 20–120 с с rms выше порога, ставит EXIT в его начало, DEPLOYMENT/CANOPY после; confidence 0.6, source `"audio"`. Пусто, если такого интервала нет.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_phases_audio.py
from tandemista.engine.phases import PhaseName, detect_phases_from_audio
from tandemista.engine.signals import Sample, SignalSeries


def rms_series(profile: list[tuple[int, int, float]], total: int) -> SignalSeries:
    values = [0.05] * total
    for start, end, level in profile:
        for t in range(start, end):
            values[t] = level
    return SignalSeries("audio_rms", [Sample(float(t), v) for t, v in enumerate(values)])


def test_wind_roar_detected_as_freefall():
    # speech at 10..40, wind roar 300..360, quiet canopy after
    rms = rms_series([(10, 40, 0.4), (300, 360, 0.95)], total=650)
    phases = detect_phases_from_audio(rms)
    ff = next(p for p in phases if p.name == PhaseName.FREEFALL)
    assert abs(ff.start - 300) <= 3
    assert abs(ff.end - 360) <= 3
    assert ff.source == "audio"
    assert any(p.name == PhaseName.CANOPY for p in phases)


def test_no_roar_no_phases():
    rms = rms_series([(10, 40, 0.4)], total=200)
    assert detect_phases_from_audio(rms) == []
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ImportError)

- [ ] **Step 3: Реализация** (добавить в `phases.py`)

```python
WIND_RMS = 0.7
MIN_FF_AUDIO_S = 20.0
MAX_FF_AUDIO_S = 120.0


def detect_phases_from_audio(rms: SignalSeries) -> list[Phase]:
    s = rms.resample(1.0).samples
    runs: list[tuple[float, float]] = []
    start = None
    for p in s:
        if p.value >= WIND_RMS:
            start = p.t if start is None else start
        elif start is not None:
            runs.append((start, p.t))
            start = None
    if start is not None:
        runs.append((start, s[-1].t))
    runs = [r for r in runs if MIN_FF_AUDIO_S <= r[1] - r[0] <= MAX_FF_AUDIO_S]
    if not runs:
        return []
    ff_start, ff_end = max(runs, key=lambda r: r[1] - r[0])
    conf = 0.6
    return [
        Phase(PhaseName.CLIMB, s[0].t, ff_start, conf, "audio"),
        Phase(PhaseName.EXIT, ff_start, ff_start, conf, "audio"),
        Phase(PhaseName.FREEFALL, ff_start, ff_end, conf, "audio"),
        Phase(PhaseName.DEPLOYMENT, ff_end, ff_end, conf, "audio"),
        Phase(PhaseName.CANOPY, ff_end, s[-1].t, conf, "audio"),
    ]
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_phases_audio.py -q` → `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/phases.py backend/tests/engine/test_phases_audio.py
git commit -m "feat: audio-based freefall phase detector (no-telemetry fallback)"
```

---

### Task 7: GPMF-парсер и извлечение телеметрии GoPro

**Files:**
- Create: `backend/tandemista/engine/gpmf.py`
- Test: `backend/tests/engine/test_gpmf.py`

**Interfaces:**
- Consumes: `SignalSeries`; ffmpeg-извлечение через `subprocess` (по образцу `media.py`).
- Produces: `parse_gpmf(data: bytes) -> dict[bytes, list]` — минимальный KLV-парсер (структура GPMF: key 4 байта, type 1 байт, size 1 байт, repeat 2 байта BE; type `\x00` — вложенный контейнер; payload выровнен до 4 байт); `telemetry_from_gopro(path: Path) -> dict[str, SignalSeries]` — находит ffprobe'ом поток `gpmd`, извлекает `ffmpeg -codec copy -f data`, парсит `GPS5` (int32 ×5: lat, lon, alt, speed2d, speed3d) с делителями `SCAL`, возвращает `{"altitude_m": ..., "vspeed_ms": ...}` (vspeed — численная производная altitude, сглаженная окном 3 с); пустой dict, если потока нет.

- [ ] **Step 1: Падающий тест** (GPMF-байты собираем вручную `struct.pack` — формат открытый)

```python
# backend/tests/engine/test_gpmf.py
import struct
from pathlib import Path

import pytest

from tandemista.engine.gpmf import gps5_series, parse_gpmf


def klv(key: bytes, type_: bytes, size: int, repeat: int, payload: bytes) -> bytes:
    pad = (-len(payload)) % 4
    return key + type_ + bytes([size]) + struct.pack(">H", repeat) + payload + b"\x00" * pad


def make_stream(alts_m: list[float]) -> bytes:
    # SCAL for GPS5: divisors [1e7, 1e7, 1000, 1000, 100]
    scal = klv(b"SCAL", b"l", 4, 5, struct.pack(">5l", 10000000, 10000000, 1000, 1000, 100))
    gps_payload = b"".join(
        struct.pack(">5l", 0, 0, int(alt * 1000), 0, 0) for alt in alts_m
    )
    gps5 = klv(b"GPS5", b"l", 20, len(alts_m), gps_payload)
    strm = klv(b"STRM", b"\x00", 1, len(scal) + len(gps5), scal + gps5)
    return klv(b"DEVC", b"\x00", 1, len(strm), strm)


def test_parse_nested_and_scaled():
    data = make_stream([4000.0, 3950.0, 3900.0])
    tree = parse_gpmf(data)
    series = gps5_series(tree, packet_rate_hz=1.0)
    alt = series["altitude_m"]
    assert [round(s.value) for s in alt.samples] == [4000, 3950, 3900]


def test_vspeed_derivative():
    data = make_stream([4000.0, 3950.0, 3900.0, 3850.0])
    series = gps5_series(parse_gpmf(data), packet_rate_hz=1.0)
    v = series["vspeed_ms"]
    assert v.value_at(1.5) == pytest.approx(-50.0, abs=5.0)


SAMPLE = Path(__file__).parent.parent / "samples" / "gopro.mp4"


@pytest.mark.skipif(not SAMPLE.exists(), reason="no real GoPro sample provided")
def test_real_gopro_file():
    from tandemista.engine.gpmf import telemetry_from_gopro

    series = telemetry_from_gopro(SAMPLE)
    assert "altitude_m" in series
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/gpmf.py
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

from .media import require_ffmpeg
from .signals import Sample, SignalSeries

_INT_TYPES = {b"l": (">l", 4), b"L": (">L", 4), b"s": (">h", 2), b"S": (">H", 2)}


def parse_gpmf(data: bytes) -> dict[bytes, list]:
    """Parse GPMF KLV into {key: [values...]}; nested containers are flattened."""
    out: dict[bytes, list] = {}
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos : pos + 4]
        type_ = data[pos + 4 : pos + 5]
        size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6 : pos + 8])[0]
        payload_len = size * repeat
        payload = data[pos + 8 : pos + 8 + payload_len]
        if type_ == b"\x00":
            nested = parse_gpmf(payload)
            for k, v in nested.items():
                out.setdefault(k, []).extend(v)
        elif type_ in _INT_TYPES:
            fmt, width = _INT_TYPES[type_]
            n = size // width
            for r in range(repeat):
                chunk = payload[r * size : (r + 1) * size]
                values = [struct.unpack(fmt, chunk[i * width : (i + 1) * width])[0] for i in range(n)]
                out.setdefault(key, []).append(values if n > 1 else values[0])
        # other types (strings, floats) are irrelevant for altitude and skipped
        pos += 8 + payload_len + ((-payload_len) % 4)
    return out


def gps5_series(tree: dict[bytes, list], packet_rate_hz: float = 18.0) -> dict[str, SignalSeries]:
    gps = tree.get(b"GPS5", [])
    scal = tree.get(b"SCAL", [])
    if not gps or len(scal) < 5:
        return {}
    alt_div = float(scal[2] if not isinstance(scal[2], list) else scal[2][0])
    step = 1.0 / packet_rate_hz
    alt = [Sample(i * step, row[2] / alt_div) for i, row in enumerate(gps)]
    alt_series = SignalSeries("altitude_m", alt).resample(1.0)
    vs: list[Sample] = []
    window = 3
    pts = alt_series.samples
    for i in range(1, len(pts)):
        lo = max(0, i - window)
        dt = pts[i].t - pts[lo].t
        vs.append(Sample(pts[i].t, (pts[i].value - pts[lo].value) / dt if dt else 0.0))
    return {"altitude_m": alt_series, "vspeed_ms": SignalSeries("vspeed_ms", vs)}


def telemetry_from_gopro(path: Path) -> dict[str, SignalSeries]:
    require_ffmpeg()
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    idx = next(
        (s["index"] for s in probe["streams"] if s.get("codec_tag_string") == "gpmd"), None
    )
    if idx is None:
        return {}
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-codec", "copy",
         "-map", f"0:{idx}", "-f", "data", "-"],
        check=True, capture_output=True,
    ).stdout
    return gps5_series(parse_gpmf(raw))
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_gpmf.py -q` → `2 passed, 1 skipped`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/gpmf.py backend/tests/engine/test_gpmf.py
git commit -m "feat: minimal GPMF parser and GoPro telemetry extraction"
```

---

### Task 8: CV-аннотатор (интерфейс + адаптер Claude Vision)

Перед реализацией свериться со скиллом `claude-api` (актуальный формат vision-запросов и model id).

**Files:**
- Create: `backend/tandemista/engine/cv.py`
- Test: `backend/tests/engine/test_cv.py`

**Interfaces:**
- Consumes: `extract_frames` (Task 5), `PhaseName`/`Phase` (Task 4).
- Produces:
  - `Moment(t: float, score: float, kind: str)` — kind: `"emotion" | "exit" | "deployment" | "scenic"`.
  - `CVAnnotation(phases: list[Phase], moments: list[Moment])` (source фаз = `"cv"`, confidence 0.7).
  - `CVAnnotator` (Protocol) c методом `annotate(video: Path) -> CVAnnotation`.
  - `ClaudeVisionAnnotator(client: object | None = None, model: str = "claude-sonnet-5", fps: float = 0.5)` — извлекает кадры, шлёт батчами до 20 кадров с промптом, требующим строгий JSON `{"frames": [{"t": 0, "phase": "freefall", "emotion": 0.8, "notable": "exit"}]}`, склеивает соседние кадры одной фазы в `Phase`, кадры с `emotion >= 0.6` или `notable` — в `Moment`. Клиент инжектится для тестов (fake), по умолчанию — `anthropic.Anthropic()`.

- [ ] **Step 1: Падающий тест** (fake-клиент, сеть не нужна)

```python
# backend/tests/engine/test_cv.py
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
```

- [ ] **Step 2: Убедиться, что падает** — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/cv.py
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
    def __init__(self, client: object | None = None, model: str = "claude-sonnet-5", fps: float = 0.5):
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
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_cv.py -q` → `1 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/cv.py backend/tests/engine/test_cv.py
git commit -m "feat: CV annotator protocol with Claude vision adapter"
```

---

### Task 9: Фьюжн таймлайна прыжка

**Files:**
- Create: `backend/tandemista/engine/timeline.py`
- Test: `backend/tests/engine/test_timeline.py`

**Interfaces:**
- Consumes: `Phase`, `PhaseName` (Task 4), `Moment` (Task 8).
- Produces:
  - `SourceFile(path: Path, role: str, duration: float, clock_offset: float, phases: list[Phase], moments: list[Moment])` — role: `"handcam" | "outside" | "ground_interview" | "ground_landing"`; `phases`/`moments` — в локальном времени файла.
  - `JumpTimeline(files: list[SourceFile], phases: list[Phase])` — фазы в общем времени (`local + clock_offset`).
  - `build_timeline(files: list[SourceFile]) -> JumpTimeline`: для каждой фазы берёт вариант с максимальным приоритетом источника (`telemetry` > `audio` > `cv`), а при равном источнике — с большей confidence; ground_interview-файл целиком даёт INTERVIEW, ground_landing — LANDING (source `"role"`, confidence 0.9), если файлы прыжковых камер эту фазу не дали.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_timeline.py
from pathlib import Path

from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.timeline import SourceFile, build_timeline


def sf(role, offset, phases, duration=600.0):
    return SourceFile(Path(f"/{role}.mp4"), role, duration, offset, phases, [])


def test_telemetry_beats_cv_and_offsets_applied():
    handcam = sf(
        "handcam", 100.0,
        [Phase(PhaseName.FREEFALL, 200.0, 260.0, 0.9, "telemetry")],
    )
    outside = sf(
        "outside", 0.0,
        [Phase(PhaseName.FREEFALL, 310.0, 365.0, 0.7, "cv")],
    )
    tl = build_timeline([handcam, outside])
    ff = next(p for p in tl.phases if p.name == PhaseName.FREEFALL)
    assert ff.source == "telemetry"
    assert ff.start == 300.0  # 200 local + 100 offset


def test_ground_roles_fill_missing_phases():
    interview = sf("ground_interview", -1800.0, [], duration=120.0)
    handcam = sf("handcam", 0.0, [Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry")])
    tl = build_timeline([interview, handcam])
    iv = next(p for p in tl.phases if p.name == PhaseName.INTERVIEW)
    assert iv.source == "role"
    assert iv.start == -1800.0 and iv.end == -1680.0
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/timeline.py
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
        if name is not None and name not in candidates:
            offer(
                Phase(name, f.clock_offset, f.clock_offset + f.duration, 0.9, "role")
            )
    phases = sorted(candidates.values(), key=lambda p: p.start)
    return JumpTimeline(files, phases)
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_timeline.py -q` → `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/timeline.py backend/tests/engine/test_timeline.py
git commit -m "feat: jump timeline fusion across sources"
```

---

### Task 10: Шаблоны монтажа и генератор EDL

**Files:**
- Create: `backend/tandemista/engine/templates.py`, `backend/tandemista/engine/edl.py`
- Test: `backend/tests/engine/test_edl.py`

**Interfaces:**
- Consumes: `JumpTimeline`, `SourceFile` (Task 9), `PhaseName` (Task 4), `Moment` (Task 8).
- Produces:
  - `Slot(phase: PhaseName, min_s: float, max_s: float, required: bool = False, prefer_roles: tuple[str, ...] = (), lead_in: float = 0.0, lead_out: float = 0.0)`.
  - `Template(variant: str, aspect: str, slots: tuple[Slot, ...])`; `TEMPLATES: dict[str, Template]` с ключами `"full_16x9"`, `"emotions_16x9"`, `"highlights_9x16"`.
  - `Clip(source: Path, src_in: float, src_out: float)`; `EDL(variant: str, aspect: str, clips: list[Clip])`.
  - `generate_edl(timeline: JumpTimeline, template: Template) -> EDL`: на каждый слот выбирает файл (первый из `prefer_roles`, чья запись покрывает фазу в общем времени; иначе любой покрывающий; у ground-ролей их фаза считается покрытой всегда), окно = фаза ± lead_in/lead_out, длительность клипа зажимается в `[min_s, max_s]`; центр окна — у лучшего `Moment` внутри фазы, если есть. Слот пропускается, если нечем заполнить и `required=False`; если `required=True` и нечем — `raise SlotUnfillableError(slot)`.

- [ ] **Step 1: Падающий тест**

```python
# backend/tests/engine/test_edl.py
from pathlib import Path

import pytest

from tandemista.engine.cv import Moment
from tandemista.engine.edl import SlotUnfillableError, generate_edl
from tandemista.engine.phases import Phase, PhaseName
from tandemista.engine.templates import TEMPLATES
from tandemista.engine.timeline import SourceFile, build_timeline


def full_jump_files():
    handcam_phases = [
        Phase(PhaseName.CLIMB, 0.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.EXIT, 300.0, 300.0, 0.9, "telemetry"),
        Phase(PhaseName.FREEFALL, 300.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.DEPLOYMENT, 360.0, 360.0, 0.9, "telemetry"),
        Phase(PhaseName.CANOPY, 360.0, 600.0, 0.9, "telemetry"),
        Phase(PhaseName.LANDING, 600.0, 615.0, 0.9, "telemetry"),
    ]
    moments = [Moment(320.0, 0.95, "emotion")]
    return [
        SourceFile(Path("/interview.mp4"), "ground_interview", 90.0, -2000.0, [], []),
        SourceFile(Path("/handcam.mp4"), "handcam", 640.0, 0.0, handcam_phases, moments),
        SourceFile(Path("/landing.mp4"), "ground_landing", 60.0, 590.0, [], []),
    ]


def test_full_template_covers_slots_in_order():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["full_16x9"])
    assert edl.aspect == "16:9"
    assert edl.clips[0].source == Path("/interview.mp4")
    assert any(c.source == Path("/handcam.mp4") for c in edl.clips)
    for c in edl.clips:
        assert c.src_out > c.src_in >= 0.0


def test_freefall_clip_centered_on_best_moment():
    edl = generate_edl(build_timeline(full_jump_files()), TEMPLATES["emotions_16x9"])
    ff = [c for c in edl.clips if c.source == Path("/handcam.mp4")]
    assert any(c.src_in <= 320.0 <= c.src_out for c in ff)


def test_required_slot_without_footage_raises():
    files = [SourceFile(Path("/interview.mp4"), "ground_interview", 90.0, 0.0, [], [])]
    with pytest.raises(SlotUnfillableError):
        generate_edl(build_timeline(files), TEMPLATES["full_16x9"])
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/templates.py
from __future__ import annotations

from dataclasses import dataclass, field

from .phases import PhaseName


@dataclass(frozen=True)
class Slot:
    phase: PhaseName
    min_s: float
    max_s: float
    required: bool = False
    prefer_roles: tuple[str, ...] = ()
    lead_in: float = 0.0
    lead_out: float = 0.0


@dataclass(frozen=True)
class Template:
    variant: str
    aspect: str  # "16:9" | "9:16"
    slots: tuple[Slot, ...]


TEMPLATES: dict[str, Template] = {
    "full_16x9": Template(
        "full_16x9", "16:9",
        (
            Slot(PhaseName.INTERVIEW, 15, 25, prefer_roles=("ground_interview",)),
            Slot(PhaseName.CLIMB, 5, 10, prefer_roles=("handcam",)),
            Slot(PhaseName.EXIT, 5, 13, required=True, lead_in=3, lead_out=10,
                 prefer_roles=("outside", "handcam")),
            Slot(PhaseName.FREEFALL, 30, 60, prefer_roles=("outside", "handcam")),
            Slot(PhaseName.CANOPY, 5, 15, prefer_roles=("handcam",)),
            Slot(PhaseName.LANDING, 5, 15, prefer_roles=("ground_landing", "outside")),
        ),
    ),
    "emotions_16x9": Template(
        "emotions_16x9", "16:9",
        (
            Slot(PhaseName.INTERVIEW, 5, 10, prefer_roles=("ground_interview",)),
            Slot(PhaseName.EXIT, 5, 13, required=True, lead_in=3, lead_out=10,
                 prefer_roles=("outside", "handcam")),
            Slot(PhaseName.FREEFALL, 15, 25, prefer_roles=("handcam", "outside")),
            Slot(PhaseName.DEPLOYMENT, 3, 6, lead_out=4, prefer_roles=("handcam",)),
            Slot(PhaseName.LANDING, 4, 8, prefer_roles=("ground_landing",)),
        ),
    ),
    "highlights_9x16": Template(
        "highlights_9x16", "9:16",
        (
            Slot(PhaseName.EXIT, 3, 5, required=True, lead_in=1, lead_out=4,
                 prefer_roles=("outside", "handcam")),
            Slot(PhaseName.FREEFALL, 6, 8, prefer_roles=("handcam", "outside")),
            Slot(PhaseName.LANDING, 2, 4, prefer_roles=("ground_landing",)),
        ),
    ),
}
```

```python
# backend/tandemista/engine/edl.py
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
    for slot in template.slots:
        phase = by_name.get(slot.phase)
        f = _pick_file(timeline, slot, phase) if phase else None
        if f is None:
            if slot.required:
                raise SlotUnfillableError(slot)
            continue
        start = phase.start - slot.lead_in
        end = phase.end + slot.lead_out
        length = min(max(end - start, slot.min_s), slot.max_s)
        best = max(
            (m for m in f.moments
             if phase.start <= m.t + f.clock_offset <= phase.end),
            key=lambda m: m.score, default=None,
        )
        if best is not None:
            start = best.t + f.clock_offset - length / 2
        # convert to file-local time and clamp to the file bounds
        local_in = max(0.0, start - f.clock_offset)
        local_out = min(f.duration, local_in + length)
        if local_out - local_in < 1.0:
            if slot.required:
                raise SlotUnfillableError(slot)
            continue
        clips.append(Clip(f.path, round(local_in, 3), round(local_out, 3)))
    return EDL(template.variant, template.aspect, clips)
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_edl.py -q` → `3 passed`. Если строка с `center` вызывает вопросы линтера — убрать переменную, оставить только ветку со `start`.

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/templates.py backend/tandemista/engine/edl.py backend/tests/engine/test_edl.py
git commit -m "feat: cut templates and EDL generator"
```

---

### Task 11: Рендерер EDL (ffmpeg)

**Files:**
- Create: `backend/tandemista/engine/render.py`
- Test: `backend/tests/engine/test_render.py`

**Interfaces:**
- Consumes: `EDL`, `Clip` (Task 10); `probe_duration`, `require_ffmpeg` (Task 5).
- Produces: `render_edl(edl: EDL, out_path: Path, height: int = 720) -> Path` — один вызов ffmpeg с filter_complex: на каждый клип `trim` + `setpts` + масштабирование (`16:9` → `scale=-2:height`, pad до кратности; `9:16` → `crop=ih*9/16:ih` центр, затем `scale`), `concat` видео и аудио (`atrim`/`asetpts`), кодеки `libx264` + `aac`.

- [ ] **Step 1: Падающий тест** (исходники — три цветных клипа из lavfi)

```python
# backend/tests/engine/test_render.py
import subprocess
from pathlib import Path

import pytest

from tandemista.engine.edl import EDL, Clip
from tandemista.engine.media import probe_duration
from tandemista.engine.render import render_edl


def make_clip(path: Path, color: str, seconds: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("render")
    return [
        make_clip(d / "a.mp4", "red", 5),
        make_clip(d / "b.mp4", "green", 5),
    ]


def test_render_16x9_duration(sources, tmp_path):
    edl = EDL("full_16x9", "16:9", [
        Clip(sources[0], 0.0, 3.0),
        Clip(sources[1], 1.0, 4.0),
    ])
    out = render_edl(edl, tmp_path / "out.mp4", height=360)
    assert probe_duration(out) == pytest.approx(6.0, abs=0.5)


def test_render_9x16_aspect(sources, tmp_path):
    edl = EDL("highlights_9x16", "9:16", [Clip(sources[0], 0.0, 2.0)])
    out = render_edl(edl, tmp_path / "v.mp4", height=640)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    assert abs(w / h - 9 / 16) < 0.02
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/engine/render.py
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
```

- [ ] **Step 4: Тесты зелёные** — `pytest tests/engine/test_render.py -q` → `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/engine/render.py backend/tests/engine/test_render.py
git commit -m "feat: ffmpeg EDL renderer with 16:9 and 9:16 output"
```

---

### Task 12: CLI `analyze-and-cut` — сквозная сборка

**Files:**
- Create: `backend/tandemista/cli.py`
- Test: `backend/tests/test_cli.py`
- Modify: `backend/pyproject.toml` (добавить `[project.scripts]`), `README.md` (раздел Usage)

**Interfaces:**
- Consumes: всё выше — `telemetry_from_gopro`, `extract_audio_rms`, `detect_phases_from_telemetry`, `detect_phases_from_audio`, `probe_duration`, `SourceFile`, `build_timeline`, `TEMPLATES`, `generate_edl`, `render_edl`, опционально `ClaudeVisionAnnotator`.
- Produces: консольная команда `tandemista <jump_dir> --out <dir> [--height N] [--cv]`. Роли файлов — по префиксу имени: `interview_*` → ground_interview, `handcam_*` → handcam, `outside_*` → outside, `landing_*` → ground_landing (конвенция MVP; в облачном каркасе роли придут из Device). Смещения часов: interview ставится до прыжковых файлов, landing — после, по порядку (MVP: `interview=-3600`, прыжковые=0, `landing=+3600` — реальная синхронизация придёт с метчингом). Для каждого шаблона: если генерация упала (`SlotUnfillableError`) — пропустить вариант с warning, остальные рендерить. `--cv` включает `ClaudeVisionAnnotator` (нужен `ANTHROPIC_API_KEY`); без флага CV не вызывается.

- [ ] **Step 1: Падающий тест** (синтетические клипы; телеметрии нет, аудио сделаем «ветер» на нужном интервале — сработает audio-fallback)

```python
# backend/tests/test_cli.py
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
```

- [ ] **Step 2: Убедиться, что падает** — FAIL

- [ ] **Step 3: Реализация**

```python
# backend/tandemista/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.edl import SlotUnfillableError, generate_edl
from .engine.gpmf import telemetry_from_gopro
from .engine.media import extract_audio_rms, probe_duration
from .engine.phases import detect_phases_from_audio, detect_phases_from_telemetry
from .engine.render import render_edl
from .engine.templates import TEMPLATES
from .engine.timeline import SourceFile, build_timeline

ROLE_PREFIX = {
    "interview": "ground_interview",
    "handcam": "handcam",
    "outside": "outside",
    "landing": "ground_landing",
}
# MVP ordering offsets until real matching arrives (plan 3)
ROLE_OFFSET = {"ground_interview": -3600.0, "handcam": 0.0, "outside": 0.0,
               "ground_landing": 3600.0}


def analyze_file(path: Path, role: str, use_cv: bool) -> SourceFile:
    duration = probe_duration(path)
    phases = []
    if role in ("handcam", "outside"):
        telemetry = telemetry_from_gopro(path)
        if "vspeed_ms" in telemetry:
            phases = detect_phases_from_telemetry(telemetry["vspeed_ms"])
        if not phases:
            phases = detect_phases_from_audio(extract_audio_rms(path))
    moments = []
    if use_cv:
        from .engine.cv import ClaudeVisionAnnotator

        ann = ClaudeVisionAnnotator().annotate(path)
        moments = ann.moments
        if not phases:
            phases = ann.phases
    return SourceFile(path, role, duration, ROLE_OFFSET[role], phases, moments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tandemista")
    parser.add_argument("jump_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cv", action="store_true", help="use Claude vision annotator")
    args = parser.parse_args(argv)

    files: list[SourceFile] = []
    for p in sorted(args.jump_dir.glob("*.mp4")):
        role = next(
            (r for pref, r in ROLE_PREFIX.items() if p.name.startswith(pref)), None
        )
        if role is None:
            print(f"skip (unknown role): {p.name}", file=sys.stderr)
            continue
        files.append(analyze_file(p, role, args.cv))
    if not files:
        print("no recognizable files in jump_dir", file=sys.stderr)
        return 1

    timeline = build_timeline(files)
    ok = 0
    for name, template in TEMPLATES.items():
        try:
            edl = generate_edl(timeline, template)
        except SlotUnfillableError as e:
            print(f"warn: {name} skipped: {e}", file=sys.stderr)
            continue
        out = render_edl(edl, args.out / f"{name}.mp4", height=args.height)
        print(f"rendered {out}")
        ok += 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

Добавить в `backend/pyproject.toml`:
```toml
[project.scripts]
tandemista = "tandemista.cli:main"
```

Добавить в `README.md` раздел:
```markdown
## Usage (engine CLI)
    tandemista /path/to/jump_dir --out /tmp/cuts
    # file naming: interview_*.mp4, handcam_*.mp4, outside_*.mp4, landing_*.mp4
    # --cv enables Claude vision highlights (needs ANTHROPIC_API_KEY)
```

- [ ] **Step 4: Все тесты зелёные**

Run: `cd backend && pip install -e ".[dev]" -q && pytest -q`
Expected: все тесты (Task 1–12) PASS, кроме skipped-теста реального GoPro.

- [ ] **Step 5: Commit**

```bash
git add backend/tandemista/cli.py backend/tests/test_cli.py backend/pyproject.toml README.md
git commit -m "feat: analyze-and-cut CLI wiring the full engine pipeline"
```

---

## Верификация плана целиком

1. `cd backend && pytest -q` — всё зелёное.
2. Прогнать CLI на синтетике из `tests/test_cli.py` (fixture-папку можно собрать теми же ffmpeg-командами руками) — получить `full_16x9.mp4`, `emotions_16x9.mp4`, `highlights_9x16.mp4`, открыть глазами.
3. Положить реальные файлы прыжка в папку с префиксами ролей, прогнать `tandemista <dir> --out cuts --cv` — проверить фазы и монтаж на настоящем материале (появление `backend/tests/samples/gopro.mp4` включит интеграционный тест GPMF).

## Отложено в следующие планы

- Облачный каркас: FastAPI, PostgreSQL, Celery, StorageBackend, админка ревью, клиентская страница (план 2).
- Аплоадер Tauri, автометчинг, clock_offset-калибровка (план 3).
- Доставка, вотермарки, PaymentProvider (ЮKassa, Stripe) (план 4).
- Музыка с даккингом, титры, кроп 9:16 по лицам (после пилотной обратной связи; заготовки — `Moment`, `aspect` в EDL).
