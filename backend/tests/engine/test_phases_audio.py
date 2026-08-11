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
