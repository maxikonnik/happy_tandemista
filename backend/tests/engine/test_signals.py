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
