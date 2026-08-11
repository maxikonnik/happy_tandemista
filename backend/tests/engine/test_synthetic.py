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
