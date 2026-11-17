from traffic.config import SignalConfig
from traffic.signal import SignalController, green_times


def test_example_from_the_project():
    assert green_times([24, 14, 10, 2]) == [58, 34, 24, 10]


def test_quiet_road_gets_the_minimum_and_busy_road_the_maximum():
    g = green_times([100, 1, 1, 1], cycle=120, min_green=10, max_green=60)
    assert g[0] == 60 and g[1:] == [10, 10, 10]


def test_no_traffic_splits_evenly():
    assert green_times([0, 0, 0, 0], cycle=120) == [30, 30, 30, 30]


def run(ctl, seconds, dt=0.1):
    for _ in range(int(seconds / dt)):
        ctl.tick(dt)


def test_next_cycle_uses_counts_from_this_cycle():
    cfg = SignalConfig(cycle=40, min_green=5, max_green=20, yellow=1, all_red=1, first_green=5)
    ctl = SignalController(roads=4, cfg=cfg)
    for _ in range(30):
        ctl.add_count(0)
    for _ in range(10):
        ctl.add_count(1)
    run(ctl, 4 * (5 + 2) + 2)
    assert ctl.cycle == 2
    assert ctl.greens == [20, 10, 5, 5]


def test_fixed_timer_ignores_counts():
    cfg = SignalConfig(first_green=8, yellow=1, all_red=1)
    ctl = SignalController(roads=4, cfg=cfg, adaptive=False)
    ctl.add_count(0, 50)
    run(ctl, 4 * 10 + 2)
    assert ctl.greens == [8, 8, 8, 8]


def test_emergency_switches_safely_then_holds_green():
    cfg = SignalConfig(first_green=30, yellow=3, all_red=2)
    ctl = SignalController(roads=4, cfg=cfg)
    run(ctl, 5)                      # north has green
    ctl.emergency(2)                 # siren on south
    assert ctl.light(0) == "yellow"  # north is not cut straight to red
    run(ctl, 3 + 2 + 0.2)
    assert ctl.light(2) == "green"
    run(ctl, 100)
    assert ctl.light(2) == "green"   # held while the siren lasts
    ctl.release_emergency()
    run(ctl, 3 + 2 + 0.2)
    assert ctl.light(0) == "green"   # north gets the rest of its green back
