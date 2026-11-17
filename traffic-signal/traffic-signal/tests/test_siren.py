import numpy as np

from traffic.siren import SirenDetector

RATE = 16000


def noise(seconds, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, int(seconds * RATE))
    return np.convolve(x, np.ones(40) / 40, mode="same") * 2  # low rumble, like traffic


def wail(seconds):
    t = np.arange(int(seconds * RATE)) / RATE
    f = 1050 + 400 * np.sin(2 * np.pi * t / 2.2)
    return 0.5 * np.sin(2 * np.pi * np.cumsum(f) / RATE)


def test_traffic_noise_alone_is_not_a_siren():
    sd = SirenDetector()
    assert sd.intervals(sd.scan(noise(10), RATE)) == []


def test_siren_in_noise_is_found_at_the_right_time():
    x = noise(20)
    s = int(8 * RATE)
    x[s:s + int(6 * RATE)] += wail(6)
    sd = SirenDetector()
    spans = sd.intervals(sd.scan(x, RATE))
    assert len(spans) == 1
    start, end = spans[0]
    assert 8 <= start <= 9.5 and 13.5 <= end <= 17


def test_short_horn_is_ignored():
    x = noise(10)
    t = np.arange(int(0.4 * RATE)) / RATE
    x[RATE * 4:RATE * 4 + len(t)] += 0.3 * np.sign(np.sin(2 * np.pi * 420 * t))
    sd = SirenDetector()
    assert sd.intervals(sd.scan(x, RATE)) == []
