from traffic.tracker import LineCounter


def test_counts_once_when_crossing_down():
    c = LineCounter(line_y=100, max_jump=30)
    for y in range(40, 200, 8):
        c.update([(50, y)])
    assert c.down == 1 and c.up == 0


def test_ignores_wrong_direction_and_counts_it_separately():
    c = LineCounter(line_y=100, max_jump=30, direction="down")
    for y in range(200, 0, -8):
        c.update([(50, y)])
    assert c.total == 0 and c.up == 1


def test_two_vehicles_side_by_side_are_two_counts():
    c = LineCounter(line_y=100, max_jump=30)
    for y in range(40, 200, 8):
        c.update([(50, y), (150, y + 4)])
    assert c.down == 2


def test_short_dropout_keeps_the_track():
    c = LineCounter(line_y=100, max_jump=40, keep_for=3)
    ys = [60, 70, 80, None, None, 110, 120]
    for y in ys:
        c.update([] if y is None else [(50, y)])
    assert c.down == 1
