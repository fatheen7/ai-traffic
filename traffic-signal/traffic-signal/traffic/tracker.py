"""Follow each detected vehicle from frame to frame and count it once when it crosses a line."""

import math
from dataclasses import dataclass, field


@dataclass
class Track:
    id: int
    x: float
    y: float
    miss: int = 0
    counted: bool = False
    history: list = field(default_factory=list)


class LineCounter:
    """Greedy nearest-centroid tracker with a horizontal counting line.

    direction: "down" counts vehicles moving down the frame (towards a camera that faces the junction),
    "up" counts the other way, "both" counts either.
    """

    def __init__(self, line_y: float, max_jump: float, direction: str = "down", keep_for: int = 8):
        self.line_y = line_y
        self.max_jump = max_jump
        self.direction = direction
        self.keep_for = keep_for
        self.tracks: list[Track] = []
        self._next = 1
        self.down = 0
        self.up = 0

    @property
    def total(self) -> int:
        if self.direction == "down":
            return self.down
        if self.direction == "up":
            return self.up
        return self.down + self.up

    def update(self, centres: list[tuple[float, float]]) -> list[int]:
        """Match this frame's centres to existing tracks. Returns the ids of tracks counted on this frame."""
        counted_now = []
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for ci, (x, y) in enumerate(centres):
                d = math.hypot(x - tr.x, y - tr.y)
                if d <= self.max_jump:
                    pairs.append((d, ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for d, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            tr = self.tracks[ti]
            prev_y = tr.y
            tr.x, tr.y = centres[ci]
            tr.miss = 0
            tr.history.append((tr.x, tr.y))
            if tr.counted:
                continue
            # Both directions are tallied; only the chosen direction is reported as a new count.
            if prev_y < self.line_y <= tr.y:
                self.down += 1
                tr.counted = True
                if self.direction in ("down", "both"):
                    counted_now.append(tr.id)
            elif prev_y > self.line_y >= tr.y:
                self.up += 1
                tr.counted = True
                if self.direction in ("up", "both"):
                    counted_now.append(tr.id)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.miss += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.keep_for]
        for ci, (x, y) in enumerate(centres):
            if ci not in used_c:
                self.tracks.append(Track(self._next, x, y, history=[(x, y)]))
                self._next += 1
        return counted_now
