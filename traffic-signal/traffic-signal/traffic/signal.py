"""Signal logic: turn vehicle counts into green times, run the phases, and handle emergency vehicles.

    green_i = clamp(count_i / total * cycle, min_green, max_green)

The minimum stops a quiet road from being starved; the maximum stops a busy road from holding the junction.
One road has green at a time (one phase per approach), in the order the approaches are given.
"""

from dataclasses import dataclass, field

from .config import SignalConfig


def green_times(counts: list[int], cycle: float = 120, min_green: float = 10, max_green: float = 60) -> list[int]:
    """Share `cycle` seconds of green in proportion to `counts`, clamped to [min_green, max_green]."""
    if not counts:
        return []
    total = sum(counts)
    if total == 0:
        return [round(min(max_green, max(min_green, cycle / len(counts))))] * len(counts)
    return [round(min(max_green, max(min_green, n / total * cycle))) for n in counts]


@dataclass
class Event:
    t: float
    kind: str
    text: str


@dataclass
class CycleRecord:
    number: int
    counts: list
    greens: list
    started: float


@dataclass
class SignalController:
    """A time-driven controller. Call tick(dt) every frame and add_count(road) whenever a camera counts a vehicle."""

    roads: int = 4
    names: list | None = None
    cfg: SignalConfig = field(default_factory=SignalConfig)
    adaptive: bool = True

    def __post_init__(self):
        self.t = 0.0
        self.phase = 0
        self.stage = "green"
        self.greens = [self.cfg.first_green] * self.roads
        self.left = self.greens[0]
        self.counts = [0] * self.roads  # counted during the current cycle
        self.cycle = 1
        self.cycles = [CycleRecord(1, None, list(self.greens), 0.0)]
        self.events: list[Event] = []
        self.emergency_road = None  # road being held green for an ambulance
        self._resume_phase = None
        self._resume_left = None
        self.names = self.names or [f"road {k + 1}" for k in range(self.roads)]

    # ---- inputs -------------------------------------------------------------
    def add_count(self, road: int, n: int = 1):
        self.counts[road] += n

    def emergency(self, road: int):
        """Siren heard on `road`: end the current green safely and give `road` green until release_emergency()."""
        if self.emergency_road == road:
            return
        self.emergency_road = road
        self._log("emergency", f"Siren on {self.names[road]}: clearing the junction for the ambulance")
        if self.stage == "green" and self.phase != road:
            self._resume_phase = self.phase
            self._resume_left = max(self.cfg.min_green, self.left)
            self._to("yellow", self.cfg.yellow)
        elif self.stage == "green" and self.phase == road:
            self.left = float("inf")

    def release_emergency(self):
        if self.emergency_road is None:
            return
        road = self.emergency_road
        self.emergency_road = None
        self._log("emergency", f"Siren gone on {self.names[road]}: normal timing resumes")
        if self.stage == "green" and self.phase == road:
            self._to("yellow", self.cfg.yellow)

    # ---- state --------------------------------------------------------------
    def light(self, road: int) -> str:
        if road != self.phase:
            return "red"
        return {"green": "green", "yellow": "yellow"}.get(self.stage, "red")

    def tick(self, dt: float):
        self.t += dt
        self.left -= dt
        if self.left > 0:
            return
        if self.stage == "green":
            self._to("yellow", self.cfg.yellow)
        elif self.stage == "yellow":
            self._to("all_red", self.cfg.all_red)
        else:
            self._next_phase()

    # ---- internals ----------------------------------------------------------
    def _to(self, stage: str, seconds: float):
        self.stage = stage
        self.left = seconds

    def _next_phase(self):
        if self.emergency_road is not None:
            self.phase = self.emergency_road
            self._to("green", float("inf"))
            return
        if self._resume_phase is not None:
            # after an emergency, give the road that was cut short the rest of its green
            self.phase = self._resume_phase
            self._resume_phase = None
            self._to("green", self._resume_left)
            return
        else:
            self.phase = (self.phase + 1) % self.roads
            if self.phase == 0:
                self._new_cycle()
        self._to("green", self.greens[self.phase])

    def _new_cycle(self):
        counts = self.counts
        self.counts = [0] * self.roads
        self.cycle += 1
        c = self.cfg
        if self.adaptive:
            self.greens = green_times(counts, c.cycle, c.min_green, c.max_green)
        else:
            self.greens = [c.first_green] * self.roads
        self.cycles[-1].counts = counts
        self.cycles.append(CycleRecord(self.cycle, None, list(self.greens), round(self.t, 2)))
        self._log("cycle", f"Cycle {self.cycle}: counted {counts} -> greens {self.greens} s")

    def _log(self, kind, text):
        self.events.append(Event(round(self.t, 2), kind, text))
