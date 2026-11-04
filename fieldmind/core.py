"""Lightweight screen-space football belief state and tactical decisions.

Coordinates are normalized SCREEN coordinates, not Roblox world coordinates.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Optional, Sequence

ACTIONS = ("idle", "intercept", "block", "tackle", "dribble", "pass", "shoot")
FEATURE_DIM = 20

@dataclass(frozen=True)
class Vec2:
    x: float
    y: float
    def distance(self, other: "Vec2") -> float:
        return hypot(self.x - other.x, self.y - other.y)

@dataclass(frozen=True)
class Player:
    foot: Vec2
    team: str = "unknown"  # ally | enemy | unknown; never guess a team from a box
    confidence: float = 1.0

@dataclass
class World:
    ball: Optional[Vec2] = None
    ball_velocity: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    ball_confidence: float = 0.0
    self_pos: Vec2 = field(default_factory=lambda: Vec2(0.5, 0.82))
    players: list[Player] = field(default_factory=list)
    attack_goal: Optional[Vec2] = None
    defend_goal: Optional[Vec2] = None
    possession: Optional[bool] = None  # measured if supplied, else approximate

@dataclass(frozen=True)
class Decision:
    action: str
    target: Optional[Vec2] = None
    reason: str = ""
    confidence: float = 0.0


def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


class BallDynamics:
    """Online alpha-beta-gamma-like fit to observed image motion, NO engine access.

    Resets on discontinuities/long gaps and estimates only screen-space motion.
    Accelerations naturally include camera and perspective artifacts.
    """
    def __init__(self, smoothing: float = 0.38) -> None:
        self.alpha = smoothing
        self.history: deque[tuple[float, Vec2]] = deque(maxlen=12)
        self.velocity = Vec2(0.0, 0.0)
        self.acceleration = Vec2(0.0, 0.0)
        self.last: Optional[Vec2] = None

    def update(self, timestamp: float, ball: Optional[Vec2]) -> None:
        if ball is None:
            return
        if self.history:
            old_t, old_p = self.history[-1]
            dt = timestamp - old_t
            if dt <= 0 or dt > 0.7 or old_p.distance(ball) > 0.38:
                self.history.clear()
                self.velocity = Vec2(0.0, 0.0)
                self.acceleration = Vec2(0.0, 0.0)
            else:
                raw_vx, raw_vy = (ball.x - old_p.x) / dt, (ball.y - old_p.y) / dt
                ax = (raw_vx - self.velocity.x) / dt
                ay = (raw_vy - self.velocity.y) / dt
                a = self.alpha
                self.acceleration = Vec2(_clamp((1-a)*self.acceleration.x+a*ax,-8,8),
                                         _clamp((1-a)*self.acceleration.y+a*ay,-8,8))
                self.velocity = Vec2((1-a)*self.velocity.x + a*raw_vx,
                                     (1-a)*self.velocity.y + a*raw_vy)
        self.history.append((timestamp, ball))
        self.last = ball

    def predict(self, horizon: float = 0.25) -> Optional[Vec2]:
        if self.last is None:
            return None
        t = _clamp(horizon, 0.0, 0.6)
        return Vec2(_clamp(self.last.x + self.velocity.x*t + 0.5*self.acceleration.x*t*t, 0, 1),
                    _clamp(self.last.y + self.velocity.y*t + 0.5*self.acceleration.y*t*t, 0, 1))


def _point_to_segment(p: Vec2, a: Vec2, b: Vec2) -> float:
    ux, uy = b.x-a.x, b.y-a.y
    q = ux*ux + uy*uy
    t = 0 if q < 1e-8 else _clamp(((p.x-a.x)*ux+(p.y-a.y)*uy)/q, 0, 1)
    return p.distance(Vec2(a.x+t*ux, a.y+t*uy))


class TacticalPlanner:
    """Conservative rule-based teacher, usable before the student is trained."""
    def __init__(self, possession_radius: float = 0.095, tackle_radius: float = 0.075):
        self.possession_radius = possession_radius
        self.tackle_radius = tackle_radius

    def decide(self, w: World, predicted_ball: Optional[Vec2] = None) -> Decision:
        if w.ball is None or w.ball_confidence < 0.43:
            return Decision("idle", reason="ball not reliably visible")
        b = w.ball
        predicted = predicted_ball or Vec2(_clamp(b.x+w.ball_velocity.x*0.2,0,1),
                                           _clamp(b.y+w.ball_velocity.y*0.2,0,1))
        enemies = [p for p in w.players if p.team == "enemy" and p.confidence >= .5]
        allies = [p for p in w.players if p.team == "ally" and p.confidence >= .5]
        has_ball = w.possession if w.possession is not None else w.self_pos.distance(b) <= self.possession_radius
        if has_ball:
            nearest_enemy = min((p.foot.distance(b) for p in enemies), default=1.0)
            if w.attack_goal is not None and w.self_pos.distance(w.attack_goal) < .31 and nearest_enemy > .10:
                return Decision("shoot", w.attack_goal, "close, low-pressure shot", .72)
            safe = []
            for p in allies:
                line_clearance = min((_point_to_segment(o.foot, b, p.foot) for o in enemies), default=1.0)
                if line_clearance >= .075 and p.foot.distance(b) > .10:
                    progress = 0 if w.attack_goal is None else w.self_pos.distance(w.attack_goal)-p.foot.distance(w.attack_goal)
                    safe.append((progress+line_clearance*.3, p))
            if safe and (nearest_enemy < .23 or max(x[0] for x in safe) > .16):
                ally = max(safe, key=lambda pair: pair[0])[1]
                return Decision("pass", ally.foot, "available passing lane", .65)
            return Decision("dribble", predicted, "retain possession; no safe pass/shot", .61)
        close = w.self_pos.distance(predicted)
        if close <= self.tackle_radius and enemies:
            return Decision("tackle", predicted, "opponent close and ball reachable", .65)
        if w.defend_goal is not None and enemies:
            carrier = min(enemies, key=lambda p: p.foot.distance(b))
            if carrier.foot.distance(b) <= .125:
                block = Vec2((carrier.foot.x+w.defend_goal.x)/2, (carrier.foot.y+w.defend_goal.y)/2)
                return Decision("block", block, "close passing/shooting lane", .78)
        return Decision("intercept", predicted, "move toward short-horizon ball projection", .67)


def features(w: World) -> list[float]:
    """Fixed-dimension state used by tiny CPU/XPU student; missing data explicit."""
    b = w.ball or Vec2(.5, .5)
    enemies = [p for p in w.players if p.team == "enemy"]
    allies = [p for p in w.players if p.team == "ally"]
    nearest_e = min(enemies, key=lambda p:p.foot.distance(b)).foot if enemies else Vec2(.5,.5)
    nearest_a = min(allies, key=lambda p:p.foot.distance(b)).foot if allies else Vec2(.5,.5)
    ag = w.attack_goal or Vec2(.5,0)
    dg = w.defend_goal or Vec2(.5,1)
    values = [b.x,b.y,w.ball_velocity.x,w.ball_velocity.y,w.ball_confidence,
              w.self_pos.x,w.self_pos.y,w.self_pos.distance(b),
              nearest_e.x,nearest_e.y,nearest_e.distance(b),float(bool(enemies)),
              nearest_a.x,nearest_a.y,nearest_a.distance(b),float(bool(allies)),
              ag.x,ag.y,dg.x,dg.y]
    assert len(values) == FEATURE_DIM
    return [float(_clamp(v,-4,4)) for v in values]
