"""
drone.py — a fly-BRAIN autopilot for a (simulated) drone.

Two of the fly's real circuits are textbook autopilot primitives, and we already built both:

  * HEADING-HOLD  — the central-complex compass + PFL3->DNa02 steering (fly.steer_signal).
                    Released at any heading, it stabilizes the craft onto a steady course and
                    holds it (yaw stabilization). This is the circuit's own preferred heading,
                    so the autopilot flies a stabilized straight course — it is NOT a global
                    waypoint solver (the steering signal can't reliably seek an arbitrary goal
                    point; it homes to the circuit's intrinsic heading — verified).
  * AVOIDANCE     — the looming -> Giant-Fiber reflex (swatter.circuit: LPLC2/LC4 -> DNp01).
                    When an obstacle looms past the Giant Fiber's reaction threshold it fires a
                    ballistic escape veer (a stereotyped, refractory maneuver, as in the real
                    fly), then the compass re-stabilizes the course.
  * THROTTLE      — the descending neurons (fly.BEHAVIORS: DNp09 forward).

So: cruise forward, the compass STABILIZES the heading, and the Giant-Fiber reflex DODGES
looming obstacles. Both control signals are real connectome circuits.

HONEST BOUNDARY: a 2D KINEMATIC sim, qualitative control (the heading-stabilization and the
avoidance trigger come from the toy-LIF circuits, NOT calibrated flight dynamics), and NO
hardware. It stabilizes a course + dodges; it does not navigate to an arbitrary waypoint.

CLI:
    .venv/bin/python drone.py        # release a drone; the brain stabilizes it + dodges (ASCII)
"""
from __future__ import annotations

import numpy as np

import fly
import swatter

_AP = None


def autopilot():
    """Cache the two real control signals + the circuit's intrinsic stable heading."""
    global _AP
    if _AP is not None:
        return _AP
    fly.steering_curve()                              # warm the DNa02 steering response
    gf = swatter.escape_threshold() or 70             # looming level at which the Giant Fiber fires
    _AP = {"intrinsic": float(fly.intrinsic_heading()), "gf_thresh": float(gf)}
    return _AP


def _gauntlet(intr, specs=((9, 0.0), (16, 1.7), (23, -1.7), (30, 0.8), (37, -1.0)), rad=1.7):
    """Obstacles laid down the drone's stabilized course (downrange r, lateral offset)."""
    ca, sa = np.cos(intr), np.sin(intr)
    return [(ca * r - sa * off, sa * r + ca * off, rad) for (r, off) in specs]


def fly_mission(release_deg=35.0, obstacles=None, steps=900, dt=0.05, v=1.0,
                k=0.7, cone_deg=42, ttc=2.4, veer=2.1, escape_steps=12, cross=42):
    """Release the drone at `release_deg`; the compass stabilizes its heading and the
    Giant-Fiber reflex dodges looming obstacles. Returns the trajectory + telemetry."""
    ap = autopilot()
    intr = ap["intrinsic"]
    ca, sa = np.cos(intr), np.sin(intr)
    if obstacles is None:
        obstacles = _gauntlet(intr)
    th, x, y = np.radians(release_deg), 0.0, 0.0
    commit, cdir, dodges, min_clear = 0, 0.0, 0, 9e9
    traj = [(0.0, 0.0, th, False)]
    cone = np.radians(cone_deg)
    for _ in range(steps):
        for (ox, oy, orad) in obstacles:
            min_clear = min(min_clear, np.hypot(ox - x, oy - y) - orad)
        if commit > 0:                                # ballistic Giant-Fiber escape maneuver
            th += cdir * veer * dt; commit -= 1; avoiding = True
        else:
            danger, adir = False, 0.0
            for (ox, oy, orad) in obstacles:
                dx, dy = ox - x, oy - y
                dist = np.hypot(dx, dy) - orad
                rel = (np.arctan2(dy, dx) - th + np.pi) % (2 * np.pi) - np.pi
                if abs(rel) < cone and dist > 0 and dist / v < ttc:
                    danger, adir = True, (-np.sign(rel) or 1.0)
            if danger:
                dodges += 1; cdir = adir; commit = escape_steps; avoiding = True
            else:
                th -= k * fly.steer_signal(th) * dt   # compass heading-hold (homes to course)
                avoiding = False
        x += v * np.cos(th) * dt; y += v * np.sin(th) * dt
        traj.append((x, y, th, avoiding))
        if ca * x + sa * y > cross:                   # crossed the field along the course
            break
    return {"traj": traj, "dodges": dodges, "intrinsic_deg": float(np.degrees(intr)),
            "min_clearance": float(min_clear), "crashed": bool(min_clear < 0),
            "obstacles": obstacles, "release_deg": float(release_deg)}


def _ascii(res, w=58, h=18):
    pts = res["traj"]
    xs = [p[0] for p in pts] + [o[0] for o in res["obstacles"]]
    ys = [p[1] for p in pts] + [o[1] for o in res["obstacles"]]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    s = min((w - 2) / max(1e-6, maxx - minx), (h - 2) / max(1e-6, maxy - miny))
    g = [[" "] * w for _ in range(h)]

    def put(px, py, c):
        cx = int((px - minx) * s) + 1; cy = h - 1 - int((py - miny) * s)
        if 0 <= cy < h and 0 <= cx < w:
            g[cy][cx] = c
    for (ox, oy, _r) in res["obstacles"]:
        put(ox, oy, "O")
    for p in pts:
        put(p[0], p[1], "!" if p[3] else ".")
    put(0, 0, "S")
    return "\n".join("".join(r) for r in g)


if __name__ == "__main__":
    print("Fly-brain drone autopilot: compass heading-hold + Giant-Fiber obstacle avoidance.\n")
    res = fly_mission()
    print(_ascii(res))
    print("\nreleased %+.0f deg -> compass stabilized to its %.0f deg course | obstacles dodged: %d"
          " | min clearance %.2f | %s"
          % (res["release_deg"], res["intrinsic_deg"], res["dodges"], res["min_clearance"],
             "CRASH" if res["crashed"] else "clean"))
    print("\nS=start  O=looming obstacle  .=stabilized cruise  !=Giant-Fiber AVOID veer\n"
          "The compass (EPG->PFL3->DNa02) holds the course; the looming->Giant-Fiber reflex\n"
          "(LPLC2/LC4->DNp01) dodges. 2D kinematic sim, qualitative real-circuit control — not\n"
          "calibrated flight dynamics, not waypoint guidance, no hardware.")
