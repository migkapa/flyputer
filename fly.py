"""
fly.py — drive a virtual fly by exciting real descending command neurons.

The brain's output to behavior runs through *descending neurons* (DNs) — ~1300 cells that
carry commands from the brain to the ventral nerve cord. The VNC and muscles are a separate
dataset (not loaded here), so we can't simulate the biomechanics. What we CAN do is excite a
real, named command DN on the connectome, confirm it actually fires, and translate that
genuine motor command — using the established DN->behavior dictionary — into motion of a
virtual fly body (x, y, heading). The brain is real; the body is a labelled stand-in for the
missing VNC.

Behavior -> command neuron (all verified present in FlyWire v783, with left/right sides):
    forward   DNp09   (bilateral forward walking)
    backward  MDN     ("moonwalker" — backward walking)
    left/right DNa02  (asymmetric steering; ipsilateral turn)
    escape    DNp01   (the Giant Fiber — escape takeoff / lunge)

CLI:
    .venv/bin/python fly.py                       # demo: forward, left, forward, escape
    .venv/bin/python fly.py forward left forward  # any sequence of behaviors
"""
from __future__ import annotations

import sys
import numpy as np

import flysim

# behavior -> (command DN cell_type, side, forward speed, turn rate). Speeds are in arena
# units/sec; turn in rad/sec. A turn keeps a little forward speed (fly arcs, not pivots).
BEHAVIORS = {
    "forward":  {"dn": "DNp09", "side": None,    "v": 1.0,  "w": 0.0,  "label": "walk forward"},
    "backward": {"dn": "MDN",   "side": None,    "v": -0.6, "w": 0.0,  "label": "moonwalk backward"},
    "left":     {"dn": "DNa02", "side": "left",  "v": 0.35, "w": 2.4,  "label": "turn left"},
    "right":    {"dn": "DNa02", "side": "right", "v": 0.35, "w": -2.4, "label": "turn right"},
    "escape":   {"dn": "DNp01", "side": None,    "v": -1.8, "w": 0.0,  "label": "escape lunge"},
}

ALIASES = {"walk": "forward", "ahead": "forward", "back": "backward", "reverse": "backward",
           "moonwalk": "backward", "turnleft": "left", "turnright": "right",
           "jump": "escape", "flee": "escape", "startle": "escape"}

# keyboard -> behavior, so a human can pilot the fly by driving its real command neurons
CONTROLS = {
    "ArrowUp": "forward", "w": "forward",
    "ArrowDown": "backward", "s": "backward",
    "ArrowLeft": "left", "a": "left",
    "ArrowRight": "right", "d": "right",
    " ": "escape",
}


def resolve(name):
    n = str(name).lower().strip()
    return ALIASES.get(n, n)


def pilot_setup():
    """Per-behavior data for the interactive 'fly the fly' game: the real command DN ids
    (verified to fire when driven) + the motor primitive. Computed once; the browser then
    runs the game loop client-side from these primitives (no live LIF per keypress)."""
    out = {}
    for name, beh in BEHAVIORS.items():
        cmd = step_behavior(name)                     # excites the real DN, confirms it fires
        out[name] = {"dn": beh["dn"], "side": beh["side"], "dn_ids": cmd["dn_ids"] if cmd else [],
                     "fires": bool(cmd and cmd["dn_spikes"] > 0),
                     "v": beh["v"], "w": beh["w"], "label": beh["label"]}
    return out


def dn_ids(cell_type, side=None):
    """Root IDs of a descending command neuron by exact cell_type (and optional side)."""
    flysim._ensure_ann()
    ANN = flysim.ANN
    m = ANN["cell_type"].astype(str).str.fullmatch(cell_type, case=False, na=False).to_numpy()
    if side and "side" in ANN.columns:
        m &= (ANN["side"].astype(str).str.lower() == side).to_numpy()
    return [int(r) for r in ANN.root_id[m].tolist()]


def excite(drive_ids, dur_ms=150, hops=2):
    """Drive the command DNs, run the downstream LIF, and report how hard the DNs fired
    (the realized motor command) plus the full sim for visualization/energy."""
    nodes, idx, W = flysim.build_subcircuit(drive_ids, hops=hops)
    counts, st, si = flysim.run_lif(nodes, idx, W, drive_ids, dur_ms=dur_ms, record=True)
    dn_spikes = sum(int(counts[idx[d]]) for d in drive_ids if d in idx)
    return {"nodes": nodes, "idx": idx, "W": W, "counts": counts,
            "st": st, "si": si, "dn_spikes": dn_spikes, "drive_ids": drive_ids}


def step_behavior(name, dur_ms=150):
    """Excite one behavior's command neuron and return its motor command, gated on the DN
    actually firing (no spikes -> no movement). `gain01` scales motion by how hard it fired."""
    name = resolve(name)
    beh = BEHAVIORS.get(name)
    if beh is None:
        return None
    ids = dn_ids(beh["dn"], beh["side"])
    sim = excite(ids, dur_ms=dur_ms)
    gain01 = min(1.0, sim["dn_spikes"] / 8.0)        # normalize firing -> [0,1] motor drive
    return {"name": name, "behavior": beh, "dn": beh["dn"], "side": beh["side"],
            "dn_ids": ids, "dn_spikes": sim["dn_spikes"], "gain01": gain01, "sim": sim}


def kinematics(seq, gains, dur_ms=150, substeps=10, x0=0.0, y0=0.0, theta0=np.pi / 2):
    """Pure body kinematics: move the fly through `seq` behaviors, each scaled by a
    pre-computed motor drive `gains[k]` in [0,1] (from how hard its DN fired). Returns the
    trajectory and per-command time spans. Keeps motion and brain-sim consistent when the
    caller already ran the spiking sim (see export3d.build_fly_scene)."""
    x, y, th = x0, y0, theta0
    path = [(0.0, x, y, th)]
    spans, t = [], 0.0
    seg = dur_ms / substeps
    for k, name in enumerate(seq):
        beh = BEHAVIORS[name]
        g = float(gains[k])
        t0 = t
        for _ in range(substeps):
            th += beh["w"] * g * (seg / 1000.0)
            sp = beh["v"] * g
            x += sp * np.cos(th) * (seg / 1000.0)
            y += sp * np.sin(th) * (seg / 1000.0)
            t += seg
            path.append((t, x, y, th))
        spans.append((name, beh, g, t0, t))
    return path, spans


def integrate_path(commands, dur_ms=150, substeps=10, x0=0.0, y0=0.0, theta0=np.pi / 2):
    """Run a sequence of behavior names through the body model. Returns the trajectory
    [(t_ms, x, y, theta), ...] plus the realized per-command motor drive. Movement is
    scaled by how strongly each command neuron actually fired."""
    x, y, th = x0, y0, theta0
    path = [(0.0, x, y, th)]
    realized, t = [], 0.0
    seg = dur_ms / substeps
    for name in commands:
        cmd = step_behavior(name, dur_ms=dur_ms)
        if cmd is None:
            continue
        beh, g = cmd["behavior"], cmd["gain01"]
        for _ in range(substeps):
            th += beh["w"] * g * (seg / 1000.0)
            sp = beh["v"] * g
            x += sp * np.cos(th) * (seg / 1000.0)
            y += sp * np.sin(th) * (seg / 1000.0)
            t += seg
            path.append((t, x, y, th))
        realized.append({"name": cmd["name"], "label": beh["label"], "dn": cmd["dn"],
                         "side": cmd["side"], "dn_spikes": cmd["dn_spikes"],
                         "gain01": round(cmd["gain01"], 2)})
    return {"path": path, "commands": realized, "dur_ms": dur_ms * max(1, len(realized))}


# --------------------------------------------------------------------------- #
# Stage 2 — closed-loop steering: compass -> PFL3 -> DNa02 -> turn
# --------------------------------------------------------------------------- #
# The brain steers by comparing heading (EPG ring) against a goal via PFL3, which drives
# the left/right DNa02 steering neurons. We build the REAL EPG -> PFL3 -> DNa02 induced
# circuit and read the DNa02 left/right firing as a function of heading — a steering signal
# straight from the connectome. It crosses zero (with a stabilizing slope) at the circuit's
# intrinsic preferred heading, so a fly released at any heading is steered onto it.
_STEER = None


def steering_circuit():
    """Real EPG(ring) -> PFL3 -> DNa02(L/R) induced subcircuit, cached."""
    import compass as cp
    epg = cp._ids_of_type("EPG")
    pfl3 = cp._ids_of_type("PFL3")
    dl = dn_ids("DNa02", "left")
    dr = dn_ids("DNa02", "right")
    nodes, idx, W = cp.induced_circuit(epg + pfl3 + dl + dr, min_syn=3)
    ang, is_ring, _ = cp.ring_angles(nodes, ring_set=set(epg))
    Lr = [idx[d] for d in dl if d in idx]
    Rr = [idx[d] for d in dr if d in idx]
    return {"nodes": nodes, "idx": idx, "W": W, "ang": ang, "is_ring": is_ring,
            "Lr": Lr, "Rr": Rr, "cp": cp}


def steering_curve(n=24, gains=(0.7, 0.8, 0.9)):
    """Sample DNa02 (right-left) firing vs heading from the real circuit (denoised over a
    few gains). Returns (grid_radians, steer_values). Cached."""
    global _STEER
    if _STEER is not None:
        return _STEER["grid"], _STEER["raw"]
    sc = steering_circuit()
    cp, nodes = sc["cp"], sc["nodes"]
    grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    raw = []
    for a in grid:
        cue = [nodes[r] for r in cp._sector(sc["ang"], float(a), only=sc["is_ring"])]
        tot = 0.0
        for g in gains:
            sp = flysim.run_lif(nodes, sc["idx"], sc["W"], cue, dur_ms=150, gain=g)
            tot += sum(sp[r] for r in sc["Rr"]) - sum(sp[r] for r in sc["Lr"])
        raw.append(tot / len(gains))
    _STEER = {"grid": grid, "raw": np.array(raw), "circuit": sc}
    return _STEER["grid"], _STEER["raw"]


def steer_signal(heading):
    """Connectome-derived steering command at a given heading (right-positive)."""
    grid, raw = steering_curve()
    e = (heading + np.pi) % (2 * np.pi) - np.pi
    return float(np.interp(e, grid, raw, period=2 * np.pi))


def intrinsic_heading():
    """The circuit's stable preferred heading: a zero crossing of steer() with a slope that
    restores (negative feedback). Returns radians."""
    grid, raw = steering_curve()
    g = np.concatenate([grid, grid[:1] + 2 * np.pi])
    r = np.concatenate([raw, raw[:1]])
    best = None
    for i in range(len(g) - 1):
        if r[i] < 0 and r[i + 1] >= 0:                 # - -> + crossing = stable for th-=k*steer
            frac = -r[i] / (r[i + 1] - r[i])
            cross = g[i] + frac * (g[i + 1] - g[i])
            # prefer the steepest (most strongly restoring) crossing
            slope = abs(r[i + 1] - r[i])
            if best is None or slope > best[1]:
                best = ((cross + np.pi) % (2 * np.pi) - np.pi, slope)
    return best[0] if best else 0.0


def navigate(theta0, steps=70, dt=0.1, k=0.5, v=1.0, step_ms=12,
             x0=0.0, y0=0.0):
    """Closed loop: the fly walks forward while the real steering signal turns it. Returns
    the trajectory [(t_ms, x, y, theta), ...] and the headings list. It homes onto the
    circuit's intrinsic preferred heading."""
    th, x, y, t = float(theta0), x0, y0, 0.0
    path = [(0.0, x, y, th)]
    heads = [th]
    for _ in range(steps):
        th -= k * steer_signal(th) * dt
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        t += step_ms
        path.append((t, x, y, th))
        heads.append(th)
    return path, heads


# --------------------------------------------------------------------------- #
# CLI: run a sequence and draw the trajectory as ASCII (no UI needed)
# --------------------------------------------------------------------------- #
def _ascii_arena(path, w=46, h=20):
    xs = [p[1] for p in path]; ys = [p[2] for p in path]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    sx = (w - 3) / max(1e-6, maxx - minx); sy = (h - 3) / max(1e-6, maxy - miny)
    s = min(sx, sy)
    grid = [[" "] * w for _ in range(h)]
    for i, (_, x, y, _th) in enumerate(path):
        cx = int((x - minx) * s) + 1
        cy = h - 2 - int((y - miny) * s)
        if 0 <= cy < h and 0 <= cx < w:
            grid[cy][cx] = "o" if i and i < len(path) - 1 else ("S" if i == 0 else "F")
    return "\n".join("".join(r) for r in grid)


if __name__ == "__main__":
    seq = sys.argv[1:] or ["forward", "left", "forward", "forward", "escape"]
    print("Driving a virtual fly by exciting real descending command neurons.\n")
    res = integrate_path(seq)
    print("%-10s %-9s %-6s %8s %7s" % ("behavior", "DN", "side", "spikes", "drive"))
    print("-" * 46)
    for c in res["commands"]:
        print("%-10s %-9s %-6s %8d %6.0f%%" %
              (c["label"], c["dn"], c["side"] or "-", c["dn_spikes"], 100 * c["gain01"]))
    print("\ntrajectory  (S=start, o=path, F=finish):\n")
    print(_ascii_arena(res["path"]))
    fp = res["path"][-1]
    print("\nfinal position: (%.2f, %.2f)  heading %.0f°" %
          (fp[1], fp[2], np.degrees(fp[3]) % 360))
    print("\nThe brain commands are real (DNs fired on the connectome); the body is a"
          "\nstand-in for the unloaded VNC. No spikes in a DN -> no movement.")
