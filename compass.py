"""
compass.py — does the fly's central complex hold a *memory*?

The logic gates (logic.py) are combinational: output depends only on the current input.
The central-complex "compass" is the opposite — a ring attractor that holds a bump of
activity representing heading, and *keeps* holding it after the input stops (working
memory), then rotates it when the fly turns. This is the project's first STATEFUL
computation: the brain remembering, not just calculating.

We assemble the real EPG / PEN / PEG / Delta7 neurons (the compass ring + its recurrent
partners) into an induced subcircuit straight from the connectome, lay them out by angle,
and run a 3-phase protocol on the LIF sim:

    CUE   drive one heading sector  -> a bump should form there
    HOLD  remove all drive          -> does the bump PERSIST? (the memory test)
    TURN  cue a shifted sector       -> does the bump MOVE to the new heading?

A population-vector decode reads the bump's heading over time, so we can measure
formation, persistence (decay), and rotation. Absolute rates are meaningless (toy sim,
see README), so we judge it qualitatively: is activity localized in angle, and does that
localization survive the drive being switched off?

CLI:
    .venv/bin/python compass.py            # run the protocol, sweep gain, report metrics
"""
from __future__ import annotations

import numpy as np

import flysim

# The compass cast. EPG = the ring (ACh, excitatory); PEN/PEG close the recurrent loop
# and shift the bump; Delta7 is the long-range inhibition that sharpens it to one bump.
COMPASS_TYPES = ["EPG", "PEN", "PEG", "Delta7"]


# --------------------------------------------------------------------------- #
# assemble the real circuit
# --------------------------------------------------------------------------- #
def _ids_of_type(*types):
    flysim._ensure_ann()
    ANN = flysim.ANN
    ct = ANN["cell_type"].astype(str)
    mask = np.zeros(len(ANN), dtype=bool)
    for t in types:
        mask |= ct.str.fullmatch(t, case=False, na=False).to_numpy()
    return [int(r) for r in ANN.root_id[mask].tolist()]


def compass_ids(types=COMPASS_TYPES):
    """Root IDs of the central-complex compass neurons, by exact cell_type."""
    return _ids_of_type(*types)


def ring_ids():
    """The EPG neurons — the ring that actually represents heading (cue target + readout)."""
    return _ids_of_type("EPG")


def induced_circuit(ids, min_syn=5):
    """Signed W[post, pre] over EXACTLY this neuron set (the recurrent induced subgraph),
    plus node order and index. Unlike flysim.build_subcircuit this adds no downstream
    hops — we want the closed loop, not a feedforward fan-out."""
    flysim._ensure_conn()
    C, SIGN = flysim.CONN, flysim.SIGN
    nodes = list(dict.fromkeys(int(i) for i in ids))
    nodeset = set(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    sub = C[(C.w >= min_syn) & C.pre.isin(nodeset) & C.post.isin(nodeset)]
    N = len(nodes)
    W = np.zeros((N, N), dtype=np.float32)
    for pre, post, w in zip(sub.pre.values, sub.post.values, sub.w.values):
        s = SIGN.get(int(pre), 0.0)
        if s:
            W[idx[int(post)], idx[int(pre)]] += w * s
    return nodes, idx, W


def ring_angles(nodes, ring_set=None):
    """Assign each neuron an angle around the compass ring. EPG somata don't carry a
    wedge label here, so we recover a ring coordinate from geometry: define the ring's
    principal 2D plane from the EPG (ring) somata, then take each neuron's angle about
    the EPG centroid in that plane. Returns (angles, is_ring_mask, basis)."""
    if ring_set is None:
        ring_set = set(nodes)
    P = _positions()
    pos = np.array([P.get(n, (np.nan, np.nan, np.nan)) for n in nodes], dtype=np.float64)
    ok = ~np.isnan(pos).any(axis=1)
    is_ring = np.array([n in ring_set for n in nodes]) & ok
    base = is_ring if is_ring.any() else ok        # fall back to all if no ring members
    c = pos[base].mean(axis=0)
    X = pos - c
    # principal plane of the RING points (so the angle tracks heading, not stray somata)
    _, _, vt = np.linalg.svd(X[base], full_matrices=False)
    e1, e2 = vt[0], vt[1]
    ang = np.arctan2(X @ e2, X @ e1)               # radians in (-pi, pi]
    ang[~ok] = np.nan
    return ang, is_ring, (e1, e2, c)


_POS = None


def _positions():
    """root_id -> (x,y,z); reads the same annotation TSV export3d uses."""
    global _POS
    if _POS is None:
        import pandas as pd
        p = pd.read_csv(flysim.ANN_FILE, sep="\t",
                        usecols=["root_id", "pos_x", "pos_y", "pos_z"]).dropna()
        p["root_id"] = p["root_id"].astype("int64")
        _POS = {int(r): (x * 4.0, y * 4.0, z * 40.0)
                for r, x, y, z in zip(p.root_id, p.pos_x, p.pos_y, p.pos_z)}
    return _POS


# --------------------------------------------------------------------------- #
# recurrent LIF with a time-varying drive schedule
# --------------------------------------------------------------------------- #
def run_attractor(nodes, idx, W, schedule, dur_ms, dt=0.1, gain=0.5,
                  drive=25.0, v_th=15.0, tau_m=10.0, tau_s=5.0, t_ref=2.0):
    """Same current-based LIF as flysim.run_lif, but external drive varies in time.

    `schedule` is a list of (t_start_ms, t_end_ms, drive_row_indices, amp). Returns
    (counts, spike_t, spike_i) so we can decode the bump over time.
    """
    N = len(nodes)
    V = np.zeros(N, dtype=np.float32)
    Isyn = np.zeros(N, dtype=np.float32)
    Wsc = W * gain
    spikes = np.zeros(N, dtype=np.int32)
    cool = np.zeros(N, dtype=np.int32)
    ref_steps = max(1, int(t_ref / dt))
    rec_t, rec_i = [], []
    steps = int(dur_ms / dt)
    for step in range(steps):
        t = step * dt
        Iext = np.zeros(N, dtype=np.float32)
        for t0, t1, rows, amp in schedule:
            if t0 <= t < t1:
                Iext[rows] += amp
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            spikes += fired
            fi = np.nonzero(fired)[0]
            rec_t.append(np.full(fi.shape, t, dtype=np.float32))
            rec_i.append(fi)
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
        np.subtract(cool, 1, out=cool, where=cool > 0)
    st = np.concatenate(rec_t) if rec_t else np.array([], dtype=np.float32)
    si = np.concatenate(rec_i) if rec_i else np.array([], dtype=np.int64)
    return spikes, st, si


# --------------------------------------------------------------------------- #
# population-vector decode (read the bump's heading over time)
# --------------------------------------------------------------------------- #
def decode_heading(st, si, ang, dur_ms, win_ms=40.0, step_ms=10.0, readout=None):
    """Sliding-window population vector. At each window, sum each readout neuron's spikes
    as a unit vector at its ring angle; the resultant's angle = decoded heading, its
    length / spike-count = bump concentration R in [0,1] (1 = tight bump, 0 = diffuse).
    `readout` is a boolean mask selecting which neurons to read (default: all with angle)."""
    st = np.asarray(st); si = np.asarray(si)
    valid = ~np.isnan(ang)
    if readout is not None:
        valid = valid & readout
    times, headings, R = [], [], []
    t = win_ms / 2
    while t <= dur_ms - win_ms / 2 + 1e-6:
        lo, hi = t - win_ms / 2, t + win_ms / 2
        m = (st >= lo) & (st < hi)
        ii = si[m].astype(int)
        ii = ii[valid[ii]]
        if len(ii):
            a = ang[ii]
            vx, vy = np.cos(a).sum(), np.sin(a).sum()
            res = np.hypot(vx, vy)
            headings.append(float(np.arctan2(vy, vx)))
            R.append(float(res / len(ii)))
        else:
            headings.append(np.nan); R.append(0.0)
        times.append(float(t)); t += step_ms
    return np.array(times), np.array(headings), np.array(R)


def _sector(ang, theta0, half_width=np.pi / 4, only=None):
    """Row indices whose ring angle is within half_width of theta0 (circular). If `only`
    (a boolean mask) is given, restrict to those neurons — we cue the EPG ring, not Δ7."""
    d = np.angle(np.exp(1j * (ang - theta0)))
    sel = np.abs(d) <= half_width
    if only is not None:
        sel = sel & only
    return np.nonzero(sel)[0]


# --------------------------------------------------------------------------- #
# the experiment
# --------------------------------------------------------------------------- #
# Two regimes of the SAME real circuit. "raw" uses the connectome weights as-is — the
# bump forms and steers but forgets. "memory" relaxes the global inhibition and leans on
# the recurrent EPG loop, tipping the circuit into a self-sustaining attractor that HOLDS
# heading with no input. The knob (inh_scale) is the E/I balance a tuned attractor sets.
REGIMES = {
    "raw":    {"gain": 0.6, "inh_scale": 1.0},
    "memory": {"gain": 1.0, "inh_scale": 0.0},
}


def run_protocol(gain=0.5, inh_scale=1.0, cue_ms=200, hold_ms=400, turn_ms=300,
                 theta0=0.0, turn_to=np.pi / 2, drive=30.0, turn_steps=12):
    """CUE one heading -> HOLD (no drive) -> TURN by sweeping the cue to a new heading.
    The TURN is a *moving* cue (like a rotating landmark) so it drags the bump around the
    ring, which is how a ring attractor is actually steered. `inh_scale` scales inhibitory
    (negative) weights: 1.0 = raw connectome, 0.0 = inhibition off (memory regime).
    Returns a dict with the decoded heading trace and phase boundaries."""
    nodes, idx, W = induced_circuit(compass_ids())
    if inh_scale != 1.0:
        W = W.copy()
        W[W < 0] *= inh_scale
    ang, is_ring, basis = ring_angles(nodes, ring_set=set(ring_ids()))
    cue_rows = _sector(ang, theta0, only=is_ring)
    dur = cue_ms + hold_ms + turn_ms
    schedule = [(0, cue_ms, cue_rows, drive)]               # CUE
    # HOLD: nothing
    # TURN: sweep the cued sector from theta0 to turn_to in small steps
    t_turn0 = cue_ms + hold_ms
    dθ = np.angle(np.exp(1j * (turn_to - theta0)))          # shortest signed sweep
    seg = turn_ms / turn_steps
    for k in range(turn_steps):
        th = theta0 + dθ * (k + 1) / turn_steps
        rows = _sector(ang, th, only=is_ring)
        schedule.append((t_turn0 + k * seg, t_turn0 + (k + 1) * seg, rows, drive))
    counts, st, si = run_attractor(nodes, idx, W, schedule, dur, gain=gain, drive=drive)
    t, head, R = decode_heading(st, si, ang, dur, readout=is_ring)
    return {
        "nodes": nodes, "idx": idx, "W": W, "ang": ang, "is_ring": is_ring, "basis": basis,
        "counts": counts, "st": st, "si": si,
        "t": t, "head": head, "R": R,
        "phases": {"cue": (0, cue_ms), "hold": (cue_ms, cue_ms + hold_ms),
                   "turn": (cue_ms + hold_ms, dur)},
        "theta0": theta0, "turn_to": turn_to, "gain": gain, "inh_scale": inh_scale,
        "dur_ms": dur, "n_neurons": len(nodes), "n_cue": len(cue_rows),
    }


def _phase_mask(t, lo, hi):
    return (t >= lo) & (t < hi)


def metrics(res):
    """Quantify the three claims: bump forms (localized at cue), persists through HOLD,
    and rotates toward the new heading during TURN."""
    t, head, R = res["t"], res["head"], res["R"]
    ph = res["phases"]

    def circ_mean(angs):
        angs = angs[~np.isnan(angs)]
        if not len(angs):
            return np.nan
        return float(np.angle(np.mean(np.exp(1j * angs))))

    def circ_dist(a, b):
        return float(abs(np.angle(np.exp(1j * (a - b)))))

    cue_m = _phase_mask(t, *ph["cue"])
    hold_m = _phase_mask(t, *ph["hold"])
    turn_m = _phase_mask(t, *ph["turn"])

    R_cue = float(np.nanmean(R[cue_m])) if cue_m.any() else 0.0
    R_hold = float(np.nanmean(R[hold_m])) if hold_m.any() else 0.0
    # persistence: fraction of HOLD windows that still have a localized bump
    persist_frac = float(np.mean(R[hold_m] > 0.4)) if hold_m.any() else 0.0

    cue_head = circ_mean(head[cue_m])
    # bump must point near the cued heading during CUE
    point_err = circ_dist(cue_head, res["theta0"]) if not np.isnan(cue_head) else np.pi
    # late-HOLD heading vs cue heading (did the memory drift?)
    late = hold_m & (t > ph["hold"][0] + 0.6 * (ph["hold"][1] - ph["hold"][0]))
    drift = circ_dist(circ_mean(head[late]), cue_head) if late.any() else np.pi
    # TURN: did the bump REACH the new heading by the end? (use the last 30% of the sweep)
    late_turn = turn_m & (t > ph["turn"][0] + 0.7 * (ph["turn"][1] - ph["turn"][0]))
    end_head = circ_mean(head[late_turn]) if late_turn.any() else np.nan
    turn_err = circ_dist(end_head, res["turn_to"]) if not np.isnan(end_head) else np.pi
    # how far the bump actually rotated from where it started
    rotated = circ_dist(end_head, cue_head) if not np.isnan(end_head) else 0.0

    return {
        "gain": res["gain"], "n_neurons": res["n_neurons"],
        "R_cue": R_cue, "R_hold": R_hold,
        "persist_frac": persist_frac,
        "point_err_deg": np.degrees(point_err),
        "hold_drift_deg": np.degrees(drift),
        "turn_err_deg": np.degrees(turn_err),
        "rotated_deg": np.degrees(rotated),
        "total_spikes": int(res["counts"].sum()),
    }


if __name__ == "__main__":
    print("The fly compass (real EPG/PEN/PEG/Delta7) under a CUE -> HOLD -> TURN protocol.")
    print("Same neurons, two regimes of the recurrent E/I balance:\n")
    print("%-8s %7s %7s %9s %10s %8s %9s %9s" %
          ("regime", "R_cue", "R_hold", "persist%", "point_err", "drift", "rotated", "turn_err"))
    print("-" * 76)
    for name, p in REGIMES.items():
        m = metrics(run_protocol(**p))
        print("%-8s %7.2f %7.2f %8.0f%% %9.0f° %7.0f° %8.0f° %8.0f°" %
              (name, m["R_cue"], m["R_hold"], 100 * m["persist_frac"],
               m["point_err_deg"], m["hold_drift_deg"], m["rotated_deg"], m["turn_err_deg"]))
    print("\nReading:")
    print("  R_cue   — how tight the bump is while the cue is on (1 = a sharp single bump)")
    print("  R_hold / persist% — does the bump survive the cue being switched OFF? (memory)")
    print("  point_err — bump's heading vs the cued heading;  drift — wander during HOLD")
    print("  rotated — how far the bump moved during TURN (target 90°);  turn_err — vs target")
    print("\n'raw' forms a sharp bump and steers it ~90° to track the turn, but FORGETS on")
    print("hold. 'memory' relaxes inhibition + leans on the recurrent EPG loop: the bump now")
    print("HOLDS heading with zero input — the same real neurons running as working memory.")
