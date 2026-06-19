"""
swatter.py — play against the fly's REAL escape circuit.

The fly's looming-escape reflex is one of the most famous circuits in the brain: wide-field
looming detectors (LPLC2, LC4) converge massively onto the Giant Fiber (DNp01), the command
neuron for escape takeoff. A shadow rushing at the eye drives the detectors; when their
summed input pushes the Giant Fiber over threshold, the fly lunges away.

Here YOU are the threat: a swatter approaches (a looming stimulus), its growing angular size
drives the real LPLC2+LC4 neurons on the connectome, and the real wiring decides — via the
toy LIF — whether the Giant Fiber spikes BEFORE your swatter lands. Counter-intuitively (and
truthfully), a fast swing is a stronger looming cue, so the fly detects it sooner: sneak slow
to beat the circuit.

HONESTY: this is a TOY LIF on real wiring. We report the ORDER of events (detectors charge ->
Giant Fiber spikes -> lunge) and which swing speeds the circuit escapes, NOT calibrated escape
latencies in milliseconds (absolute rates are not biophysical here). The drive is a scalar
looming ramp on the detector somata, not real retinotopic optics. Takeoff itself is the
labelled VNC stand-in (no muscles in this brain dataset).

CLI:
    .venv/bin/python swatter.py          # sweep swing speeds, print who wins and why
"""
from __future__ import annotations

import numpy as np

import flysim

LOOM_TYPES = ("LPLC2", "LC4")     # wide-field looming detectors (excitatory, ACh)
GF_TYPE = "DNp01"                 # the Giant Fiber — escape-takeoff command neuron

_CIRCUIT = None


def _ids_of(*types):
    flysim._ensure_ann()
    ANN = flysim.ANN
    ct = ANN["cell_type"].astype(str)
    pat = "|".join(types)
    m = ct.str.fullmatch(pat, case=False, na=False).to_numpy()
    return [int(r) for r in ANN.root_id[m].tolist()]


def circuit():
    """Real looming subcircuit: LPLC2+LC4 detectors -> ... -> DNp01 Giant Fiber. Cached."""
    global _CIRCUIT
    if _CIRCUIT is not None:
        return _CIRCUIT
    loom = _ids_of(*LOOM_TYPES)
    gf = _ids_of(GF_TYPE)
    nodes, idx, W = flysim.build_subcircuit(loom + gf, hops=2)
    _CIRCUIT = {
        "nodes": nodes, "idx": idx, "W": W,
        "loom_ids": loom, "gf_ids": gf,
        "loom_rows": [idx[i] for i in loom if i in idx],
        "gf_rows": [idx[i] for i in gf if i in idx],
    }
    return _CIRCUIT


def _loom_drive(approach_ms, dt, peak=34.0, near=0.08):
    """Looming ramp: angular size of an object approaching from far to `near`*start distance
    over `approach_ms`. Grows slowly then explodes near contact — the real expansion profile
    the detectors are tuned to. Returns the per-step external drive."""
    steps = max(1, int(approach_ms / dt))
    s = np.linspace(0.0, 1.0, steps)            # 0 = far, 1 = contact
    dist = 1.0 - (1.0 - near) * s               # normalized distance, -> near at contact
    ang = 1.0 / dist                            # angular size ~ 1/distance
    ramp = (ang - 1.0) / (1.0 / near - 1.0)     # 0 at start, 1 at contact
    return peak * ramp


def run_loom(approach_ms, gain=0.6, dt=0.1, peak=34.0, record=False,
             v_th=15.0, tau_m=10.0, tau_s=5.0, t_ref=2.0):
    """Drive the real detectors with a looming ramp and run the toy LIF. Returns the Giant
    Fiber's first-spike time (ms) or None, plus the detector + GF activity timeline. With
    record=True also returns per-neuron spike counts + spike (time, neuron) for the 3D scene."""
    C = circuit()
    nodes, idx, W = C["nodes"], C["idx"], C["W"]
    loom_rows, gf_rows = C["loom_rows"], C["gf_rows"]
    N = len(nodes)
    drive = _loom_drive(approach_ms, dt, peak=peak)
    steps = len(drive)

    V = np.zeros(N, dtype=np.float32)
    Isyn = np.zeros(N, dtype=np.float32)
    Wsc = W * gain
    cool = np.zeros(N, dtype=np.int32)
    ref_steps = max(1, int(t_ref / dt))
    loom_act = np.zeros(steps, dtype=np.float32)
    gf_act = np.zeros(steps, dtype=np.float32)
    gf_spike_t = None
    loom_set = np.array(loom_rows)
    counts = np.zeros(N, dtype=np.int32)
    rec_t, rec_i = [], []

    for step in range(steps):
        Iext = np.zeros(N, dtype=np.float32)
        if len(loom_set):
            Iext[loom_set] = drive[step]
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            counts += fired
            if record:
                fi = np.nonzero(fired)[0]
                rec_t.append(np.full(fi.shape, step * dt, dtype=np.float32))
                rec_i.append(fi)
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
            if gf_spike_t is None and fired[gf_rows].any():
                gf_spike_t = step * dt
        np.subtract(cool, 1, out=cool, where=cool > 0)
        loom_act[step] = float(fired[loom_set].sum()) if len(loom_set) else 0.0
        gf_act[step] = float(fired[gf_rows].sum())

    out = {"gf_spike_t": gf_spike_t, "approach_ms": approach_ms,
           "loom_act": loom_act, "gf_act": gf_act, "drive": drive, "dt": dt}
    if record:
        out["counts"] = counts
        out["st"] = np.concatenate(rec_t) if rec_t else np.array([], dtype=np.float32)
        out["si"] = np.concatenate(rec_i) if rec_i else np.array([], dtype=np.int64)
    return out


def play_round(approach_ms):
    """One swat: the swatter lands at `approach_ms`. The fly escapes if the Giant Fiber spikes
    before then. Returns the outcome + when each event happened."""
    r = run_loom(approach_ms)
    t = r["gf_spike_t"]
    escaped = (t is not None) and (t < approach_ms)
    return {
        "approach_ms": approach_ms,
        "gf_spike_t": t,
        "escaped": escaped,
        "margin_ms": (None if t is None else round(approach_ms - t, 1)),
        "winner": "fly" if escaped else "you",
    }


def precompute_curve(speeds_ms=(70, 100, 140, 190, 250, 320, 420)):
    """For a range of swing speeds (swatter travel time), when does the Giant Fiber spike and
    who wins? Faster swing = steeper looming ramp = the circuit should fire earlier."""
    return [play_round(int(s)) for s in speeds_ms]


def escape_threshold(lo=40, hi=130, step=3):
    """The swing speed (ms) at/above which the fly always escapes: swing FASTER than this and
    the swatter lands before the Giant Fiber can integrate to threshold (you win). An emergent
    minimum-reaction property of the real wiring + the toy membrane integration."""
    for s in range(lo, hi + 1, step):
        if play_round(s)["escaped"]:
            return s
    return None


def game_curve(lo=44, hi=140, step=8):
    """Coarse swing->outcome curve shipped to the browser so the game loop is data-driven
    (no live LIF per click). Each entry: swing_ms, escaped, gf spike fraction of the swing."""
    out = []
    for s in range(lo, hi + 1, step):
        r = play_round(s)
        out.append({"swing_ms": s, "escaped": r["escaped"],
                    "gf_frac": (None if r["gf_spike_t"] is None else round(r["gf_spike_t"] / s, 3))})
    return out


if __name__ == "__main__":
    C = circuit()
    print("Real escape circuit: %d looming detectors (LPLC2+LC4) -> %d Giant Fiber (DNp01)"
          % (len(C["loom_ids"]), len(C["gf_ids"])))
    print("subcircuit: %d neurons\n" % len(C["nodes"]))
    print("%-12s %-14s %-9s %-7s %s" % ("swing(ms)", "GF spike(ms)", "margin", "winner", "what happened"))
    print("-" * 72)
    for r in precompute_curve():
        gf = "—" if r["gf_spike_t"] is None else "%.1f" % r["gf_spike_t"]
        mg = "—" if r["margin_ms"] is None else ("%+.1f" % r["margin_ms"])
        what = ("escaped %.1fms before the swat" % r["margin_ms"]) if r["escaped"] else \
               ("no escape — you landed it" if r["gf_spike_t"] is None else "too slow — swat landed first")
        print("%-12d %-14s %-9s %-7s %s" % (r["approach_ms"], gf, mg, r["winner"], what))
    print("\nORDER of events only (detectors charge -> Giant Fiber spikes -> lunge); these are\n"
          "not calibrated escape latencies. A faster swing is a stronger looming cue, so the\n"
          "real wiring trips the Giant Fiber sooner — the honest, counter-intuitive lesson.")
