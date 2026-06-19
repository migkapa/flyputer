"""
export3d.py — turn a stimulation into 3D web-visualization data.

  build_data(query, limit, dur_ms) -> dict     (reused by server.py for the live UI)

CLI:
  .venv/bin/python export3d.py                       # default: olfactory
  .venv/bin/python export3d.py "mushroom body" 60 250
writes fly3d_data.js and opens fly3d.html (the standalone 3D view).
"""
import os
import sys
import json
import webbrowser
from collections import Counter

import numpy as np
import pandas as pd

import flysim
import energy

_POS = None


def _positions():
    """root_id -> (x, y, z) brain coordinates, cached after first read."""
    global _POS
    if _POS is None:
        p = pd.read_csv(flysim.ANN_FILE, sep="\t",
                        usecols=["root_id", "pos_x", "pos_y", "pos_z"]).dropna()
        p["root_id"] = p["root_id"].astype("int64")
        # positions are voxels at 4x4x40 nm; convert to nm for true brain proportions
        _POS = {int(r): (x * 4.0, y * 4.0, z * 40.0) for r, x, y, z in
                zip(p.root_id, p.pos_x, p.pos_y, p.pos_z)}
    return _POS


_TF = None
_GHOST = None


def _transform():
    """Global center + scale from ALL neuron positions, so the active circuit sits in
    its true place inside the full brain."""
    global _TF
    if _TF is None:
        a = np.array(list(_positions().values()), dtype=np.float64)
        c = a.mean(axis=0)
        s = 90.0 / (np.abs(a - c).max() + 1e-9)
        _TF = (c, s)
    return _TF


def _ghost(step=7):
    """Flat [x,y,z, ...] of a downsampled full-brain point cloud (inactive context)."""
    global _GHOST
    if _GHOST is None:
        P = _positions()
        c, s = _transform()
        pts = []
        for k in list(P.keys())[::step]:
            x, y, z = P[k]
            pts += [round((x - c[0]) * s, 1), round((y - c[1]) * s, 1),
                    round((z - c[2]) * s, 1)]
        _GHOST = pts
    return _GHOST


_CV = None
_HERO = {}
_HERO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hero_cache")


def _hero_arbor(rid, n_points=2200):
    """Fetch a neuron mesh (token-free) and return ~n_points surface points in global
    coords as a flat [x,y,z,...] list. Cached in memory + on disk; returns None if
    cloud-volume is missing or the fetch fails (so the rest of the app still works)."""
    global _CV
    rid = int(rid)
    if rid in _HERO:
        return _HERO[rid]
    cache = os.path.join(_HERO_DIR, f"{rid}.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                _HERO[rid] = json.load(f)
            return _HERO[rid]
        except Exception:
            pass
    try:
        import cloudvolume as cv
        if _CV is None:
            _CV = cv.CloudVolume("precomputed://gs://flywire_v141_m783",
                                 use_https=True, progress=False)
        m = _CV.mesh.get(rid)
        m = m[rid] if isinstance(m, dict) else m
        v = np.asarray(m.vertices, dtype=np.float64)
        if len(v) == 0:
            _HERO[rid] = None
            return None
        if len(v) > n_points:                      # stride-sample across the arbor
            v = v[::max(1, len(v) // n_points)][:n_points]
        c, s = _transform()
        v = (v - c) * s
        flat = [round(float(x), 2) for x in v.reshape(-1)]
        _HERO[rid] = flat
        try:
            os.makedirs(_HERO_DIR, exist_ok=True)
            with open(cache, "w") as f:
                json.dump(flat, f)
        except Exception:
            pass
        return flat
    except Exception:
        _HERO[rid] = None
        return None


def warm():
    """Initialize cloud-volume up front so the first hero fetch isn't slow."""
    global _CV
    try:
        import cloudvolume as cv
        if _CV is None:
            _CV = cv.CloudVolume("precomputed://gs://flywire_v141_m783",
                                 use_https=True, progress=False)
    except Exception:
        pass


def resting_data():
    """Starter scene: the whole brain at rest, nothing firing."""
    return {
        "title": "FlyWire brain (at rest)",
        "query": "", "dur_ms": 200,
        "n_input": 0, "n_downstream": 0, "top_downstream_types": [],
        "neurons": [], "edges": [], "ghost": _ghost(), "heroes": [],
    }


def build_data(query="olfactory", limit=40, dur_ms=200, heroes=6):
    """Stimulate `query` neurons and return a JSON-able dict for the 3D viewer."""
    found = flysim.find_neurons(query, limit=limit)["neurons"]
    if not found:
        raise ValueError(f"No neurons matched '{query}'. Try: olfactory, sugar, "
                         "'mushroom body', gustatory, descending, motor.")
    ids = [n["root_id"] for n in found]
    sset = set(ids)
    nodes, idx, W = flysim.build_subcircuit(ids, hops=2)
    counts, st, si = flysim.run_lif(nodes, idx, W, ids, dur_ms=dur_ms, record=True)

    P = _positions()
    active = [r for r in range(len(nodes)) if counts[r] > 0 and nodes[r] in P]
    row2new = {r: k for k, r in enumerate(active)}

    spk = [[] for _ in active]
    for t, r in zip(st.tolist(), si.tolist()):
        k = row2new.get(int(r))
        if k is not None:
            spk[k].append(round(float(t), 1))

    c, s = _transform()
    xyz = np.array([P[nodes[r]] for r in active], dtype=np.float64)
    if len(xyz):
        xyz = (xyz - c) * s

    neurons, dtypes = [], Counter()
    for k, r in enumerate(active):
        nid = nodes[r]
        ts = spk[k]
        if len(ts) > 120:
            ts = ts[::max(1, len(ts) // 120)]
        role = "input" if nid in sset else "downstream"
        lab = flysim.LABEL.get(nid, "?")
        if role == "downstream":
            dtypes[lab] += int(counts[r])
        neurons.append({"x": round(float(xyz[k, 0]), 2),
                        "y": round(float(xyz[k, 1]), 2),
                        "z": round(float(xyz[k, 2]), 2),
                        "role": role, "type": lab, "t": ts})

    edges = []
    for post, pre in np.argwhere(W != 0):
        a, b = row2new.get(int(pre)), row2new.get(int(post))
        if a is not None and b is not None:
            edges.append([a, b])
            if len(edges) >= 4000:
                break

    # hero neurons: real 3D arbors for the inputs + most-active downstream cells
    hero_data = []
    if heroes:
        inp_rows = [r for r in active if nodes[r] in sset][:max(1, heroes // 2)]
        down_rows = sorted((r for r in active if nodes[r] not in sset),
                           key=lambda r: -int(counts[r]))[:heroes - len(inp_rows)]
        for r in inp_rows + down_rows:
            pts = _hero_arbor(nodes[r])
            if pts:
                k = row2new[r]
                hero_data.append({"role": neurons[k]["role"], "type": neurons[k]["type"],
                                  "t": neurons[k]["t"], "pts": pts})

    ev = energy.synaptic_events(counts, W)
    dense = energy.dense_synapse_updates(W, dur_ms, counts)
    return {
        "title": f"Fly brain: '{query}' response",
        "query": query, "dur_ms": dur_ms,
        "n_input": sum(1 for n in neurons if n["role"] == "input"),
        "n_downstream": sum(1 for n in neurons if n["role"] == "downstream"),
        "top_downstream_types": [t for t, _ in dtypes.most_common(6)],
        "neurons": neurons, "edges": edges, "ghost": _ghost(), "heroes": hero_data,
        "energy": energy.summary(ev, dense),
    }


def _robustness_label(width):
    """Turn a gate's gain-plateau width (from logic.find_gate) into a one-word verdict.
    A wider band of gains that all compute the gate = a more robust, less knife-edge gate."""
    if width is None:
        return None
    if width >= 0.5:
        return "robust"
    if width >= 0.3:
        return "stable"
    return "fragile"


def build_gate_scene(gate, phase_ms=200):
    """3D scene for a logic gate: A/B/O as real arbors, cycling through the four input
    combinations on one looped timeline, plus the truth table and energy ledger.
    `gate` is a dict from logic.find_gate()."""
    A, B, O = gate["A"], gate["B"], gate["O"]
    nodes = [A, B, O]
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((3, 3), dtype=np.float32)
    W[idx[O], idx[A]] = gate["wA"] * gate["sA"]
    W[idx[O], idx[B]] = gate["wB"] * gate["sB"]

    P = _positions()
    c, s = _transform()
    phases = [[], [A], [B], [A, B]]            # 00, 10, 01, 11
    spk = {A: [], B: [], O: []}
    total = np.zeros(3)
    for p, drive in enumerate(phases):
        sp, st, si = flysim.run_lif(nodes, idx, W, drive, dur_ms=phase_ms,
                                    gain=gate["gain"], record=True)
        total += sp
        off = p * phase_ms
        for t, ni in zip(st.tolist(), si.tolist()):
            spk[nodes[int(ni)]].append(round(float(t) + off, 1))

    def _lab(n):
        v = flysim.LABEL.get(n, "")
        return v if v and v != "?" else ("cell " + str(n)[-5:])

    roles = {A: "input", B: "input", O: "output"}
    neurons, heroes = [], []
    for n in nodes:
        pts = _hero_arbor(n)                    # mesh exists token-free for any proofread cell
        if pts:
            x, y, z = np.asarray(pts, dtype=np.float64).reshape(-1, 3).mean(axis=0)
        elif n in P:
            x, y, z = (np.array(P[n]) - c) * s
        else:
            continue
        lab = _lab(n)
        neurons.append({"x": round(float(x), 2), "y": round(float(y), 2),
                        "z": round(float(z), 2), "role": roles[n], "type": lab, "t": spk[n]})
        if pts:
            heroes.append({"role": roles[n], "type": lab, "t": spk[n], "pts": pts})

    labels = {"A": _lab(A), "B": _lab(B), "O": _lab(O)}
    ev = energy.synaptic_events(total, W)
    dense = energy.dense_synapse_updates(W, phase_ms * 4, total)
    return {
        "title": "Fly-brain logic gate: %s" % gate["kind"],
        "query": gate["kind"], "dur_ms": phase_ms * 4,
        "n_input": 2, "n_downstream": 1, "top_downstream_types": [labels["O"]],
        "neurons": neurons, "edges": [], "ghost": _ghost(), "heroes": heroes,
        "energy": energy.summary(ev, dense),
        "gate": {"kind": gate["kind"], "labels": labels, "truth": gate["truth"],
                 "gain": gate.get("gain"), "plateau_width": gate.get("plateau_width"),
                 "robustness": _robustness_label(gate.get("plateau_width")),
                 "phase_ms": phase_ms, "rows": [[0, 0], [1, 0], [0, 1], [1, 1]]},
    }


def build_compass_scene(regime="raw"):
    """3D scene of the central-complex compass: the real EPG ring + PEN/PEG/Delta7 running
    a CUE -> HOLD -> TURN protocol. The EPG bump lights up, holds, and steers to track the
    turning cue. Returns neurons + spike timeline + a `compass` block (decoded heading over
    time, phase boundaries) + the energy ledger. `regime` is 'raw' or 'memory'."""
    import compass
    params = compass.REGIMES.get(regime, compass.REGIMES["raw"])
    res = compass.run_protocol(**params)
    nodes, counts = res["nodes"], res["counts"]
    is_ring, ang = res["is_ring"], res["ang"]

    # per-neuron spike times over the full looped timeline
    spk = {r: [] for r in range(len(nodes))}
    for t, r in zip(res["st"].tolist(), res["si"].tolist()):
        spk[int(r)].append(round(float(t), 1))

    P = _positions()
    c, s = _transform()
    active = [r for r in range(len(nodes)) if counts[r] > 0 and nodes[r] in P]
    neurons = []
    for r in active:
        x, y, z = (np.array(P[nodes[r]]) - c) * s
        ts = spk[r]
        if len(ts) > 140:
            ts = ts[::max(1, len(ts) // 140)]
        neurons.append({"x": round(float(x), 2), "y": round(float(y), 2),
                        "z": round(float(z), 2),
                        "role": "input" if is_ring[r] else "downstream",
                        "type": flysim.LABEL.get(nodes[r], "?"), "t": ts})

    # decoded heading trace (downsampled), degrees; null where the bump is silent
    t_arr, head, R = res["t"], res["head"], res["R"]
    step = max(1, len(t_arr) // 90)
    trace = []
    for i in range(0, len(t_arr), step):
        h = head[i]
        trace.append([round(float(t_arr[i]), 0),
                      None if np.isnan(h) else round(float(np.degrees(h)), 1),
                      round(float(R[i]), 3)])

    ph = res["phases"]
    ev = energy.synaptic_events(counts, res["W"])
    dense = energy.dense_synapse_updates(res["W"], res["dur_ms"], counts)
    return {
        "title": "Fly compass: heading memory (%s)" % regime,
        "query": "compass", "dur_ms": res["dur_ms"],
        "n_input": int(is_ring.sum()),
        "n_downstream": int(len(neurons) - sum(n["role"] == "input" for n in neurons)),
        "top_downstream_types": ["EPG ring"],
        "neurons": neurons, "edges": [], "ghost": _ghost(), "heroes": [],
        "energy": energy.summary(ev, dense),
        "compass": {
            "regime": regime, "dur_ms": res["dur_ms"],
            "phases": {k: [round(float(a), 0), round(float(b), 0)] for k, (a, b) in ph.items()},
            "theta0_deg": round(float(np.degrees(res["theta0"])), 0),
            "turn_to_deg": round(float(np.degrees(res["turn_to"])), 0),
            "n_ring": int(is_ring.sum()), "trace": trace,
        },
    }


def build_fly_scene(commands, seg_ms=160):
    """3D scene of a virtual fly driven by exciting real descending command neurons. Each
    behavior drives its DN on the connectome for one phase; the brain lights up the DNs +
    downstream, and the SAME firing moves a virtual body. Returns neurons + spike timeline,
    a `fly` block (trajectory + per-command spans), and the energy ledger."""
    import fly
    seq = [fly.resolve(c) for c in (commands or [])]
    seq = [c for c in seq if c in fly.BEHAVIORS] or ["forward"]

    # one combined subcircuit over all command DNs, so positions/indices are stable
    cmd_ids = [fly.dn_ids(fly.BEHAVIORS[c]["dn"], fly.BEHAVIORS[c]["side"]) for c in seq]
    all_drive = list(dict.fromkeys(i for ids in cmd_ids for i in ids))
    nodes, idx, W = flysim.build_subcircuit(all_drive, hops=2)

    # phase k drives command k's DNs; collect spikes per neuron with a phase time offset
    spk = {r: [] for r in range(len(nodes))}
    total = np.zeros(len(nodes))
    gains = []
    for k, ids in enumerate(cmd_ids):
        drive = [i for i in ids if i in idx]
        sp, st, si = flysim.run_lif(nodes, idx, W, drive, dur_ms=seg_ms, record=True)
        total += sp
        dn_spikes = sum(int(sp[idx[d]]) for d in drive)
        gains.append(min(1.0, dn_spikes / 8.0))
        off = k * seg_ms
        for t, ni in zip(st.tolist(), si.tolist()):
            spk[int(ni)].append(round(float(t) + off, 1))

    path, spans = fly.kinematics(seq, gains, dur_ms=seg_ms)
    drive_set = set(all_drive)

    P = _positions()
    c, s = _transform()
    active = [r for r in range(len(nodes)) if total[r] > 0 and nodes[r] in P]
    neurons = []
    for r in active:
        x, y, z = (np.array(P[nodes[r]]) - c) * s
        ts = spk[r]
        if len(ts) > 140:
            ts = ts[::max(1, len(ts) // 140)]
        neurons.append({"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 2),
                        "role": "input" if nodes[r] in drive_set else "downstream",
                        "type": flysim.LABEL.get(nodes[r], "?"), "t": ts})

    # downsample the path for the wire
    step = max(1, len(path) // 160)
    traj = [[round(t, 0), round(float(px), 4), round(float(py), 4),
             round(float(np.degrees(th)), 1)] for (t, px, py, th) in path[::step]]
    xs = [p[1] for p in traj]; ys = [p[2] for p in traj]
    cmds = [{"label": beh["label"], "dn": beh["dn"], "side": (beh["side"] or ""),
             "gain01": round(g, 2), "t0": round(t0, 0), "t1": round(t1, 0)}
            for (name, beh, g, t0, t1) in spans]

    ev = energy.synaptic_events(total, W)
    dense = energy.dense_synapse_updates(W, seg_ms * len(seq), total)
    return {
        "title": "Fly driven by descending command neurons",
        "query": "fly", "dur_ms": seg_ms * len(seq),
        "n_input": sum(1 for n in neurons if n["role"] == "input"),
        "n_downstream": sum(1 for n in neurons if n["role"] == "downstream"),
        "top_downstream_types": [c["dn"] for c in cmds][:6],
        "neurons": neurons, "edges": [], "ghost": _ghost(), "heroes": [],
        "energy": energy.summary(ev, dense),
        "fly": {
            "dur_ms": seg_ms * len(seq), "seg_ms": seg_ms, "commands": cmds, "traj": traj,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
        },
    }


def build_navigate_scene(start_heading_deg=120.0, steps=64, phases=10):
    """Closed-loop scene: a fly released at start_heading walks while the REAL
    EPG -> PFL3 -> DNa02 loop steers it onto the circuit's intrinsic heading. The brain
    lights up the steering circuit (heading bump + DNa02 L/R) phase-by-phase along the
    trajectory; the body follows the same steering signal. Returns neurons + spike timeline,
    a `fly` block (homing trajectory + goal heading), and the energy ledger."""
    import fly
    theta0 = np.radians(float(start_heading_deg))
    path, heads = fly.navigate(theta0, steps=steps)
    goal = fly.intrinsic_heading()

    sc = fly.steering_circuit()
    nodes, idx, W, cp = sc["nodes"], sc["idx"], sc["W"], sc["cp"]
    drive_rows = set(np.nonzero(sc["is_ring"])[0]) | set(sc["Lr"]) | set(sc["Rr"])

    # phase the brain sim along the trajectory: drive EPG at the heading at each waypoint
    seg_ms = max(1, int(path[-1][0] / phases))
    spk = {r: [] for r in range(len(nodes))}
    total = np.zeros(len(nodes))
    for p in range(phases):
        h = heads[min(len(heads) - 1, int((p + 0.5) / phases * len(heads)))]
        cue = [nodes[r] for r in cp._sector(sc["ang"], float(h), only=sc["is_ring"])]
        spcount, st, si = flysim.run_lif(nodes, idx, W, cue, dur_ms=seg_ms, gain=0.8, record=True)
        total += spcount
        off = p * seg_ms
        for t, ni in zip(st.tolist(), si.tolist()):
            spk[int(ni)].append(round(float(t) + off, 1))

    P = _positions()
    c, s = _transform()
    active = [r for r in range(len(nodes)) if total[r] > 0 and nodes[r] in P]
    neurons = []
    for r in active:
        x, y, z = (np.array(P[nodes[r]]) - c) * s
        ts = spk[r]
        if len(ts) > 140:
            ts = ts[::max(1, len(ts) // 140)]
        lab = flysim.LABEL.get(nodes[r], "?")
        neurons.append({"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 2),
                        "role": "input" if r in drive_rows else "downstream",
                        "type": lab, "t": ts})

    stepd = max(1, len(path) // 160)
    traj = [[round(t, 0), round(float(px), 4), round(float(py), 4),
             round(float(np.degrees(th)), 1)] for (t, px, py, th) in path[::stepd]]
    xs = [p[1] for p in traj]; ys = [p[2] for p in traj]
    final_err = abs(((heads[-1] - goal + np.pi) % (2 * np.pi)) - np.pi)
    dur = path[-1][0]
    cmds = [{"label": "homing → %d°" % round(np.degrees(goal)), "dn": "DNa02",
             "side": "L/R", "gain01": 1.0, "t0": 0.0, "t1": dur}]

    ev = energy.synaptic_events(total, W)
    dense = energy.dense_synapse_updates(W, dur, total)
    return {
        "title": "Fly homing via the real compass→DNa02 steering loop",
        "query": "navigate", "dur_ms": dur,
        "n_input": sum(1 for n in neurons if n["role"] == "input"),
        "n_downstream": sum(1 for n in neurons if n["role"] == "downstream"),
        "top_downstream_types": ["DNa02"],
        "neurons": neurons, "edges": [], "ghost": _ghost(), "heroes": [],
        "energy": energy.summary(ev, dense),
        "fly": {
            "dur_ms": dur, "seg_ms": seg_ms, "commands": cmds, "traj": traj,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "goal_deg": round(float(np.degrees(goal)), 0),
            "start_deg": round(float(start_heading_deg), 0),
            "final_err_deg": round(float(np.degrees(final_err)), 0), "closed_loop": True,
        },
    }


def build_math_scene(x, y, op="add", phase_ms=200):
    """3D scene of the fly-neuron 'calculator' computing x (+ or x) y: the gate motif
    neurons as arbors, plus the binary result and the energy the computation cost."""
    import flymath
    res = flymath.compute(int(x), int(y), op)
    gates = flymath._gates()
    c, s = _transform()
    P = _positions()
    fire = {"AND": (1, 1), "OR": (1, 0), "AND-NOT": (1, 0)}   # an input that lights each gate

    neurons, heroes, seen = [], [], set()
    for kind, g in gates.items():
        nodes = [g["A"], g["B"], g["O"]]
        idx = {n: i for i, n in enumerate(nodes)}
        W = np.zeros((3, 3), dtype=np.float32)
        W[idx[g["O"]], idx[g["A"]]] = g["wA"] * g["sA"]
        W[idx[g["O"]], idx[g["B"]]] = g["wB"] * g["sB"]
        drive = [n for n, on in zip((g["A"], g["B"]), fire[kind]) if on]
        sp, st, si = flysim.run_lif(nodes, idx, W, drive, dur_ms=phase_ms,
                                    gain=g["gain"], record=True)
        spk = {n: [] for n in nodes}
        for t, ni in zip(st.tolist(), si.tolist()):
            spk[nodes[int(ni)]].append(round(float(t), 1))
        roles = {g["A"]: "input", g["B"]: "input", g["O"]: "output"}
        for n in nodes:
            if n in seen:
                continue
            seen.add(n)
            pts = _hero_arbor(n)
            if pts:
                xx, yy, zz = np.asarray(pts, dtype=np.float64).reshape(-1, 3).mean(axis=0)
            elif n in P:
                xx, yy, zz = (np.array(P[n]) - c) * s
            else:
                continue
            lab = flysim.LABEL.get(n, "")
            lab = lab if lab and lab != "?" else ("cell " + str(n)[-5:])
            neurons.append({"x": round(float(xx), 2), "y": round(float(yy), 2),
                            "z": round(float(zz), 2), "role": roles[n], "type": lab, "t": spk[n]})
            if pts:
                heroes.append({"role": roles[n], "type": lab, "t": spk[n], "pts": pts})

    return {
        "title": "Fly brain computing: %d %s %d = %d" % (x, res["sym"], y, res["result"]),
        "query": "math", "dur_ms": phase_ms,
        "n_input": 0, "n_downstream": 0, "top_downstream_types": [],
        "neurons": neurons, "edges": [], "ghost": _ghost(), "heroes": heroes,
        # chip equivalent: each real gate op is its (wA+wB)-synapse circuit, clock-evaluated
        # for phase_ms; average the gate synapse counts and scale by the number of ops.
        "energy": energy.summary(
            res["events"],
            res["gate_ops"] * energy.dense_synapse_updates(
                np.full((1, 1), np.mean([g["wA"] + g["wB"] for g in gates.values()])), phase_ms)),
        "math": {"x": x, "y": y, "result": res["result"], "sym": res["sym"],
                 "x_bin": res["x_bin"], "y_bin": res["y_bin"],
                 "result_bin": res["result_bin"], "gate_ops": res["gate_ops"]},
    }


def export3d(query="olfactory", limit=40, dur_ms=200, out="fly3d_data.js",
             open_browser=True):
    data = build_data(query, limit, dur_ms)
    with open(out, "w") as f:
        f.write("window.FLY_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";")
    nspk = sum(len(n["t"]) for n in data["neurons"])
    print(f"wrote {out}: {len(data['neurons'])} neurons, {len(data['edges'])} edges, "
          f"{nspk} spikes")
    html = os.path.abspath("fly3d.html")
    print(f"open in a browser: file://{html}")
    if open_browser and os.path.exists(html):
        webbrowser.open("file://" + html)
    return data


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "olfactory"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 200
    try:
        export3d(q, lim, dur)
    except ValueError as e:
        sys.exit(str(e))
