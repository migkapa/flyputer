"""
looming_surface.py — LINK 4: a PRECOMPUTED real-time response surface for see->avoid.

The browser game loop cannot run a fresh LIF every frame. So we run the REAL
vision -> looming-detector -> Giant-Fiber pipeline ONCE over a grid of
(obstacle azimuth, obstacle angular size) and bake whether/when DNp01 fires into a
table. The live loop is then a fast lookup (like swatter.game_curve / drone's cached
escape_threshold) — O(1) per frame, no LIF.

REAL pipeline per grid cell (no geometry shortcut):
  1. Render an obstacle disk on the REAL L1 lamina sheet (789 columns) at the given
     azimuth, with radius set by the angular size.
  2. Relay it through the REAL L1->Mi1 matrix W (~66k synapses)  -> medulla activity.
  3. Spatially GATE the REAL LPLC2/LC4 detectors: drive each detector by how much the
     illuminated visual field overlaps its retinotopic location (flattened soma u).
     A bigger / closer obstacle illuminates more columns and drives more detectors.
  4. Run the REAL swatter LIF subcircuit and record if/when the REAL DNp01 Giant Fiber
     spikes.

HONESTY: toy LIF, ~13% NT error, absolute rates not meaningful. The detector receptive
fields here are a soma-position retinotopic PROXY (LPLC2/LC4 are genuinely wide-field;
we use flattened soma u as an azimuth proxy to make azimuth gating meaningful), and the
medulla->detector step is a spatial-overlap drive, not the toy's weak 4-hop propagation
(Mi1->Tm5e->LPLC2 barely carries signal — see fallback note). Vision (L1->Mi1), the
LPLC2/LC4 detectors, and DNp01 are all the REAL connectome.
"""
from __future__ import annotations

import time
import numpy as np

import swatter
import optic
import flysim
import export3d


_DET = None  # cached detector retinotopic layout


def detector_layout():
    """Flatten the REAL LPLC2/LC4 soma positions onto a per-eye 2D retinotopic sheet so
    each detector gets a (u=azimuth proxy, v) coordinate. Cached."""
    global _DET
    if _DET is not None:
        return _DET
    C = swatter.circuit()
    loom = C["loom_ids"]
    idx = C["idx"]
    flysim._ensure_ann()
    side = flysim.ANN.set_index("root_id")["side"]
    P = export3d._positions()
    layout = {}
    for eye in ("right", "left"):
        ids = [i for i in loom if str(side.get(i)) == eye and i in P and i in idx]
        X = np.array([P[i] for i in ids], dtype=float)
        c = X.mean(0)
        _, _, vt = np.linalg.svd(X - c, full_matrices=False)
        g = (X - c) @ vt[:2].T
        g -= g.min(0)
        g /= (g.max(0) + 1e-9)
        rows = np.array([idx[i] for i in ids])
        layout[eye] = {"ids": ids, "rows": rows, "u": g[:, 0], "v": g[:, 1]}
    _DET = layout
    return _DET


def _render_and_relay(azimuth, ang_size, O, L1u, L1v):
    """Paint an obstacle disk centered at (azimuth, 0.5) with radius ~ang_size on the REAL
    L1 columns, relay through the REAL L1->Mi1 matrix, return (l1_drive, mi_act)."""
    r = max(0.02, 0.5 * ang_size)              # disk radius in [0,1] sheet units
    d = np.hypot(L1u - azimuth, L1v - 0.5)
    drive = (d < r).astype(np.float32)         # obstacle silhouette on the lamina
    mi = O["W"] @ drive                        # REAL retinotopic relay L1->Mi1
    return drive, mi


def _detector_drive(azimuth, ang_size, mi_act, O, det, peak=34.0):
    """Map the illuminated medulla field onto the REAL detectors. Each detector at
    retinotopic u is driven in proportion to how much of the obstacle's azimuth window it
    overlaps, scaled by the medulla activity there. Returns a drive value per detector row."""
    half = max(0.03, 0.5 * ang_size)
    du = det["u"]
    # fraction of this detector's RF (a tophat of width ~2*half around azimuth) that is lit
    overlap = np.clip(1.0 - np.abs(du - azimuth) / (half + 0.12), 0.0, 1.0)
    # global medulla brightness in the lit window (the actual relayed vision signal)
    mi_u = O["mig"][:, 0] if O["mig"].ndim == 2 else O["mig"]
    lit = np.isfinite(mi_u) & (np.abs(mi_u - azimuth) < (half + 0.12))
    bright = float(mi_act[lit].sum()) / max(1.0, float(lit.sum()))
    return peak * overlap * bright, bright, int(lit.sum())


def run_vision_loom(azimuth, ang_size, approach_ms=180.0, gain=0.6, dt=0.1,
                    eye="right", peak=34.0, v_th=15.0, tau_m=10.0, tau_s=5.0, t_ref=2.0):
    """The REAL see->detect->GF pipeline for ONE obstacle (azimuth, ang_size).
    Returns dict with gf_spike_t (ms or None), detectors driven, medulla brightness."""
    C = swatter.circuit()
    nodes, idx, W = C["nodes"], C["idx"], C["W"]
    gf_rows = C["gf_rows"]
    O = optic.optic()
    det = detector_layout()[eye]
    L1u, L1v = O["l1g"][:, 0], O["l1g"][:, 1]

    _, mi_act = _render_and_relay(azimuth, ang_size, O, L1u, L1v)
    det_drive_vec, bright, n_lit = _detector_drive(azimuth, ang_size, mi_act, O, det, peak=peak)
    n_driven = int((det_drive_vec > 0.5).sum())

    # looming RAMP: the obstacle's angular size grows over the approach. Scale the
    # spatially-gated detector drive by the same looming profile swatter uses.
    N = len(nodes)
    steps = max(1, int(approach_ms / dt))
    s = np.linspace(0.0, 1.0, steps)
    near = 0.08
    dist = 1.0 - (1.0 - near) * s
    ang = 1.0 / dist
    ramp = (ang - 1.0) / (1.0 / near - 1.0)            # 0 -> 1 contact (same as swatter)

    V = np.zeros(N, dtype=np.float32)
    Isyn = np.zeros(N, dtype=np.float32)
    Wsc = W * gain
    cool = np.zeros(N, dtype=np.int32)
    ref_steps = max(1, int(t_ref / dt))
    rows = det["rows"]
    base = det_drive_vec.astype(np.float32)
    gf_spike_t = None
    gf_arr = np.array(gf_rows)

    for step in range(steps):
        Iext = np.zeros(N, dtype=np.float32)
        Iext[rows] = base * ramp[step]
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
            if gf_spike_t is None and fired[gf_arr].any():
                gf_spike_t = step * dt
                break
        np.subtract(cool, 1, out=cool, where=cool > 0)

    return {"azimuth": azimuth, "ang_size": ang_size, "gf_spike_t": gf_spike_t,
            "n_driven": n_driven, "bright": bright, "n_lit": n_lit,
            "approach_ms": approach_ms}


def precompute_surface(n_az=12, n_sz=10, approach_ms=180.0, eye="right", verbose=False):
    """Run the REAL pipeline over a grid of (azimuth, angular size) and bake a lookup table.
    Returns the grid axes + a fires[n_az,n_sz] bool table + gf_t[n_az,n_sz] spike-time table."""
    azs = np.linspace(0.05, 0.95, n_az)
    szs = np.linspace(0.10, 0.95, n_sz)
    fires = np.zeros((n_az, n_sz), dtype=bool)
    gf_t = np.full((n_az, n_sz), np.nan, dtype=np.float32)
    n_driven = np.zeros((n_az, n_sz), dtype=np.int32)
    t0 = time.time()
    for i, a in enumerate(azs):
        for j, sz in enumerate(szs):
            r = run_vision_loom(a, sz, approach_ms=approach_ms, eye=eye)
            fires[i, j] = r["gf_spike_t"] is not None
            if r["gf_spike_t"] is not None:
                gf_t[i, j] = r["gf_spike_t"]
            n_driven[i, j] = r["n_driven"]
            if verbose:
                print("az=%.2f sz=%.2f driven=%3d fire=%s t=%s"
                      % (a, sz, r["n_driven"], fires[i, j], r["gf_spike_t"]))
    elapsed = time.time() - t0
    return {"azs": azs, "szs": szs, "fires": fires, "gf_t": gf_t,
            "n_driven": n_driven, "elapsed": elapsed, "n_cells": n_az * n_sz,
            "approach_ms": approach_ms, "eye": eye}


def save_surface(surf, path="looming_surface.npz"):
    """Bake the response surface to disk — the ONLY data the live game loop needs."""
    np.savez(path, azs=surf["azs"], szs=surf["szs"], fires=surf["fires"],
             gf_t=surf["gf_t"], n_driven=surf["n_driven"],
             approach_ms=surf["approach_ms"])
    return path


def make_lookup(surf):
    """Return an O(1) per-frame closure (az,size)->(fires, gf_spike_ms). No LIF at runtime."""
    azs, szs = surf["azs"], surf["szs"]
    fires, gf_t = surf["fires"], surf["gf_t"]

    def lookup(az, size):
        i = int(np.clip(np.searchsorted(azs, az), 0, len(azs) - 1))
        j = int(np.clip(np.searchsorted(szs, size), 0, len(szs) - 1))
        return bool(fires[i, j]), float(gf_t[i, j])

    return lookup


if __name__ == "__main__":
    print("LINK 4 — precomputed real vision->detector->GF response surface\n")
    t0 = time.time()
    surf = precompute_surface(n_az=12, n_sz=10, verbose=True)
    print("\ngrid %dx%d = %d cells, precompute %.2fs (%.1f ms/cell)"
          % (len(surf["azs"]), len(surf["szs"]), surf["n_cells"],
             surf["elapsed"], 1000 * surf["elapsed"] / surf["n_cells"]))
    print("cells that fire DNp01: %d / %d" % (surf["fires"].sum(), surf["n_cells"]))
