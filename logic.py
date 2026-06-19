"""
logic.py — find real fly-neuron logic gates and read their truth tables.

A spiking neuron is a threshold gate. Drive combinations of input neurons, measure an
output neuron, and the boolean function falls out. Reuses flysim (connectome + LIF sim).
The same convergent motif can act as AND or OR depending on excitability (gain vs. threshold).
"""
import numpy as np

import flysim

GATES = {
    (0, 0, 0, 1): "AND",
    (0, 1, 1, 1): "OR",
    (0, 1, 1, 0): "XOR",
    (1, 1, 1, 0): "NAND",
    (1, 0, 0, 0): "NOR",
    (0, 1, 0, 0): "AND-NOT",        # A and not B
    (1, 1, 0, 0): "NOT-B",
}


def classify(out4):
    """out4 = output spike counts for inputs [(0,0),(1,0),(0,1),(1,1)]."""
    key = tuple(1 if v > 0 else 0 for v in out4)
    return GATES.get(key, "custom")


def truth_table(A, B, O, wA, wB, gain, sA=1.0, sB=1.0, dur_ms=200):
    """Run the 3-neuron circuit A,B -> O for all four input combinations.
    sA/sB are the signs of A/B onto O (+1 excitatory, -1 inhibitory)."""
    nodes = [A, B, O]
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((3, 3), dtype=np.float32)
    W[idx[O], idx[A]] = wA * sA
    W[idx[O], idx[B]] = wB * sB
    out = []
    for drive in ([], [A], [B], [A, B]):
        sp = flysim.run_lif(nodes, idx, W, drive, dur_ms=dur_ms, gain=gain)
        out.append(int(sp[idx[O]]))
    return out   # [v00, v10, v01, v11]


def _candidates(kind, min_w=25):
    """Yield (O, A, B, wA, wB, sA, sB) motifs from the real connectome."""
    flysim._ensure_conn()
    C = flysim.CONN
    SIGN = flysim.SIGN
    strong = C[C.w >= min_w].copy()
    strong["s"] = strong["pre"].map(lambda r: SIGN.get(int(r), 0.0))
    exc = strong[strong["s"] > 0]

    if kind in ("AND", "OR"):
        top2 = exc.sort_values("w", ascending=False).groupby("post").head(2)
        for O, grp in top2.groupby("post"):
            if len(grp) < 2:
                continue
            g = grp.sort_values("w", ascending=False)
            A = int(g.iloc[0].pre); wA = float(g.iloc[0].w)
            B = int(g.iloc[1].pre); wB = float(g.iloc[1].w)
            if wB >= 0.6 * wA:                  # comparable inputs
                yield (int(O), A, B, wA, wB, 1.0, 1.0)
    else:   # AND-NOT: strongest excitatory A + strongest inhibitory B onto the same O
        inh = strong[strong["s"] < 0]
        exc_best = exc.loc[exc.groupby("post")["w"].idxmax()]
        inh_best = inh.loc[inh.groupby("post")["w"].idxmax()]
        m = exc_best.merge(inh_best, on="post", suffixes=("_e", "_i"))
        for r in m.itertuples():
            yield (int(r.post), int(r.pre_e), int(r.pre_i),
                   float(r.w_e), float(r.w_i), 1.0, -1.0)


_GAIN_GRID = np.linspace(0.2, 1.6, 29)   # gain sweep used to measure a motif's robustness


def _gain_plateau(A, B, O, wA, wB, sA, sB, kind):
    """Sweep gain and find the WIDEST contiguous band where the motif classifies as
    `kind`. Return (center_gain, width). A wide band = a robust gate that doesn't
    depend on a knife-edge gain; we pick the band's CENTER, not the first value that
    happens to work, so jitter in either direction stays inside the band."""
    passes = [classify(truth_table(A, B, O, wA, wB, float(G), sA, sB)) == kind
              for G in _GAIN_GRID]
    best_lo = best_hi = -1
    best_w = -1.0
    i, n = 0, len(_GAIN_GRID)
    while i < n:
        if passes[i]:
            j = i
            while j + 1 < n and passes[j + 1]:
                j += 1
            w = float(_GAIN_GRID[j] - _GAIN_GRID[i])
            if w > best_w:
                best_w, best_lo, best_hi = w, i, j
            i = j + 1
        else:
            i += 1
    if best_lo < 0:
        return None, 0.0
    return float((_GAIN_GRID[best_lo] + _GAIN_GRID[best_hi]) / 2), best_w


def find_gate(kind="AND", max_try=400, good_plateau=0.4):
    """Find a real fly-neuron motif that computes `kind`, preferring ROBUST ones.

    For each candidate motif we sweep gain and measure how wide a band classifies as
    `kind`. We accept the first motif with a band at least `good_plateau` wide (using
    the band's center gain); if none reaches that, we return the widest-band motif seen.
    This replaces "first gain that works" — which could land on a fragile knife-edge —
    with "the most gain-robust motif, run at the center of its stable band."
    """
    tried = 0
    best = None
    for O, A, B, wA, wB, sA, sB in _candidates(kind):
        tried += 1
        if tried > max_try:
            break
        center, width = _gain_plateau(A, B, O, wA, wB, sA, sB, kind)
        if center is None:
            continue
        cand = {
            "kind": kind, "A": A, "B": B, "O": O,
            "wA": wA, "wB": wB, "sA": sA, "sB": sB, "gain": center,
            "truth": truth_table(A, B, O, wA, wB, center, sA, sB),
            "plateau_width": width,
            "labels": {"A": flysim.LABEL.get(A, "?"),
                       "B": flysim.LABEL.get(B, "?"),
                       "O": flysim.LABEL.get(O, "?")},
        }
        if width >= good_plateau:
            return cand                      # robust enough — take it now
        if best is None or width > best["plateau_width"]:
            best = cand                      # otherwise remember the widest band so far
    return best


if __name__ == "__main__":
    for kind in ("AND", "OR", "AND-NOT"):
        g = find_gate(kind)
        if g:
            t = g["truth"]
            print("%-8s  %s + %s -> %s   truth[00,10,01,11]=%s  gain=%.2f" %
                  (kind, g["labels"]["A"], g["labels"]["B"], g["labels"]["O"], t, g["gain"]))
        else:
            print("%-8s  (no clean motif found)" % kind)
