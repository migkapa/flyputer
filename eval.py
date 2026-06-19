"""
eval.py — does flyputer excite the *right* neurons, and do the gates really compute?

Three independent layers, each printed as a small report with a soft verdict:

  1. find_neurons precision  — are the returned neurons actually the cell type asked
                               for? (ground-truthed against the FlyWire annotations)
  2. stimulation controls    — positive controls (a known pathway should light up the
                               expected downstream types) and negative controls (recurrent
                               circuits should barely propagate from a feedforward poke).
  3. gate robustness         — do the AND/OR/AND-NOT motifs classify correctly across a
                               BAND of gains (not one cherry-picked value), and do they
                               survive weight jitter + the known ~13% neurotransmitter
                               labelling error? Plus an end-to-end arithmetic check.

Nothing here validates absolute firing rates or biophysics — the sim is a toy by design
(see README "Honest caveats"). These checks validate the qualitative claims only.

    .venv/bin/python eval.py            # run everything
    .venv/bin/python eval.py neurons    # just layer 1
    .venv/bin/python eval.py controls   # just layer 2
    .venv/bin/python eval.py gates      # just layer 3
"""
import sys

import numpy as np

import flysim
import logic


# --------------------------------------------------------------------------- #
# small reporting helpers
# --------------------------------------------------------------------------- #
def _hdr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _verdict(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL' if ok is False else '….'}] {msg}")


def _neuron_text(root_id):
    """All searchable annotation text for one neuron, lowercased — the ground truth
    we judge find_neurons against (richer than the 3 fields it echoes back)."""
    flysim._ensure_ann()
    row = flysim.ANN[flysim.ANN.root_id == int(root_id)]
    if row.empty:
        return ""
    parts = [str(row.iloc[0].get(c, "")) for c in flysim._TEXT_COLS]
    return " ".join(parts).lower()


# --------------------------------------------------------------------------- #
# Layer 1 — find_neurons precision
# --------------------------------------------------------------------------- #
# query -> substrings that MUST appear in a correct hit's annotation text.
_PRECISION_CASES = {
    "gustatory":       ["gustatory"],
    "olfactory":       ["olfactory"],
    "Kenyon":          ["kenyon"],
    "mushroom body":   ["kenyon", "mbon", "mbin"],   # alias expands to these
    "motor":           ["motor"],
    "descending":      ["descending"],
}


def eval_find_neurons(limit=15):
    _hdr("LAYER 1 — find_neurons precision (vs. FlyWire annotations)")
    print("  precision = fraction of returned neurons whose real annotation text")
    print("  actually contains the expected cell-type term.\n")
    scores = []
    for query, expect in _PRECISION_CASES.items():
        res = flysim.find_neurons(query, limit=limit)
        hits = res["neurons"]
        if not hits:
            _verdict(False, f"{query!r:18} → 0 matches (search returned nothing)")
            scores.append(0.0)
            continue
        good = 0
        for n in hits:
            text = _neuron_text(n["root_id"])
            if any(term in text for term in expect):
                good += 1
        prec = good / len(hits)
        scores.append(prec)
        _verdict(prec >= 0.8,
                 f"{query!r:18} → {prec:5.0%} precision "
                 f"({good}/{len(hits)} of {res['n_matches']} total matches)")
    mean = float(np.mean(scores)) if scores else 0.0
    print(f"\n  mean precision across {len(scores)} queries: {mean:.0%}")
    return mean


# --------------------------------------------------------------------------- #
# Layer 2 — stimulation controls
# --------------------------------------------------------------------------- #
# Positive: stimulate a seed population, expect these substrings among top responders.
_POSITIVE = {
    "olfactory":  ["kenyon", "mbon", "projection", "pn", "mushroom"],
    "gustatory":  ["descending", "motor", "feeding", "dn", "an"],
}
# Negative: recurrent / inhibitory circuits should NOT propagate much from a single poke.
_NEGATIVE = ["central complex", "clock"]

_QUIET_RATIO = 0.15   # fired/subcircuit below this = "quiet", as expected for recurrent nets


def _seed_ids(query, n=8):
    res = flysim.find_neurons(query, limit=n)
    return [x["root_id"] for x in res["neurons"]]


def eval_controls():
    _hdr("LAYER 2 — stimulation controls (positive + negative)")
    ok_all = True

    print("\n  POSITIVE controls — a known pathway should light up expected downstream types:")
    for query, expect in _POSITIVE.items():
        seeds = _seed_ids(query)
        if not seeds:
            _verdict(False, f"{query!r}: no seeds found")
            ok_all = False
            continue
        res = flysim.stimulate(seeds, dur_ms=200)
        responders = " ".join(res["top_responding_cell_types"].keys()).lower()
        matched = [t for t in expect if t in responders]
        ok = bool(matched)
        ok_all &= ok
        _verdict(ok, f"{query!r:12} → fired {res['neurons_that_fired']:>4}/"
                     f"{res['subcircuit_size']:<4} neurons; "
                     f"expected-type match: {matched or 'none'}")
        print(f"          top responders: {list(res['top_responding_cell_types'])[:6]}")

    print("\n  NEGATIVE controls — recurrent circuits should barely propagate from a poke:")
    for query in _NEGATIVE:
        seeds = _seed_ids(query)
        if not seeds:
            _verdict(None, f"{query!r}: no seeds found (skipped)")
            continue
        res = flysim.stimulate(seeds, dur_ms=200)
        size = max(1, res["subcircuit_size"])
        ratio = res["neurons_that_fired"] / size
        ok = ratio < _QUIET_RATIO
        ok_all &= ok
        _verdict(ok, f"{query!r:16} → {ratio:5.1%} of subcircuit fired "
                     f"({res['neurons_that_fired']}/{size})  "
                     f"{'quiet, as expected' if ok else 'unexpectedly active'}")
    return ok_all


# --------------------------------------------------------------------------- #
# Layer 3 — gate robustness
# --------------------------------------------------------------------------- #
def _plateau(gains, passes, center):
    """Width of the contiguous band of passing gains that contains `center`."""
    i = int(np.argmin(np.abs(gains - center)))
    if not passes[i]:
        return 0.0, (center, center)
    lo = hi = i
    while lo > 0 and passes[lo - 1]:
        lo -= 1
    while hi < len(gains) - 1 and passes[hi + 1]:
        hi += 1
    return float(gains[hi] - gains[lo]), (float(gains[lo]), float(gains[hi]))


def _gate_robustness(g, n_trials=200, jitter=0.15, nt_err=0.13, seed=0):
    """Return dict of robustness metrics for one gate motif."""
    rng = np.random.default_rng(seed)
    kind = g["kind"]
    A, B, O, wA, wB, sA, sB, gain = (g["A"], g["B"], g["O"],
                                     g["wA"], g["wB"], g["sA"], g["sB"], g["gain"])

    # (a) gain sweep — does it hold across a band, or only at one magic value?
    gains = np.linspace(0.2, 1.6, 29)
    passes = [logic.classify(logic.truth_table(A, B, O, wA, wB, float(G), sA, sB)) == kind
              for G in gains]
    width, (glo, ghi) = _plateau(gains, np.array(passes), gain)
    frac_gains = float(np.mean(passes))

    # (b) perturbation survival — isolate each failure mode, then combine.
    def survival(do_gain, do_w, do_nt):
        ok = 0
        for _ in range(n_trials):
            jg = gain * (1 + rng.uniform(-jitter, jitter)) if do_gain else gain
            jA = wA * (1 + rng.uniform(-jitter, jitter)) if do_w else wA
            jB = wB * (1 + rng.uniform(-jitter, jitter)) if do_w else wB
            fA = -sA if (do_nt and rng.random() < nt_err) else sA
            fB = -sB if (do_nt and rng.random() < nt_err) else sB
            ok += logic.classify(logic.truth_table(A, B, O, jA, jB, jg, fA, fB)) == kind
        return ok / n_trials

    return {
        "kind": kind,
        "gain": gain,
        "plateau_width": width,
        "plateau_band": (glo, ghi),
        "frac_gains_pass": frac_gains,
        "surv_gain": survival(True, False, False),
        "surv_weight": survival(False, True, False),
        "surv_nt": survival(False, False, True),
        "surv_all": survival(True, True, True),
    }


def eval_gates(n_trials=200):
    _hdr("LAYER 3 — gate robustness (band of gains + jitter + 13% NT error)")
    print("  A real gate should classify correctly across a BAND of gains and survive")
    print("  weight jitter and the known ~13% neurotransmitter-labelling error — not just")
    print("  one cherry-picked setting.\n")
    all_ok = True
    for kind in ("AND", "OR", "AND-NOT"):
        g = logic.find_gate(kind)
        if g is None:
            _verdict(False, f"{kind}: no motif found in connectome")
            all_ok = False
            continue
        m = _gate_robustness(g, n_trials=n_trials)
        # "robust enough": a non-trivial gain plateau and decent combined survival.
        ok = m["plateau_width"] >= 0.2 and m["surv_all"] >= 0.5
        all_ok &= ok
        lab = g["labels"]
        _verdict(ok, f"{kind:8} {lab['A']}+{lab['B']}→{lab['O']}  (chosen gain {m['gain']:.2f})")
        print(f"          gain plateau:  width {m['plateau_width']:.2f} "
              f"over [{m['plateau_band'][0]:.2f}, {m['plateau_band'][1]:.2f}]  "
              f"({m['frac_gains_pass']:.0%} of swept gains pass)")
        print(f"          survival   :  gain-jitter {m['surv_gain']:.0%} | "
              f"weight-jitter {m['surv_weight']:.0%} | "
              f"NT-flip {m['surv_nt']:.0%} | combined {m['surv_all']:.0%}")
    return all_ok


def eval_arithmetic():
    """End-to-end: real composed gates must produce correct binary sums/products."""
    _hdr("LAYER 3b — arithmetic via composed real gates (end-to-end)")
    try:
        import flymath
    except Exception as e:                       # noqa: BLE001
        _verdict(None, f"flymath unavailable ({e}) — skipped")
        return None
    cases = [(2, 3, "add"), (1, 1, "add"), (5, 6, "add"), (6, 7, "mul"), (3, 4, "mul")]
    ok_all = True
    for x, y, op in cases:
        r = flymath.compute(x, y, op=op)
        got = r["result"] if isinstance(r, dict) and "result" in r else r
        want = x + y if op == "add" else x * y
        ok = int(got) == want
        ok_all &= ok
        sym = "+" if op == "add" else "×"
        _verdict(ok, f"{x} {sym} {y} = {got}  (expected {want})")
    return ok_all


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if which in ("all", "neurons"):
        results["find_neurons precision"] = eval_find_neurons()
    if which in ("all", "controls"):
        results["stimulation controls"] = eval_controls()
    if which in ("all", "gates"):
        results["gate robustness"] = eval_gates()
        results["arithmetic"] = eval_arithmetic()

    _hdr("SUMMARY")
    for name, val in results.items():
        if val is None:
            line = "skipped"
        elif isinstance(val, float):
            line = f"{val:.0%}"
        else:
            line = "PASS" if val else "FAIL"
        print(f"  {name:28} {line}")
    print()


if __name__ == "__main__":
    main()
