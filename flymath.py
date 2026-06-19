"""
flymath.py — do binary arithmetic with real fly-neuron logic gates.

Each gate (AND, OR, AND-NOT) is a real motif from the connectome (logic.find_gate),
evaluated on the live LIF sim. We compose them into a half/full adder and ripple across
bits to add numbers. Every gate evaluation is counted for the energy ledger, so we can say
exactly how much energy the fly brain spent to do the sum.

(Named flymath, not math, so it doesn't shadow Python's stdlib math module.)
"""
import numpy as np

import flysim
import logic
import energy

_GATES = None


def _gates():
    global _GATES
    if _GATES is None:
        g = {k: logic.find_gate(k) for k in ("AND", "OR", "AND-NOT")}
        if any(v is None for v in g.values()):
            raise RuntimeError("could not find all gate motifs in the connectome")
        _GATES = g
    return _GATES


def _eval(g, a, b):
    """Run one real gate motif. Returns (output_bit, synaptic_events)."""
    nodes = [g["A"], g["B"], g["O"]]
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((3, 3), dtype=np.float32)
    W[idx[g["O"]], idx[g["A"]]] = g["wA"] * g["sA"]
    W[idx[g["O"]], idx[g["B"]]] = g["wB"] * g["sB"]
    drive = [n for n, on in ((g["A"], a), (g["B"], b)) if on]
    sp = flysim.run_lif(nodes, idx, W, drive, dur_ms=200, gain=g["gain"])
    return (1 if sp[idx[g["O"]]] > 0 else 0), energy.synaptic_events(sp, W)


class _Counter:
    """Runs gates on real neurons and tallies how many operations + synaptic events."""
    def __init__(self):
        self.events = 0.0
        self.ops = 0

    def gate(self, kind, a, b):
        bit, ev = _eval(_gates()[kind], a, b)
        self.events += ev
        self.ops += 1
        return bit

    def and_(self, a, b):
        return self.gate("AND", a, b)

    def or_(self, a, b):
        return self.gate("OR", a, b)

    def xor(self, a, b):                                   # (a or b) and not (a and b)
        return self.gate("AND-NOT", self.or_(a, b), self.and_(a, b))


def _full_adder(c, a, b, cin):
    axb = c.xor(a, b)
    s = c.xor(axb, cin)
    cout = c.or_(c.and_(a, b), c.and_(cin, axb))
    return s, cout


def _ripple_add(c, x, y):
    """Ripple-carry add using the gate counter `c`; returns the sum."""
    bits = max(1, max(x, y).bit_length()) + 1
    cin, out = 0, 0
    for i in range(bits):
        s, cin = _full_adder(c, (x >> i) & 1, (y >> i) & 1, cin)
        out |= (s << i)
    return out


def _result(x, y, out, op, sym, c):
    w = max(out, x, y, 1).bit_length()
    return {
        "x": x, "y": y, "result": out, "op": op, "sym": sym,
        "x_bin": format(x, "0{}b".format(w)),
        "y_bin": format(y, "0{}b".format(w)),
        "result_bin": format(out, "0{}b".format(w)),
        "gate_ops": c.ops, "events": c.events,
        "correct": out == (x + y if op == "add" else x * y),
    }


def add(x, y):
    """Add two non-negative integers with real fly-neuron gates (ripple-carry)."""
    c = _Counter()
    return _result(x, y, _ripple_add(c, x, y), "add", "+", c)


def multiply(x, y):
    """Multiply via shift-and-add: each partial sum runs through the fly-neuron adder."""
    c = _Counter()
    acc = 0
    for i in range(max(1, y.bit_length())):
        if (y >> i) & 1:
            acc = _ripple_add(c, acc, x << i)
    return _result(x, y, acc, "mul", "×", c)


def compute(x, y, op="add"):
    return multiply(x, y) if op in ("mul", "*", "x", "X", "times", "multiply") else add(x, y)


if __name__ == "__main__":
    for op, pairs in [("add", [(0, 0), (2, 3), (7, 1)]),
                      ("mul", [(2, 3), (3, 3), (6, 7)])]:
        for x, y in pairs:
            r = compute(x, y, op)
            print("%d %s %d = %-3d (%s %s %s = %s) | %2d gate ops, %s  %s" % (
                x, r["sym"], y, r["result"], r["x_bin"], r["sym"], r["y_bin"],
                r["result_bin"], r["gate_ops"],
                energy.humanize(r["events"] * energy.E_SYN_BIO),
                "OK" if r["correct"] else "WRONG"))
