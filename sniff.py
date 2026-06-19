"""
sniff.py — the fly learns smell B without forgetting smell A.

Catastrophic forgetting is modern AI's most stubborn embarrassment: train a network on task
B and it clobbers task A. The fruit fly solved it ~hundreds of millions of years ago, with
ARCHITECTURE, not optimization. In the mushroom body, ~685 antennal-lobe projection neurons
(ALPN) fan out onto ~5177 Kenyon cells (KC) so sparsely that each odor lights up a tiny,
near-disjoint ~3-5% of them. Learning an odor depresses just *that* odor's KC→MBON synapses
(gated by a dopaminergic "this was bad" signal), so a new memory barely touches an old one.

Here we run that real wiring: take two odors, build their real Kenyon-cell codes from the
FlyWire ALPN→KC connectivity, and show each odor occupies its own near-disjoint sliver of the
mushroom body — so teaching many odors barely makes an untaught one look "already learned".

HONESTY: this demonstrates the architectural MECHANISM (sparse, near-disjoint codes), it is
NOT a rigged race — a dense net on random high-dimensional odors can separate them too; the
point is HOW the fly does it. The KC code is a thresholded coincidence code tuned to be sparse
(threshold 3 ≈ a Kenyon cell needing several coincident inputs), not a calibrated firing rate,
and the disjointness is a MEAN property (rare odor pairs collide more).

CLI:
    .venv/bin/python sniff.py
"""
from __future__ import annotations

import numpy as np

import flysim

PN_CLASS = "ALPN"          # antennal-lobe projection neurons (odor input)
KC_CLASS = "Kenyon_Cell"   # mushroom-body Kenyon cells (sparse expansion layer)
MBON_CLASS = "MBON"        # mushroom-body output neurons (valence readout)

_CIRC = None


def _cls_ids(val):
    flysim._ensure_ann()
    ANN = flysim.ANN
    m = ANN["cell_class"].astype(str).str.fullmatch(val, case=False, na=False).to_numpy()
    return [int(r) for r in ANN.root_id[m].tolist()]


def circuit(min_syn=3):
    """Real olfactory-learning slice: ALPN -> KC -> MBON connectivity matrices. Cached."""
    global _CIRC
    if _CIRC is not None:
        return _CIRC
    flysim._ensure_conn()
    C = flysim.CONN
    pn = sorted(_cls_ids(PN_CLASS)); kc = sorted(_cls_ids(KC_CLASS)); mb = sorted(_cls_ids(MBON_CLASS))
    pi = {p: i for i, p in enumerate(pn)}
    ki = {k: i for i, k in enumerate(kc)}
    mi = {m: i for i, m in enumerate(mb)}
    e1 = C[(C.w >= min_syn) & C.pre.isin(set(pn)) & C.post.isin(set(kc))]
    e2 = C[(C.w >= min_syn) & C.pre.isin(set(kc)) & C.post.isin(set(mb))]
    Wpk = np.zeros((len(kc), len(pn)), dtype=np.float32)        # KC x PN
    for pre, post in zip(e1.pre.values, e1.post.values):
        Wpk[ki[int(post)], pi[int(pre)]] += 1.0                 # connectivity (partner count)
    Wkm = np.zeros((len(mb), len(kc)), dtype=np.float32)        # MBON x KC (synapse-weighted)
    for pre, post, w in zip(e2.pre.values, e2.post.values, e2.w.values):
        Wkm[mi[int(post)], ki[int(pre)]] += float(w)
    _CIRC = {"pn": pn, "kc": kc, "mb": mb, "Wpk": Wpk, "Wkm": Wkm, "ki": ki}
    return _CIRC


def odor(seed, n_pn=None, active_frac=0.11):
    """An odor = a random subset of projection neurons firing (one input pattern)."""
    C = circuit()
    n = len(C["pn"])
    r = np.random.default_rng(seed)
    return (r.random(n) < active_frac).astype(np.float32)


def kc_code(odor_vec, coincidence=3):
    """Sparse KC code: a Kenyon cell fires if >= `coincidence` of its ALPN inputs are active
    (coincidence detection -> a tiny, near-disjoint subset of KCs per odor)."""
    C = circuit()
    active_partners = (C["Wpk"] > 0) @ odor_vec
    return active_partners >= coincidence


class FlyMemory:
    """Mushroom-body learning: a dopamine-gated DEPRESSION of KC->MBON output for whichever
    KCs were active during a punished odor. New memories only touch their own (near-disjoint)
    KCs, so they don't overwrite old ones."""

    def __init__(self):
        C = circuit()
        self.depressed = np.zeros(len(C["kc"]), dtype=bool)   # KCs whose output has been weakened

    def teach(self, code):
        self.depressed |= code                                 # pair odor with "bad" -> depress its KCs

    def danger_score(self, code):
        """Learned avoidance strength for an odor = fraction of its KCs that are depressed."""
        n = int(code.sum())
        return float((code & self.depressed).sum() / n) if n else 0.0


def specificity_test(seeds=range(1, 13), coincidence=3):
    """The honest, non-trivial test: after the fly learns several odors (depressing each
    one's Kenyon cells), how often does an UNTAUGHT odor falsely read as 'already learned'?
    Sparse, near-disjoint codes keep that false-memory rate low — that's the real protection
    against interference (retaining a taught odor is trivial under monotonic depression; not
    corrupting the OTHERS is the hard part)."""
    seeds = list(seeds)
    fly = FlyMemory()
    taught = []
    for s in seeds[: len(seeds) // 2]:                # teach the first half
        c = kc_code(odor(s), coincidence)
        fly.teach(c); taught.append(c)
    # false-alarm: untaught odors whose KCs already look depressed by coincidence
    false = [fly.danger_score(kc_code(odor(s), coincidence)) for s in seeds[len(seeds) // 2:]]
    return {"n_taught": len(taught), "n_untaught": len(false),
            "mean_false_memory": float(np.mean(false)) if false else 0.0,
            "kc_active": float(np.mean([c.mean() for c in taught]))}


def run_experiment(seedA=1, seedB=2, coincidence=3):
    """Two odors -> two sparse Kenyon-cell codes. Returns the codes + their overlap, and a
    false-memory rate from specificity_test. The headline number is the OVERLAP: each odor
    occupies its own near-disjoint sliver of the mushroom body, which is how a new memory
    avoids overwriting an old one."""
    oA, oB = odor(seedA), odor(seedB)
    cA, cB = kc_code(oA, coincidence), kc_code(oB, coincidence)
    kc_overlap = (cA & cB).sum() / max(1, (cA | cB).sum())
    spec = specificity_test(coincidence=coincidence)
    return {
        "kc_overlap": float(kc_overlap),
        "kc_active_A": float(cA.mean()), "kc_active_B": float(cB.mean()),
        "shared_kc": int((cA & cB).sum()),
        "false_memory": spec["mean_false_memory"], "n_taught": spec["n_taught"],
        "codes": {"A": cA, "B": cB},
    }


if __name__ == "__main__":
    C = circuit()
    print("Mushroom body: %d projection neurons -> %d Kenyon cells -> %d MBONs\n"
          % (len(C["pn"]), len(C["kc"]), len(C["mb"])))
    r = run_experiment()
    print("Two odors light up two near-disjoint sparse codes in the Kenyon-cell layer:")
    print("  A %.1f%% / B %.1f%% of the %d cells active — only %.1f%% overlap (%d shared cells)\n"
          % (100 * r["kc_active_A"], 100 * r["kc_active_B"], len(C["kc"]),
             100 * r["kc_overlap"], r["shared_kc"]))
    print("Teach the fly %d odors (depress each one's cells); an untaught odor reads as"
          % r["n_taught"])
    print("'already learned' only %.0f%% of the time — new memories barely touch old ones.\n"
          % (100 * r["false_memory"]))
    print("That near-disjoint sparse coding is the architectural reason fly memories don't\n"
          "interfere the way a dense network's shared weights do (catastrophic forgetting).\n"
          "Mechanism, not a benchmark — a dense net on random odors can separate them too;\n"
          "the point is HOW the fly does it: one tiny, near-private code per odor.")
