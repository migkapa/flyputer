"""
energy.py — how much energy did that computation cost the fly brain?

A spiking neuron computes with *synaptic events*: each time a neuron fires, every synapse
it makes performs one analog operation (a weighted transmission ~ a multiply-accumulate).
We count those events straight from the simulation and price them three ways:
  - the fly brain   (bioenergetics)
  - a digital chip  (one MAC + a memory access)
  - the Landauer floor (the physical minimum energy for one irreversible bit operation)

The numbers are order-of-magnitude estimates; the *gap* between them is robust to the exact
constants. Sources:
  - Attwell & Laughlin 2001, "An energy budget for signaling in grey matter"
  - Laughlin 2003, energy cost of information in the fly retina
  - Horowitz 2014, "Computing's energy problem" (digital op / memory energies)
  - Landauer 1961 (kT ln2 minimum)
"""
import numpy as np

# --- energy per one synaptic operation, in joules ---
E_SYN_BIO = 1e-14        # ~10 fJ / synaptic event (Attwell & Laughlin; the fly is small + efficient)
E_SYN_SILICON = 1e-12    # ~1 pJ / synapse update incl. local memory access (Horowitz 2014)
KB = 1.380649e-23        # Boltzmann constant (J/K)
T_ROOM = 300.0           # K
E_LANDAUER = KB * T_ROOM * np.log(2)   # ~2.9e-21 J, physical floor per irreversible bit op

# A conventional chip simulating a spiking network is CLOCKED: it re-evaluates every synapse
# on every tick, active or not. We price that at a physically-motivated neural update rate
# (NOT our internal 0.1 ms sim step, which is an arbitrary numerical choice), so the chip's
# cost — and the fly/chip ratio — reflects real activity sparsity instead of a fixed constant.
CHIP_CLOCK_HZ = 1000.0   # 1 kHz: a standard rate for digital spiking-network simulation


def synaptic_events(spike_counts, W):
    """Count synaptic transmission events the simulated circuit performed (EVENT-DRIVEN: a
    synapse only costs when its presynaptic neuron fires).

    For each neuron: (times it fired) x (number of synapses it sends). W[post, pre] holds
    signed syn_count, so a neuron's outgoing synapse count = sum_post |W[post, neuron]|.
    """
    spikes = np.asarray(spike_counts, dtype=np.float64)
    out_syn = np.abs(np.asarray(W, dtype=np.float64)).sum(axis=0)
    return float(np.dot(spikes, out_syn))


def dense_synapse_updates(W, dur_ms, spike_counts=None):
    """Work a CLOCKED chip does to run the same circuit: every synapse re-evaluated on every
    tick for the whole duration — regardless of whether it carried a spike. Counted on the
    SAME per-synapse basis as synaptic_events (sum of synapse counts, |W|), so their ratio
    (utilization) is a true 0..1 active fraction. This dense clocked cost is exactly what the
    fly avoids by being event-driven, and it's what makes the comparison vary by activity.

    Pass `spike_counts` to restrict to the engaged subnetwork (neurons that fired at least
    once) — otherwise a big inactive 2-hop dragnet of never-firing synapses would inflate the
    chip's bill for synapses no computation ever touched, on either substrate."""
    W = np.abs(np.asarray(W, dtype=np.float64))
    if spike_counts is not None:
        active = np.asarray(spike_counts) > 0
        if active.any():
            W = W[np.ix_(active, active)]
    ticks = max(1.0, float(dur_ms) * CHIP_CLOCK_HZ / 1000.0)
    return float(W.sum()) * ticks


def compare(events, dense_ops=None):
    """Price the fly's `events` synaptic transmissions against a clocked chip doing
    `dense_ops` synapse updates for the same circuit. If dense_ops is omitted we fall back
    to charging the chip per fly-event (the old fixed-100x behavior)."""
    events = float(events)
    if dense_ops is None:
        dense_ops = events                    # legacy fallback -> constant ratio
    dense_ops = float(dense_ops)
    bio = events * E_SYN_BIO
    chip = dense_ops * E_SYN_SILICON
    floor = events * E_LANDAUER
    return {
        "events": events,
        "dense_ops": dense_ops,
        "utilization": (events / dense_ops) if dense_ops else 0.0,   # active fraction (sparsity)
        "fly_joules": bio,
        "chip_joules": chip,
        "floor_joules": floor,
        "chip_vs_fly": (chip / bio) if bio else 0.0,
        "fly_above_floor": (bio / floor) if floor else 0.0,
    }


def humanize(j):
    """Joules -> friendly SI string."""
    if j <= 0:
        return "0 J"
    for scale, name in [(1e-21, "zJ"), (1e-18, "aJ"), (1e-15, "fJ"), (1e-12, "pJ"),
                        (1e-9, "nJ"), (1e-6, "uJ"), (1e-3, "mJ"), (1.0, "J")]:
        if j < scale * 1000:
            return "%.1f %s" % (j / scale, name)
    return "%.2e J" % j


def summary(events, dense_ops=None):
    """UI-ready energy summary. Pass `dense_ops` (from dense_synapse_updates) so the chip is
    priced on the clocked work it actually does — making the fly/chip ratio reflect real
    activity sparsity instead of the fixed 100x. Omit it for the legacy constant-ratio."""
    c = compare(events, dense_ops)
    util = c["utilization"]
    if dense_ops is None:
        tail = ("about %.0fx less than a chip would (%s), and only ~%.0e x above the "
                "physical minimum." % (c["chip_vs_fly"], humanize(c["chip_joules"]),
                                       c["fly_above_floor"]))
    else:
        tail = ("%.0fx less than a chip clock-evaluating the same circuit (%s): only %.1f%% "
                "of synapses were active, and the brain pays for those instead of every "
                "synapse every tick." % (c["chip_vs_fly"], humanize(c["chip_joules"]),
                                         100 * util))
    return {
        "events": int(round(events)),
        "dense_ops": int(round(c["dense_ops"])),
        "utilization": util,
        "fly_joules": c["fly_joules"],
        "chip_joules": c["chip_joules"],
        "floor_joules": c["floor_joules"],
        "fly": humanize(c["fly_joules"]),
        "chip": humanize(c["chip_joules"]),
        "floor": humanize(c["floor_joules"]),
        "chip_vs_fly": c["chip_vs_fly"],
        "fly_above_floor": c["fly_above_floor"],
        "headline": "The fly brain spent %s on %d synaptic events - %s" %
                    (humanize(c["fly_joules"]), int(round(events)), tail),
    }


if __name__ == "__main__":
    # sanity: 1000 synaptic operations
    import json
    print(json.dumps(summary(1000), indent=2))
    print("\nLandauer floor per op: %.3e J" % E_LANDAUER)
