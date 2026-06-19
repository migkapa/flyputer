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
E_SYN_SILICON = 1e-12    # ~1 pJ / equivalent MAC incl. local memory access (Horowitz 2014)
KB = 1.380649e-23        # Boltzmann constant (J/K)
T_ROOM = 300.0           # K
E_LANDAUER = KB * T_ROOM * np.log(2)   # ~2.9e-21 J, physical floor per irreversible bit op


def synaptic_events(spike_counts, W):
    """Count synaptic transmission events the simulated circuit performed.

    For each neuron: (times it fired) x (number of synapses it sends). W[post, pre] holds
    signed syn_count, so a neuron's outgoing synapse count = sum_post |W[post, neuron]|.
    """
    spikes = np.asarray(spike_counts, dtype=np.float64)
    out_syn = np.abs(np.asarray(W, dtype=np.float64)).sum(axis=0)
    return float(np.dot(spikes, out_syn))


def compare(events):
    """Price `events` synaptic operations three ways; return raw joules + ratios."""
    events = float(events)
    bio = events * E_SYN_BIO
    chip = events * E_SYN_SILICON
    floor = events * E_LANDAUER
    return {
        "events": events,
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


def summary(events):
    """UI-ready energy summary for a computation of `events` synaptic operations."""
    c = compare(events)
    return {
        "events": int(round(events)),
        "fly_joules": c["fly_joules"],
        "chip_joules": c["chip_joules"],
        "floor_joules": c["floor_joules"],
        "fly": humanize(c["fly_joules"]),
        "chip": humanize(c["chip_joules"]),
        "floor": humanize(c["floor_joules"]),
        "chip_vs_fly": c["chip_vs_fly"],
        "fly_above_floor": c["fly_above_floor"],
        "headline": ("The fly brain spent %s on %d synaptic operations - about %.0fx less "
                     "than a chip would (%s), and only ~%.0e x above the physical minimum."
                     % (humanize(c["fly_joules"]), int(round(events)), c["chip_vs_fly"],
                        humanize(c["chip_joules"]), c["fly_above_floor"])),
    }


if __name__ == "__main__":
    # sanity: 1000 synaptic operations
    import json
    print(json.dumps(summary(1000), indent=2))
    print("\nLandauer floor per op: %.3e J" % E_LANDAUER)
