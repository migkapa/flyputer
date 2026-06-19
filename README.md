# Ask the Fly Brain 🪰🧠

Let a local **Gemma** model drive a simulation of the **FlyWire** fruit-fly brain
connectome. You ask a question in plain English; Gemma calls tools that pull the *real*
connectome (the ~140k-neuron wiring diagram of an adult *Drosophila*) and run a small
spiking simulation, then explains what lit up — gates, arithmetic, a heading memory, a
moving fly, shortest paths, even a game you play against the fly's real escape reflex.

```
you (English) ─▶ Gemma (local, via Ollama) ─▶ {"tool": ...}
                                                  │
        find_neurons ─ stimulate ─ show_logic_gate ─ do_math ─ show_compass
        ─ move_fly ─ navigate_fly ─ show_path ─ dodge_swatter ─ neuroglancer
                                                  │
                          real FlyWire data + a tiny LIF sim
                                                  │
                                results ─▶ Gemma ─▶ plain-English answer
```

## Setup

1. **Ollama + a Gemma model.** Defaults to `gemma4:latest`. Check what you have:
   ```bash
   ollama list
   # don't have it? `ollama pull gemma3:4b` and set MODEL in agent.py
   ```
2. **Python deps** (uses a local venv with system site-packages):
   ```bash
   python3 -m venv .venv --system-site-packages
   .venv/bin/pip install -r requirements.txt
   ```
3. **Connectome data** (one-time, ~852 MB, no login):
   ```bash
   bash get_data.sh
   ```
   The neuron annotation file (~32 MB) auto-downloads on first run.

## Run — live chat + 3D (start here)

```bash
.venv/bin/python server.py        # then open http://localhost:8000
```
Chat on the right, a live 3D fly brain on the left. (First start loads the connectome, ~5s,
then warms the gates / routing graph / escape circuit in the background.)

Things to ask — each lights up the real connectome and comes back with an energy ledger:

- **"What happens when the fly smells something?"** — stimulates olfactory neurons and you
  watch the signal cascade through the brain.
- **"Show me the fly brain computing an AND gate"** — three real neurons cycle through every
  input combination with a live truth table, a robustness chip, and an energy cost. The
  same convergent motif computes **AND or OR** depending on excitability; gates that work:
  AND, OR, AND-NOT. (`logic.py` finds the motifs; `energy.py` does the accounting.)
- **"Add 2 + 3 with the fly brain"** — composes those real gates into a half/full adder and
  ripple-carries across bits: `2 + 3 = 5` (`010 + 011 = 101`). It also **multiplies**
  ("multiply 6 × 7" → 42). `flymath.py` is the adder/multiplier.
- **"Show me the fly's compass"** — the central-complex ring of EPG/PEN/PEG/Δ7 neurons forms
  a **heading bump** that holds like a memory and **steers to track a turn**, with a live
  heading dial. First *stateful* computation in the project. (`compass.py`)
- **"Walk the fly forward, turn left, then make it escape"** — drives the real **descending
  command neurons** (DNp09 forward, MDN backward, DNa02 steering, DNp01 Giant Fiber escape)
  and a virtual fly body moves in a top-down arena. (`fly.py`)
- **"Release the fly at heading 120 and let it navigate"** — closed loop: the real
  **compass → PFL3 → DNa02** steering pathway homes the fly onto a stable heading. (`fly.py`)
- **"Trace the path from sugar to motor"** — "six degrees of the fly brain": one shortest
  *wiring* path lights up hop-by-hop. The graph recovers textbook circuits on its own
  (EPG→PFL3→DNa02 steering, olfactory ORN→PN→Kenyon). Pure topology, zero firing claims.
- **"Let me try to swat the fly"** — a **playable game**: a looming swatter drives the real
  LPLC2+LC4 detectors converging onto the Giant Fiber; swing faster than the circuit's
  reaction limit to land it, slower and the real escape reflex jumps first. (`swatter.py`)
- **"How does the fly remember two smells without forgetting?"** — two odors light up two
  near-disjoint sparse sets of Kenyon cells (~0.6% each, ~2% overlap), so a new memory barely
  touches an old one — the architecture behind not suffering catastrophic forgetting. (`sniff.py`)
- **"Show a heart on the fly's eye"** — paint a picture onto the ~789 L1 lamina columns and
  it travels along ~66k real L1→Mi1 synapses into the medulla: a recognizable image glowing
  on the real optic lobe (a ~750-column brain's-eye view, not a camera). (`optic.py`)
- **"Let me pilot the fly with the keyboard"** — a **playable game**: arrow keys / WASD fire
  the real descending command neurons (DNp09 forward, MDN back, DNa02 steer, DNp01 escape) —
  they glow as you pilot a **real 3D fly model** (procedural, with flapping wings) around the
  brain, foraging food. Drop in a glTF (e.g. a CC-BY micro-CT *Drosophila*) via `loadFlyGLB()`.
  (`fly.py` pilot mode + a procedural three.js fly)

## Or use the pieces directly

```bash
.venv/bin/python agent.py                      # default: find sugar neurons → stimulate → explain
.venv/bin/python agent.py "Stimulate visual projection neurons and tell me what responds"
.venv/bin/python agent.py -i                   # interactive chat
.venv/bin/python visualize.py                  # plot the response → fly_response.png
.venv/bin/python visualize.py "mushroom body" 30 300   # any term, #seeds, duration(ms)
.venv/bin/python export3d.py olfactory 40 200          # 3D web view: opens fly3d.html

.venv/bin/python logic.py        # find AND / OR / AND-NOT gate motifs, with robustness
.venv/bin/python flymath.py      # add & multiply through real composed gates (demo set)
.venv/bin/python compass.py      # CUE → HOLD → TURN: heading-memory regime comparison
.venv/bin/python fly.py forward left forward escape   # ASCII trajectory from real DNs
.venv/bin/python swatter.py      # sweep swing speeds: who wins vs the escape circuit, and why
.venv/bin/python sniff.py        # two odors -> near-disjoint sparse Kenyon-cell codes
.venv/bin/python optic.py heart  # relay an image through the real optic lobe (ASCII in/out)

# find_neurons understands region names: sugar, gustatory, "mushroom body",
# "central complex", clock, olfactory, descending, motor, Kenyon, MBON.
```

Smoke-test the backend alone (no LLM needed):
```bash
.venv/bin/python flysim.py
```

Validate the claims (precision, control circuits, gate robustness, arithmetic):
```bash
.venv/bin/python eval.py         # neurons | controls | gates  to run one layer
```

## How it works

- **`flysim.py`** loads the connectome edge list + neuron annotations and exposes the core
  tools: `find_neurons` (search by cell type / class / region), `stimulate` (inject current,
  run a tiny leaky-integrate-and-fire sim over the 2-hop downstream subcircuit),
  `shortest_path` (BFS routing for "six degrees"), and `neuroglancer` (a 3D viewer URL).
- **`logic.py` / `flymath.py`** — real gate motifs and arithmetic composed from them.
- **`compass.py`** — the central-complex ring attractor (heading memory + steering).
- **`fly.py`** — descending-neuron → behavior mapping, body kinematics, and the closed-loop
  compass→DNa02 steering controller.
- **`swatter.py`** — the looming-escape circuit (LPLC2/LC4 → Giant Fiber) and the swat game.
- **`sniff.py`** — the mushroom-body olfactory-learning slice and sparse-coding analysis.
- **`optic.py`** — the retinotopic lamina→medulla (L1→Mi1) image relay.
- **`energy.py`** — the energy ledger (see below). **`export3d.py`** builds the 3D scenes;
  **`server.py`** is the web app; **`agent.py`** is the version-agnostic JSON tool-calling
  loop around Gemma. **`eval.py`** is the validation harness.

No CAVE token, no `fafbseg`, no GPU — just the Zenodo file + the GitHub annotation TSV.

### The energy ledger

The brain is **event-driven**: a synapse only costs energy when its neuron fires. A
conventional chip simulating the same circuit is **clocked** — it re-evaluates every synapse
every tick whether it fired or not. So the fly-vs-chip ratio isn't a fixed number; it tracks
**activity sparsity**, a real measured quantity. Sparse computations (a logic gate, ~3% of
synapses active) show the brain ~thousands of times cheaper; dense ones (a big olfactory
cascade, ~44% active) much less. Per-synapse cost ≈ 10 fJ (Attwell & Laughlin) vs ≈ 1 pJ per
clocked MAC (Horowitz); the chip clock is priced at a physical 1 kHz, not the sim's `dt`.

## Honest caveats

- The simulation is a **toy**: Shiu-style sign convention (ACh = excitatory, GABA/Glu =
  inhibitory, neuromodulators ≈ 0), ~13% neurotransmitter-prediction error, and absolute
  firing rates are **not** meaningful. It's for *qualitative* "what's downstream of what"
  exploration and motif-level computation, not biophysics.
- It runs a bounded **2-hop subcircuit**, not the whole brain at biophysical fidelity.
- **The VNC + muscles are not in this dataset (brain only).** The moving fly and the swatter
  game read the brain's *real* motor commands (descending neurons), but the body itself is a
  labelled stand-in for the missing ventral nerve cord.
- **"Six degrees" is topology, not timing** — it returns *a* shortest wiring path (one of
  possibly many, at ≥5 synapses/edge), not a causal/temporal signal path.
- **Dodge the swatter** reports the *order* of events (detectors charge → Giant Fiber spike →
  lunge) and which swing speeds escape — **not** calibrated millisecond latencies.
- The compass forms and steers a sharp bump faithfully; a *self-sustained* held heading needs
  finer excitation/inhibition tuning than this toy provides.
- **Two smells** demonstrates the *mechanism* (sparse, near-disjoint Kenyon-cell codes), not a
  rigged benchmark — a dense net on random odors can separate them too; the point is *how* the
  fly does it. The Kenyon code is a tuned coincidence threshold, not a calibrated firing rate.
- **The fly's eye** is a ~750-column retinotopic sensor / brain's-eye view, **not** a camera:
  no ommatidial optics, no T4/T5 motion detection, uncalibrated rates. The medulla is read out
  at each cell's lamina column (a full undistorted 2D reconstruction needs de-warping the
  curved hex lattice, which a plain PCA flatten can't do — verified retinotopic only in 1D).
- Root IDs are pinned to FlyWire materialization **v783** (the Oct 2024 *Nature* release).

## License / attribution

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Data:** the FlyWire connectome is **CC BY-NC 4.0 (non-commercial)**. See
  [`CITATION.md`](CITATION.md) for the papers to cite and the non-commercial terms.
- Independent hobby project — **not affiliated with the FlyWire consortium.**

## What's next

A sibling project turns the ~88-neuron central-complex "compass" circuit into an actual
**chip blueprint** (connectome → Verilog → GDSII layout, free on a laptop). The compass and
closed-loop steering demos here are the on-ramp to it.
