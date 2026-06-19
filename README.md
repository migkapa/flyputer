# Ask the Fly Brain 🪰🧠

Let a local **Gemma** model drive a simulation of the **FlyWire** fruit-fly brain
connectome. You ask a question in plain English; Gemma calls tools that pull the *real*
connectome (the ~140k-neuron wiring diagram of an adult *Drosophila*) and run a small
spiking simulation, then explains what lit up.

```
you (English) ─▶ Gemma (local, via Ollama) ─▶ {"tool": ...}
                                                  │
                   find_neurons ── stimulate ── neuroglancer
                                                  │
                          real FlyWire data + a tiny LIF sim
                                                  │
                                results ─▶ Gemma ─▶ plain-English answer
```

## Setup

1. **Ollama + a Gemma model.** Defaults to `gemma4:e4b`. Check what you have:
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
Chat on the right, a live 3D fly brain on the left. Ask *"what happens when the fly
smells something?"* and Gemma runs the simulation — the scene lights up and you watch
the signal travel through the brain. (First start loads the connectome, ~5s.)

Ask *"show me the fly brain computing an AND gate, and how little energy it uses"* and it
demonstrates that **real fly neurons compute logic**: three neurons in 3D cycle through
every input combination, with a live **truth table** and an **energy ledger** showing the
fly brain spends ~100× less than a silicon chip per operation (and sits remarkably close to
the physical/Landauer limit). Gates that work: AND, OR, AND-NOT — and every scene reports
its energy cost. (`logic.py` finds real gate motifs; `energy.py` does the accounting.)

And it can **do arithmetic**: ask *"add 2 + 3 with the fly brain"* and it composes those real
gates into a half/full adder (`XOR = (A OR B) AND-NOT (A AND B)`) and ripple-carries across
bits — `2 + 3 = 5` (`010 + 011 = 101`) in ~27 real gate operations, for ~146 pJ (a chip would
spend ~100x more). It also **multiplies** (shift-and-add) — *"multiply 6 × 7"* → `42` in 135
gate operations. `flymath.py` is the adder/multiplier, and the energy panel shows a live
**race**: the fly brain's bar is a sliver next to the chip's (~100x longer).

## Or use the pieces directly

```bash
.venv/bin/python agent.py                      # default: find sugar neurons → stimulate → explain
.venv/bin/python agent.py "Stimulate visual projection neurons and tell me what responds"
.venv/bin/python agent.py -i                   # interactive chat
.venv/bin/python visualize.py                  # plot the response → fly_response.png
.venv/bin/python visualize.py "mushroom body" 30 300   # any term, #seeds, duration(ms)
.venv/bin/python visualize.py olfactory 40 200         # a big, flashy ~800-neuron cascade
.venv/bin/python export3d.py olfactory 40 200          # 3D web view: opens fly3d.html, drag to rotate

# find_neurons understands region names: sugar, gustatory, "mushroom body",
# "central complex", clock, olfactory, descending, motor, Kenyon, MBON.
# Recurrent/inhibitory circuits (central complex, clock) barely propagate from a
# simple feedforward poke — that's expected, not a bug.
```

Smoke-test the backend alone (no LLM needed):
```bash
.venv/bin/python flysim.py
```

## How it works

- **`flysim.py`** loads the connectome edge list + neuron annotations and exposes three
  tools:
  - `find_neurons(query)` — search neurons by cell type / class / region
  - `stimulate(drive_ids, dur_ms)` — inject current, run a tiny leaky-integrate-and-fire
    sim over the 2-hop downstream subcircuit, report which cell types fired most
  - `neuroglancer(ids)` — a 3D viewer URL
- **`agent.py`** is a version-agnostic JSON tool-calling loop around Gemma (it uses
  Ollama's forced-JSON mode so even small models stay parseable).

No CAVE token, no `fafbseg`, no GPU — just the Zenodo file + the GitHub annotation TSV.

## Honest caveats

- The simulation is a **toy**: Shiu-style sign convention (ACh = excitatory, GABA/Glu =
  inhibitory, neuromodulators ≈ 0), ~13% neurotransmitter-prediction error, and absolute
  firing rates are **not** meaningful. It's for *qualitative* "what's downstream of what"
  exploration, not biophysics. If you get silence or runaway, tune `gain` / `drive` in
  `flysim.run_lif`.
- It runs a bounded **2-hop subcircuit**, not the whole brain — that's what keeps it fast.
- Small models occasionally fumble a tool call; `gemma4:e4b` is the default, `gemma4:e2b`
  is lighter. Ollama's forced-JSON mode keeps every reply parseable.
- Root IDs are pinned to FlyWire materialization **v783** (the Oct 2024 *Nature* release).

## License / attribution

FlyWire data is **CC BY-NC 4.0** (non-commercial). If you publish anything from this,
cite **Dorkenwald et al., _Nature_ 2024** and **Schlegel et al., _Nature_ 2024**.

## What's next

A sibling project turns this same ~88-neuron central-complex "compass" circuit into an
actual **chip blueprint** (connectome → Verilog → GDSII layout, free on a laptop). Ask to
scaffold "option 2".
