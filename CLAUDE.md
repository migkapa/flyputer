# flyputer — notes for Claude Code

A local app that runs logic gates and binary arithmetic on the real **FlyWire** fruit-fly
connectome, visualizes it in 3D, and measures the energy cost vs. silicon. Driven by a local
**Gemma** model via Ollama.

## Rules
- **Git commits: NEVER add a "Co-Authored-By: Claude" trailer or any Claude/AI attribution to
  commit messages.** Commit as the user only.
- Never commit large data files (`*.feather`, `annotations_783.tsv`), `.venv/`, caches, or
  generated artifacts — they're in `.gitignore`; data is fetched via `get_data.sh`.
- Data is **CC BY-NC 4.0** (non-commercial). Keep the framing factual (see `CITATION.md`):
  we *measure* efficiency; we don't claim to prove anything metaphysical.

## Layout
- `flysim.py` — connectome load + LIF sim + neuron lookup tools
- `logic.py` / `flymath.py` — logic gates + arithmetic on real neurons
- `energy.py` — energy ledger (fly brain vs chip vs Landauer limit)
- `export3d.py` — builds 3D scenes (response / gate / math)
- `server.py` + `chat3d.html` — the live chat + 3D app (`.venv/bin/python server.py`)
- `agent.py` / `visualize.py` / `fly3d.html` — CLI agent, matplotlib plot, standalone 3D viewer
