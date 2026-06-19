"""
agent.py — let a local Gemma model drive the FlyWire fly-brain simulation.

Prereqs:
  1. Install Ollama:        https://ollama.com/download
  2. Pull a Gemma model:    ollama pull gemma3:4b      (or gemma3:12b)
  3. pip install -r requirements.txt
  4. Download the connectome data (see README / get_data.sh).

Run:
  python agent.py                       # default feeding-circuit demo
  python agent.py "your question"       # ask your own
  python agent.py -i                    # interactive chat
"""
import sys
import json
import re

try:
    import ollama
except ImportError:
    sys.exit("Missing dep: pip install ollama  "
             "(also install the Ollama app and run `ollama pull gemma3:4b`)")

import flysim
import visualize

MODEL = "gemma4:latest"   # your local Gemma 4; try "gemma3:4b" for a lighter/faster run


def _tool_visualize(query="sugar", dur_ms=200, seeds=30):
    """Stimulate neurons matching `query` and save a plot. Returns a summary dict."""
    try:
        return visualize.visualize(query=query, limit=int(seeds),
                                   dur_ms=float(dur_ms), quiet=True)
    except ValueError as e:
        return {"error": str(e)}


TOOLS = dict(flysim.TOOLS)
TOOLS["visualize"] = _tool_visualize

SYSTEM = """You are a curious neuroscientist exploring the FlyWire fruit-fly brain \
connectome.

You can ONLY act by emitting a single JSON object per turn — nothing else, no prose:
  {"tool": "find_neurons", "args": {"query": "gustatory"}}
  {"tool": "stimulate",    "args": {"drive_ids": [720..., 720...], "dur_ms": 200}}
  {"tool": "visualize",    "args": {"query": "olfactory", "seeds": 40, "dur_ms": 200}}
  {"tool": "neuroglancer", "args": {"ids": [720..., 720...]}}
When you are done, emit your conclusion as:
  {"final": "your plain-English explanation here"}

Tools:
- find_neurons(query): search neurons by cell type / class / brain region (e.g.
  "mushroom body", "olfactory", "sugar"). Returns root_ids.
- stimulate(drive_ids, dur_ms): inject current into those neurons, simulate the
  downstream circuit, and return which cell types fired most.
- visualize(query, seeds, dur_ms): stimulate neurons matching `query` and SAVE a plot
  (spike raster + top responders) to an image file. Use this whenever the user asks
  to see, draw, plot, or visualize something.
- neuroglancer(ids): get a 3D viewer URL for those neurons.

Strategy: if the user asks to SEE / DRAW / PLOT / VISUALIZE something, just call
visualize with the right query. Otherwise find ids first, then stimulate, then explain.
Emit exactly ONE json object each turn."""

DEFAULT_Q = (
    "Visualize what happens when the fly smells something: stimulate the olfactory "
    "neurons, draw the response, and explain in simple words where the signal goes."
)


def _parse_json(text):
    """Tolerant JSON extraction — Gemma sometimes wraps output in ```json fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)   # first balanced-ish object
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def chat(user_msg: str, max_steps: int = 10, verbose: bool = True) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    last_image = None
    for _ in range(max_steps):
        reply = ollama.chat(model=MODEL, messages=msgs,
                            format="json")["message"]["content"]
        call = _parse_json(reply)
        if call is None:
            msgs.append({"role": "user",
                         "content": "Reply with exactly one valid JSON object — no prose, no code fences."})
            continue

        if "final" in call:
            answer = call["final"]
            if last_image:
                answer += f"\n\n[plot saved to: {last_image}]"
            return answer

        tool = call.get("tool")
        args = call.get("args", {}) or {}
        if verbose:
            print(f"  -> {tool}({json.dumps(args)[:90]})", file=sys.stderr)
        fn = TOOLS.get(tool)
        result = fn(**args) if fn else {"error": f"unknown tool '{tool}'"}
        if isinstance(result, dict) and result.get("image_file"):
            last_image = result["image_file"]
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user",
                     "content": "TOOL RESULT:\n" + json.dumps(result, default=str)[:3500]})
    return "(stopped: hit max_steps without a final answer)"


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("-i", "--interactive"):
        print("Interactive mode — ask about the fly brain (Ctrl-C to quit).")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                print("\n" + chat(q))
    else:
        q = " ".join(argv) if argv else DEFAULT_Q
        print(f"Q: {q}\n")
        print(chat(q))


if __name__ == "__main__":
    main()
