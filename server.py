"""
server.py — chat with Gemma and watch the fly brain respond live in 3D.

  .venv/bin/python server.py        # then open http://localhost:8000

Left: a 3D view of the FlyWire brain. Right: chat. Ask it to show / stimulate something
("what happens when the fly smells?") and the agent runs the simulation and the 3D scene
updates live with the response.
"""
import os
import sys
import json
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import ollama

import flysim
import agent
import export3d
import logic

PORT = 8000
HERE = os.path.dirname(os.path.abspath(__file__))
_INITIAL = None   # a starter scene served on first page load

SYSTEM = """You are a friendly fly-brain guide. The user sees a live 3D view of the
FlyWire fruit-fly connectome next to this chat.

Act by emitting exactly ONE JSON object per turn, nothing else:
  {"tool": "show3d", "args": {"query": "olfactory", "seeds": 40, "dur_ms": 200}}
  {"tool": "show_logic_gate", "args": {"kind": "AND"}}
  {"tool": "do_math", "args": {"a": 6, "b": 7, "op": "mul"}}
  {"tool": "show_compass", "args": {"regime": "raw"}}
  {"tool": "move_fly", "args": {"moves": ["forward", "left", "forward", "escape"]}}
  {"tool": "navigate_fly", "args": {"start_heading": 120}}
  {"tool": "show_path", "args": {"start": "sugar", "end": "motor"}}
  {"tool": "find_neurons", "args": {"query": "mushroom body"}}
  {"final": "your short, friendly plain-English answer"}

Tools:
- show3d(query, seeds, dur_ms): stimulate neurons matching `query` and DISPLAY the
  response in the 3D view. Use for "what happens when...". seeds 20-60, dur_ms 150-300.
- show_logic_gate(kind): demonstrate that REAL fly neurons compute a logic gate. kind is
  "AND", "OR", or "AND-NOT". Shows the 3 neurons in 3D cycling through every input
  combination, with the truth table and how little energy it costs. Use whenever the user
  asks about logic, computing, gates, the brain as a computer, or energy / efficiency.
- do_math(a, b, op): do arithmetic with real fly-neuron gates wired into an adder/multiplier.
  op is "add" or "mul". Keep a, b small (0-12). Shows the gate neurons in 3D, the binary
  working, and the energy it cost. Use when the user asks to add, multiply, or compute numbers.
- show_compass(regime): demonstrate the central-complex COMPASS — real EPG ring neurons
  that hold a heading like a memory. Shows a bump of activity form, then steer to track a
  90-degree turn, in 3D with a live heading dial. regime is "raw" (sharp bump that forms
  and steers) or "memory" (relaxed inhibition so the bump self-sustains with no input). Use
  when the user asks about the compass, heading, navigation, memory, working memory, ring
  attractor, the central complex, or "remembering" / state.
- move_fly(moves): drive a virtual fly by EXCITING REAL DESCENDING COMMAND NEURONS. `moves`
  is a list of behaviors from: "forward" (DNp09), "backward"/"moonwalk" (MDN), "left"/"right"
  (DNa02 steering), "escape"/"jump" (the Giant Fiber DNp01). The brain lights up the real
  command neurons and a virtual fly body walks/turns/lunges accordingly. Use when the user
  asks to move/walk/turn/steer the fly, make it jump or escape, or drive behavior. The VNC +
  muscles aren't in this dataset, so the body is a stand-in driven by the real brain commands.
- navigate_fly(start_heading): CLOSED LOOP. Release the fly at start_heading (degrees) and
  the real compass->PFL3->DNa02 steering circuit turns it onto the circuit's intrinsic
  preferred heading while it walks — it homes and then holds a straight course. Use when the
  user asks the fly to navigate, home, hold a course, steer itself, find its heading, or
  asks about closed-loop / the compass driving behavior.
- show_path(start, end): "six degrees of the fly brain" — trace ONE shortest WIRING path
  between two regions/cell-types and light it up hop-by-hop in 3D (e.g. sugar->motor,
  smell->escape, eye->wing, clock->everything). It is pure connectivity (how many synapses
  apart two neurons are), NOT firing/timing — say "A shortest path", never "the" path, and
  never claim signal timing. Use when the user asks how two things connect, how far apart
  neurons are, what links X to Y, or to trace/route a signal across the brain.
- find_neurons(query): look up neurons by name/region.

Every scene comes back with an energy ledger comparing the fly brain to a computer chip.
When relevant, tell the user how little energy the brain used (the brain spends ~100x less
than a chip per operation, near the physical limit). Keep `final` short (2-3 sentences).
Region names that work: olfactory, sugar, gustatory, "mushroom body", "central complex",
clock, descending, motor."""


def run_agent(message, max_steps=8):
    """Run the Gemma tool loop. Returns (answer_text, viz_data_or_None)."""
    viz = {}

    def show3d(query="olfactory", seeds=40, dur_ms=200):
        data = export3d.build_data(query, int(seeds), float(dur_ms))
        viz["data"] = data
        return {"shown_in_3d": True, "query": data["query"],
                "n_input": data["n_input"], "n_downstream": data["n_downstream"],
                "top_types": data.get("top_downstream_types", []),
                "energy": data["energy"]["headline"]}

    def show_logic_gate(kind="AND"):
        k = str(kind).upper().replace(" ", "-").replace("_", "-")
        if k in ("NOT", "ANDNOT", "NAND-NOT"):
            k = "AND-NOT"
        if k not in ("AND", "OR", "AND-NOT"):
            k = "AND"
        g = logic.find_gate(k)
        if not g:
            return {"error": "no clean %s gate found in the connectome" % k}
        data = export3d.build_gate_scene(g)
        viz["data"] = data
        return {"shown_in_3d": True, "gate": data["gate"]["kind"],
                "neurons": data["gate"]["labels"], "truth_table": data["gate"]["truth"],
                "energy": data["energy"]["headline"]}

    def do_math(a=2, b=3, op="add"):
        try:
            a, b = max(0, int(a)), max(0, int(b))
        except Exception:
            return {"error": "a and b must be small whole numbers"}
        data = export3d.build_math_scene(a, b, op=str(op))
        viz["data"] = data
        m = data["math"]
        return {"shown_in_3d": True,
                "result": "%d %s %d = %d" % (m["x"], m["sym"], m["y"], m["result"]),
                "binary": "%s %s %s = %s" % (m["x_bin"], m["sym"], m["y_bin"], m["result_bin"]),
                "gate_ops": m["gate_ops"], "energy": data["energy"]["headline"]}

    def show_compass(regime="raw"):
        r = str(regime).lower().strip()
        if r not in ("raw", "memory"):
            r = "raw"
        data = export3d.build_compass_scene(r)
        viz["data"] = data
        cm = data["compass"]
        return {"shown_in_3d": True, "scene": "compass", "regime": r,
                "ring_neurons": cm["n_ring"],
                "cued_heading_deg": cm["theta0_deg"], "turned_to_deg": cm["turn_to_deg"],
                "energy": data["energy"]["headline"]}

    def move_fly(moves=None):
        if isinstance(moves, str):
            moves = moves.replace(",", " ").split()
        if not moves:
            moves = ["forward", "left", "forward", "escape"]
        data = export3d.build_fly_scene([str(m) for m in moves])
        viz["data"] = data
        f = data["fly"]
        end = f["traj"][-1] if f["traj"] else [0, 0, 0, 0]
        return {"shown_in_3d": True, "scene": "fly",
                "moves": [c["label"] for c in f["commands"]],
                "command_neurons": [c["dn"] for c in f["commands"]],
                "ended_at": {"x": end[1], "y": end[2], "heading_deg": end[3]},
                "energy": data["energy"]["headline"]}

    def navigate_fly(start_heading=120):
        try:
            h = float(start_heading)
        except Exception:
            h = 120.0
        data = export3d.build_navigate_scene(h)
        viz["data"] = data
        f = data["fly"]
        return {"shown_in_3d": True, "scene": "navigate",
                "released_at_deg": f["start_deg"], "homed_to_deg": f["goal_deg"],
                "final_error_deg": f["final_err_deg"], "energy": data["energy"]["headline"]}

    def show_path(start="sugar", end="motor"):
        data = export3d.build_path_scene(str(start), str(end))
        viz["data"] = data
        p = data["path"]
        if not p.get("found"):
            return {"error": "no path %s -> %s: %s" % (start, end, p.get("reason", ""))}
        return {"shown_in_3d": True, "scene": "path", "start": start, "end": end,
                "n_synapses": p["n_synapses"], "chain": " -> ".join(p["hops"]),
                "note": "a shortest wiring path (topology, not signal timing)"}

    tools = dict(flysim.TOOLS)
    tools["show3d"] = show3d
    tools["show_logic_gate"] = show_logic_gate
    tools["do_math"] = do_math
    tools["show_compass"] = show_compass
    tools["move_fly"] = move_fly
    tools["navigate_fly"] = navigate_fly
    tools["show_path"] = show_path
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": message}]
    for _ in range(max_steps):
        reply = ollama.chat(model=agent.MODEL, messages=msgs,
                            format="json")["message"]["content"]
        call = agent._parse_json(reply)
        if call is None:
            msgs.append({"role": "user", "content": "Reply with ONE JSON object only."})
            continue
        if "final" in call:
            return call["final"], viz.get("data")
        fn = tools.get(call.get("tool"))
        args = call.get("args", {}) or {}
        try:
            result = fn(**args) if fn else {"error": "unknown tool"}
        except Exception as e:
            result = {"error": str(e)}
        msgs.append({"role": "assistant", "content": reply})
        note = "TOOL RESULT:\n" + json.dumps(result, default=str)[:2200]
        if viz.get("data"):       # once something is shown, push to conclude
            note += ('\n\nThe 3D view is now updated. Reply ONLY with '
                     '{"final": "<2-3 sentence simple explanation>"}.')
        msgs.append({"role": "user", "content": note})

    # ran out of steps: synthesize a friendly answer from whatever was shown
    data = viz.get("data")
    if data:
        tops = ", ".join(data.get("top_downstream_types", [])[:4]) or "several cell types"
        return (f"I stimulated the {data['query']} neurons ({data['n_input']} inputs) and "
                f"the signal spread to {data['n_downstream']} downstream neurons, including "
                f"{tops}. You can watch it travel in the 3D view.", data)
    return ("Hmm, I couldn't pin that down — try naming a region like 'olfactory', "
            "'sugar', or 'mushroom body'.", None)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                html = open(os.path.join(HERE, "chat3d.html")).read()
            except OSError:
                self._send(500, "chat3d.html not found", "text/plain")
                return
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/initial":
            self._send(200, _INITIAL if _INITIAL else "null")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, "not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            message = json.loads(self.rfile.read(n) or b"{}").get("message", "")
        except Exception:
            self._send(400, json.dumps({"error": "bad json"}))
            return
        try:
            answer, data = run_agent(message)
            self._send(200, json.dumps({"answer": answer, "viz": data}))
        except Exception as e:
            self._send(200, json.dumps({"answer": f"(error: {e})", "viz": None}))

    def log_message(self, *args):   # keep the console quiet
        pass


def serve(open_browser=True):
    print("Loading connectome (first start is slow, ~5s)...", file=sys.stderr)
    flysim._ensure_conn()
    global _INITIAL
    print("Building the resting brain...", file=sys.stderr)
    try:
        _INITIAL = json.dumps(export3d.resting_data())
    except Exception as e:
        print("starter scene failed:", e, file=sys.stderr)
    # warm the mesh service + logic-gate motifs in the background so first requests are fast
    def _warm():
        export3d.warm()
        try:
            import flymath
            flymath._gates()
        except Exception:
            pass
        try:
            flysim._ensure_adj()          # routing adjacency, so first show_path is instant
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()
    url = f"http://localhost:{PORT}"
    print(f"Ready -> {url}", file=sys.stderr)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except OSError as e:
        print(f"\nCould not start on port {PORT}: {e}\n"
              f"Another server may still be running. Free it with:\n"
              f"  lsof -ti tcp:{PORT} | xargs kill\n", file=sys.stderr)
        raise


if __name__ == "__main__":
    serve()
