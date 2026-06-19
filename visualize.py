"""
visualize.py — stimulate a set of FlyWire neurons and PLOT the response.

  .venv/bin/python visualize.py                      # default: sugar neurons
  .venv/bin/python visualize.py gustatory            # any search term
  .venv/bin/python visualize.py "mushroom body" 30 300   # term, #seeds, dur_ms

Writes fly_response.png, prints a 3D Neuroglancer URL, and a plain-English summary.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # save to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import flysim

ORANGE, BLUE = "#e8730c", "#1f77b4"


def visualize(query="sugar", limit=25, dur_ms=200, out="fly_response.png", quiet=False):
    found = flysim.find_neurons(query, limit=limit)["neurons"]
    if not found:
        raise ValueError(f"No neurons matched '{query}'. Try: sugar, gustatory, "
                         "'mushroom body', 'central complex', clock, olfactory, descending, motor.")
    ids = [n["root_id"] for n in found]
    nodes, idx, W = flysim.build_subcircuit(ids, hops=2)
    counts, st, si = flysim.run_lif(nodes, idx, W, ids, dur_ms=dur_ms, record=True)

    driven = {idx[i] for i in ids if i in idx}
    labels = [flysim.LABEL.get(n, "?") for n in nodes]

    # neurons that fired, stimulated ones first then by activity
    active = [r for r in range(len(nodes)) if counts[r] > 0]
    active.sort(key=lambda r: (r not in driven, -int(counts[r])))
    row_of = {r: k for k, r in enumerate(active)}

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})

    # ---------- left: spike raster (each tick = one neuron firing) ----------
    for r in active:
        m = si == r
        if m.any():
            axL.scatter(st[m], np.full(int(m.sum()), row_of[r]), s=8,
                        c=(ORANGE if r in driven else BLUE), marker="|")
    axL.set_xlabel("time (ms)")
    axL.set_ylabel(f"neuron  (the {len(active)} that fired, of {len(nodes)})")
    axL.set_title(f"Stimulate '{query}' neurons → signal spreads through the brain")
    axL.invert_yaxis()
    axL.legend(handles=[
        Line2D([0], [0], marker="|", color=ORANGE, lw=0, label="stimulated (the input)"),
        Line2D([0], [0], marker="|", color=BLUE, lw=0, label="downstream response"),
    ], loc="lower right", fontsize=9)

    # ---------- right: top downstream cell types ----------
    df = pd.DataFrame({"row": range(len(nodes)), "cell": labels, "sp": counts})
    df = df[(df.sp > 0) & (~df.row.isin(driven))]
    top = df.groupby("cell").sp.sum().sort_values(ascending=False).head(12)
    if len(top):
        axR.barh(range(len(top)), top.values[::-1], color=BLUE)
        axR.set_yticks(range(len(top)))
        axR.set_yticklabels(top.index[::-1], fontsize=9)
        axR.set_xlabel("total spikes")
        axR.set_title("Which downstream cell types lit up")
    else:
        axR.text(0.5, 0.5, "no downstream firing", ha="center", va="center")
        axR.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # ---------- 3D view + plain-words summary ----------
    downstream_ids = [nodes[r] for r in active if r not in driven]
    show = ids[:5] + downstream_ids[:5]
    url = flysim.neuroglancer(show)["url"]
    top_names = list(top.index[:6]) if len(top) else []
    summary = (f"Pressed {len(driven)} '{query}' input neurons; over {dur_ms:g} ms the "
               f"signal reached {len(downstream_ids)} downstream neurons. Top responders: "
               f"{', '.join(top_names) if top_names else '(none)'}.")
    if not quiet:
        print(f"saved {out}")
        print("\nIn plain words: " + summary)
        print(f"\n3D view of these actual neurons (open in Chrome):\n{url}")
    return {
        "image_file": out,
        "n_input_neurons": len(driven),
        "n_downstream": len(downstream_ids),
        "top_downstream_types": top_names,
        "neuroglancer_url": url,
        "summary": summary,
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "sugar"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 200
    try:
        visualize(q, lim, dur)
    except ValueError as e:
        sys.exit(str(e))
