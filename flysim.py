"""
flysim.py — a tiny, laptop-friendly FlyWire connectome + spiking-sim backend.

Exposes three tools (find_neurons / stimulate / neuroglancer) that a local LLM
(see agent.py) can call to explore the fruit-fly brain.

Data (downloaded once, NO CAVE token needed):
  - proofread_connections_783.feather  (~852 MB) from Zenodo record 10676866
        https://zenodo.org/records/10676866   (run ./get_data.sh)
  - Supplemental_file1_neuron_annotations.tsv  (~32 MB, auto-downloaded on first use)

License: FlyWire data is CC BY-NC 4.0 (non-commercial). Cite Dorkenwald et al. 2024
and Schlegel et al. 2024 if you publish anything from this.
"""

from __future__ import annotations
import os
import sys
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONN_FILE = os.path.join(HERE, "proofread_connections_783.feather")
ANN_FILE = os.path.join(HERE, "annotations_783.tsv")
ANN_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

# Lazily-populated module globals (loaded on first tool call, not at import).
CONN: pd.DataFrame | None = None        # edge list: pre, post, w
ANN: pd.DataFrame | None = None         # neuron annotations
SIGN: dict[int, float] | None = None    # root_id -> synaptic sign (+1/-1/0)
LABEL: dict[int, str] | None = None     # root_id -> cell-type label
_TEXT_COLS: list[str] | None = None     # searchable text columns in ANN


# --------------------------------------------------------------------------- #
# data loading (lazy)
# --------------------------------------------------------------------------- #
def _find_col(df: pd.DataFrame, *candidates: str) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"none of {candidates} found in columns {list(df.columns)}")


def _load_annotations() -> pd.DataFrame:
    if not os.path.exists(ANN_FILE):
        print("flysim: downloading annotation TSV (~32 MB)…", file=sys.stderr)
        urllib.request.urlretrieve(ANN_URL, ANN_FILE)
    df = pd.read_csv(ANN_FILE, sep="\t", low_memory=False)
    df = df.dropna(subset=["root_id"])
    df["root_id"] = df["root_id"].astype("int64")
    # FlyWire's annotation file calls the predicted transmitter "top_nt".
    if "top_nt" in df.columns and "nt_type" not in df.columns:
        df = df.rename(columns={"top_nt": "nt_type"})
    keep = [
        c for c in ["root_id", "nt_type", "cell_type", "cell_class",
                    "cell_sub_class", "super_class", "supertype",
                    "hemibrain_type", "side"]
        if c in df.columns
    ]
    return df[keep].reset_index(drop=True)


def _load_connectome() -> pd.DataFrame:
    if not os.path.exists(CONN_FILE):
        sys.exit(
            f"\nflysim: missing {os.path.basename(CONN_FILE)}.\n"
            "Download it once (~852 MB, no login) — run ./get_data.sh, or grab the\n"
            "connections feather from https://zenodo.org/records/10676866\n"
        )
    print("flysim: loading connectome edges (first use only)…", file=sys.stderr)
    df = pd.read_feather(CONN_FILE)
    pre = _find_col(df, "pre_pt_root_id", "pre_root_id", "pre_id", "pre")
    post = _find_col(df, "post_pt_root_id", "post_root_id", "post_id", "post")
    w = _find_col(df, "syn_count", "n_syn", "weight", "count", "syns", "n")
    df = df[[pre, post, w]].rename(columns={pre: "pre", post: "post", w: "w"})
    df["pre"] = df["pre"].astype("int64")
    df["post"] = df["post"].astype("int64")
    df["w"] = df["w"].astype("int32")
    return df


def _nt_sign(nt) -> float:
    """Sign convention (a deliberate simplification): GABA & glutamate are inhibitory,
    everything else excitatory. NT predictions are unreliable for some cell types
    (notably Kenyon cells, which get mislabelled 'dopamine'), so we default to
    excitatory rather than zeroing those neurons out and killing all downstream signal."""
    nt = str(nt).upper()
    if nt.startswith("GABA"):
        return -1.0           # inhibitory
    if nt.startswith("GLU"):
        return -1.0           # inhibitory in this toy (GluCl-dominated)
    return 1.0                # ACh + dopamine/serotonin/octopamine + unknown


def _ensure_ann() -> None:
    global ANN, SIGN, LABEL, _TEXT_COLS
    if ANN is not None:
        return
    ANN = _load_annotations()
    if "nt_type" in ANN.columns:
        SIGN = {int(r): _nt_sign(nt) for r, nt in zip(ANN.root_id, ANN.nt_type)}
    else:
        SIGN = {}
    # Label: best available name (cell_type -> cell_class -> super_class).
    def _col_or_na(name):
        if name not in ANN.columns:
            return pd.Series(pd.NA, index=ANN.index, dtype="string")
        s = ANN[name].astype("string")
        return s.mask(s.str.strip().eq(""), pd.NA)

    lab = (_col_or_na("cell_type")
           .combine_first(_col_or_na("cell_class"))
           .combine_first(_col_or_na("super_class"))
           .fillna("?"))
    LABEL = {int(r): str(v) for r, v in zip(ANN.root_id, lab)}
    _TEXT_COLS = [
        c for c in ["cell_type", "cell_class", "cell_sub_class", "super_class",
                    "supertype", "hemibrain_type"]
        if c in ANN.columns
    ]


def _ensure_conn() -> None:
    global CONN
    _ensure_ann()
    if CONN is None:
        CONN = _load_connectome()


# --------------------------------------------------------------------------- #
# subcircuit extraction + tiny leaky-integrate-and-fire engine
# --------------------------------------------------------------------------- #
def build_subcircuit(seed_ids, hops=2, max_neurons=1500, min_syn=5):
    """BFS the downstream neighborhood of the seeds in DISCOVERY order (so truncation
    keeps the seeds and their nearest targets); return a signed W[post, pre]."""
    _ensure_conn()
    seeds = list(dict.fromkeys(int(s) for s in seed_ids))
    edges = CONN[CONN.w >= min_syn]
    order = list(seeds)
    seen = set(seeds)
    frontier = list(seeds)
    for _ in range(hops):
        if not frontier or len(order) >= max_neurons:
            break
        targets = edges[edges.pre.isin(frontier)].post.values
        new = []
        for x in targets:
            xi = int(x)
            if xi not in seen:
                seen.add(xi)
                order.append(xi)
                new.append(xi)
                if len(order) >= max_neurons:
                    break
        frontier = new
    nodes = order[:max_neurons]
    idx = {n: i for i, n in enumerate(nodes)}
    nodeset = set(nodes)
    sub = edges[edges.pre.isin(nodeset) & edges.post.isin(nodeset)]
    N = len(nodes)
    W = np.zeros((N, N), dtype=np.float32)            # W[post, pre], signed
    for pre, post, w in zip(sub.pre.values, sub.post.values, sub.w.values):
        s = SIGN.get(int(pre), 0.0)
        if s:
            W[idx[int(post)], idx[int(pre)]] += w * s
    return nodes, idx, W


def run_lif(nodes, idx, W, drive_ids, drive=25.0, dur_ms=200.0, dt=0.1,
            gain=0.5, v_th=15.0, tau_m=10.0, tau_s=5.0, t_ref=2.0, record=False):
    """Vectorized current-based LIF with a refractory period. Returns per-neuron
    spike counts. If record=True, also returns (counts, t_ms, neuron_idx)."""
    N = len(nodes)
    V = np.zeros(N, dtype=np.float32)
    Isyn = np.zeros(N, dtype=np.float32)
    Iext = np.zeros(N, dtype=np.float32)
    for d in drive_ids:
        if int(d) in idx:
            Iext[idx[int(d)]] = drive
    Wsc = W * gain
    spikes = np.zeros(N, dtype=np.int32)
    cool = np.zeros(N, dtype=np.int32)               # refractory countdown (steps)
    ref_steps = max(1, int(t_ref / dt))
    rec_t, rec_i = [], []
    for step in range(int(dur_ms / dt)):
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0                             # not in refractory
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            spikes += fired
            if record:
                fi = np.nonzero(fired)[0]
                rec_t.append(np.full(fi.shape, step * dt, dtype=np.float32))
                rec_i.append(fi)
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
        np.subtract(cool, 1, out=cool, where=cool > 0)
    if record:
        t = np.concatenate(rec_t) if rec_t else np.array([], dtype=np.float32)
        i = np.concatenate(rec_i) if rec_i else np.array([], dtype=np.int64)
        return spikes, t, i
    return spikes


# --------------------------------------------------------------------------- #
# tools the LLM can call
# --------------------------------------------------------------------------- #
# Common brain-region names -> the cell-type substrings FlyWire actually uses,
# so you can search "mushroom body" instead of "Kenyon"/"MBON".
_ALIASES = {
    "mushroom body": ["Kenyon", "MBON", "MBIN"],
    "kenyon cell": ["Kenyon"],
    "central complex": ["EPG", "PEN", "PEG", "Delta7", "PFL", "PFN", "ExR", "ring neuron"],
    "compass": ["EPG", "PEN", "PEG", "Delta7"],
    "heading": ["EPG", "PEN", "PEG", "Delta7"],
    "ellipsoid body": ["ring neuron", "EPG", "ExR"],
    "clock": ["LNv", "LNd", "DN1"],
    "circadian": ["LNv", "LNd", "DN1"],
}

_SUGGESTIONS = ["sugar", "gustatory", "mushroom body", "central complex", "clock",
                "olfactory", "descending", "motor", "Kenyon", "MBON"]


def find_neurons(query: str, limit: int = 15):
    """Search neurons by cell type / class / brain region. Understands a few common
    region names (e.g. 'mushroom body', 'central complex', 'clock')."""
    _ensure_ann()
    q = str(query).lower().strip()
    terms = []
    for key, syns in _ALIASES.items():
        if key in q:
            terms.extend(syns)
    if not terms:
        terms = [query]

    mask = pd.Series(False, index=ANN.index)
    for term in terms:
        t = str(term).lower()
        for c in _TEXT_COLS:
            mask |= ANN[c].astype(str).str.lower().str.contains(t, na=False, regex=False)

    hits = ANN[mask].head(limit)
    neurons = [
        {
            "root_id": int(r.root_id),
            "cell_type": None if pd.isna(r.get("cell_type")) else str(r.get("cell_type")),
            "cell_class": None if pd.isna(r.get("cell_class")) else str(r.get("cell_class")),
            "nt_type": None if pd.isna(r.get("nt_type")) else str(r.get("nt_type")),
        }
        for _, r in hits.iterrows()
    ]
    out = {"query": query, "matched_terms": terms,
           "n_matches": int(mask.sum()), "neurons": neurons}
    if mask.sum() == 0:
        out["suggestions"] = _SUGGESTIONS
    return out


def stimulate(drive_ids, dur_ms: float = 200, hops: int = 2):
    """Inject current into drive_ids, simulate the downstream subcircuit, and
    report which cell types fired the most."""
    drive_ids = [int(x) for x in drive_ids]
    nodes, idx, W = build_subcircuit(drive_ids, hops=hops)
    sp = run_lif(nodes, idx, W, drive_ids, dur_ms=dur_ms)
    df = pd.DataFrame({"root_id": nodes, "spikes": sp})
    df["cell_type"] = df.root_id.map(lambda r: LABEL.get(int(r), "?"))
    active = df[df.spikes > 0]
    top = (active.groupby("cell_type").spikes.sum()
           .sort_values(ascending=False).head(12))
    return {
        "subcircuit_size": len(nodes),
        "neurons_that_fired": int((sp > 0).sum()),
        "total_spikes": int(sp.sum()),
        "top_responding_cell_types": {k: int(v) for k, v in top.items()},
    }


def neuroglancer(ids):
    """Return a FlyWire Neuroglancer 3D-viewer URL for these neurons (use Chrome)."""
    ids = [str(int(i)) for i in ids][:50]
    state = {
        "layers": [{
            "type": "segmentation",
            "name": "FlyWire",
            "source": ("graphene://https://segmentation.prod.flywire-daf.com/"
                       "segmentation/table/fly_v141"),
            "segments": ids,
        }],
        "layout": "3d",
    }
    return {"url": "https://ngl.flywire.ai/#!" + urllib.parse.quote(json.dumps(state))}


TOOLS = {
    "find_neurons": find_neurons,
    "stimulate": stimulate,
    "neuroglancer": neuroglancer,
}


if __name__ == "__main__":
    # Backend smoke test (no LLM needed).
    print("Searching for 'gustatory' neurons…")
    res = find_neurons("gustatory", limit=5)
    print(json.dumps(res, indent=2)[:900])
    if res["neurons"]:
        ids = [n["root_id"] for n in res["neurons"]]
        print("\nStimulating them (this loads the 852 MB connectome on first call)…")
        print(json.dumps(stimulate(ids), indent=2))
