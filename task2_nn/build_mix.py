"""Release candidate: task2/build.py (v1 recipe, unchanged options) with network probabilities mixed in just before
the decision. task2/ is not modified: the source of build.predict_split_v1 is copied and two expressions are
replaced (each must match exactly once):
  onset    best_k_set(p)        -> best_k_set((1 - w_on) * p + w_on * p_nn[T+30])
  ongoing  (p >= th)            -> ((1 - w_go) * p + w_go * p_nn[h, link]) >= th
With --w-onset 0 --w-ongoing 0 the output must equal the plain build.py file (checked by --check-plain).

--trunc: windows come from task2/trunc_test.py (raw release files with rows after T removed); each window's
13-row arrays are asserted to equal the arrays the network predicted from (prep.py), so the network part is the
same function of the same inputs. The output is compared cell by cell with --compare.

Usage: build_mix.py --nn <release.pkl> [...] --w-onset a --w-ongoing b --out <csv> [--trunc --compare <csv>] -- <build.py options>
"""
from __future__ import annotations

import argparse
import inspect
import os
import pickle
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
T2 = Path(os.environ.get("T2_DIR", HERE.parent / "task2"))   # task2 code + cache (run_all: the fresh task2 build)
sys.path.insert(0, str(T2)); sys.path.insert(0, str(HERE))
DATA = Path(os.environ.get("NN_DATA", HERE / "data"))         # prep.py output
import numpy as np
import pandas as pd

import build
import adaptive_gate
import features_v1 as F1

STATS = {"windows": 0, "arrays_checked": 0}


def main() -> None:
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", type=Path, nargs="+", required=True)
    ap.add_argument("--w-onset", type=float, required=True)
    ap.add_argument("--w-ongoing", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trunc", action="store_true")
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--gate-clim", type=float, help="ongoing: the mix may not add a queue (0 -> 1 against the plain "
                    "decision) on links free at T (T row, empty cells from T-5) whose train climatology of a queue at T+h "
                    "is >= this value")
    ap.add_argument("--gate-wide", action="store_true", help="with --gate-clim: in the gate cells a queue needs both the "
                    "plain decision and the network alone (p_nn >= th), so plain-decision queues there can also be removed")
    ap.add_argument("--gate-adaptive", action="store_true", help="per-window gate strength from the split's own "
                    "observations before the window's day (adaptive_gate.py); overrides --gate-clim / --gate-wide")
    ap.add_argument("--gate-log", type=Path, help="with --gate-adaptive: write the per-window h, ratio and gate (json)")
    ap.add_argument("--dump", type=Path, help="save LightGBM / network ongoing probabilities per window (diagnosis)")
    args = ap.parse_args(argv)
    rels = [pickle.load(open(p, "rb")) for p in args.nn]
    NN = {w: np.mean([r[w]["p"] for r in rels], 0) for w in rels[0]}
    arrays = {}
    for p in build.PANELS:
        z = np.load(DATA / f"{p}.npz", allow_pickle=False)
        for s in ("validation", "private"):
            for i, w in enumerate(z[f"rel_{s}_wid"]):
                arrays[str(w)] = {a: z[f"rel_{s}_{a}"][i] for a in ("speed", "flow", "occ", "pct")} | {"t": int(z[f"rel_{s}_t"][i])}

    DUMP = {}

    def nn_on(w, p):
        STATS["windows"] += 1
        return (1 - args.w_onset) * p + args.w_onset * NN[w.window_id][-1]

    GATE = {}
    GLOG = {}

    def gate_tables(panel):
        if panel not in GATE:
            z = np.load(DATA / f"{panel}.npz", allow_pickle=False)
            GATE[panel] = (z["vcut"], z["clim_all_q"])
        return GATE[panel]

    def nn_go(w, df, p, th):
        STATS["windows"] += 1
        h, link = df.h.to_numpy(), df.link.to_numpy()
        q = NN[w.window_id][h - 1, link]
        if args.dump:
            DUMP[w.window_id] = {"p": np.asarray(p, float), "q": q, "h": df.h.to_numpy(), "link": df.link.to_numpy()}
        mix = (1 - args.w_ongoing) * p + args.w_ongoing * q
        gate_c, gate_wide = args.gate_clim, args.gate_wide
        if args.gate_adaptive:
            panel = w.window_id[3:].rsplit("_", 2)[0]
            info = adaptive_gate.window_gate(panel, "validation" if "_validation_" in w.window_id else "private",
                                             arrays[w.window_id]["t"], gate_tables(panel)[1])
            gate_c, gate_wide = info["c"], info["wide"]
            GLOG[w.window_id] = info
        if gate_c is not None:
            vcut, clim = gate_tables(w.window_id[3:].rsplit("_", 2)[0])
            a = arrays[w.window_id]; sp = a["speed"]
            row = np.where(np.isnan(sp[12]), sp[11], sp[12])
            free = ~(np.nan_to_num(row, nan=999) <= vcut)
            t = a["t"] + h.astype(np.int64); dw = ((t // 288 + F1.T0.day_of_week) % 7 >= 5).astype(int)
            g = free[link] & (clim[dw, t % 288, link] >= gate_c)
            if gate_wide:
                keep = (np.asarray(p) >= th) & (q >= th)
                STATS["gated"] = STATS.get("gated", 0) + int((g & ((mix >= th) != keep)).sum())
                mix = np.where(g, np.where(keep, 1.0, 0.0), mix)
            else:
                block = g & (mix >= th) & (np.asarray(p) < th)
                STATS["gated"] = STATS.get("gated", 0) + int(block.sum())
                mix = np.where(block, p, mix)
        return mix

    src = textwrap.dedent(inspect.getsource(build.predict_split_v1))
    for old, new in (("pred = best_k_set(p)", "pred = best_k_set(_NN_ON(w, p))"),
                     ('"queue_pred": (p >= th).astype(int)', '"queue_pred": (_NN_GO(w, df, p, th) >= th).astype(int)')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    build._NN_ON, build._NN_GO = nn_on, nn_go
    exec(compile(src, build.__file__, "exec"), build.__dict__)

    if args.trunc:
        import trunc_test
        orig = trunc_test.release_windows_trunc

        def checked(panel, split, st):
            for w, arr, t_idx in orig(panel, split, st):
                a = arrays[w.window_id]
                assert a["t"] == t_idx
                for k in ("speed", "flow", "occ", "pct"):
                    assert np.array_equal(a[k], arr[k], equal_nan=k != "pct"), (w.window_id, k)
                STATS["arrays_checked"] += 1
                yield w, arr, t_idx
        F1.release_windows = checked
        build.online_source = trunc_test.online_source_trunc

        def no_cache(*a, **k):
            raise AssertionError("split-level cache read in the truncated run")
        build.split_rate_source = no_cache
    sys.argv = ["build.py", "--split", "all", "--onset-variant", "v1", "--ongoing-variant", "v1"] + rest + \
               ["--out", str(args.out), "--overwrite"]
    build.main()
    print("hooked windows", STATS)
    if args.dump:
        pickle.dump(DUMP, open(args.dump, "wb"))
    if args.gate_log:
        import json
        json.dump(GLOG, open(args.gate_log, "w"), indent=1)
    if args.compare:
        a = pd.read_csv(args.compare); b = pd.read_csv(args.out)
        m = a.merge(b, on=["window_id", "timestamp", "link_id"])
        d = m.queue_pred_x != m.queue_pred_y
        print(f"compare with {args.compare}: cells {len(m):,}, different {int(d.sum())}, windows {m[d].window_id.nunique()}"
              + (f", rows after T dropped {trunc_test_stats()}" if args.trunc else ""))


def trunc_test_stats():
    import trunc_test
    return f"{trunc_test.STATS['rows_after_T_dropped']:,} over {trunc_test.STATS['windows']} windows"


if __name__ == "__main__":
    main()
