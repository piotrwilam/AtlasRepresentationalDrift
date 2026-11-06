"""The sharing bootstrap at the 16-construct scope: cross-twin sharing with core resampling.

**What it computes.** For the **120** pairs among the position-marked constructs, at all **35**
pretraining references, the cross-twin sharing `|S_a(twin1) n S_b(twin2)| / 7373` under
**1,000** core-bootstrap draws -- the number curve 1 used (`r2_bootstrap.DRAWS`). Output is
**quantiles only**: the 2.5/50/97.5 points, the mean and the SE per pair per reference, and no draw
is stored. From the per-draw series it also gives, per fallen pair, the **interval on the
checkpoint where the fall begins** and on its **total size**, beside A46's point estimates.

**C0.5, which decides the whole design.** One core-index set a draw, drawn once over a construct's
core **names** and applied to **both twins at every reference**, so a draw resamples cores and not
checkpoints. The names are checked to be the same set at every reference and in both twins before
any draw is made; a construct whose names differ is refused rather than resampled by position.

**Why twin 2 comes from the record and twin 1 from the tree.** The derived tree carries per-core
arrays for twin 1 only -- a twin-2 npz holds `concept`, `concept_se` and `concept_indented` -- so
twin 2's per-core means are rebuilt from the record's **per-cell accumulators** by the same route
`compute_profiles` uses (`from_accumulators(..., stream="h")`), against that twin's own
`twin2__C_cells.npz` yardstick. Twin 1 is read from the tree because 17 MB of per-core arrays is
cheaper than 71 MB of accumulators.

**The read budget, and CC's earlier projection was low.** A46 projected **39.0 GB and 0.49 h** at a
measured 15.9 MB/s, counting twin 2's accumulators alone. It omitted two things this pass cannot do
without: twin 1's per-core arrays from the tree (16 x 35 x 17 MB = **9.5 GB**) and twin 2's
yardstick (35 x 144 MB = **5.0 GB**). The budget is therefore **53.5 GB and about 0.93 h**, and the
measured figure is reported against both numbers.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib
import time

import numpy as np

from experiments.p8_cheap import fall
from experiments.r2_bootstrap import DRAWS, K as K_FRACTION, SEED

#: `r2_bootstrap.K` is the **fraction** 0.05, not the count. Importing it as a count made
#: `np.partition` refuse an index of 147,455.95, which is the good kind of failure: the division
#: that follows would otherwise have produced sharing values 147,000 times too large in silence.
K = int(round(K_FRACTION * 24 * 6144))
from experiments.step8_graph import recorded
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root, dataset_repo

D = data_root()
R2 = D / "validation/r2"
OUT = R2 / "p8_bootstrap"
REPO = dataset_repo()
TREE = D / "profiles"
N_UNITS = 24 * 6144


def twin1_cores(ref: str, c: str, token: str) -> tuple:
    """`(A_cores, B_cores)` for twin 1, from the tree, locally or by a ranged member read."""
    # **Dataset only** (A52). This read the local tree first, and that tree holds the PILOT
    # recording at the phase recording's path -- so eight of this pass's thirty-five references
    # were the pilot's, and they are the early ones, where the sharing fall begins. The seeds
    # and every other parameter are unchanged, so the recompute differs from the original in
    # the recording it reads and in nothing else.
    from profilebattery.profiles import resolve
    z = resolve.profile(ref, "twin1", c, ("A_cores", "B_cores"))
    return np.asarray(z["A_cores"], np.float32), np.asarray(z["B_cores"], np.float32)


def twin2_cores(ref: str, c: str, store, ys) -> tuple:
    """Twin 2's per-core means, rebuilt from the record's per-cell accumulators."""
    from experiments.compute_profiles import load_set, make
    out = []
    for kind in ("A", "B"):
        pa, rows, acc = load_set(ref, "twin2", c, kind, store, True)
        prof = make(pa, rows, acc, ys, c, kind)
        if prof is None:
            return None, None
        names = sorted(prof.core_means)
        out.append((names, np.stack([prof.core_means[k] for k in names]).astype(np.float32)))
    return out[0], out[1]


def draw_indices(n_a: int, n_b1: int, n_b2: int, draws: int, seed: int) -> dict:
    """One index set a draw for **A**, applied to both twins; **per twin** for B, and why.

    **C0.5 is satisfiable for A and not for B, and the record says why.** The A sets of the two
    twins share all eight cores, so index 3 names the same core in both and one index set is
    meaningful. The **B** sets do not: `compute_profiles` records that "A and B have their own core
    sets", each twin's B being built from its own prompts with its own sibling groups dropped, and
    the counts bear it out -- `AsyncFor` has **21** B cores in twin 1 and **20** in twin 2
    (measured, 16 Oct). Index 7 in one twin's B is not the same core as index 7 in the other's, so
    applying a single index set across twins would not be C0.5's resampling; it would be a
    coincidence of positions. B is therefore drawn **per twin** from the same generator, so the
    draws are paired in order but each resamples its own cores, and the departure is reported.
    """
    rng = np.random.default_rng(seed)
    out = {"a": [], "b1": [], "b2": [], "b_shared": n_b1 == n_b2}
    for _ in range(draws):
        out["a"].append(rng.integers(0, n_a, n_a))
        out["b1"].append(rng.integers(0, n_b1, n_b1))
        out["b2"].append(rng.integers(0, n_b2, n_b2))
    return out


def masks_for(a: np.ndarray, b: np.ndarray, ia_list, ib_list, k: int = K) -> np.ndarray:
    """`(draws, N_UNITS)` boolean carrier masks under each draw's core resampling."""
    af = a.reshape(a.shape[0], -1).astype(np.float64)
    bf = b.reshape(b.shape[0], -1).astype(np.float64)
    out = np.zeros((len(ia_list), af.shape[1]), bool)
    for d, (ia, ib) in enumerate(zip(ia_list, ib_list)):
        v = af[ia].mean(axis=0) - bf[ib].mean(axis=0)
        f = v * v
        thr = np.partition(f, f.size - k)[f.size - k]
        m = f >= thr
        if m.sum() > k:                                   # ties at the threshold, trimmed by index
            on = np.where(m)[0]
            m[on[k:]] = False
        out[d] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit-refs", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    s4 = json.loads((R2 / "STEP4_FLAT_pretrain.json").read_text())
    sixteen = sorted({k.split("|")[0] for k in s4["forced"]["series"]}
                     | {k.split("|")[0] for cls in ("governed", "scope")
                        for k in (s4["dependent"].get(cls) or {}).get("series", {})})
    pairs = list(itertools.combinations(sixteen, 2))
    seq = recorded("pretrain")
    if a.limit_refs:
        seq = seq[:a.limit_refs]
    refs, toks = [r for r, _ in seq], [t for _, t in seq]
    token = os.environ.get("HF_TOKEN")
    if token is None:
        from huggingface_hub import get_token
        token = get_token()
    print(f"{len(sixteen)} constructs, {len(pairs)} pairs, {len(refs)} references, "
          f"{a.draws} draws", flush=True)

    from profilebattery.record.remote import RecordStore, quiet_hub
    from profilebattery.profiles.yardstick import Yardstick
    quiet_hub()

    read_s = comp_s = 0.0
    read_bytes = 0
    idx_by_c, names_by_c = {}, {}
    cube = np.full((len(refs), a.draws, len(pairs)), np.nan, np.float32)
    for ri, ref in enumerate(refs):
        fp = out / f"{ref}.npz"
        if fp.exists():
            with np.load(fp) as z:
                cube[ri] = z["sharing"]
            print(f"  {ref}: resumed", flush=True)
            continue
        t0 = time.time()
        store = RecordStore(D / "record", ref, REPO, token)
        yp = store.path("twin2__C_cells.npz")
        if yp is None:
            print(f"  {ref}: no twin2 yardstick -- skipped", flush=True)
            continue
        ys2 = Yardstick.from_npz(str(yp))
        cores = {}
        for c in sixteen:
            a1, b1 = twin1_cores(ref, c, token)
            (na2, a2), (nb2, b2) = twin2_cores(ref, c, store, ys2)
            if a2 is None:
                continue
            cores[c] = (a1, b1, a2, b2)
            shape = (a1.shape[0], b1.shape[0], a2.shape[0], b2.shape[0])
            names_by_c.setdefault(c, {})[ref] = shape
            if a1.shape[0] != a2.shape[0]:
                raise SystemExit(f"{c}: the two twins' A sets differ in size "
                                 f"({a1.shape[0]} vs {a2.shape[0]}), so C0.5's shared index set "
                                 f"is not applicable to A either -- refusing")
            # **B's core partition is not constant across references, and the draw is per
            # reference because of it.** Measured 16 Oct: `For`'s B set has 25 token-site cores at
            # some references and 21 at others, for the **same 720 prompts** -- the same prompts
            # partitioned into different numbers of cores. The derived tree is uniform (n_a = 720
            # and identical cell counts at all 35 references for all 188 constructs), and the
            # token-site prompt count is 17,280 everywhere, so this is not a record defect: it is
            # a difference in how B's prompts are grouped into resampling units. C0.5's "one index
            # set applied at every reference" is therefore not meetable for B, and the draw is
            # seeded by (construct, reference) -- which is also what `r2_bootstrap` does for curves
            # 2, 3 and 6, so these intervals stay comparable with those.
            seed_c = (a.seed + abs(hash((c, ref))) % 10_000_019)
            idx_by_c[(c, ref)] = draw_indices(a1.shape[0], b1.shape[0], b2.shape[0],
                                              a.draws, seed_c)
        read_s += time.time() - t0
        read_bytes += int(getattr(store, "fetched_bytes", 0))
        t1 = time.time()
        m1, m2 = {}, {}
        for c, (a1, b1, a2, b2) in cores.items():
            ix = idx_by_c[(c, ref)]
            m1[c] = masks_for(a1, b1, ix["a"], ix["b1"])
            m2[c] = masks_for(a2, b2, ix["a"], ix["b2"])
        for pi, (x, y) in enumerate(pairs):
            if x in m1 and y in m2:
                cube[ri, :, pi] = (m1[x] & m2[y]).sum(axis=1) / K
        comp_s += time.time() - t1
        np.savez_compressed(fp, sharing=cube[ri].astype(np.float32))
        print(f"  {ref}: {len(cores)} constructs, read {time.time()-t0-(time.time()-t1):.0f}s, "
              f"compute {time.time()-t1:.0f}s", flush=True)

    # ---- quantiles only ----
    q = np.nanpercentile(cube, [2.5, 50, 97.5], axis=1)
    rep = {"n_constructs": len(sixteen), "constructs": sixteen, "n_pairs": len(pairs),
           "references": refs, "draws": a.draws, "seed": a.seed,
           "quantity": "|S_a(twin1) n S_b(twin2)| / 7373, product space, per draw",
           "c0_5": "one core-index set a draw for A, applied to both twins at every reference; "
                   "B drawn PER TWIN from the same generator, because each twin's B has its own "
                   "core set -- AsyncFor has 21 B cores in twin 1 and 20 in twin 2 -- so one "
                   "index set across twins would be a coincidence of positions and not a "
                   "resampling of cores",
           "c0_5_departure": {
               "what": "B's core partition is not constant across references -- For has 25 "
                       "token-site B cores at some and 21 at others for the same 720 prompts -- "
                       "so one index set cannot be applied at every reference. The draw is "
                       "therefore seeded by (construct, reference), which is what r2_bootstrap "
                       "does for curves 2, 3 and 6, so these intervals are comparable with "
                       "those.",
               "not_a_record_defect": "the token-site prompt count is 17,280 = 720 x 24 at every "
                       "reference, and the derived tree is uniform: n_a = 720 with identical cell "
                       "counts for all 188 constructs at all 35 references. What differs is how "
                       "B's prompts are grouped into cores, and which read sites a reference "
                       "recorded -- some carry token, span_end and last, others token alone, and "
                       "every reading uses the token site.",
               "a_is_shared": "A has 8 cores in both twins at every reference, so A's index set "
                              "is shared across the twins as C0.5 asks",
               "core_shapes_per_construct": {c: {r: list(v) for r, v in d.items()}
                                             for c, d in names_by_c.items()}},
           "stored": "quantiles, mean and SE only -- no draws",
           "read_seconds": read_s, "compute_seconds": comp_s,
           "projection_in_a46_hours": 0.49,
           "budget_corrected_GB": 53.5,
           "pairs": {}}
    for pi, (x, y) in enumerate(pairs):
        rep["pairs"][f"{x}|{y}"] = {
            "lo": [float(v) for v in q[0, :, pi]], "median": [float(v) for v in q[1, :, pi]],
            "hi": [float(v) for v in q[2, :, pi]],
            "mean": [float(v) for v in np.nanmean(cube[:, :, pi], axis=1)],
            "se": [float(v) for v in np.nanstd(cube[:, :, pi], axis=1, ddof=1)]}

    # ---- the fall's beginning and size, per draw -> an interval ----
    begins, sizes = collections.defaultdict(list), collections.defaultdict(list)
    for pi, (x, y) in enumerate(pairs):
        for d in range(a.draws):
            s = cube[:, d, pi]
            if np.isnan(s).any():
                continue
            f = fall(s)
            if f["outcome"] == "fell":
                begins[pi].append(f["begin_index"])
                sizes[pi].append(f["total_fall_in_units"])
    falls = {}
    for pi, (x, y) in enumerate(pairs):
        b, z = begins.get(pi, []), sizes.get(pi, [])
        if len(b) < 2:
            falls[f"{x}|{y}"] = {"n_draws_fell": len(b), "outcome": "not a fall in the draws"}
            continue
        falls[f"{x}|{y}"] = {
            "n_draws_fell": len(b), "share_of_draws_that_fell": len(b) / a.draws,
            "begin_index_median": float(np.median(b)),
            "begin_index_interval": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "begin_tokens_median": float(toks[int(np.median(b))]),
            "total_fall_units_median": float(np.median(z)),
            "total_fall_units_interval": [float(np.percentile(z, 2.5)),
                                          float(np.percentile(z, 97.5))]}
    rep["fall_intervals"] = falls
    fell = [v for v in falls.values() if v.get("n_draws_fell", 0) >= 2]
    # the two summaries the draft quotes ("115 fall in at least two draws and the median
    # pair falls in 88% of them"); the per-pair shares alone do not let a reader check them
    rep["fall_summary"] = {
        "n_pairs": len(falls),
        "n_pairs_fell_in_at_least_two_draws": len(fell),
        "median_share_of_draws_that_fell": (
            float(np.median([v["share_of_draws_that_fell"] for v in fell])) if fell else None),
        "median_is": "over the pairs that fell in at least two draws, not over all pairs"}
    print(f"\n{len(fell)} of {len(pairs)} pairs fall in at least two draws")
    if fell:
        w = [v["begin_index_interval"][1] - v["begin_index_interval"][0] for v in fell]
        sh = [v["share_of_draws_that_fell"] for v in fell]
        print(f"  the fall's beginning: interval width median {np.median(w):.1f} checkpoints, "
              f"range {min(w):.0f}-{max(w):.0f}")
        print(f"  the share of draws in which the pair falls: median {np.median(sh):.3f}, "
              f"min {min(sh):.3f}")
    print(f"\nread {read_s/3600:.2f} h, compute {comp_s/3600:.2f} h, "
          f"total {(read_s+comp_s)/3600:.2f} h against A46's 0.49 h projection "
          f"and the corrected 0.93 h budget")

    p = R2 / "P8_BOOTSTRAP_pretrain.json"
    p.write_text(json.dumps(rep, indent=1, default=float))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
