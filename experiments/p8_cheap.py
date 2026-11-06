"""P8, the cheap reading: does a pair's sharing fall only after the rarer construct's loss settles?

**Exploratory throughout, and the marks are not decoration.** No intervals: every figure is a
point estimate, because curve 5's intervals need the 14,196 pairs' per-core arrays at **both**
twins and the tree carries them for twin 1 only (A41's structural limit). The population is the
**16 position-marked constructs** -- 120 pairs -- because the loss side is read at marked obliged
and dependent positions and only those 16 carry any (A43). And P8's demand condition is read on
**timing alone**: whether the sharing fall begins before or after the loss stops falling, with no
claim that either causes the other.

**The four definitions this reading needs, each a choice CC made and names.**

1. **A pair's sharing falls** if there is a reference after which its cross-twin sharing stays
   below its 4.19 B level by more than one **point-to-point unit** -- the median absolute
   step-to-step change of that pair's own series. The first such reference is the **fall's
   beginning**, and the total fall is reported in those units.
2. **The rarer construct** of a pair is the one with the lower **role frequency per file**, which
   is the quantity P2 and P3 are read on. Where a rate is missing the pair is reported unreadable
   rather than guessed.
3. **The rarer construct's loss** is its **dependent** series where it has one -- the 4 constructs
   whose role gap clears C0.14 -- and its **forced** (obliged) series otherwise. Where a construct
   has several forced roles, the one with the largest remaining drop at 4.19 B is taken, that being
   the series with the most room to fall, and the choice is recorded per pair.
4. **The loss stops falling** at the first reference from which the series stays within one of its
   own noise units of its final value, to the end. For a series flat from the first reference this
   is reference 0, which is the case step 4's leg analysis reports as "flat from the start".

**What the comparison can and cannot say.** It is a comparison of two **checkpoint indices** on a
35-point grid whose spacing is 1,000 steps in the dense block and up to 91,000 in the tail, so a
difference of one index is not a fixed amount of training. Differences are therefore reported in
**checkpoints and in decades of tokens**, and no pair is called simultaneous on the strength of a
single index.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.r2_curve4 import read_positions, series as pos_series
from experiments.step8_graph import membership, recorded, sets_at
from profilebattery.profiles.names import contract_name, tree_name
from profilebattery.paths import data_root, dataset_repo

D = data_root()
R2 = D / "validation/r2"
STEP4_DL = D / "_step4_dl"
K = 7373
#: a settle point needs this many points after it, or it is vacuous (A38's refusal)
MIN_AFTER = 2


def sharing_series(seq, constructs) -> tuple:
    """`(pairs, (refs x pairs) array)` of cross-twin sharing, curve 5's own quantity."""
    refs = [r for r, _ in seq]
    rows = []
    for ref in refs:
        a = membership(sets_at(ref, constructs, twin="twin1"), constructs).astype(np.float32)
        b = membership(sets_at(ref, constructs, twin="twin2"), constructs).astype(np.float32)
        rows.append(a @ b.T / K)
    A = np.stack(rows)
    iu = np.triu_indices(len(constructs), 1)
    pairs = [(constructs[i], constructs[j]) for i, j in zip(*iu)]
    return pairs, A[:, iu[0], iu[1]]


def fall(series: np.ndarray) -> dict:
    """Where a pair's sharing begins to stay below its first level, and by how much.

    The unit is the pair's own median absolute point-to-point change, so a pair whose series is
    noisy needs a larger fall to count -- which is the point of using its own noise rather than a
    pooled one.
    """
    v = np.asarray(series, float)
    d = np.abs(np.diff(v))
    unit = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else np.nan
    if not np.isfinite(unit) or unit == 0:
        return {"outcome": "no_unit", "why": "the series has no point-to-point movement"}
    below = v < v[0] - unit
    begin = None
    for i in range(len(v)):
        if below[i:].all():
            begin = i
            break
    total = (v[0] - v[-1]) / unit
    if begin is None:
        return {"outcome": "rose" if v[-1] > v[0] + unit else "did_not_move",
                "unit": unit, "total_change_in_units": float(total),
                "level_first": float(v[0]), "level_last": float(v[-1])}
    return {"outcome": "fell", "begin_index": int(begin), "unit": unit,
            "total_fall_in_units": float(total), "level_first": float(v[0]),
            "level_last": float(v[-1])}


def loss_series_for(construct: str, per_ref: dict, refs, s4: dict) -> dict:
    """The rarer construct's loss series: dependent where it has one, forced otherwise."""
    dep = s4["dependent"]
    for cls in ("governed", "scope"):
        for key, e in (dep.get(cls) or {}).get("series", {}).items():
            if key.split("|")[0] == construct and e.get("counts_under_c0_14") is True:
                vals, units, ok = pos_series(per_ref, refs, tuple(key.split("|")[:3]))
                if ok and len(vals) == len(refs):
                    return {"which": "dependent", "key": key, "values": vals, "units": units}
    best = None
    for key, e in s4["forced"]["series"].items():
        if key.split("|")[0] != construct or "remaining_drop_at_early" not in e:
            continue
        if best is None or e["remaining_drop_at_early"] > best[1]["remaining_drop_at_early"]:
            best = (key, e)
    if best is None:
        return {"which": None, "why": "no forced or dependent series for this construct"}
    vals, units, ok = pos_series(per_ref, refs, tuple(best[0].split("|")[:3]))
    if not ok or len(vals) != len(refs):
        return {"which": None, "why": "the series does not cover every reference"}
    return {"which": "forced", "key": best[0], "values": vals, "units": units,
            "chosen_by": "the largest remaining drop at 4.19 B among this construct's forced roles"}


def stopped_falling(values, units) -> dict:
    """The first reference from which the series stays within one unit of its final value."""
    v = np.asarray(values, float)
    u = [x for x in (units or []) if x is not None and np.isfinite(x) and x > 0]
    unit = float(np.median(u)) if u else float(np.median(np.abs(np.diff(v))))
    if not np.isfinite(unit) or unit == 0:
        return {"index": None, "why": "no usable noise unit"}
    # **A settle point at the final index is free, and A38 is why that matters.** The slice from
    # the last point contains only the last point, so every series "stays within one unit of its
    # final value" there. A38 refused a stability date at the last point of its series for exactly
    # this reason; the same refusal applies here, and it removes `Try`, whose first qualifying
    # index was 34 of 34.
    for i in range(max(0, len(v) - MIN_AFTER)):
        if np.all(np.abs(v[i:] - v[-1]) <= unit):
            return {"index": int(i), "unit": unit,
                    "flat_from_the_start": i == 0,
                    "n_after": int(len(v) - 1 - i),
                    "level_there": float(v[i]), "level_last": float(v[-1])}
    return {"index": None,
            "why": f"the series never stays within one unit of its final value with at least "
                   f"{MIN_AFTER} points after it; a settle point at the last index is vacuous"}


def demand_magnitude(rows: dict, loss_legs: dict, draws: int = 10_000,
                     seed: int = 20261016) -> dict:
    """Does a pair's sharing fall by more when the rarer construct's loss drops by more?

    **Two legs, the second a control.** Over **4.19 -> 83.9 B** the loss is falling (step 4
    measured the obliged class falling for 15 of 24 there); over **83.9 B -> the end** it is flat
    (flat for 20 of 24, falling for 0). So a demand account predicts a relation on the first leg
    and none on the second, and the second leg is reported as the control rather than as a second
    test.

    **The sign convention.** Both quantities are signed changes, so a pair whose sharing falls and
    whose rarer construct's loss drops has two negative numbers: a demand relation is a
    **positive** rank correlation between them.

    **The effective sample is 16, not 87, and both are reported.** The 87 pairs share 16 rarer
    constructs, so the loss column repeats and a permutation over pairs understates its own null.
    The per-construct form -- each rarer construct once, its sharing change averaged over its
    pairs -- is the honest n, and it is given beside the per-pair form rather than instead of it.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    out = {"note": "point estimates, exploratory; leg 2 is the control, the loss being flat there",
           "sign_convention": "both are signed changes, so a demand relation is POSITIVE",
           "legs": {}}
    for leg in ("leg1_4p19_to_83p9B", "leg2_83p9B_to_end"):
        xs, ys, who = [], [], []
        for k, e in rows.items():
            if e["sharing"]["outcome"] != "fell":
                continue
            ds = e.get("sharing_change", {}).get(leg)
            dl = (loss_legs.get(e["rarer"]) or {}).get(leg)
            if ds is None or dl is None:
                continue
            xs.append(dl)
            ys.append(ds)
            who.append(e["rarer"])
        if len(xs) < 10:
            out["legs"][leg] = {"n": len(xs), "spearman": None,
                                "why": "fewer than ten pairs carry both changes"}
            continue
        x, y = np.array(xs), np.array(ys)
        rho = float(spearmanr(x, y).statistic)
        null = []
        idx = np.arange(x.size)
        for _ in range(draws):
            rng.shuffle(idx)
            null.append(spearmanr(x, y[idx]).statistic)
        null = np.array(null)
        # the per-construct form: each rarer construct once, its pairs averaged
        agg = collections.defaultdict(list)
        for c, yy in zip(who, y):
            agg[c].append(yy)
        cx = np.array([loss_legs[c][leg] for c in agg])
        cy = np.array([np.mean(v) for v in agg.values()])
        crho = float(spearmanr(cx, cy).statistic) if len(cx) >= 4 else None
        cnull = []
        if crho is not None:
            ci = np.arange(cx.size)
            for _ in range(draws):
                rng.shuffle(ci)
                cnull.append(spearmanr(cx, cy[ci]).statistic)
            cnull = np.array(cnull)
        out["legs"][leg] = {
            "n_pairs": int(x.size), "n_rarer_constructs": len(agg),
            "spearman_per_pair": rho,
            "permutation_interval_per_pair": [float(np.percentile(null, 2.5)),
                                              float(np.percentile(null, 97.5))],
            "permutation_p_per_pair": float((np.abs(null) >= abs(rho)).mean()),
            "spearman_per_construct": crho,
            "permutation_interval_per_construct": (
                [float(np.percentile(cnull, 2.5)), float(np.percentile(cnull, 97.5))]
                if crho is not None else None),
            "permutation_p_per_construct": (float((np.abs(cnull) >= abs(crho)).mean())
                                            if crho is not None else None),
            "median_loss_change": float(np.median(x)),
            "median_sharing_change": float(np.median(y)),
            "which_n_to_believe": "the per-construct one: the per-pair null treats 87 rows as "
                                  "independent when the loss column has 16 distinct values"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()

    s4 = json.loads((R2 / "STEP4_FLAT_pretrain.json").read_text())
    seq = recorded("pretrain")
    refs, toks = [r for r, _ in seq], [t for _, t in seq]
    # the 16 position-marked constructs: step 4's forced and dependent classes reach exactly these
    sixteen = sorted({k.split("|")[0] for k in s4["forced"]["series"]}
                     | {k.split("|")[0] for cls in ("governed", "scope")
                        for k in (s4["dependent"].get(cls) or {}).get("series", {})})
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {c: (({**ast, **bi}).get(contract_name(c)) or {}).get("per_file") for c in sixteen}
    print(f"the position-marked population: {len(sixteen)} constructs, "
          f"{len(sixteen)*(len(sixteen)-1)//2} pairs")
    print(f"  {sixteen}")

    per_ref = {}
    for r in refs:
        p = STEP4_DL / f"record/{r}/POSITIONS.json"
        if p.exists():
            per_ref[r] = read_positions(json.load(open(p)))
    rr = [r for r in refs if r in per_ref]
    print(f"  POSITIONS.json present for {len(rr)} of {len(refs)} references")

    pairs, S = sharing_series(seq, sixteen)
    i84 = int(np.argmin([abs(t - 84e9) for t in toks]))
    print(f"  the 83.9 B anchor is index {i84} ({refs[i84]})")
    rep = {"exploratory": "no intervals; the 16 position-marked constructs only; P8's demand "
                          "condition read on timing alone",
           "n_constructs": len(sixteen), "n_pairs": len(pairs), "constructs": sixteen,
           "role_frequency_per_file": rates, "references": refs, "tokens": toks}

    # ---- the loss side, once per construct ----
    loss = {}
    for c in sixteen:
        ls = loss_series_for(c, per_ref, rr, s4)
        if ls.get("which") is None:
            loss[c] = ls
            continue
        st = stopped_falling(ls["values"], ls["units"])
        loss[c] = {"which": ls["which"], "key": ls["key"], "stopped": st,
                   "chosen_by": ls.get("chosen_by"),
                   "level_first": float(ls["values"][0]), "level_last": float(ls["values"][-1])}
    loss_legs = {}
    for c in sixteen:
        ls = loss_series_for(c, per_ref, rr, s4)
        if ls.get("which") is None:
            continue
        v = np.asarray(ls["values"], float)
        j = int(np.argmin([abs(toks[refs.index(r)] - 84e9) for r in rr]))
        loss_legs[c] = {"leg1_4p19_to_83p9B": float(v[j] - v[0]),
                        "leg2_83p9B_to_end": float(v[-1] - v[j]),
                        "which": ls["which"]}
    rep["loss_legs"] = loss_legs
    rep["loss_side"] = loss
    nd = sum(1 for v in loss.values() if v.get("which") == "dependent")
    print(f"\nloss side: {nd} constructs read on a dependent series, "
          f"{sum(1 for v in loss.values() if v.get('which') == 'forced')} on a forced one, "
          f"{sum(1 for v in loss.values() if v.get('which') is None)} unreadable")
    for c, v in sorted(loss.items()):
        if v.get("which") is None:
            print(f"   {c:14s} UNREADABLE: {v['why'][:48]}")
            continue
        i = v["stopped"].get("index")
        print(f"   {c:14s} {v['which']:9s} {v['key'].split('|', 1)[1]:22s} stopped at "
              + (f"index {i:2d} ({rr[i] if i is not None else '--'}, "
                 f"{toks[i]/1e9:7.1f} B)" if i is not None else "NEVER")
              + ("  flat from the start" if v["stopped"].get("flat_from_the_start") else ""))

    # ---- the sharing side and the comparison ----
    rows, cases = {}, collections.Counter()
    for (x, y), s in zip(pairs, S.T):
        f = fall(s)
        rarer = (x if (rates.get(x) or np.inf) < (rates.get(y) or np.inf) else y)
        e = {"pair": f"{x}|{y}", "rarer": rarer,
             "rarer_frequency": rates.get(rarer), "sharing": f,
             "sharing_change": {"leg1_4p19_to_83p9B": float(s[i84] - s[0]),
                                "leg2_83p9B_to_end": float(s[-1] - s[i84])}}
        lv = loss.get(rarer) or {}
        e["loss_which"] = lv.get("which")
        e["loss_stopped_index"] = (lv.get("stopped") or {}).get("index")
        if f["outcome"] == "fell" and e["loss_stopped_index"] is not None:
            d = f["begin_index"] - e["loss_stopped_index"]
            e["fall_begins_minus_loss_stopped_checkpoints"] = int(d)
            e["in_decades"] = float(np.log10(toks[f["begin_index"]]
                                             / toks[e["loss_stopped_index"]]))
            e["case"] = "before" if d < 0 else ("at" if d == 0 else "after")
            e["loss_still_falling_when_the_fall_began"] = bool(d < 0)
            cases[e["case"]] += 1
        elif f["outcome"] == "fell":
            e["case"] = "loss_unreadable"
            cases["loss_unreadable"] += 1
        else:
            e["case"] = f["outcome"]
            cases[f["outcome"]] += 1
        rows[e["pair"]] = e
    rep["pairs"] = rows
    rep["case_counts"] = dict(cases)
    fell = [e for e in rows.values() if e["sharing"]["outcome"] == "fell"]
    print(f"\nsharing side: {len(fell)} of {len(pairs)} pairs fell beyond their own noise; "
          f"cases {dict(cases)}")
    if fell:
        tf = [e["sharing"]["total_fall_in_units"] for e in fell]
        bi_ = [e["sharing"]["begin_index"] for e in fell]
        print(f"  total fall: median {np.median(tf):.1f} units, range "
              f"{min(tf):.1f}-{max(tf):.1f}")
        print(f"  the fall begins: median index {int(np.median(bi_))} "
              f"({rr[int(np.median(bi_))] if int(np.median(bi_)) < len(rr) else '--'}), "
              f"range {min(bi_)}-{max(bi_)}")
    comp = [e for e in fell if e.get("case") in ("before", "at", "after")]
    pred = [e for e in comp if e["loss_still_falling_when_the_fall_began"]]
    print(f"\nthe comparison, on {len(comp)} readable pairs:")
    for case in ("before", "at", "after"):
        n = sum(1 for e in comp if e["case"] == case)
        print(f"   the fall begins {case:6s} the loss stopped falling: {n:3d}")
    print(f"  P8's predicted case -- the loss still falling when the fall began: "
          f"{len(pred)} of {len(comp)}")
    if pred:
        print(f"    e.g. {[e['pair'] for e in pred[:4]]}")
    begins = collections.Counter(e["sharing"]["begin_index"] for e in comp)
    stops = collections.Counter(e["loss_stopped_index"] for e in comp)
    rep["predicted_case"] = {
        "n": len(pred), "of": len(comp),
        "pairs": [e["pair"] for e in pred],
        "definition": "the sharing fall begins at an earlier checkpoint than the one from which "
                      "the rarer construct's loss stays within one unit of its final value",
        "begin_index_counts": dict(sorted(begins.items())),
        "loss_stopped_index_counts": dict(sorted(stops.items())),
        "how_much_the_marginals_decide_it": "the test is close to determined by the two marginal "
                                            "distributions: most sharing falls begin at the first "
                                            "available checkpoint while most of these constructs' "
                                            "losses stop very late, so 'before' is nearly "
                                            "automatic. The informative part is the exceptions, "
                                            "not the majority."}

    # ---- the demand-magnitude reading, beside the ordering ----
    rep["demand_magnitude"] = demand_magnitude(rows, loss_legs)
    print("\nthe demand-magnitude reading (exploratory, beside the ordering):")
    for leg, v in rep["demand_magnitude"]["legs"].items():
        if v.get("spearman_per_pair") is None:
            print(f"   {leg:22s} {v['why']}")
            continue
        print(f"   {leg:22s} per pair      rho {v['spearman_per_pair']:+.3f} "
              f"(n={v['n_pairs']}, perm p={v['permutation_p_per_pair']:.4f}, interval "
              f"[{v['permutation_interval_per_pair'][0]:+.3f},"
              f"{v['permutation_interval_per_pair'][1]:+.3f}])")
        print(f"   {'':22s} per construct rho {v['spearman_per_construct']:+.3f} "
              f"(n={v['n_rarer_constructs']}, perm p={v['permutation_p_per_construct']:.4f}, "
              f"interval [{v['permutation_interval_per_construct'][0]:+.3f},"
              f"{v['permutation_interval_per_construct'][1]:+.3f}])")
        print(f"   {'':22s} median loss change {v['median_loss_change']:+.4f}, "
              f"median sharing change {v['median_sharing_change']:+.4f}")

    # ---- the pairwise bootstrap's cost, measured ----
    rep["bootstrap_estimate"] = bootstrap_estimate(len(pairs), len(rr))
    b = rep["bootstrap_estimate"]
    print(f"\nthe pairwise bootstrap, if it is wanted:")
    for k, v in b.items():
        if not isinstance(v, dict):
            print(f"   {k}: {v}")

    out = pathlib.Path(a.out) / "P8_CHEAP_pretrain.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


def bootstrap_estimate(n_pairs: int, n_refs: int) -> dict:
    """What a pairwise bootstrap would cost, from the sizes and a measured read rate."""
    import time

    from huggingface_hub import HfFileSystem, get_token
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or get_token())
    rate = None
    try:
        p = (f"datasets/{dataset_repo()}/record/pretrain_iter_0020000/"
             "twin2__For_A_acc.npz")
        size = fs.info(p)["size"]
        t = time.time()
        with fs.open(p, "rb") as f:
            raw = f.read()
        dt = time.time() - t
        rate = {"file": p.split("/")[-1], "bytes": int(size), "seconds": dt,
                "MB_per_s": size / 1e6 / dt, "n_read": len(raw)}
    except Exception as e:                                       # noqa: BLE001
        rate = {"error": f"{type(e).__name__}: {e}"}
    # twin 2 has no per-core arrays in the tree, so the bootstrap reads the record's accumulators
    per_set_mb = 23.5 + 47.8                                     # A_acc + B_acc, measured means
    n_constructs = 16
    gb = per_set_mb * n_constructs * n_refs / 1024
    hours = (gb * 1024 / rate["MB_per_s"] / 3600) if rate.get("MB_per_s") else None
    out_mb = n_pairs * n_refs * 5 * 8 / 1e6
    return {"why_the_record_and_not_the_tree": "twin 2's npz carries concept, concept_se and "
                                              "concept_indented and no per-core arrays, so a "
                                              "pairwise interval cannot be drawn from the tree",
            "measured_read": rate,
            "bytes_to_read_GB": gb,
            "n_accumulator_files": n_constructs * n_refs * 2,
            "projected_hours_single_stream": hours,
            "projected_hours_at_12_workers_IF_bandwidth_allowed": hours / 12 if hours else None,
            "parallel_caveat": "the 12-worker figure is naive scaling and will not be delivered: "
                               "these are 35-40 MB files, so the pass is bandwidth-bound rather "
                               "than latency-bound and workers divide one pipe. The single-stream "
                               "figure is the one to plan against; the earlier ranged-read passes "
                               "parallelised well because they read 0.6-2.4 MB members, where "
                               "latency dominates.",
            "output_MB_quantiles_only": out_mb,
            "output_is": "2.5/50/97.5 quantiles, mean and SE per pair per reference -- five "
                         "floats, no draws stored",
            "scope": f"the {n_constructs} position-marked constructs at {n_refs} references; all "
                     f"169 would be {per_set_mb * 169 * n_refs / 1024:.0f} GB"}


if __name__ == "__main__":
    main()
