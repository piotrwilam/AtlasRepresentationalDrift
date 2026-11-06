"""Step 7 -- P4, P5, P7 and P9, in the forms the plan and the record allow.

**P4, the learning-rate bend -- a MEASUREMENT ROW, never a verdict on clause 4.** The plan already
records the limit: the learning rate is **constant at 3e-4 across the whole pretraining branch**,
so no within-branch bend exists to read, and the annealing branch changes the learning rate **and
the data mix at once**, so a bend at the branch point cannot be attributed to the rate. What is
measured instead: **whole-graph rewiring (R12)** -- the total change of all carrier sets per
checkpoint, one curve for the model -- and **every construct's fixed-step turnover across the
branch point**, with the branch point (`pretrain_iter_0373000`) and the three **epoch boundaries**
(about 508, 1,016 and 1,524 B tokens, from four epochs over 508 B unique tokens) marked on every
curve. Any bend common to constructs at an epoch boundary is reported. The confound is named in
the artefact, in the print, and in every summary field.

**P5, rate by frequency.** Turnover level against log role frequency at each **dense** checkpoint,
Spearman per checkpoint with a permutation interval, with and without the pilot's 30. The
prediction is **rate falling with frequency at every checkpoint**; the pilot's **+0.54** at the
last pair is disclosed as the plan requires. The dense block is where a fixed step actually exists
-- A38 measured the tail's spacing to drive the level by itself -- so the reading is confined to
it and the tail is reported separately rather than mixed in.

**P7, strength with loss.** The raw norm's direction between **84 B** and its own date, read
against the **obliged** class's flat leg. P7 is refuted if the raw norm **grows** while the loss is
flat. Step 4 measured the obliged class flat over 84 B -> last for **20 of 24** series with
**falling 0**, so the loss side of the comparison is settled; what this adds is the norm's
direction over the same stretch, per construct, with the count in each direction.

**P9, concentration before settling -- reported UNREADABLE.** The comparison is the concentration
date (curve 6) against the **stability date** (curve 1) on P4, and A38 leaves **no construct with
a stability date** that survives the spacing check, on either branch. So P9 has nothing to compare
and is reported unreadable, with **A34's ordering on the 14 constructs whose three profile dates
are all distinct** beside it -- which is an ordering among *profile* curves and not the
concentration-against-stability comparison P9 names.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import branches
from experiments.r1_turnover import load_index, overlap
from experiments.r2_curves import fixed_partner
from experiments.step3_senses import DRAWS, SEED, spearman
from profilebattery.profiles.names import contract_name, tree_name
from profilebattery.paths import data_root

D = data_root()
INDEX = D / "_carriers_dl/validation/carriers"
R2 = D / "validation/r2"
#: four epochs over 508 billion unique tokens (the plan's arithmetic)
EPOCHS = (508e9, 1016e9, 1524e9)
BRANCH_POINT = "pretrain_iter_0373000"
ANCHOR_84B = 84e9
#: the dense block: the references spaced a fixed 1,000 training steps apart
DENSE_GAP = 1000


def recorded(branch: str) -> list:
    have = {p.stem for p in INDEX.glob("*.npz")}
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    return [(r, t) for r, t in branches(sched)[branch] if r in have]


def dense_indices(refs) -> list:
    """The indices whose fixed partner really is 1,000 steps away.

    A38 measured curve 1's level to fall monotonically with the spacing it is read at, so mixing
    the dense block with the tail mixes two rulers. Only 56% of the pretraining points carry the
    nominal spacing; this returns the ones that do.
    """
    out = []
    for i in range(len(refs)):
        j, gap = fixed_partner(refs, i)
        if j is not None and gap == DENSE_GAP:
            out.append((i, j))
    return out


def whole_graph(idx: dict, refs, pairs, p1, space: str = "product") -> dict:
    """R12: the total change of all carrier sets between two checkpoints, one curve.

    The quantity is the share of all carrier membership that is **not** retained:
    `1 - |union of intersections| / |union of twin-1 sets|`, pooled over every construct, so a
    construct with a larger carrier set weighs more -- it is a statement about the graph, not an
    average over constructs.
    """
    out = []
    for i, j in pairs:
        kept = tot = 0
        for c in p1:
            u1 = idx.get(refs[i], {}).get(f"twin1/{c}|{space}")
            u2 = idx.get(refs[j], {}).get(f"twin1/{c}|{space}")
            if u1 is None or u2 is None:
                continue
            sa = set(u1.tolist())
            kept += len(sa & set(u2.tolist()))
            tot += len(sa)
        if tot:
            out.append({"from": refs[i], "to": refs[j],
                        "retained": kept / tot, "turnover": 1 - kept / tot, "n_units": tot})
    return {"points": out, "space": space,
            "quantity": "1 - retained share of all carrier membership, pooled over constructs"}


def per_construct_turnover(idx: dict, refs, pairs, p1, space: str = "product") -> dict:
    """Every construct's fixed-step retention, as the curve P4's bend would appear in."""
    out = {}
    for c in p1:
        pts = []
        for i, j in pairs:
            u1 = idx.get(refs[i], {}).get(f"twin1/{c}|{space}")
            u2 = idx.get(refs[j], {}).get(f"twin1/{c}|{space}")
            if u1 is None or u2 is None:
                continue
            pts.append({"index": i, "from": refs[i], "to": refs[j],
                        "retained": float(overlap(u1, u2))})
        if len(pts) >= 2:
            out[c] = pts
    return out


def mark(toks, points, branch_token: float | None) -> dict:
    """Where the branch point and the epoch boundaries fall among a curve's points."""
    out = {"epoch_boundaries_tokens": list(EPOCHS), "branch_point_tokens": branch_token,
           "epoch_nearest_point": {}, "branch_nearest_point": None}
    if not points:
        return out
    pt = [toks[p["index"]] if "index" in p else None for p in points]
    pt = [x for x in pt if x is not None] or toks[:len(points)]
    for e in EPOCHS:
        k = int(np.argmin([abs(x - e) for x in pt]))
        out["epoch_nearest_point"][f"{e/1e9:.0f}B"] = {
            "point": k, "tokens": float(pt[k]),
            "distance_decades": float(abs(np.log10(pt[k] / e)))}
    if branch_token:
        k = int(np.argmin([abs(x - branch_token) for x in pt]))
        out["branch_nearest_point"] = {"point": k, "tokens": float(pt[k])}
    return out


def bend_at(points, toks, target: float, window: int = 3, gaps=None) -> dict:
    """The change in slope of a curve across a marked token count.

    The slope is a least-squares fit of retention against log tokens on the `window` points each
    side. A bend is the difference; whether it is common is answered by how many constructs show
    the same sign, not by any one curve.
    """
    x = np.array([np.log10(toks[p["index"]]) for p in points], float)
    y = np.array([p["retained"] for p in points], float)
    k = int(np.argmin(np.abs(x - np.log10(target))))
    lo, hi = x[max(0, k - window):k + 1], x[k:k + window + 1]
    ylo, yhi = y[max(0, k - window):k + 1], y[k:k + window + 1]
    if len(lo) < 2 or len(hi) < 2:
        return {"readable": False, "why": "fewer than two points on one side of the mark"}
    s1 = float(np.polyfit(lo, ylo, 1)[0])
    s2 = float(np.polyfit(hi, yhi, 1)[0])
    out = {"readable": True, "at_tokens": float(target), "point": k,
           "slope_before": s1, "slope_after": s2, "bend": s2 - s1,
           "sign": "steeper" if s2 < s1 else "flatter"}
    # **A bend is only a bend if the ruler did not change across it.** A38 measured retention to
    # fall monotonically with the spacing it is read at, and the fixed-step spacing in the sparse
    # tail runs from 29,865 to 91,000 steps. So each bend carries the spacing either side, and is
    # marked confounded when they differ -- which is the case at every epoch boundary, all three
    # of which fall in the tail.
    if gaps is not None:
        g = np.array(gaps, float)
        gb = g[max(0, k - window):k + 1]
        ga = g[k:k + window + 1]
        out["spacing_steps_before"] = [float(np.min(gb)), float(np.max(gb))]
        out["spacing_steps_after"] = [float(np.min(ga)), float(np.max(ga))]
        out["spacing_constant_across_the_mark"] = bool(np.ptp(np.concatenate([gb, ga])) == 0)
        if not out["spacing_constant_across_the_mark"]:
            out["confounded"] = ("the fixed-step spacing is not constant across this mark, and "
                                 "retention falls monotonically with spacing (A38), so this "
                                 "bend is confounded with the schedule")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()

    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    p2 = sorted(set(json.loads((D / "validation/p2/P2_GATE_S15.json").read_text())["p2"]) & set(p1))
    from experiments.step3_senses import pilot_30
    pilot = pilot_30()
    seq = recorded("pretrain")
    refs, toks = [r for r, _ in seq], [t for _, t in seq]
    aseq = recorded("anneal")
    idx = load_index(INDEX, refs + [r for r, _ in aseq])
    branch_token = dict(seq).get(BRANCH_POINT)
    print(f"|P1| = {len(p1)}, |P2 n P1| = {len(p2)}; {len(refs)} pretraining references, "
          f"{len(aseq)} annealing")

    rep = {"n_p1": len(p1), "n_p2": len(p2),
           "confound": "P4 is a MEASUREMENT ROW, not a verdict on clause 4: the learning rate is "
                       "constant at 3e-4 across the whole pretraining branch, so there is no "
                       "within-branch bend to attribute to it, and the annealing branch changes "
                       "the learning rate AND the data mix at once, so a bend at the branch "
                       "point cannot be attributed to the rate alone"}

    # ---------------- P4 ----------------
    pairs = dense_indices(refs)
    allpairs, allgaps = [], []
    for i in range(len(refs)):
        j, gap = fixed_partner(refs, i)
        if j is not None:
            allpairs.append((i, j))
            allgaps.append(gap)
    wg = whole_graph(idx, refs, allpairs, p1)
    pc = per_construct_turnover(idx, refs, allpairs, p1)
    print(f"\n=== P4 (measurement row) ===")
    print(f"  R12 whole-graph: {len(wg['points'])} fixed-step points, retention "
          f"{wg['points'][0]['retained']:.4f} -> {wg['points'][-1]['retained']:.4f}, "
          f"{wg['points'][0]['n_units']:,} carrier memberships pooled")
    marks = mark(toks, [{"index": i} for i, _ in allpairs], branch_token)
    bends = {}
    for e in EPOCHS:
        bd = bend_at([{"index": i, "retained": p["retained"]}
                      for (i, _), p in zip(allpairs, wg["points"])], toks, e, gaps=allgaps)
        bends[f"{e/1e9:.0f}B"] = bd
        if bd["readable"]:
            print(f"  R12 bend at {e/1e9:.0f} B: slope {bd['slope_before']:+.4f} -> "
                  f"{bd['slope_after']:+.4f} ({bd['sign']})"
                  + ("  CONFOUNDED: spacing "
                     f"{bd['spacing_steps_before']} -> {bd['spacing_steps_after']} steps"
                     if not bd["spacing_constant_across_the_mark"] else "  spacing constant"))
    # is any bend COMMON across constructs?
    common = {}
    for e in EPOCHS:
        signs = []
        for c, pts in pc.items():
            bd = bend_at(pts, toks, e, gaps=allgaps)
            if bd["readable"]:
                signs.append(bd["bend"])
        if signs:
            arr = np.array(signs)
            common[f"{e/1e9:.0f}B"] = {
                "n_constructs": len(arr), "n_steeper": int((arr < 0).sum()),
                "n_flatter": int((arr > 0).sum()), "median_bend": float(np.median(arr)),
                "common": bool(max((arr < 0).mean(), (arr > 0).mean()) >= 0.8)}
            k = common[f"{e/1e9:.0f}B"]
            print(f"      across constructs: {k['n_steeper']} steeper / {k['n_flatter']} flatter "
                  f"of {k['n_constructs']}, median {k['median_bend']:+.5f} -> "
                  f"common: {k['common']}")
    rep["p4"] = {"r12_whole_graph": wg, "marks": marks, "r12_bends": bends,
                 "fixed_step_gaps": allgaps,
                 "epoch_boundaries_all_in_the_sparse_tail": bool(
                     all(min(toks[i] for i, _ in pairs) < e for e in EPOCHS)
                     and all(e > max(toks[i] for i, _ in pairs) for e in EPOCHS)),
                 "bend_common_across_constructs": common,
                 "n_construct_curves": len(pc), "per_construct": pc,
                 "branch_point": BRANCH_POINT, "branch_point_tokens": branch_token,
                 "epoch_boundaries_tokens": list(EPOCHS),
                 "epoch_arithmetic": "four epochs over 508 billion unique tokens",
                 "confound": rep["confound"]}

    # ---------------- P5 ----------------
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {**ast, **bi}
    freq = {c: float(np.log(r["per_file"])) for c in p2
            if (r := rates.get(contract_name(c))) and (r.get("per_file") or 0) > 0}
    print(f"\n=== P5: turnover level against log role frequency, per DENSE checkpoint ===")
    print(f"  {len(pairs)} dense pairs (a true {DENSE_GAP}-step spacing) of "
          f"{len(allpairs)} fixed-step points; {len(freq)} constructs carry a rate")
    rep["p5"] = {"n_dense_pairs": len(pairs), "n_all_pairs": len(allpairs),
                 "prediction": "the rate falls with frequency at every checkpoint: retention "
                               "RISES with frequency, so the Spearman of retention against log "
                               "frequency is positive",
                 "pilot_disclosure": {"spearman": 0.54, "n": 30, "p": 0.002,
                                      "where": "the last fixed-ratio pair of the dense block",
                                      "status": "exploratory"},
                 "dense": {}, "tail": {}}
    rng = np.random.default_rng(SEED)
    for which, members in (("with_pilot", p2), ("without_pilot", [c for c in p2 if c not in pilot])):
        for label, pp in (("dense", pairs), ("tail", [x for x in allpairs if x not in pairs])):
            rows = []
            for i, j in pp:
                xs, ys = [], []
                for c in members:
                    if c not in freq:
                        continue
                    # NOT `a`/`b`: argparse's namespace is `a` in this module, and shadowing it
                    # inside a loop is the fifth instance of this exact defect in this codebase
                    u1 = idx.get(refs[i], {}).get(f"twin1/{c}|product")
                    u2 = idx.get(refs[j], {}).get(f"twin1/{c}|product")
                    if u1 is None or u2 is None:
                        continue
                    xs.append(freq[c])
                    ys.append(float(overlap(u1, u2)))
                if len(xs) < 10:
                    continue
                x, y = np.array(xs), np.array(ys)
                rho = spearman(y, x)
                null = np.array([spearman(rng.permutation(y), x) for _ in range(400)])
                rows.append({"from": refs[i], "tokens": toks[i], "n": len(xs),
                             "spearman": float(rho),
                             "permutation_interval": [float(np.percentile(null, 2.5)),
                                                      float(np.percentile(null, 97.5))],
                             "p": float((np.abs(null) >= abs(rho)).mean())})
            sig = [r for r in rows if r["p"] < 0.05]
            pos = [r for r in sig if r["spearman"] > 0]
            rep["p5"][label][which] = {
                "n_checkpoints": len(rows), "n_significant": len(sig),
                "n_positive_significant": len(pos),
                "median_spearman": float(np.median([r["spearman"] for r in rows])) if rows else None,
                "every_checkpoint_positive": bool(rows and all(r["spearman"] > 0 for r in rows)),
                "points": rows}
            if rows:
                print(f"  {label:5s} [{which:13s}] {len(rows):2d} checkpoints, median rho "
                      f"{np.median([r['spearman'] for r in rows]):+.3f}, {len(sig)} significant "
                      f"({len(pos)} positive) -- all positive: "
                      f"{rep['p5'][label][which]['every_checkpoint_positive']}")

    # ---------------- P7 ----------------
    rep2 = json.loads((R2 / "R2_REPORT.json").read_text())
    nb = rep2["curves"]["pretrain|curve2_norm"]
    s4 = json.loads((R2 / "STEP4_FLAT_pretrain.json").read_text())
    leg = s4["legs"]["forced"]["from_84b_to_last"]
    print(f"\n=== P7: the raw norm between 84 B and its date, against the obliged loss's flat leg")
    print(f"  the loss side (step 4): obliged 84 B -> last is {leg['majority']} "
          f"{leg['counts']} -- falling {leg['counts'].get('falling', 0)}")
    rows, dirs = {}, collections.Counter()
    for c, ser in nb["series"].items():
        d = nb["dates"].get(c) or {}
        if (d.get("flat") or d.get("label") or d.get("outcome")) != "dated":
            continue
        rs, vs, us = ser["references"], ser["values"], ser["units"]
        tk = [dict(seq).get(r) for r in rs]
        i84 = int(np.argmin([abs((t or 0) - ANCHOR_84B) for t in tk]))
        dt = d.get("tokens")
        if dt is None or dt not in tk:
            continue
        idate = tk.index(dt)
        if idate <= i84:
            dirs["date_at_or_before_84B"] += 1
            rows[c] = {"outcome": "date_at_or_before_84B", "date_tokens": dt}
            continue
        ch = float(vs[idate] - vs[i84])
        unit = [u for u in us[i84:idate + 1] if u]
        nu = float(np.median(unit)) if unit else None
        lab = ("no_unit" if nu is None else
               "flat" if abs(ch) <= nu else "rising" if ch > 0 else "falling")
        dirs[lab] += 1
        rows[c] = {"outcome": lab, "change": ch, "noise_unit": nu,
                   "level_84b": float(vs[i84]), "level_at_date": float(vs[idate]),
                   "date_tokens": dt}
    print(f"  the norm side: {dict(dirs)} over {len(rows)} dated norms")
    refuted = dirs["rising"] > dirs["falling"] + dirs["flat"]
    rep["p7"] = {"loss_leg_from_84b": leg, "norm_directions": dict(dirs),
                 "n_dated_norms": len(rows),
                 "refutation_test": "P7 is refuted if the raw norm GROWS while both loss classes "
                                    "are flat",
                 "refuted": bool(refuted),
                 "reading": ("the raw norm rises over the same stretch on which the obliged loss "
                             "is flat for the majority: P7's refutation condition is met"
                             if refuted else
                             "the raw norm does not rise for the majority over the stretch on "
                             "which the obliged loss is flat: P7 is not refuted"),
                 "constructs": rows}
    print(f"  -> P7 refuted: {refuted} -- {rep['p7']['reading'][:88]}")

    # ---------------- P9 ----------------
    sp = json.loads((R2 / "CURVE1_SPACING_pretrain.json").read_text())
    spa = json.loads((R2 / "CURVE1_SPACING_anneal.json").read_text())
    a34 = json.loads((R2 / "A34_AGREEMENT_pretrain.json").read_text())
    rep["p9"] = {
        "outcome": "UNREADABLE",
        "why": "P9 compares the concentration date (curve 6) with the STABILITY date (curve 1) "
               "on P4, and A38's spacing check leaves no construct with a stability date on "
               "either branch: 0 dated on pretraining in both spaces, and the annealing branch "
               "spans less than one pretraining checkpoint step so no date there is carried",
        "n_stability_dated": {"pretrain_product": sp["product"]["n_dated"],
                              "pretrain_layer": sp["layer"]["n_dated"],
                              "anneal_usable_for_dating": spa["resolution"]["usable_for_dating"]},
        "beside_it_a34": {
            "population": a34["n_population"], "n_strict_order": a34["n_strict_order"],
            "order_counts": a34["order_counts_strict_only"],
            "n_sparsity_first": sum(n for k, n in a34["order_counts_strict_only"].items()
                                    if k.startswith("sparsity")),
            "what_it_is": "an ordering among the three PROFILE curves (norm, depth, sparsity) on "
                          "the 14 constructs whose three dates are all distinct -- NOT the "
                          "concentration-against-stability comparison P9 names"}}
    k = rep["p9"]["beside_it_a34"]
    print(f"\n=== P9: UNREADABLE (no stability date survives A38 on either branch)")
    print(f"  beside it, A34 on the {k['n_strict_order']} strict of {k['population']}: "
          f"sparsity dates first in {k['n_sparsity_first']} -- an ordering among the profile "
          f"curves, not P9's comparison")

    out = pathlib.Path(a.out) / "STEP7_pretrain.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
