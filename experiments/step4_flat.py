"""Step 4 -- P1 FLAT, in the two forms A36 rules on.

**The prediction** (`PLAN-Analysis` part 5, step 4; C2 as amended by A27): *competence ends before
settling ends* -- over the window in which a construct's carrier set turns over, its loss is
already flat.

**Form A, the date-to-date form, is reported as NO VERDICT with its reason**, because the
population where both dates exist is empty and that is a result about the record:

* `|P3| = 0` in the product space, the plan's primary;
* `|P3| = 12` in the verdict layer -- and all 12 are builtin **exception names**, while the
  positional classes reach **16** constructs, all **AST syntactic**. Disjoint by construction;
* `P3(layer)` against the construct-token dated set: **0** as well;
* part 6 needs `|P3| >= 43` for "settling is the end of drift", so that conclusion stays unstated.

**Form B, the window form, is the reading** (A36). Settling ends nowhere in the record -- curve 1
is `moving` for **169 of 169**, R >= 9.4 everywhere in the product space against the datable band's
5 -- so the window "while the carriers turn over" is the whole recorded branch for every construct,
measured rather than assumed. The question becomes: per class, is the loss flat over that window?

**The direction is checked, not assumed.** Curve 1 is *turnover*, but the quantity stored is the
**retention ratio** `overlap / ceiling`. It **falls** for 148 of 169 significantly on pretraining
and rises for none, so turnover **rises** -- the window the plan names. Reading the stored ratio as
"curve 1" inverts it, so this module re-derives the direction and refuses to run if it has flipped.

**The three class readings are the plan's, with A27's narrowing.**

* **forced** -- the `obliged` class: the loss at `0001000` (4.19 B), at `0020000` (83.89 B) and at
  the last recorded checkpoint (2,033.67 B); flat if the change is within
  `max(0.10 * remaining drop at 0001000, one noise unit)`. The floor is zero here, so the remaining
  drop is the loss itself -- A27's reason for confining the test to this class. The four
  sensitivities (5%, 20%, 0.5 and 2 units) are reported beside the default, not instead of it.
* **the construct-token class** (A29) beside it, the same test on all 188.
* **expected** -- each realised piece's curve dated by **A30's rule on its own**, not by distance
  to a floor, which is what A27 settled when the battery's continuation entropy turned out not to
  be the loss's limit.
* **dependent** -- `governed` and `scope` separately, after **C0.14**: the role gap must exceed one
  noise unit for the position to count, and a series that fails is reported as **not counting**,
  never as a negative.

**One subtlety the forced class carries** (A36, CC's call). Of the 38 `obliged` roles, **23
realise exactly one piece** at n >= 20 and **15 realise two or more** -- mostly tokenisation
variants of one syntactic thing (`AsyncFor|header_colon` admits `:\\n`, `):\\n`, `"]:\\n`,
`():\\n`). The zero floor is airtight only on the 23, so the test runs on all 38 as the record
labels them and the 23 are reported beside as the strict subset.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import branches, label_curve
from experiments.r2_curve4 import DEPENDENT, FORCED, read_positions, role_gaps, series
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
DL = D / "_step4_dl"
R2 = D / "validation/r2"
#: the plan's three anchors: `0001000`, "84 B", and the last recorded checkpoint
ANCHOR_EARLY = "pretrain_iter_0001000"
ANCHOR_84B = 84e9
#: the default threshold and the four sensitivities step 4 asks for beside it
FRACTIONS = (0.05, 0.10, 0.20)
UNIT_MULTIPLES = (0.5, 1.0, 2.0)
DEFAULT = (0.10, 1.0)
MIN_PIECE_N = 20


def recorded(branch: str) -> list:
    """`[(reference, tokens)]` for the references the record actually carries, in order.

    The schedule lists 509 planned checkpoints and the record carries 61; asking the Hub for an
    unrecorded reference is a 404, which is how the first fetch of this pass failed.
    """
    have = {p.stem for p in (D / "_carriers_dl/validation/carriers").glob("*.npz")}
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    return [(r, t) for r, t in branches(sched)[branch] if r in have]


def anchors(seq) -> dict:
    """The three anchor indices: `0001000`, the reference nearest 84 B, and the last.

    The nearest-to-84 B choice is explicit because no recorded checkpoint sits exactly there:
    `pretrain_iter_0020000` is 83.89 B, and the plan's "84 B" names it.
    """
    refs = [r for r, _ in seq]
    toks = [t for _, t in seq]
    if ANCHOR_EARLY not in refs:
        raise ValueError(f"{ANCHOR_EARLY} is not in the recorded references")
    i0 = refs.index(ANCHOR_EARLY)
    i84 = min(range(len(toks)), key=lambda i: abs(toks[i] - ANCHOR_84B))
    return {"early": {"i": i0, "reference": refs[i0], "tokens": toks[i0]},
            "at_84b": {"i": i84, "reference": refs[i84], "tokens": toks[i84]},
            "last": {"i": len(refs) - 1, "reference": refs[-1], "tokens": toks[-1]}}


def noise_unit(values, units) -> dict:
    """The threshold's unit, with **which** unit it is named.

    A30 uses the **core-bootstrap SE** where one exists -- `series()` produces it per checkpoint --
    and its median over the curve is the unit here. Where no SE exists (a single core, or core
    names that differ across references) the fallback is the median absolute point-to-point change,
    which is a *different* quantity and is labelled so rather than silently substituted.
    """
    fin = [u for u in (units or []) if u is not None and np.isfinite(u) and u > 0]
    if fin:
        return {"unit": float(np.median(fin)), "unit_is": "median core-bootstrap SE over the curve"}
    v = np.asarray(values, float)
    if v.size < 2:
        return {"unit": None, "unit_is": "no unit: fewer than two checkpoints"}
    d = np.abs(np.diff(v))
    d = d[np.isfinite(d)]
    if d.size == 0 or float(np.median(d)) == 0.0:
        return {"unit": None, "unit_is": "no unit: no core-bootstrap SE and no point-to-point "
                                         "movement to form one from"}
    return {"unit": float(np.median(d)),
            "unit_is": "median |point-to-point change| -- NOT the core-bootstrap SE, used only "
                       "because this series carries no usable SE"}


def drop_test(values, units, idx: dict) -> dict:
    """The remaining-drop test at a forced position, at every threshold step 4 asks for.

    The floor is **zero** (A27: one piece is admissible and a competent model's loss goes to
    nothing), so the remaining drop at `0001000` is the loss itself. A change is flat if it is
    within `max(fraction * remaining_drop, multiple * unit)`; the default is `(0.10, 1.0)` and the
    other combinations are the sensitivities, all reported.

    The **sign** is kept: a change that exceeds the threshold is `falling` or `rising`, never just
    "not flat" -- step 4's outcome table has a rising branch of its own.
    """
    v = np.asarray(values, float)
    i0, i84, il = idx["early"]["i"], idx["at_84b"]["i"], idx["last"]["i"]
    if max(i0, i84, il) >= v.size or not np.isfinite(v[[i0, i84, il]]).all():
        return {"outcome": "unreadable",
                "why": "one of the three anchors is missing from this series"}
    rem = float(v[i0])
    nu = noise_unit(values, units)
    out = {"remaining_drop_at_early": rem, "level_early": float(v[i0]),
           "level_84b": float(v[i84]), "level_last": float(v[il]),
           "change_to_84b": float(v[i84] - v[i0]), "change_to_last": float(v[il] - v[i0]),
           # **The second leg, 84 B to the last checkpoint, is a reading in its own right.**
           # The plan names 84 B as an anchor, and the two legs answer different questions: the
           # first covers 4.19 -> 84 B, where everything is still learning, and the second covers
           # 84 -> 2,034 B, 96% of the recorded tokens and the stretch over which the carriers
           # keep turning over. A whole-window label averages the two and reports the first.
           "change_84b_to_last": float(v[il] - v[i84]),
           **nu, "thresholds": {}}
    if nu["unit"] is None:
        out["outcome"] = "unreadable"
        out["why"] = nu["unit_is"]
        return out
    for f in FRACTIONS:
        for m in UNIT_MULTIPLES:
            thr = max(f * rem, m * nu["unit"])
            lab = {}
            for leg, ch in (("to_84b", out["change_to_84b"]),
                            ("to_last", out["change_to_last"]),
                            ("from_84b_to_last", out["change_84b_to_last"])):
                lab[leg] = ("flat" if abs(ch) <= thr else
                            "falling" if ch < 0 else "rising")
            out["thresholds"][f"{int(f*100)}pct|{m}u"] = {"threshold": float(thr), **lab}
    key = f"{int(DEFAULT[0]*100)}pct|{DEFAULT[1]}u"
    out["default_key"] = key
    out["outcome"] = out["thresholds"][key]["to_last"]
    out["outcome_to_84b"] = out["thresholds"][key]["to_84b"]
    out["outcome_from_84b"] = out["thresholds"][key]["from_84b_to_last"]
    return out


def piece_multiplicity(per_ref: dict, refs) -> dict:
    """`{(construct, class, role): {pieces}}` over pieces realised at n >= 20 anywhere.

    This is what separates a genuinely **forced** position from one the record labels `obliged`
    but which admits several tokens: the zero floor the drop test needs holds only where one
    piece is admissible. Reported as a sensitivity, never as a redefinition of the class.
    """
    out = collections.defaultdict(set)
    for r in refs:
        for (c, cls, role, piece), e in per_ref[r]["by_piece"].items():
            if (e.get("n") or 0) >= MIN_PIECE_N:
                out[(c, cls, role)].add(piece)
    return dict(out)


def window_is_turnover(branch: str, space: str = "product", shape: dict | None = None) -> dict:
    """A36's premise, re-derived: curve 1 moving for all of P1, and the ratio falling not rising.

    Refuses rather than proceeds if the direction has flipped, because the window form's whole
    meaning is that the carriers keep turning over across the recorded branch.
    """
    if shape is None:
        idxc = json.loads((R2 / "R2_INDEX_CURVES.json").read_text())
        shape = idxc["curve1"][f"{branch}|{space}|ratio"]["shape"]
    sh = shape
    moving = sum(1 for s in sh.values() if (s.get("flatness") or s.get("label")) == "moving")
    sig = [s for s in sh.values() if (s.get("permutation_p") or 1) < 0.05]
    falling = sum(1 for s in sig if s["spearman_vs_log_tokens"] < 0)
    rising = len(sig) - falling
    rmin = min(s["R"] for s in sh.values())
    out = {"n": len(sh), "n_moving": moving, "min_R": float(rmin),
           "n_significant": len(sig), "n_ratio_falling": falling, "n_ratio_rising": rising,
           "turnover_rises": falling > rising,
           "note": "the stored quantity is the RETENTION ratio overlap/ceiling; it falling means "
                   "turnover RISING, which is the window the plan names 'curve 1 rising'"}
    if moving != len(sh) or not out["turnover_rises"]:
        out["refused"] = ("the window form requires curve 1 moving for every construct and "
                          "turnover rising for the majority; one of those does not hold")
    return out


def c0_14(gap: dict | None) -> dict:
    """C0.14's verdict on one dependent series, with a failure named as *not counting*.

    The role gap must exceed one noise unit for the position to count. A series that fails is
    reported as `does_not_count` -- an absence of a readable position, not a negative result about
    the loss there. The three ways a verdict can be absent (no finite gap, no recorded SE, a
    single core whose SE is exactly zero) carry their own reasons from `R2_CURVE4.json`.
    """
    counts = None if not gap else gap.get("exceeds_one_unit")
    out = {"counts_under_c0_14": counts,
           "role_gap_mean": None if not gap else gap.get("mean"),
           "role_gap_noise_unit": None if not gap else gap.get("noise_unit")}
    if counts is True:
        return out
    out["outcome"] = "does_not_count"
    out["why"] = ("C0.14: the role gap does not exceed one noise unit, so this position does not "
                  "count -- reported as not counting, not as a negative result") if counts is False \
        else ((gap or {}).get("why") or "no role gap is recorded for this series")
    return out


def expected_label(values, units, toks, untrained, step: str) -> dict:
    """One realised piece's curve, dated by A30's rule, with the single-core case named.

    A piece realised by **one** core has no between-core spread, so C0.5's unit does not exist and
    A30's rule is not applicable -- which is a different outcome from A30's `unreadable`, the
    measured 1 < R < 5 band. Both were arriving under one label. The case is named and a
    sensitivity is run with the point-to-point unit, marked as not the unit A30 divides by.
    """
    d = label_curve(values, units, toks[:len(values)], untrained=untrained, step=step)
    if d.get("why") == "no checkpoint has a noise unit":
        d["outcome"] = "no_core_bootstrap_unit"
        d["why"] = ("this piece is realised by a single core, so there is no between-core spread "
                    "and C0.5's unit does not exist; A30's rule is not applicable")
        nu = noise_unit(values, units)
        if nu["unit"] is not None:
            d["sensitivity_point_to_point_unit"] = label_curve(
                values, [nu["unit"]] * len(values), toks[:len(values)],
                untrained=untrained, step=step)
            d["sensitivity_unit_is"] = nu["unit_is"]
    d["n_checkpoints"] = len(values)
    d["level_first"], d["level_last"] = values[0], values[-1]
    return d


def form_a() -> dict:
    """The date-to-date form: no verdict, with the empty population measured and named."""
    g = json.loads((R2 / "R0B_GATES_pretrain.json").read_text())
    c4 = json.loads((R2 / "R2_CURVE4.json").read_text())
    ctl = json.loads((D / "validation/loss_offset0/CONSTRUCT_TOKEN_LOSS.json").read_text())
    from experiments.r0b_gates import C3_VERDICT, curve4_dated
    p3 = {s: set(v["members"]) for s, v in g["p3"].items()}
    positional = set()
    for k, v in (c4.get("classes") or {}).items():
        if k.startswith("pretrain"):
            positional |= {s.split("|")[0] for s in (v.get("series") or {})}
    h = curve4_dated(c4, ctl, "pretrain", "hashed")
    ct = curve4_dated(c4, ctl, "pretrain", "construct_token")
    # **The verdict layer's 12 are a spacing artefact (A38), and the figure carries that now.**
    # Curve 1's level is a monotone function of the partner spacing (Spearman -0.813), the twelve
    # clear theta only where the available spacing shortens to 1.33x at the last point that has a
    # partner, and once two spacing bands must agree the count is 0 in both spaces. The 1.5x-rule
    # counts are kept because they are what the earlier tables carried, and annotated.
    sp = R2 / "CURVE1_SPACING_pretrain.json"
    spacing = json.loads(sp.read_text()) if sp.exists() else None
    return {"outcome": "NO VERDICT",
            "why": "step 4's reading as written ends in a date-to-date clause in all three "
                   "classes, and the population where both dates exist is empty",
            "n_p3_under_the_spacing_check": (
                None if spacing is None else
                {k: spacing[k]["n_dated"] for k in ("product", "layer")}),
            "spacing_check": (
                "A38: the 1.5x-rule counts below are superseded -- curve 1's level is a monotone "
                "function of the partner spacing, and no construct of P1 has a stability date "
                "that survives requiring two spacing bands to agree. Step 4's answer is "
                "unaffected: it never used a stability date, which is why A36 put it on the "
                "recorded window."),
            "n_p3": {s: len(v) for s, v in p3.items()},
            "n_positional_constructs": len(positional),
            "overlap_p3_layer_with_positional": len(p3["layer"] & positional),
            "overlap_p3_layer_with_hashed_dated": len(p3["layer"] & h),
            "overlap_p3_layer_with_construct_token_dated": len(p3["layer"] & ct),
            "p3_layer_members": sorted(p3["layer"]), "positional_constructs": sorted(positional),
            "part6_condition": f"|P3| >= {C3_VERDICT}",
            "part6_conclusion": "UNSTATED: 'settling is the end of drift, not of learning' needs "
                                f"|P3| >= {C3_VERDICT} and |P3| is "
                                f"{len(p3['product'])} in the primary space"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()
    br = a.branch

    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = {tree_name(c) for c in split["p1"]}
    seq = recorded(br)
    refs, toks = [r for r, _ in seq], [t for _, t in seq]
    print(f"branch {br}: {len(refs)} recorded references; |P1| = {len(p1)}")

    per_ref, gaps = {}, {}
    for r in refs + ["untrained"]:
        p = DL / f"record/{r}/POSITIONS.json"
        if not p.exists():
            continue
        doc = json.load(open(p))
        per_ref[r] = read_positions(doc)
        gaps[r] = role_gaps(doc)
    refs = [r for r in refs if r in per_ref]
    toks = [t for r, t in seq if r in per_ref]
    print(f"{len(per_ref)} POSITIONS.json read from {DL}")

    rep = {"branch": br, "n_p1": len(p1), "n_references": len(refs),
           "form_a_date_to_date": form_a(), "window": window_is_turnover(br)}
    w = rep["window"]
    print(f"\nFORM A  {rep['form_a_date_to_date']['outcome']}: |P3| "
          f"{rep['form_a_date_to_date']['n_p3']}, positional constructs "
          f"{rep['form_a_date_to_date']['n_positional_constructs']}, overlap "
          f"{rep['form_a_date_to_date']['overlap_p3_layer_with_positional']}")
    print(f"WINDOW  curve 1 moving {w['n_moving']}/{w['n']} (min R {w['min_R']:.1f}); "
          f"ratio falling {w['n_ratio_falling']} rising {w['n_ratio_rising']} of "
          f"{w['n_significant']} significant -> turnover rises: {w['turnover_rises']}")
    if "refused" in w:
        raise SystemExit(f"REFUSED: {w['refused']}")

    idx = anchors(list(zip(refs, toks)))
    print(f"anchors: {idx['early']['reference']} ({idx['early']['tokens']/1e9:.2f} B), "
          f"{idx['at_84b']['reference']} ({idx['at_84b']['tokens']/1e9:.2f} B), "
          f"{idx['last']['reference']} ({idx['last']['tokens']/1e9:.2f} B)")
    rep["anchors"] = idx

    mult = piece_multiplicity(per_ref, refs)

    # ---- forced: the drop test ----
    keys = sorted({k for r in refs for k in per_ref[r]["by_role"] if k[1] == FORCED})
    inp1 = [k for k in keys if k[0] in p1]
    forced, strict = {}, {}
    for k in inp1:
        vals, units, ok = series(per_ref, refs, k)
        t = drop_test(vals, units, idx) if ok and len(vals) == len(refs) else {
            "outcome": "unreadable",
            "why": "the series does not cover every reference, or its core names differ across "
                   "references so no core-bootstrap unit can be formed"}
        t["n_pieces_at_least_20"] = len(mult.get(k, ()))
        t["single_piece"] = t["n_pieces_at_least_20"] == 1
        forced["|".join(k)] = t
        if t["single_piece"]:
            strict["|".join(k)] = t
    fc = collections.Counter(v["outcome"] for v in forced.values())
    sc = collections.Counter(v["outcome"] for v in strict.values())
    print(f"\nFORCED  {len(forced)} obliged roles in P1 "
          f"({len({k.split('|')[0] for k in forced})} constructs): {dict(fc)}")
    print(f"        of which {len(strict)} realise ONE piece (floor strictly zero): {dict(sc)}")
    rep["forced"] = {"n_series": len(forced), "counts": dict(fc),
                     "n_single_piece": len(strict), "counts_single_piece": dict(sc),
                     "series": forced}

    # ---- the construct-token class beside it (A29) ----
    ctl = json.loads((D / "validation/loss_offset0/CONSTRUCT_TOKEN_LOSS.json").read_text())
    ct = {}
    for c, e in (ctl.get("curves") or {}).items():
        b = (e.get("branch") or {}).get(br) or {}
        # A29's artefact names the level `mean` and the per-checkpoint unit `noise_unit`; reading
        # it as `values`/`units` is how all 188 first came back "unreadable"
        vals, units, rr = b.get("mean"), b.get("noise_unit"), b.get("references")
        if not vals or not rr:
            ct[c] = {"outcome": "unreadable", "why": "no series on this branch"}
            continue
        sub = anchors([(r, dict(seq)[r]) for r in rr if r in dict(seq)]) if ANCHOR_EARLY in rr \
            else None
        ct[c] = (drop_test(vals, units, sub) if sub else
                 {"outcome": "unreadable", "why": f"{ANCHOR_EARLY} is not in this series"})
    cc = collections.Counter(v["outcome"] for v in ct.values())
    # **The drop test's floor is not zero at offset 0, and the test says so rather than pretend.**
    # A27 confined the remaining-drop test to forced positions because only there is one piece
    # admissible. A29's class is the loss on the construct's own token at **offset 0**, where the
    # model has made no commitment to produce that construct at all and many continuations are
    # admissible -- so the loss at `0001000` is not a remaining drop, and 10% of it is not a
    # tenth of anything. Measured: that loss has median 9.97 nats, the resulting threshold is
    # 1.02 nats, and the FRACTION rather than the noise unit sets it in 170 of 188 series, against
    # a median absolute change of 0.59. The "flat" count is therefore produced by the threshold.
    # It is reported with the assumption named, and A30's own dating of the same curves beside it.
    dom = sum(1 for v in ct.values() if v.get("outcome") in ("flat", "rising", "falling")
              and abs(v["thresholds"][v["default_key"]]["threshold"]
                      - FRACTIONS[1] * v["remaining_drop_at_early"]) < 1e-12)
    a30 = collections.Counter()
    for c, e in (ctl.get("curves") or {}).items():
        d = ((e.get("branch") or {}).get(br) or {}).get("date") or {}
        a30[d.get("label") or d.get("outcome") or "none"] += 1
    print(f"CONSTRUCT-TOKEN (A29, all {len(ct)}): {dict(cc)}")
    print(f"        floor-zero assumption NOT satisfied at offset 0: the fraction sets the "
          f"threshold in {dom} of {len(ct)}; A30's own dating of the same curves: {dict(a30)}")
    rep["construct_token"] = {
        "n": len(ct), "counts": dict(cc), "series": ct,
        "floor_zero_assumption": "NOT SATISFIED",
        "why": "A29's class is the loss on the construct's own token at offset 0, where many "
               "continuations are admissible, so the loss there is not a remaining drop to a "
               "zero floor -- A27's argument, applied to this class",
        "n_threshold_set_by_fraction": dom,
        "a30_dating_of_the_same_curves": dict(a30),
        "read_as": "NOT a flatness verdict; the drop test is reported here because the ruling "
                   "asks for it beside the forced class, and A30's dating is the reading that "
                   "does not need a floor"}

    # ---- expected: each realised piece dated by A30's rule ----
    untr = per_ref.get("untrained", {"by_role": {}, "by_piece": {}})
    # **The n >= 20 filter is A27's, and it is not optional.** `by_class_role_piece` carries every
    # piece realised at all -- 6,484 entries a reference, of which only 128 reach n = 20. A curve
    # built without the filter is mostly singletons: the first run produced 4,788 piece curves of
    # which 3,849 were unreadable, which is not a reading, it is the filter missing.
    thin = {r: {"by_role": v["by_role"],
                "by_piece": {k: e for k, e in v["by_piece"].items()
                             if (e.get("n") or 0) >= MIN_PIECE_N}}
            for r, v in per_ref.items()}
    pkeys = sorted({k for r in refs for k in thin[r]["by_piece"] if k[0] in p1})
    pieces = {}
    for k in pkeys:
        vals, units, ok = series(thin, refs, k, "by_piece")
        if len(vals) < 2 or not ok:
            pieces["|".join(k)] = {
                "outcome": "unreadable",
                "why": f"fewer than two references realise this piece at n >= {MIN_PIECE_N}, "
                       f"or the core names differ across references"}
            continue
        d = expected_label(vals, units, toks, (untr["by_piece"].get(k) or {}).get("mean"),
                           "consecutive" if br == "anneal" else "factor")
        d["multi_piece_position"] = len(mult.get(k[:3], ())) > 1
        d["position_class"] = k[1]
        pieces["|".join(k)] = d
    pc = collections.Counter((v.get("flat") or v.get("label") or v["outcome"])
                             for v in pieces.values())
    print(f"EXPECTED (per realised piece at n >= {MIN_PIECE_N}, A30's rule, {len(pieces)} "
          f"curves): {dict(pc)}")
    nru = sum(1 for v in pieces.values() if v["outcome"] == "no_core_bootstrap_unit")
    if nru:
        alt = collections.Counter(
            (v.get("sensitivity_point_to_point_unit") or {}).get("label")
            or (v.get("sensitivity_point_to_point_unit") or {}).get("outcome") or "none"
            for v in pieces.values() if v["outcome"] == "no_core_bootstrap_unit")
        print(f"         {nru} are single-core, so C0.5's unit does not exist -- under the "
              f"point-to-point unit instead (NOT A30's): {dict(alt)}")
    multi = {k: v for k, v in pieces.items() if v.get("multi_piece_position")}
    mc = collections.Counter((v.get("flat") or v.get("label") or v["outcome"])
                             for v in multi.values())
    print(f"         of which {len(multi)} sit at a position admitting several pieces -- the "
          f"class A27 reads this way: {dict(mc)}")
    rep["expected_pieces"] = {"n": len(pieces), "counts": dict(pc), "min_piece_n": MIN_PIECE_N,
                              "n_multi_piece_position": len(multi),
                              "counts_multi_piece_position": dict(mc), "series": pieces}

    # ---- dependent: governed and scope, after C0.14 ----
    # **C0.14's verdict is an aggregate, not a per-reference field.** `record/<ref>/POSITIONS.json`
    # carries the gap's `mean`, `se` and `core_means` at that checkpoint; whether the gap exceeds
    # one noise unit is decided on the whole curve (the median SE over references) and lives in
    # `R2_CURVE4.json`'s `role_gap` block, together with the three named reasons a verdict can be
    # absent. Reading `exceeds_one_unit` off a single reference returns None for every series,
    # which is how all 11 first came back as not counting.
    c4r = json.loads((R2 / "R2_CURVE4.json").read_text()).get("role_gap", {}).get(br, {})
    dep = {}
    for cls in DEPENDENT:
        keys = sorted({k for r in refs for k in per_ref[r]["by_role"] if k[1] == cls})
        inp1 = [k for k in keys if k[0] in p1]
        rows = {}
        for k in inp1:
            e = c0_14(c4r.get(f"{k[0]}|{cls}"))
            vals, units, ok = series(per_ref, refs, k)
            if "outcome" in e:
                pass
            elif len(vals) == len(refs) and ok:
                e.update(drop_test(vals, units, idx))
            else:
                e["outcome"] = "unreadable"
                e["why"] = "the series does not cover every reference, or core names differ"
            rows["|".join(k)] = e
        c = collections.Counter(v["outcome"] for v in rows.values())
        n_cnt = sum(1 for v in rows.values() if v["counts_under_c0_14"] is True)
        print(f"DEPENDENT {cls:9s} {len(rows)} series, {n_cnt} count under C0.14: {dict(c)}")
        dep[cls] = {"n_series": len(rows), "n_counting": n_cnt, "counts": dict(c),
                    "series": rows}
    rep["dependent"] = dep

    # ---- P1's majority answer, on the window form, by class ----
    def legs(rows) -> dict:
        """Each leg's counts at the default threshold, and across all nine thresholds."""
        out = {}
        for leg in ("to_84b", "to_last", "from_84b_to_last"):
            c = collections.Counter(v["thresholds"][v["default_key"]][leg] for v in rows
                                    if "thresholds" in v and v.get("default_key"))
            sens = {key: dict(collections.Counter(v["thresholds"][key][leg] for v in rows
                                                  if "thresholds" in v))
                    for key in sorted({k for v in rows if "thresholds" in v
                                       for k in v["thresholds"]})}
            tot = sum(c.values())
            out[leg] = {"counts": dict(c), "n": tot,
                        "majority": max(c, key=c.get) if c else None,
                        "share": (max(c.values()) / tot) if tot else None,
                        "across_thresholds": sens}
        return out

    maj = {}
    for name, counts, n in (("forced", fc, len(forced)),
                            ("forced_single_piece", sc, len(strict)),
                            ("construct_token", cc, len(ct)),
                            ("dependent_counting", collections.Counter(
                                v["outcome"] for cls in DEPENDENT
                                for v in dep[cls]["series"].values()
                                if v["counts_under_c0_14"] is True), None)):
        read = {k: v for k, v in counts.items() if k in ("flat", "falling", "rising")}
        tot = sum(read.values())
        top = max(read, key=read.get) if read else None
        maj[name] = {"n_series": n if n is not None else tot, "n_readable": tot,
                     "counts": read, "majority": top,
                     "share": (read[top] / tot) if tot else None}
        if name == "construct_token":
            maj[name]["caveat"] = ("the floor-zero assumption the drop test needs is NOT "
                                   "satisfied at offset 0; this row is not a flatness verdict")
    rep["legs"] = {
        "forced": legs(list(forced.values())),
        "forced_single_piece": legs(list(strict.values())),
        "dependent_counting": legs([v for cls in DEPENDENT
                                    for v in dep[cls]["series"].values()
                                    if v["counts_under_c0_14"] is True]),
        "construct_token": legs(list(ct.values()))}
    print("\nThe two legs, forced class, default threshold:")
    for leg in ("to_84b", "from_84b_to_last"):
        e = rep["legs"]["forced"][leg]
        share = "n/a" if e["share"] is None else f"{100 * e['share']:.0f}%"
        print(f"  {leg:18s} {e['majority']} on {e['n']} ({share})  {e['counts']}")

    # **A36's final form fixes which leg is the answer and which classes carry it.** The answer
    # is the `84 B -> last` leg -- the plan's own anchor, and the stretch "while the carriers turn
    # over" -- with the first leg reported beside it and both counts given; the whole-window label
    # is not the answer. The construct-token class is withdrawn from the flatness reading, its
    # floor assumption unsatisfied, with A30's dating of it beside instead.
    ANSWER_LEG = "from_84b_to_last"
    BESIDE_LEG = "to_84b"
    CARRY = ("forced", "forced_single_piece", "dependent_counting")
    answer = {c: {"answer": rep["legs"][c][ANSWER_LEG], "beside": rep["legs"][c][BESIDE_LEG],
                  "whole_window_not_the_answer": rep["legs"][c]["to_last"]}
              for c in CARRY}
    verdicts = {v["answer"]["majority"] for v in answer.values() if v["answer"]["majority"]}
    rep["p1_answer"] = {
        "answer_leg": ANSWER_LEG, "carried_by": list(CARRY),
        "by_class": answer,
        "majority_answer": (verdicts.pop() if len(verdicts) == 1 else
                            "the carrying classes disagree: " + ", ".join(
                                f"{c}={answer[c]['answer']['majority']}" for c in CARRY)),
        "withdrawn": {"construct_token": rep["construct_token"]["read_as"],
                      "counts_if_it_had_been_read": dict(cc),
                      "a30_dating_beside": rep["construct_token"]
                                              ["a30_dating_of_the_same_curves"]},
        "per_class_counts_all_legs": maj,
        "qualification": "this is flatness over a STRETCH of the record, NOT flatness before a "
                         "settling date: no construct has a settling date inside the record "
                         "(|P3| = 0), which is why the window is the recorded branch (A36)",
        "part6_conclusion": "UNSTATED -- |P3| >= 43 is not met",
        "licensed_statement": "competence stops improving within the first 4% of the recorded "
                              "tokens and does not resume, while the carriers keep turning over "
                              "through the remaining 96%"}
    print(f"\nP1's ANSWER ({ANSWER_LEG}, A36's final form): "
          f"{rep['p1_answer']['majority_answer']}")
    for c in CARRY:
        ans, bes = answer[c]["answer"], answer[c]["beside"]
        print(f"  {c:20s} answer {ans['majority']:8s} {ans['counts']}"
              f"   | beside (to 84 B) {bes['majority']}: {bes['counts']}")
    print(f"  construct_token      WITHDRAWN -- floor assumption not satisfied; A30's dating "
          f"beside: {rep['construct_token']['a30_dating_of_the_same_curves']}")

    out = pathlib.Path(a.out) / f"STEP4_FLAT_{br}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
