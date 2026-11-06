"""Curve 1's level is a function of the spacing it is read at, and the pretraining dates are not.

**Why this check exists.** Twelve constructs date at **one and the same checkpoint** in the verdict
layer -- `pretrain_iter_0364000`, 1.527 T tokens -- which is likelier to be a property of that
reference, or of how the point is formed, than twelve independent dates. The Settle chat asked for
the reference to be checked before 1.527 T entered any table as a date.

**The reference is healthy.** Its twin ceiling and twin agreement sit flat against its neighbours:
over all 169 constructs the median ceiling runs 0.7486 (233k), 0.7436 (291k), **0.7466 (364k)**,
0.7414 (455k), 0.7400 (485k), and the agreement 0.9245, 0.9242, **0.9249**, 0.9225, 0.9203. Over
the twelve the ceiling at 364k is **slightly lower** than its neighbours (0.6635 against
0.6755-0.6803), which *raises* the ratio, since the ceiling is its denominator -- but by about
1.5%, against a jump of 20%. The reference is not the cause.

**The spacing is.** Curve 1's point compares twin 1 at `t` with twin 2 at a partner checkpoint, and
the partner is chosen by a token factor inside `FACTOR_BAND = (1.25, 2.0)`. Measured over the
twelve, across all 1,584 (point, partner) pairs in that band:

| step factor | median ratio |
|---|---|
| 1.20-1.35x | **0.7382** |
| 1.35-1.50x | 0.7128 |
| 1.50-1.65x | 0.6680 |
| 1.65-2.10x | 0.6113 |

Spearman of ratio against step factor: **-0.837**, p ~ 0. Less training between the two
checkpoints leaves more of the carrier set in common, so a shorter step reads as *more stable*.
**And the twelve dates sit exactly where the available spacing shortens.** At indices 29-31 the
only in-band partner is ~1.56x; at index 32 -- 0364000 -- it is **1.332x**, because the 1.5x
partner (2.29 T) is past the end of training. The same construct at index 29 reads **0.611 at
1.56x and 0.717 at 1.25x**: a gap of 0.106, larger than the whole "jump" at index 32 (0.119), and
already above theta at 0.78 T.

**Worse, index 32 is the last point the series has**, so A31's "crosses and stays above theta to
the end" is **vacuous** there: nothing follows it to stay above. The same holds for A32's
fixed-step form, whose dates concentrate at index 33 (`0455000`) -- 13 of 13 in the product space,
40 of 53 in the verdict layer -- whose fixed partner is the **smallest gap in the tail** (29,865
steps) and which is likewise the last point of its series. Only **56%** of the pretraining points
have the nominal 1,000-step spacing at all (`share_at_fixed_step`).

**So a date needs a rule that the spacing cannot manufacture**, and that is what this module
applies: a construct is dated at index `i` only if, **from `i` to the end of the series, the ratio
stays above theta at every spacing available at every index** -- not only at the one partner the
band happened to pick. Four constructs pass on the pretraining branch and they pass unmistakably,
rising from 0.3-0.4 and sitting in the 0.74-0.86 band for every remaining point at spacings from
1,000 to 91,000 steps. The twelve do not pass at all.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import FACTOR_BAND, branches
from experiments.r1_turnover import THETA, ceiling, load_index, overlap
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
INDEX = D / "_carriers_dl/validation/carriers"
R2 = D / "validation/r2"


def recorded(branch: str) -> list:
    have = {p.stem for p in INDEX.glob("*.npz")}
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    return [(r, t) for r, t in branches(sched)[branch] if r in have]


def all_partners(toks, i: int, band=FACTOR_BAND) -> list:
    """**Every** partner inside the band, not the one a single rule picks.

    The band exists because no checkpoint sits exactly 1.5x later. Taking one partner from it
    makes the point depend on which spacings the schedule happens to offer at that index, and the
    measured dependence is strong (Spearman -0.837 of ratio against factor). Reading all of them
    turns that dependence from a hidden confound into a reported spread.
    """
    lo, hi = band
    return [j for j in range(i + 1, len(toks)) if lo <= toks[j] / toks[i] <= hi]


def point_spread(idx: dict, refs, toks, c: str, i: int, space: str, band=FACTOR_BAND) -> dict:
    """One index's ratio at every available spacing: the min, the max and the factors."""
    out = []
    for j in all_partners(toks, i, band):
        a = idx.get(refs[i], {}).get(f"twin1/{c}|{space}")
        b = idx.get(refs[j], {}).get(f"twin2/{c}|{space}")
        if a is None or b is None:
            continue
        c1 = ceiling(idx, refs[i], c, space)
        c2 = ceiling(idx, refs[j], c, space)
        if not (np.isfinite(c1) and np.isfinite(c2) and c1 > 0 and c2 > 0):
            continue
        out.append({"partner": refs[j], "factor": float(toks[j] / toks[i]),
                    "ratio": float(overlap(a, b) / np.sqrt(c1 * c2))})
    if not out:
        return {"n": 0}
    r = [p["ratio"] for p in out]
    return {"n": len(out), "min": float(min(r)), "max": float(max(r)),
            "spread": float(max(r) - min(r)), "points": out}


#: the band is split into four, because one threshold cannot serve a 1.25x and a 2.0x window
BANDS = ((1.25, 1.35), (1.35, 1.50), (1.50, 1.65), (1.65, 2.00))
#: the annealing branch spans x1.0611 in tokens, so no factor band exists there at all (A33).
#: Its spacing is counted in **references**, and the same exposure is tested at 1, 2, 4 and 8.
STEP_BANDS = (1, 2, 4, 8)
MIN_POINTS = 4
MIN_FOLLOWING = 2
#: a date needs an observed crossing: at least one point BELOW theta before it
MIN_BELOW_BEFORE = 1


def _date_from_points(pts, refs, toks, theta: float, label: dict) -> dict:
    """A31's crosses-and-stays on a prepared point list, with both vacuous cases refused.

    Two refusals, symmetric, and each was found in the record:

    * **nothing after it.** A crossing at the series' last point satisfies "stays above theta to
      the end" by having nothing to stay above -- the twelve pretraining dates at 1.527 T.
    * **nothing before it.** A series that is *already* above theta at its first point never
      crosses; calling that a date at the first checkpoint is calling the start of the window a
      settling date. On the annealing branch the ratio is above theta for **99.9%** of all points
      at one-reference spacing, and A32's fixed form duly dated **169 of 169 constructs, 166 of
      them at the first annealing reference**. That is the threshold failing to discriminate on a
      x1.06 span, not 169 settlings.
    """
    if len(pts) < MIN_POINTS:
        return {"outcome": "too_few_points", "n": len(pts), **label}
    for k in range(len(pts)):
        if all(r > theta for _, _, r in pts[k:]):
            below = sum(1 for _, _, r in pts[:k] if r <= theta)
            if below < MIN_BELOW_BEFORE:
                return {"outcome": "above_from_the_start", "n": len(pts), **label,
                        "index": pts[k][0], "reference": refs[pts[k][0]],
                        "n_below_before": below,
                        "why": f"the ratio is above theta from the first point, so no crossing "
                               f"is observed; a 'date' here is the start of the window and not "
                               f"a settling date"}
            if len(pts) - k - 1 < MIN_FOLLOWING:
                return {"outcome": "vacuous", "n": len(pts), **label,
                        "index": pts[k][0], "reference": refs[pts[k][0]],
                        "tokens": toks[pts[k][0]], "n_following": len(pts) - k - 1,
                        "why": f"fewer than {MIN_FOLLOWING} points follow the crossing in this "
                               f"band, so 'stays above theta' is near-vacuous"}
            return {"outcome": "dated", "n": len(pts), **label,
                    "index": pts[k][0], "reference": refs[pts[k][0]],
                    "tokens": toks[pts[k][0]], "n_following": len(pts) - k - 1,
                    "n_below_before": below,
                    "min_from_here": float(min(r for _, _, r in pts[k:]))}
    return {"outcome": "censored", "n": len(pts), **label,
            "max_ratio": float(max(r for _, _, r in pts))}


def _point(idx, refs, c, space, i, j):
    a = idx.get(refs[i], {}).get(f"twin1/{c}|{space}")
    b = idx.get(refs[j], {}).get(f"twin2/{c}|{space}")
    if a is None or b is None:
        return None
    c1, c2 = ceiling(idx, refs[i], c, space), ceiling(idx, refs[j], c, space)
    if not (np.isfinite(c1) and np.isfinite(c2) and c1 > 0 and c2 > 0):
        return None
    return float(overlap(a, b) / np.sqrt(c1 * c2))


def date_in_step_band(idx: dict, refs, toks, c: str, space: str, k: int,
                      theta: float = THETA) -> dict:
    """A31's rule at a spacing of exactly `k` references -- the annealing branch's form.

    A33 made the annealing branch consecutive because no 1.5x step exists on a x1.06 span. The
    spacing exposure does not go away with the factor: measured over P1, the median ratio runs
    **0.8936** at one reference apart, 0.8704 at two, 0.8439 at four and 0.7630 at sixteen. So
    the branch is read at several step counts and they must agree, exactly as the factor bands
    must on pretraining.
    """
    pts = []
    for i in range(len(refs) - k):
        r = _point(idx, refs, c, space, i, i + k)
        if r is not None:
            pts.append((i, float(toks[i + k] / toks[i]), r))
    return _date_from_points(pts, refs, toks, theta, {"step_band": k})


def date_in_band(idx: dict, refs, toks, c: str, space: str, lo: float, hi: float,
                 theta: float = THETA) -> dict:
    """A31's rule applied to the points whose spacing falls inside one narrow band.

    **Why per band and not over the whole of `FACTOR_BAND`.** The ratio is a monotone function of
    the spacing (Spearman -0.81 here), so a single theta cannot serve a 1.25x window and a 2.0x
    window: at 1.25x the median ratio is 0.73 and at 1.65-2.0x it is 0.61, and theta is 0.70.
    Requiring the minimum over all spacings to clear theta would refuse every construct for a
    reason that is about the window length, not about settling. Requiring it **within** a narrow
    band compares like with like, and the date is then trustworthy only if the bands **agree**.

    A crossing with fewer than `MIN_FOLLOWING` points after it is `vacuous`, not a date: that is
    what produced the twelve at 1.527 T and the fixed form's thirteen at 1.908 T.
    """
    pts = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            f = toks[j] / toks[i]
            if not lo <= f <= hi:
                continue
            a = idx.get(refs[i], {}).get(f"twin1/{c}|{space}")
            b = idx.get(refs[j], {}).get(f"twin2/{c}|{space}")
            if a is None or b is None:
                continue
            c1, c2 = ceiling(idx, refs[i], c, space), ceiling(idx, refs[j], c, space)
            if not (np.isfinite(c1) and np.isfinite(c2) and c1 > 0 and c2 > 0):
                continue
            pts.append((i, float(f), float(overlap(a, b) / np.sqrt(c1 * c2))))
            break                                    # one point an index: the shortest in band
    return _date_from_points(pts, refs, toks, theta, {"band": [lo, hi]})


def date_across_bands(idx: dict, refs, toks, c: str, space: str,
                      theta: float = THETA, branch: str = "pretrain") -> dict:
    """The date each band gives, and whether the bands agree -- which is the actual test.

    A date that only one band produces is a date on that spacing. A date several bands produce at
    the same reference is a date on the model. The output carries both so neither can be read as
    the other.
    """
    if branch == "anneal":
        per = {f"{k}ref": date_in_step_band(idx, refs, toks, c, space, k, theta)
               for k in STEP_BANDS}
    else:
        per = {f"{lo}-{hi}": date_in_band(idx, refs, toks, c, space, lo, hi, theta)
               for lo, hi in BANDS}
    dated = {k: v for k, v in per.items() if v["outcome"] == "dated"}
    readable = [k for k, v in per.items() if v["outcome"] in ("dated", "censored")]
    out = {"per_band": per, "n_bands_readable": len(readable), "n_bands_dated": len(dated),
           "band_outcomes": {k: v["outcome"] for k, v in per.items()}}
    if not dated:
        # **"unreadable" must not absorb the two vacuous outcomes.** A band that is above theta
        # from its first point, or whose crossing has nothing after it, is a *named* refusal and
        # not a missing measurement; collapsing them into "unreadable" hides exactly the defect
        # this module was written to expose. The dominant refusal is reported instead.
        if readable:
            out["outcome"] = "censored"
            out["why"] = ("no band has the ratio above theta from any point to the end of that "
                          "band's series")
            return out
        kinds = collections.Counter(v["outcome"] for v in per.values())
        top, n = kinds.most_common(1)[0]
        out["outcome"] = top if top != "too_few_points" else "unreadable"
        out["n_bands_with_that_outcome"] = n
        out["why"] = (next(v["why"] for v in per.values() if v["outcome"] == top)
                      if any(v.get("why") for v in per.values() if v["outcome"] == top)
                      else "no band carries enough points to read")
        return out
    refsd = {v["reference"] for v in dated.values()}
    out["agreeing"] = len(dated) >= 2 and len(refsd) == 1
    out["outcome"] = "dated" if out["agreeing"] else "dated_in_one_band_only"
    out["references"] = sorted(refsd)
    out["tokens"] = sorted({v["tokens"] for v in dated.values()})
    if not out["agreeing"]:
        out["why"] = (f"{len(dated)} of {len(readable)} readable bands date this construct, at "
                      f"{sorted(refsd)}: a date one spacing produces and others do not is a "
                      f"date on the spacing")
    return out


def spacing_robust_date(idx: dict, refs, toks, c: str, space: str, theta: float = THETA,
                        band=FACTOR_BAND) -> dict:
    """The first index from which the ratio stays above theta **at every available spacing**.

    Two refusals are built in, both of which the 1.5x rule missed:

    * an index whose only partner is shorter than every earlier index's partner cannot carry a
      date on its own -- it is reported, with the spacing named, but `dated` requires the
      *minimum* over spacings to clear theta;
    * a crossing at the **last** index with any partner is `vacuous`, not `dated`: "stays above
      theta to the end" is satisfied by having nothing after it. The twelve dates at 1.527 T and
      the fixed form's thirteen at 1.908 T are both of this kind.
    """
    spreads = {i: point_spread(idx, refs, toks, c, i, space, band) for i in range(len(toks))}
    have = [i for i, s in spreads.items() if s["n"]]
    if not have:
        return {"outcome": "unreadable", "why": "no index has a partner inside the band"}
    last = max(have)
    for i in have:
        tail = [j for j in have if j >= i]
        if all(spreads[j]["min"] > theta for j in tail):
            if i == last:
                return {"outcome": "vacuous", "index": i, "reference": refs[i],
                        "tokens": toks[i], "n_following": 0,
                        "min_ratio": spreads[i]["min"], "max_ratio": spreads[i]["max"],
                        "why": "the crossing is at the last index that has a partner, so "
                               "'stays above theta to the end' is satisfied by having nothing "
                               "after it; this is not a date"}
            return {"outcome": "dated", "index": i, "reference": refs[i], "tokens": toks[i],
                    "n_following": len(tail) - 1,
                    "min_ratio_from_here": float(min(spreads[j]["min"] for j in tail)),
                    "min_spacing": float(min(p["factor"] for j in tail
                                             for p in spreads[j]["points"])),
                    "max_spacing": float(max(p["factor"] for j in tail
                                             for p in spreads[j]["points"]))}
    best = max(have, key=lambda i: spreads[i]["min"])
    return {"outcome": "censored", "best_index": best, "best_reference": refs[best],
            "best_min_ratio": spreads[best]["min"], "best_max_ratio": spreads[best]["max"],
            "why": "no index has the ratio above theta at every spacing from there to the end"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()
    br = a.branch

    seq = recorded(br)
    refs, toks = [r for r, _ in seq], [t for _, t in seq]
    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    idx = load_index(INDEX, refs)
    print(f"branch {br}: {len(refs)} references, |P1| = {len(p1)}, theta = {THETA}, "
          f"band = {FACTOR_BAND}")

    # the spacing dependence, measured on whatever constructs the old rule dated
    old = json.loads((R2 / "R2_INDEX_CURVES.json").read_text())["curve1"]
    suspect = sorted(c for c, v in (old.get(f"{br}|layer|fixed" if br == "anneal"
                                             else f"{br}|layer|ratio") or {}).get("dates",
                                                                                  {}).items()
                     if (v.get("label") or v.get("outcome")) == "dated")
    pairs = []
    sample = (suspect or p1)[:12]
    if br == "anneal":
        # on annealing the x-axis is the reference step count, since every factor is ~1.003
        for c in sample:
            for k in STEP_BANDS:
                for i in range(len(refs) - k):
                    r = _point(idx, refs, c, "layer", i, i + k)
                    if r is not None:
                        pairs.append((float(k), r))
    else:
        for c in sample:
            for i in range(len(refs)):
                sp = point_spread(idx, refs, toks, c, i, "layer")
                pairs += [(p["factor"], p["ratio"]) for p in sp.get("points", [])]
    rep = {"branch": br, "theta": THETA, "band": list(FACTOR_BAND),
           "n_suspect_constructs": len(suspect), "suspect_constructs": suspect}
    if pairs:
        arr = np.array(pairs)
        from scipy.stats import spearmanr
        rho, pv = spearmanr(arr[:, 0], arr[:, 1])
        bands = {}
        edges = (((0.5, 1.5), (1.5, 2.5), (3.5, 4.5), (7.5, 8.5)) if br == "anneal"
                 else ((1.20, 1.35), (1.35, 1.50), (1.50, 1.65), (1.65, 2.10)))
        for lo, hi in edges:
            m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if m.sum():
                bands[f"{lo}-{hi}"] = {"n": int(m.sum()),
                                       "median_ratio": float(np.median(arr[m, 1]))}
        rep["spacing_dependence"] = {"n_pairs": len(pairs), "spearman": float(rho),
                                     "p": float(pv), "by_band": bands,
                                     "reading": "the ratio is a monotone function of the spacing "
                                                "it is read at, so a date produced where the "
                                                "available spacing shortens is a date on the "
                                                "schedule and not on the model"}
        print(f"\nspacing dependence over {len(pairs)} (point, partner) pairs: "
              f"spearman {rho:+.3f} (p = {pv:.1e})")
        for k, v in bands.items():
            print(f"   factor {k}: n = {v['n']:5d}  median ratio {v['median_ratio']:.4f}")

    for space in ("product", "layer"):
        rows = {c: date_across_bands(idx, refs, toks, c, space, branch=br) for c in p1}
        counts = collections.Counter(v["outcome"] for v in rows.values())
        one = sorted(c for c, v in rows.items() if v["outcome"] == "dated_in_one_band_only")
        perband = collections.Counter(
            k for v in rows.values() for k, e in v["per_band"].items()
            if e["outcome"] == "dated")
        vac = collections.Counter(
            k for v in rows.values() for k, e in v["per_band"].items()
            if e["outcome"] == "vacuous")
        print(f"\n{space:8s} per band -- dated: {dict(perband)}")
        print(f"{'':8s} per band -- vacuous crossings: {dict(vac)}")
        if one:
            print(f"{'':8s} dated in ONE band only ({len(one)}): {one[:6]}")
        dated = sorted(c for c, v in rows.items() if v["outcome"] == "dated")
        print(f"\n{space:8s} ({'PRIMARY' if space == 'product' else 'beside it'}): {dict(counts)}")
        for c in dated:
            v = rows[c]
            print(f"   {c:22s} {v['references'][0]:22s} {v['tokens'][0]/1e9:8.2f} B  "
                  f"{v['n_bands_dated']} of {v['n_bands_readable']} readable bands agree")
        rep[space] = {"counts": dict(counts), "n_dated": len(dated), "dated": dated,
                      "dated_in_one_band_only": one,
                      "n_dated_per_band": dict(perband), "n_vacuous_per_band": dict(vac),
                      "rule": "a construct is dated only where at least two spacing bands date "
                              "it at the SAME reference; a date one band produces and others do "
                              "not is a date on the spacing, and a crossing with fewer than "
                              f"{MIN_FOLLOWING} points after it in its band is vacuous",
                      "constructs": rows}

    # **The branch's own resolution, because a date is only as fine as the window it sits in.**
    # The annealing branch's 24 references span x1.0611 = 0.026 decades in total -- less than a
    # third of ONE pretraining checkpoint step (median 0.094 decades, A34). A date inside it
    # cannot be ordered against a pretraining date, or against another annealing date, at this
    # record's resolution, whatever the band agreement says.
    span = float(np.log10(toks[-1] / toks[0]))
    gaps = [np.log10(toks[i + 1] / toks[i]) for i in range(len(toks) - 1)]
    rep["resolution"] = {
        "branch_span_decades": span, "median_step_decades": float(np.median(gaps)),
        "pretraining_step_decades": 0.094,
        "usable_for_dating": bool(span > 0.094),
        "note": ("the whole branch spans less than one pretraining checkpoint step, so a date "
                 "inside it is below the record's resolution and is not carried"
                 if span <= 0.094 else
                 "the branch spans more than one pretraining checkpoint step")}
    print(f"\nresolution: the branch spans {span:.3f} decades (median step {np.median(gaps):.4f}); "
          f"one pretraining step is 0.094 -> usable for dating: {rep['resolution']['usable_for_dating']}")

    out = pathlib.Path(a.out) / f"CURVE1_SPACING_{br}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
