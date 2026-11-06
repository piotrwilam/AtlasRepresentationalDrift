"""Step 5 -- P2 FREQUENCY (C3): is the settling date set by how often the model met the construct?

**The prediction** (`PLAN-Analysis` part 5, step 5): the settling date is set by role frequency,
not by how the construct is built -- frequent constructs settle first, and a composite is not
delayed by its parts. **The rival** is assembly: a composite settles no earlier than the latest of
the constructs it obliges.

**The reading, as the plan writes it**: the **stability date** -- curve 1, censored beyond the
record -- against log role frequency and the composite indicator, frame class as covariate, in an
interval-censored AFT (log-normal, Weibull as check), core-bootstrap intervals, a permutation null
of 10,000; **beside it** the Spearman of date rank against log frequency with censored constructs
tied last, *which is primary if step 1 found the dates reference-dependent*; and the **assembly
test**, each composite's date against the latest among the constructs it obliges.

**Three things are settled before any number is produced, and each is a measurement.**

1. **C3's condition is not met, so there is no verdict.** `|P3| >= 43` is required for a verdict on
   P2 and `|P3| = 0` in the product space, 12 in the verdict layer. Step 0b's own clause then
   applies: *P2 is reported as the correlation with its interval and no verdict*. That is what this
   module produces -- not a weaker verdict, and not a substituted quantity.
2. **Step 1's clause about which reading is primary is UNDETERMINED, not satisfied.** R1's
   `rank_agreement` is `n = 0, spearman = null`: fewer than three constructs were dated under both
   thetas, so step 1 neither found the dates reference-dependent nor found them stable. The clause
   promotes the Spearman only on a positive finding, so the AFT stays primary by default and the
   undetermined status is reported rather than resolved by assumption.
3. **The Spearman on the primary space is undefined, not zero.** Curve 1's product column has 0
   dated and 169 censored: one distinct rank, so a rank correlation against it carries no
   information. Reported as undefined with the rank count, as step 3's second sense did.

**The assembly test is reported as empty where it is empty, with the reason.** 15 constructs carry
`components`. The test needs a date for the composite *and* for at least one component; the 12
constructs with a stability date in the verdict layer are all builtin exception names and all 15
composites are AST constructs, so the intersection is empty by construction.

**What reads beside it, marked exploratory.** Step 3's third sense fitted this same regression to
every date column, and the **norm** column carries log-frequency **-0.274** [-0.426, -0.141],
permutation p = 0.000. So frequency does predict a settling-like date in this record -- just not
the one C3 names. That is a cross-reference to A35, not a substitution for P2.

**Every reading is reported with and without the pilot's 30**, and the pilot's own two forms are
disclosed as the plan requires: the extrapolated-date form Spearman -0.30 (n = 30, p = 0.10) and
the measured-turnover form +0.54 (p = 0.002), both exploratory.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.r0b_gates import C3_POWERED, C3_VERDICT
from experiments.step3_senses import (DRAWS, SEED, aft_column, date_columns, pilot_30,
                                      predictors, ranks_tied_last, spearman)
from profilebattery.profiles.names import contract_name, tree_name
from profilebattery.paths import data_root

D = data_root()
R1 = D / "validation/r1"
R2 = D / "validation/r2"
#: the pilot's two forms, disclosed as the plan requires
PILOT = {"extrapolated_date_form": {"spearman": -0.30, "n": 30, "p": 0.10},
         "measured_turnover_form": {"spearman": +0.54, "n": 30, "p": 0.002},
         "status": "both exploratory; the phase battery's reading is reported with and without "
                   "these 30 constructs"}


def step1_clause(branch: str, space: str) -> dict:
    """Whether step 1 found the dates reference-dependent -- the clause that picks the primary.

    The plan promotes the Spearman to primary *if* step 1 found the dates reference-dependent.
    R1 found neither: with fewer than three constructs dated under both thetas, the rank agreement
    between the two definitions has no value at all. An undetermined condition does not promote,
    so the AFT stays primary -- stated, because treating "not found" as "found" would silently
    swap the two readings.
    """
    p = R1 / f"R1_{branch}_{space}.json"
    if not p.exists():
        return {"status": "UNKNOWN", "why": f"{p.name} is absent", "primary": "aft"}
    ra = (json.loads(p.read_text()).get("rank_agreement") or {})
    if ra.get("spearman") is None:
        return {"status": "UNDETERMINED", "n": ra.get("n"), "why": ra.get("why"),
                "primary": "aft",
                "note": "the clause promotes the Spearman only on a positive finding; an "
                        "undetermined condition leaves the AFT primary and is reported as "
                        "undetermined, not as a negative"}
    # **The plan gives no numeric cut, so none is invented.** It says the Spearman becomes
    # primary "if step 1 found the dates reference-dependent", and discloses the pilot's observed
    # range between definitions (0.89-1.00) without naming a threshold. The measured value is
    # reported against that range and the condition is left **not decidable by the contract**;
    # which reading is primary is then only material if the two disagree, so that is checked and
    # reported instead of resolved by a threshold CC would have chosen.
    return {"status": "NOT DECIDABLE BY THE CONTRACT",
            "spearman_between_thetas": ra["spearman"], "n_dated_under_both": ra.get("n"),
            "thetas": [ra.get("theta_a"), ra.get("theta_b")],
            "pilot_observed_range": [0.89, 1.00],
            "below_the_pilot_range": ra["spearman"] < 0.89,
            "primary": "aft",
            "note": "the plan names no cut for 'reference-dependent', so the AFT stays primary "
                    "by default and the measured agreement is reported; whether the choice "
                    "matters is answered by `primary_choice_is_material`, not by a threshold"}


def frequency(members) -> dict:
    """`{construct: log per-file role frequency}`, the predictor C3 is about."""
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {**ast, **bi}
    out = {}
    for c in members:
        r = rates.get(contract_name(c)) or {}
        if (r.get("per_file") or 0) > 0:
            out[c] = float(np.log(r["per_file"]))
    return out


def spearman_form(col: dict, members, freq: dict, draws: int = DRAWS,
                  seed: int = SEED) -> dict:
    """Date rank against log frequency, censored constructs tied last, permutation interval.

    Returns `spearman: None` with the reason when the date column has a single distinct rank --
    which is the case in the primary space, where every construct is censored. A near-zero
    correlation against a constant column is an absence of information, not a measured
    independence, so it is never reported as a number.
    """
    shared = [c for c in members if c in col and c in freq]
    if len(shared) < 10:
        return {"n": len(shared), "spearman": None,
                "why": "fewer than ten constructs carry both a date column entry and a rate"}
    ranks = ranks_tied_last(col, shared)
    nd = len(set(np.round(ranks, 9)))
    cens = sum(1 for c in shared if col[c] is None)
    out = {"n": len(shared), "distinct_ranks": nd, "n_censored": cens,
           "n_dated": len(shared) - cens}
    if nd < 2:
        out["spearman"] = None
        out["why"] = (f"the date column has a single rank ({nd}): every construct is censored, so "
                      f"a rank correlation against it is undefined rather than zero")
        return out
    # **With two ranks the correlation is not about ordering, and saying so is the whole point.**
    # Every dated construct sharing one checkpoint leaves exactly two ranks -- dated and censored
    # -- so the coefficient measures whether a construct is **dated at all** against its
    # frequency, and carries no information about which dates come first. P2 is a claim about
    # order. The quantity is named in the output so the number cannot be read as the claim.
    dates = {col[c] for c in shared if col[c] is not None}
    out["n_distinct_dates"] = len(dates)
    if nd == 2 or len(dates) < 2:
        out["measures"] = ("whether a construct is DATED AT ALL against its frequency -- not "
                           "which dates come first: all dated constructs share one checkpoint, "
                           "so there is no order among the dates to correlate with")
        out["is_the_p2_quantity"] = False
    else:
        out["measures"] = "date rank against log frequency, censored tied last"
        out["is_the_p2_quantity"] = True
    x = np.array([freq[c] for c in shared], float)
    rho = spearman(ranks, x)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(rng.permutation(ranks), x) for _ in range(draws)])
    lo, hi = np.percentile(null, [2.5, 97.5])
    out.update({"spearman": float(rho), "permutation_interval": [float(lo), float(hi)],
                "permutation_p": float((np.abs(null) >= abs(rho)).mean()), "n_draws": draws})
    return out


def assembly_test(col: dict, members) -> dict:
    """Each composite's date against the latest date among the constructs it obliges.

    The rival C3 names. Reported as **empty with the reason** when no composite has both its own
    date and at least one component's: an empty test is a fact about the record, and saying
    "composites are not later than their parts" on zero pairs would be a claim about nothing.
    """
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]
    comp = {}
    for c in members:
        e = inv.get(contract_name(c)) or {}
        if e.get("components"):
            comp[c] = [tree_name(x) for x in e["components"]]
    dated = {c: t for c, t in col.items() if t is not None}
    rows, usable = {}, 0
    for c, parts in sorted(comp.items()):
        have = {p: dated[p] for p in parts if p in dated}
        e = {"components": parts, "n_components_dated": len(have),
             "composite_date": dated.get(c)}
        if e["composite_date"] is not None and have:
            latest = max(have.values())
            e["latest_component_date"] = latest
            e["composite_not_later"] = e["composite_date"] <= latest
            e["separation_decades"] = float(abs(np.log10(e["composite_date"] / latest)))
            usable += 1
        else:
            e["readable"] = False
            e["why"] = ("the composite has no date inside the record" if
                        e["composite_date"] is None else
                        "none of the constructs it obliges has a date inside the record")
        rows[c] = e
    out = {"n_composites": len(comp), "n_usable_pairs": usable, "composites": rows}
    if not usable:
        out["outcome"] = "EMPTY"
        out["why"] = ("no composite has both its own stability date and a date for any construct "
                      "it obliges, so the assembly rival cannot be tested on this record; the "
                      "dated set and the composite set do not intersect")
    else:
        ok = [v["composite_not_later"] for v in rows.values() if "composite_not_later" in v]
        out["outcome"] = "assembly" if sum(ok) < len(ok) / 2 else "not_later_than_parts"
        out["n_not_later"] = int(sum(ok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--aft-draws", type=int, default=200)
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()
    br = a.branch

    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    p2 = set(json.loads((D / "validation/p2/P2_GATE_S15.json").read_text())["p2"]) & set(p1)
    pilot = pilot_30()
    sets = {"with_pilot": sorted(p2), "without_pilot": sorted(p2 - pilot)}
    gates = json.loads((R2 / f"R0B_GATES_{br}.json").read_text())
    print(f"branch {br}: |P2 n P1| = {len(p2)}; without the pilot's 30: "
          f"{len(sets['without_pilot'])}")

    rep = {"branch": br, "n_p2": len(p2), "pilot_30": sorted(pilot), "pilot_disclosure": PILOT,
           "c3": {"n_p3": {s: v["n"] for s, v in gates["p3"].items()},
                  "verdict_threshold": C3_VERDICT, "powered_threshold": C3_POWERED,
                  "verdict": "NO VERDICT on P2: |P3| is below C3's threshold in both spaces, so "
                             "step 0b's clause applies -- P2 is reported as the correlation with "
                             "its interval and no verdict"},
           "step1_clause": {}, "spearman": {}, "aft": {}, "assembly": {},
           "exploratory_cross_reference": {}}
    # **A38 withdraws the verdict layer's coefficients, and the artefact says so.** The twelve
    # dates the layer carried are a partner-spacing artefact: curve 1's level falls monotonically
    # with the spacing it is read at (Spearman -0.813), the twelve clear theta only where the
    # available spacing shortens at the series' last point, and requiring two spacing bands to
    # agree leaves 0 dated in both spaces. Every coefficient computed on those twelve is
    # therefore a coefficient on which constructs happened to clear a moving threshold.
    spc = R2 / f"CURVE1_SPACING_{br}.json"
    if spc.exists():
        sp = json.loads(spc.read_text())
        rep["withdrawn_by_a38"] = {
            "n_dated_under_the_spacing_check": {k: sp[k]["n_dated"]
                                                for k in ("product", "layer")},
            "spacing_dependence_spearman": sp["spacing_dependence"]["spearman"],
            "what": "every verdict-layer coefficient below -- the Spearman and both AFT fits -- "
                    "is WITHDRAWN: it is computed on twelve dates that are a partner-spacing "
                    "artefact, not twelve settling dates",
            "what_stands": "P2 has NO VERDICT, now because |P3| = 0 in both spaces rather than "
                           "12 < 43; the assembly rival is empty; the composite indicator is "
                           "out; step 1's clause is not decidable by the contract"}
    print(f"\nC3: |P3| {rep['c3']['n_p3']} against a verdict threshold of {C3_VERDICT} "
          f"-> NO VERDICT")

    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    from experiments.construct_token_loss import branches
    toks = sorted(t for _, t in branches(schedule)[br])

    for space in ("product", "layer"):
        cl = step1_clause(br, space)
        rep["step1_clause"][space] = cl
        print(f"\n=== curve 1's stability date, {space} space "
              f"({'PRIMARY' if space == 'product' else 'beside it'}) ===")
        print(f"  step 1's clause: {cl['status']} -> the primary reading is the {cl['primary']}"
              + (f" ({cl.get('why')})" if cl.get("why") else ""))
        col = date_columns(br, space)["curve1_turnover"]
        for which, members in sets.items():
            freq = frequency(members)
            s = spearman_form(col, members, freq, a.draws)
            rep["spearman"][f"{space}|{which}"] = s
            if s.get("spearman") is None:
                print(f"  spearman [{which:13s}] n={s['n']:3d}  UNDEFINED: {s['why'][:72]}")
            else:
                print(f"  spearman [{which:13s}] n={s['n']:3d} dated={s['n_dated']:3d} "
                      f"ranks={s['distinct_ranks']:3d}  rho {s['spearman']:+.3f} "
                      f"p={s['permutation_p']:.4f}  interval "
                      f"[{s['permutation_interval'][0]:+.3f},{s['permutation_interval'][1]:+.3f}]")

            pred = predictors(members)
            for dist in ("lognormal", "weibull"):
                r = aft_column(col, members, pred, toks, draws=a.aft_draws, dist=dist)
                rep["aft"][f"{space}|{which}|{dist}"] = r
                if r.get("fit") is None:
                    print(f"  aft [{which:13s}|{dist:9s}] n={r['n']:3d} events={r['n_events']:3d}"
                          f"  NO FIT: {r['why'][:60]}")
                    continue
                f = r["coefficients"].get("log_frequency", {})
                print(f"  aft [{which:13s}|{dist:9s}] n={r['n']:3d} events={r['n_events']:3d}  "
                      f"log_freq {f.get('beta'):+.3f}{'*' if f.get('excludes_zero') else ' '} "
                      f"[{f.get('lo'):+.3f},{f.get('hi'):+.3f}] perm "
                      f"p={r['permutation_p_log_frequency']}  composite "
                      f"{'in' if r['composite_entered'] else 'OUT'} ({r['n_composite']})")

            sp = rep["spearman"][f"{space}|{which}"]
            af = rep["aft"][f"{space}|{which}|lognormal"]
            fb = ((af.get("coefficients") or {}).get("log_frequency") or {}).get("beta")
            if sp.get("spearman") is not None and fb is not None:
                # the AFT is in log-TOKENS of the date and the Spearman is in RANKS with censored
                # LAST, so both carry the same sign convention: positive = later / censored.
                material = (np.sign(sp["spearman"]) != np.sign(fb)) or (
                    (sp["permutation_p"] < 0.05) !=
                    bool(((af["coefficients"]["log_frequency"]).get("excludes_zero"))))
                rep["step1_clause"][space][f"primary_choice_is_material|{which}"] = bool(material)
                print(f"  the choice of primary reading is "
                      f"{'MATERIAL' if material else 'immaterial'}: the Spearman "
                      f"({sp['spearman']:+.3f}) and the AFT ({fb:+.3f}) "
                      f"{'disagree' if material else 'agree in sign and in significance'}")

            asm = assembly_test(col, members)
            rep["assembly"][f"{space}|{which}"] = asm
            print(f"  assembly [{which:13s}] {asm['n_composites']} composites, "
                  f"{asm['n_usable_pairs']} usable pairs -> {asm['outcome']}"
                  + (f": {asm['why'][:64]}" if asm.get("why") else ""))

    # what does read, marked exploratory and not substituted for P2
    s3 = json.loads((R2 / f"STEP3_SENSES_{br}.json").read_text())
    for which in sets:
        e = (s3.get("third_sense") or {}).get(f"{which}|curve2_norm|lognormal") or {}
        f = (e.get("coefficients") or {}).get("log_frequency") or {}
        rep["exploratory_cross_reference"][which] = {
            "column": "curve2_norm", "beta": f.get("beta"),
            "interval": [f.get("lo"), f.get("hi")],
            "permutation_p": e.get("permutation_p_log_frequency"),
            "status": "EXPLORATORY (A35, step 3's third sense). Frequency predicts the NORM's "
                      "date, not the stability date C3 names. Reported beside P2, never as P2."}
    # **The direction, stated against the prediction rather than left to the reader.** C3
    # predicts frequent constructs settle FIRST: a negative coefficient on log frequency, and a
    # negative Spearman of date rank (censored last) against frequency. Whatever the layer shows
    # is reported with that comparison made explicit -- and with C3's unmet condition attached, so
    # a sign cannot be read as a verdict either way.
    rep["direction"] = {}
    for key, e in rep["aft"].items():
        f = ((e.get("coefficients") or {}).get("log_frequency") or {})
        if f.get("beta") is None:
            continue
        rep["direction"][key] = {
            "beta": f["beta"], "excludes_zero": f.get("excludes_zero"),
            "prediction_is": "negative: frequent constructs settle first",
            "observed": "negative, as predicted" if f["beta"] < 0 else
                        "POSITIVE -- the opposite of the prediction",
            "but": f"C3's condition |P3| >= {C3_VERDICT} is not met, and the dated constructs "
                   f"share a single checkpoint, so this is not a verdict in either direction"}
    for key, e in sorted(rep["direction"].items()):
        print(f"  direction [{key}] {e['beta']:+.3f} -> {e['observed']}")

    x = rep["exploratory_cross_reference"]["with_pilot"]
    print(f"\nBESIDE IT (exploratory, A35): frequency on the NORM's date "
          f"{x['beta']:+.3f} [{x['interval'][0]:+.3f},{x['interval'][1]:+.3f}] "
          f"perm p={x['permutation_p']} -- not the date C3 names")

    out = pathlib.Path(a.out) / f"STEP5_FREQUENCY_{br}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
