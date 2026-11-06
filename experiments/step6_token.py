"""Step 6 -- P3 TOKEN, in the two exploratory forms the record allows.

**The prediction** (`PLAN-Analysis` part 5, step 6): the date follows the **token's** exposure, not
the construct's role frequency. **The reading as written** is within each shared-token group --
`For`, `ListComp`, `GeneratorExp`, `DictComp`, `SetComp` on `for`; `If`, `IfExp` on `if`; and the
rest of the battery's token table -- asking whether the members' **dates** coincide within their
spreads or order by role frequency, and across P2 the P2 model with token exposure and role
frequency both entered.

**The within-group clause is date-bound, and the dates are gone.** A38 measured curve 1's level to
be a monotone function of the partner spacing (Spearman **-0.813** over 1,404 pairs) and found
**no construct of P1 with a stability date on the pretraining branch that survives the spacing
check** -- 0 in the product space and 0 in the verdict layer once two spacing bands are required to
agree. So "do the members' dates coincide" **has no dates to compare**, in any group, and is
reported as untestable rather than answered. This is stated first because a reader will otherwise
take what follows for the clause itself.

**What the record does carry, both marked EXPLORATORY forms of P3 and never its verdict**
(the Settle chat's instruction, 15 Oct):

1. **Within each shared-token group**, the members' **turnover levels and trends** -- the median
   ratio and the Spearman against log tokens with its permutation interval, which are A30 *shape*
   quantities and involve no partner choice beyond the one band they were computed in -- and their
   **norm dates**, which do survive, being a level curve dated by A30's rule with no partner at
   all. The question becomes: do members of one token group sit at the same turnover level and
   settle their norm together, or do they order by role frequency?
2. **Across P2**, the two-predictor regression on the **norm** date: log role frequency and token
   exposure entered together, each coefficient reported, and each fitted **alone** so that what
   each explains on its own is visible -- the plan's "each predictor's coefficient and what each
   explains alone".

**Exposure is parameterised as the multiplier, and the reason is algebra.** `exposure_multiplier`
gives each construct's cross-language token multiplier, 1.0 to 8.71: how much more its *token* is
seen than its Python role frequency implies, because other languages use the same token. Total
exposure is role frequency times that multiplier, so `log exposure = log frequency + log
multiplier` -- entering frequency and exposure together is exactly collinear. Entering **frequency
and the multiplier** is the same model without the collinearity, and the exposure coefficient is
recovered as the sum. Stated because the two parameterisations look different and are not.

**The Rust clause of block 11 is not applicable**, as the plan already records: the phase battery
carries no Rust keywords and `validation/rust/` is a two-reference screen, not a battery.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.step3_senses import (DRAWS, SEED, aft_fit, date_columns, pilot_30, spearman)
from profilebattery.profiles.names import contract_name, tree_name
from profilebattery.paths import data_root

D = data_root()
R2 = D / "validation/r2"
MIN_GROUP = 2


def groups(members) -> dict:
    """`{token: [constructs]}` for every token at least two members of `members` share."""
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]
    by = collections.defaultdict(list)
    for c in members:
        t = (inv.get(contract_name(c)) or {}).get("token")
        if t:
            by[t].append(c)
    return {t: sorted(cs) for t, cs in by.items() if len(cs) >= MIN_GROUP}


def predictors2(members) -> dict:
    """`{construct: {log_frequency, log_exposure_multiplier}}`, P3's two predictors.

    `log_exposure_multiplier` is the cross-language token multiplier in logs. Total token exposure
    is `frequency * multiplier`, so this pair spans the same model as (frequency, exposure) without
    the exact collinearity that pair carries.
    """
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {**ast, **bi}
    exp = json.loads((D / "calibration/exposure_multiplier.json").read_text())["per_construct"]
    out = {}
    for c in members:
        base = contract_name(c)
        r, e = rates.get(base), exp.get(base)
        if not r or not e or not (r.get("per_file") or 0) > 0:
            continue
        m = e.get("multiplier")
        if not m or m <= 0:
            continue
        out[c] = {"log_frequency": float(np.log(r["per_file"])),
                  "log_exposure_multiplier": float(np.log(m)),
                  "token": e.get("token")}
    return out


def interval(col: dict, members, toks_sorted) -> tuple:
    """The interval-censored (lo, hi) in log tokens, the same convention step 3 used."""
    lo, hi, keep = [], [], []
    for c in members:
        if c not in col:
            continue
        t = col[c]
        if t is None:
            lo.append(np.log(toks_sorted[-1]))
            hi.append(np.inf)
        else:
            i = toks_sorted.index(t) if t in toks_sorted else None
            if i is None:
                continue
            lo.append(np.log(toks_sorted[i - 1]) if i else -np.inf)
            hi.append(np.log(t))
        keep.append(c)
    return np.array(lo), np.array(hi), keep


def two_predictor(col: dict, members, pred: dict, toks_sorted, draws: int = 200,
                  seed: int = SEED, dist: str = "lognormal") -> dict:
    """Both predictors together, and each alone -- the plan's "what each explains alone".

    Refuses below ten events, as step 3 does: a regression on fewer is an interval and not an
    estimate.
    """
    names = ("log_frequency", "log_exposure_multiplier")
    shared = [c for c in members if c in pred and c in col]
    lo, hi, keep = interval(col, shared, toks_sorted)
    if not len(keep):
        return {"n": 0, "fit": None, "why": "no construct carries both a date column entry "
                                            "and both predictors"}
    events = int(np.isfinite(hi).sum())
    if events < 10:
        return {"n": len(keep), "n_events": events, "fit": None,
                "why": "fewer than ten events: a regression on this column would be an interval "
                       "and not an estimate"}
    rng = np.random.default_rng(seed)
    out = {"n": len(keep), "n_events": events, "models": {}}
    for label, use in (("both", names), ("frequency_alone", names[:1]),
                       ("exposure_alone", names[1:])):
        X = np.column_stack([np.ones(len(keep))] +
                            [[pred[c][n] for c in keep] for n in use])
        fit = aft_fit(X, lo, hi, dist)
        cols = ["intercept"] + list(use)
        boot = []
        for _ in range(draws):
            k = rng.integers(0, len(keep), len(keep))
            f = aft_fit(X[k], lo[k], hi[k], dist)
            if f["converged"]:
                boot.append(f["beta"])
        B = np.array(boot) if boot else np.zeros((0, len(cols)))
        coef = {}
        for j, n in enumerate(cols):
            e = {"beta": float(fit["beta"][j])}
            if len(B):
                e["lo"], e["hi"] = (float(np.percentile(B[:, j], 2.5)),
                                    float(np.percentile(B[:, j], 97.5)))
                e["excludes_zero"] = bool(e["lo"] * e["hi"] > 0)
            coef[n] = e
        out["models"][label] = {"predictors": list(use), "converged": fit["converged"],
                                # `aft_fit` returns the NEGATIVE log likelihood as `nll`;
                                # the sign is carried here so a comparison between the two
                                # single-predictor models reads the right way round
                                "log_likelihood": -float(fit["nll"]),
                                "coefficients": coef, "n_bootstrap": len(boot)}
    # the exposure coefficient in the (frequency, exposure) parameterisation, for the record
    b = out["models"]["both"]["coefficients"]
    out["exposure_total_coefficient"] = {
        "value": b["log_exposure_multiplier"]["beta"],
        "note": "in the (log frequency, log multiplier) parameterisation the multiplier's "
                "coefficient IS the coefficient on log total exposure holding frequency fixed, "
                "since log exposure = log frequency + log multiplier"}
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
    p2 = sorted(set(json.loads((D / "validation/p2/P2_GATE_S15.json").read_text())["p2"]) & set(p1))
    pilot = pilot_30()
    sets = {"with_pilot": p2, "without_pilot": [c for c in p2 if c not in pilot]}
    sp = json.loads((R2 / f"CURVE1_SPACING_{br}.json").read_text())
    print(f"branch {br}: |P2 n P1| = {len(p2)}, without the pilot's 30: "
          f"{len(sets['without_pilot'])}")

    rep = {"branch": br, "within_group_clause": {
        "status": "UNTESTABLE",
        "why": "the clause compares members' STABILITY dates, and A38's spacing check leaves no "
               "construct of P1 with one on this branch: 0 dated in the product space and 0 in "
               "the verdict layer once two spacing bands must agree",
        "n_dated_product": sp["product"]["n_dated"], "n_dated_layer": sp["layer"]["n_dated"],
        "spacing_dependence_spearman": sp["spacing_dependence"]["spearman"]},
        "exploratory_level_and_trend": {}, "exploratory_norm_dates": {},
        "exploratory_two_predictor": {},
        "marked": "both forms below are EXPLORATORY forms of P3 and are never its verdict "
                  "(the Settle chat, 15 Oct)"}
    print(f"\nTHE CLAUSE AS WRITTEN: UNTESTABLE -- {rep['within_group_clause']['why'][:88]}...")

    idxc = json.loads((R2 / "R2_INDEX_CURVES.json").read_text())["curve1"]
    shape = idxc[f"{br}|product|ratio"]["shape"]
    rep2 = json.loads((R2 / "R2_REPORT.json").read_text())
    norm = (rep2["curves"].get(f"{br}|curve2_norm") or {}).get("dates") or {}
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {**ast, **bi}

    gs = groups(p2)
    print(f"\n=== form 1: {len(gs)} shared-token groups inside P2 (exploratory) ===")
    for tok, cs in sorted(gs.items(), key=lambda x: -len(x[1])):
        lev = {c: shape[c]["median"] for c in cs if c in shape}
        trd = {c: (shape[c]["spearman_vs_log_tokens"], shape[c].get("permutation_p"))
               for c in cs if c in shape}
        nd = {}
        for c in cs:
            e = norm.get(c) or {}
            if (e.get("flat") or e.get("label") or e.get("outcome")) == "dated":
                nd[c] = e.get("tokens")
        fr = {c: (rates.get(contract_name(c)) or {}).get("per_file") for c in cs}
        row = {"members": cs, "levels": lev, "trends": trd, "norm_dates": nd,
               "role_frequency_per_file": fr}
        if len(lev) >= 2:
            row["level_spread"] = float(max(lev.values()) - min(lev.values()))
            row["levels_within_one_noise_unit"] = bool(
                row["level_spread"] <= float(np.median([shape[c]["noise_unit"] for c in lev])))
        if len(nd) >= 2:
            vals = [v for v in nd.values() if v]
            row["norm_dates_coincide"] = len(set(vals)) == 1
            row["norm_date_spread_decades"] = float(np.log10(max(vals) / min(vals))) if vals else None
            # do they order by role frequency? Spearman of date against frequency inside the group
            pairs = [(fr[c], nd[c]) for c in nd if fr.get(c) and nd[c]]
            if len(pairs) >= 3:
                row["spearman_date_vs_frequency"] = float(spearman(
                    np.array([np.log(x) for x, _ in pairs]),
                    np.array([np.log(y) for _, y in pairs])))
        rep["exploratory_level_and_trend"][tok] = row
        print(f"  {tok!r:8s} {len(cs)} members: levels "
              + " ".join(f"{c.split('__')[0][:10]}={lev[c]:.3f}" for c in sorted(lev))
              + (f"  spread {row.get('level_spread', float('nan')):.3f}"
                 f" ({'within' if row.get('levels_within_one_noise_unit') else 'beyond'} "
                 f"one noise unit)" if "level_spread" in row else ""))
        if nd:
            print(f"           norm dated {len(nd)} of {len(cs)}: "
                  + " ".join(f"{c.split('__')[0][:10]}={v/1e9:.0f}B" for c, v in sorted(nd.items()))
                  + (f"  coincide={row.get('norm_dates_coincide')}"
                     f" spread {row.get('norm_date_spread_decades', 0):.2f} dec" if len(nd) >= 2
                     else ""))

    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    from experiments.construct_token_loss import branches
    toks = sorted(t for _, t in branches(schedule)[br])
    col = {c: ((norm.get(c) or {}).get("tokens")
               if (norm.get(c) or {}).get("label", (norm.get(c) or {}).get("outcome")) == "dated"
               else None)
           for c in norm}

    print(f"\n=== form 2: the two-predictor regression on the NORM date (exploratory) ===")
    for which, members in sets.items():
        pred = predictors2(members)
        for dist in ("lognormal", "weibull"):
            r = two_predictor(col, members, pred, toks, draws=a.aft_draws, dist=dist)
            rep["exploratory_two_predictor"][f"{which}|{dist}"] = r
            if r.get("fit", "x") is None:
                print(f"  [{which:13s}|{dist:9s}] n={r['n']:3d} NO FIT: {r['why'][:60]}")
                continue
            both = r["models"]["both"]["coefficients"]
            print(f"  [{which:13s}|{dist:9s}] n={r['n']:3d} events={r['n_events']:3d}  "
                  f"together: freq {both['log_frequency']['beta']:+.3f}"
                  f"{'*' if both['log_frequency'].get('excludes_zero') else ' '} "
                  f"exposure {both['log_exposure_multiplier']['beta']:+.3f}"
                  f"{'*' if both['log_exposure_multiplier'].get('excludes_zero') else ' '}")
            for lab in ("frequency_alone", "exposure_alone"):
                m = r["models"][lab]
                n = m["predictors"][0]
                c = m["coefficients"][n]
                print(f"{'':32s}alone: {n:24s} {c['beta']:+.3f}"
                      f"{'*' if c.get('excludes_zero') else ' '} "
                      f"[{c.get('lo', float('nan')):+.3f},{c.get('hi', float('nan')):+.3f}]  "
                      f"loglik {m['log_likelihood']:.2f}")

    out = pathlib.Path(a.out) / f"STEP6_TOKEN_{br}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
