"""Every curve-2 reading, re-read on the RAW norm -- the form C0.12 makes primary.

**Why.** A41 measured that R2's curve 2 is the **standardised** norm: the per-core arrays the
bootstrap pass reads match `A_mean` to 4e-7 and differ from `A_raw` by up to 70, on 32 of 32 npz.
C0.12 and C10a make the **raw** form primary and date curve 2 **on the raw curve**, because the
pilot's Fact 3 was a growth that existed in the standardised form only. So four readings were
taken on the form the contract puts beside, and this module re-takes them on the primary.

**And the two forms disagree about almost every construct.** Measured over all 169 of P1 and all
35 pretraining references: the direction over the record **disagrees for 120**, and in every one of
those the **raw norm falls while the standardised rises** -- 145 of 169 raw norms fall against 144
of 169 standardised norms rising, median first-checkpoint share **1.173 raw** against **0.806
standardised**. This is sigma-0 shrinking as ordinary code becomes more predictable, at scale.

**The four readings re-taken, each against what it said before.**

* **A34**, the agreement reading on the constructs whose norm, depth and sparsity all date;
* **step 3's second sense**, the pairwise Spearman of date columns;
* **step 3's third sense**, the regression on the norm's date -- the one column that carried a
  frequency effect;
* **step 6's** group ordering and two-predictor regression, both of which read the norm's date.

**One difference that is not a finding and is marked as one.** The raw curve has **no per-core
arrays**, so its unit is the median point-to-point change rather than the core-bootstrap SE, and
its dates are therefore produced by a different noise unit as well as a different series. For
like-with-like, the standardised column here is re-derived with the **same** point-to-point unit,
so it is not R2's own curve-2 column: R2 reports 75 dates with the core bootstrap, this module's
standardised re-derivation gives 72. Comparisons below are raw-against-re-derived-standardised,
and the R2 figure is quoted whenever it differs.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.step3_senses import DRAWS, SEED, aft_column, date_columns, pair, pilot_30, predictors
from experiments.step6_token import groups, predictors2, two_predictor
from profilebattery.profiles.names import contract_name, tree_name
from profilebattery.paths import data_root

D = data_root()
R2 = D / "validation/r2"
OWED = R2 / "owed"


def strip_level(label: str | None) -> str | None:
    """A30's level sub-labels are a **loss** test and do not apply to a norm curve.

    `r2_report.py` strips them for the same reason; keeping them here would relabel 92 dated raw
    curves as `dated_never_learned` on a comparison with an untrained norm that says nothing about
    competence.
    """
    return None if label is None else label.replace("_level_unknown", "")


def raw_dates(branch: str = "pretrain", form: str = "raw") -> dict:
    """`{construct: tokens | None}` from the owed pass's curve, in the convention step 3 uses.

    `None` is censored-or-transient (a date beyond the record); a construct absent from the dict is
    unreadable on this curve. The distinction is step 3's and is preserved.
    """
    doc = json.loads((OWED / f"CURVE2_FORMS_{branch}.json").read_text())
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    from experiments.construct_token_loss import branches
    tok = dict(branches(sched)[branch])
    out = {}
    for c, v in doc["curves"].items():
        e = v.get(form) or {}
        lab = strip_level(e.get("flat") or e.get("label") or e.get("outcome"))
        if lab == "dated":
            i, refs = e.get("index"), e.get("references") or []
            out[c] = tok.get(refs[i]) if i is not None and 0 <= i < len(refs) else None
        elif lab in ("censored", "transient"):
            out[c] = None
    return out


def agreement(norm: dict, branch: str = "pretrain") -> dict:
    """A34's population and ordering, with the norm column swapped for the raw one.

    A34's own machinery re-dates from the stored series, which exists only for the standardised
    form, so what is re-taken here is the part that depends on the date: the population of
    constructs dated on all three profile curves, and the order among those with distinct dates.
    """
    cols = date_columns(branch)
    depth = {c: t for c, t in cols["curve3_depth"].items() if t is not None}
    spars = {c: t for c, t in cols["curve6_sparsity"].items() if t is not None}
    dated = {c: t for c, t in norm.items() if t is not None}
    pop = sorted(set(dated) & set(depth) & set(spars))
    order, ties = collections.Counter(), collections.Counter()
    for c in pop:
        t = {"norm": dated[c], "depth": depth[c], "sparsity": spars[c]}
        if len(set(t.values())) == 3:
            order[" < ".join(sorted(t, key=lambda k: t[k]))] += 1
        else:
            ties["=".join(sorted(k for k in t if list(t.values()).count(t[k]) > 1))] += 1
    return {"n_population": len(pop), "population": pop,
            "n_strict": sum(order.values()), "n_with_tie": sum(ties.values()),
            "order_counts_strict_only": dict(order), "tie_patterns": dict(ties),
            "n_norm_first": sum(n for k, n in order.items() if k.startswith("norm"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--aft-draws", type=int, default=200)
    ap.add_argument("--out", default=str(OWED))
    a = ap.parse_args()
    br = a.branch

    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    p2 = sorted(set(json.loads((D / "validation/p2/P2_GATE_S15.json").read_text())["p2"]) & set(p1))
    pilot = pilot_30()
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    from experiments.construct_token_loss import branches
    toks = sorted(t for _, t in branches(sched)[br])

    forms = {f: raw_dates(br, f) for f in ("raw", "standardised")}
    rep = {"branch": br, "primary": "raw, by C0.12 and C10a",
           "n_dated": {f: sum(1 for v in d.values() if v is not None) for f, d in forms.items()},
           "note": "the standardised column here is re-derived with the raw form's point-to-point "
                   "unit for like-with-like, so it is not R2's curve-2 column (75 dates with the "
                   "core bootstrap)"}
    print(f"dates: {rep['n_dated']}  (R2's own standardised curve: 75)")

    # ---- A34 ----
    rep["a34"] = {f: agreement(forms[f], br) for f in forms}
    for f, v in rep["a34"].items():
        print(f"\nA34 on the {f:12s} norm: population {v['n_population']}, strict "
              f"{v['n_strict']}, ties {v['n_with_tie']}; norm dates FIRST in {v['n_norm_first']}")
        for k, n in sorted(v["order_counts_strict_only"].items(), key=lambda x: -x[1])[:4]:
            print(f"     {k:34s} {n}")

    # ---- step 3's two senses ----
    rep["second_sense"] = {}
    cols = date_columns(br)
    for f in forms:
        c2 = forms[f]
        out = {}
        for other in ("curve3_depth", "curve6_sparsity", "curve4_loss", "curve1_turnover"):
            out[f"curve2_norm x {other}"] = pair(c2, cols[other], p1, a.draws)
        rep["second_sense"][f] = out
        sig = [(k, v["spearman"]) for k, v in out.items()
               if v.get("permutation_p") is not None and v["permutation_p"] < 0.05]
        print(f"\nsecond sense, {f:12s}: {len(sig)} of {len(out)} pairs outside their interval")
        for k, v in out.items():
            if v.get("spearman") is None:
                print(f"   {k:36s} UNDEFINED: {v['why'][:48]}")
            else:
                print(f"   {k:36s} rho {v['spearman']:+.3f} p={v['permutation_p']:.4f} "
                      f"n={v['n']}")

    rep["third_sense"] = {}
    for which, members in (("with_pilot", p2), ("without_pilot", [c for c in p2 if c not in pilot])):
        pred = predictors(members)
        for f in forms:
            r = aft_column(forms[f], members, pred, toks, draws=a.aft_draws)
            rep["third_sense"][f"{which}|{f}"] = r
            if r.get("fit") is None:
                print(f"third sense [{which:13s}|{f:12s}] NO FIT: {r['why'][:52]}")
                continue
            b = r["coefficients"]["log_frequency"]
            print(f"third sense [{which:13s}|{f:12s}] n={r['n']:3d} events={r['n_events']:3d}  "
                  f"log_freq {b['beta']:+.3f}{'*' if b.get('excludes_zero') else ' '} "
                  f"[{b.get('lo'):+.3f},{b.get('hi'):+.3f}] perm p="
                  f"{r['permutation_p_log_frequency']}")

    # ---- step 6: the group ordering and the two predictors ----
    rep["step6"] = {}
    gs = groups(p2)
    for f in forms:
        c2 = forms[f]
        rows = {}
        for tok_, cs in gs.items():
            nd = {c: c2[c] for c in cs if c2.get(c)}
            e = {"members": cs, "n_dated": len(nd), "dates": nd}
            if len(nd) >= 3:
                ast = json.loads((D / "calibration/construct_rates.json").read_text())
                bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
                rates = {**ast, **bi}
                prs = [((rates.get(contract_name(c)) or {}).get("per_file"), nd[c]) for c in nd]
                prs = [(x, y) for x, y in prs if x and y]
                if len(prs) >= 3:
                    from experiments.step3_senses import spearman
                    e["spearman_date_vs_frequency"] = float(spearman(
                        np.array([np.log(x) for x, _ in prs]),
                        np.array([np.log(y) for _, y in prs])))
                    e["spread_decades"] = float(np.log10(max(y for _, y in prs)
                                                         / min(y for _, y in prs)))
            rows[tok_] = e
        two = {}
        for which, members in (("with_pilot", p2),
                               ("without_pilot", [c for c in p2 if c not in pilot])):
            two[which] = two_predictor(c2, members, predictors2(members), toks,
                                       draws=a.aft_draws)
        rep["step6"][f] = {"groups": rows, "two_predictor": two}
        g = rows.get("for") or {}
        print(f"\nstep 6, {f:12s}: `for` group {g.get('n_dated')} dated of "
              f"{len(g.get('members', []))}, spearman date vs frequency "
              f"{g.get('spearman_date_vs_frequency')}, spread "
              f"{g.get('spread_decades')} decades")
        m = (two["with_pilot"].get("models") or {}).get("both")
        if m:
            print(f"   two predictors together: freq "
                  f"{m['coefficients']['log_frequency']['beta']:+.3f}"
                  f"{'*' if m['coefficients']['log_frequency'].get('excludes_zero') else ' '} "
                  f"exposure {m['coefficients']['log_exposure_multiplier']['beta']:+.3f}"
                  f"{'*' if m['coefficients']['log_exposure_multiplier'].get('excludes_zero') else ' '}")

    out = pathlib.Path(a.out) / f"RAW_NORM_REREAD_{br}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
