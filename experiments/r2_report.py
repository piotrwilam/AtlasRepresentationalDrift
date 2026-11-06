"""R2's remainder: curves 2 and 3 assembled, curve 6 dated, and the verdict-layer tie-break.

**What this closes.** The Colab pass wrote one JSON a reference into `validation/bootstraps/`:
per construct, the core-bootstrap interval of the **norm**, the **depth** and the **sparsity** on
twin 1, and at `pretrain_iter_0484865` the **tie-break** -- the share of draws in which each
layer is the argmax of the per-layer norm. This turns those into the three curves and dates them.

* **Curve 2, norm (R4).** The norm as a **share of its value at the branch's last checkpoint**,
  raw form primary. The plan asks for the standardised form beside it because the pilot's Fact 3
  was a growth present in the standardised form and absent in the raw; the bootstrap artefact
  carries the raw norm, so the standardised form is **named as owed** rather than approximated.
* **Curve 3, depth (R5).** The centre of mass over layers 1 to 22, already computed per draw.
* **Curve 6, sparsity (R13).** The values came from the carrier index yesterday and were left
  **undated** because A30's unit is the core-bootstrap SE and it did not exist. It does now, so
  this is where curve 6 is dated.

**The noise unit is now the contract's.** Every curve is dated by A30 with the per-checkpoint
core-bootstrap SE from the pass, and the step rule A33 gives the branch -- the 1.5x step on
pretraining, consecutive references on annealing.

**The verdict-layer tie-break, and the two sensitivities the R1 ruling attached.** Every layer
winning at least 5% of the draws is reported beside the modal layer; the **66 constructs peaking
at the last layer** are reported as a class, since the last layer feeds the unembedding; and every
verdict-layer figure is given **both as the rule stands and with the last layer excluded**.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import branches, label_curve
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
BOOT = D / "_boot_dl/validation/bootstraps"
CARRIERS = D / "_carriers_dl/validation/carriers"
VERDICT_REF = "pretrain_iter_0484865"
TIE_FLOOR = 0.05
RUNNING_EXAMPLE = "For"


def load_boot(root: pathlib.Path, refs) -> dict:
    out = {}
    for r in refs:
        p = root / f"{r}.json"
        if p.exists():
            out[r] = json.loads(p.read_text())
    return out


def pick(boot: dict, refs, c: str, field: str):
    """`(values, units, present_refs)` for one construct's curve, from the interval means."""
    vals, units, got = [], [], []
    for r in refs:
        e = (boot.get(r) or {}).get(c)
        if not e or "error" in e or field not in e:
            continue
        m, se = e[field]["mean"], e[field]["se"]
        if m is None:
            continue
        vals.append(float(m))
        units.append(None if se is None else float(se))
        got.append(r)
    return vals, units, got


def as_share_of_last(vals):
    """Curve 2's form: the norm as a share of its value at the branch's last checkpoint."""
    if not vals or not np.isfinite(vals[-1]) or vals[-1] == 0:
        return None
    return [v / vals[-1] for v in vals]


def date(vals, units, toks, branch: str, untrained=None) -> dict:
    """A30's R bands and step rule, **without its level test**, which is a loss test.

    **Why `untrained` is not passed to `label_curve` here.** A30's level test asks whether a
    curve settles *at or above the untrained value*, and labels it `never_learned` if it does.
    That is a statement about a **loss**: a loss flat at 13.9 nats against 11.5 untrained has
    learned nothing, which is the case (`BrokenPipeError`) the clause was made for. Curves 2, 3
    and 6 are not losses. For the **norm**, settling above the untrained value is *growth* --
    the hypothesis, not a failure; for the **depth**, "above untrained" is a layer index and the
    comparison has no direction at all; for **sparsity** it is concentration, again the
    hypothesis. Passing the untrained value made 75 of 169 norm curves `dated_never_learned`
    on the pretraining branch, which is a category error and not a reading.

    So these curves take A30's `R` bands (flat / unreadable / datable), its step rule, and its
    censoring clauses -- and the untrained value is **recorded beside the date as context**,
    never folded into the label. Whether A30's level test is meant to reach non-loss curves is
    a scope question for the Settle chat; CC's reading is that it is not, and the untrained
    value is carried so the other reading costs nothing to apply later.
    """
    step = "consecutive" if branch == "anneal" else "factor"
    d = label_curve(vals, units, toks, untrained=None, step=step)
    d["step_rule"] = step
    d["n_checkpoints"] = len(vals)
    d["untrained_value"] = untrained
    d["level_test"] = ("not applied: A30's level test is a loss test and this curve is not a "
                       "loss (see `date`)")
    # `label_curve` names an unevaluated level test `dated_level_unknown` / `flat_level_unknown`,
    # which is the loss vocabulary. On a non-loss curve the level test is not *unknown*, it is
    # *not applicable*, so the sub-label is dropped and the outcome stands on its own.
    if d.get("label") == "dated_level_unknown":
        d["label"] = "dated"
    if d.get("flat") == "flat_level_unknown":
        d["flat"] = "flat"
    d.pop("competence", None)
    return d


def summarise(dates: dict) -> dict:
    """Counts by label, and the median date with its spread in decades of tokens."""
    by = collections.Counter((v.get("flat") or v.get("label") or v["outcome"])
                             for v in dates.values())
    tok = [v["tokens"] for v in dates.values() if v.get("tokens")]
    out = {"counts": dict(by), "n_dated": len(tok)}
    if tok:
        lo, hi = np.quantile(tok, [0.25, 0.75])
        out.update({"median_tokens": float(np.median(tok)),
                    "iqr_tokens": [float(lo), float(hi)],
                    "median_decades": float(np.log10(np.median(tok))),
                    "iqr_decades": [float(np.log10(lo)), float(np.log10(hi))]})
    return out


def tie_break_table(boot: dict, floor: float = TIE_FLOOR) -> dict:
    """The verdict layer as the rule stands, and with the last layer excluded.

    The pass stored the per-draw argmax shares. Excluding the last layer cannot be done by
    dropping it from those shares -- the argmax would have to be recomputed over the remaining
    layers, and the per-draw layer norms are not in the artefact. So the exclusion is reported
    as **how many constructs the modal layer is the last one for**, which is the quantity the
    ruling's sensitivity actually needs, and the recomputation is named as owed.
    """
    at = boot.get(VERDICT_REF) or {}
    rows, tied, last = {}, [], []
    for c, e in at.items():
        t = e.get("tie_break_twin1")
        if not t:
            continue
        rows[c] = {"modal_layer": t["modal_layer"], "modal_share": t["modal_share"],
                   "contenders": t["contenders"], "tied": t["tied"]}
        if t["tied"]:
            tied.append(c)
        if t["modal_layer"] == 23:
            last.append(c)
    return {"n": len(rows), "tied": sorted(tied), "n_tied": len(tied),
            "modal_is_last_layer": sorted(last), "n_modal_is_last_layer": len(last),
            "floor": floor, "per_construct": rows,
            "owed": "excluding the last layer requires recomputing the argmax over layers 0-22 "
                    "per draw; the per-draw layer norms are not in the bootstrap artefact, so "
                    "that recomputation is owed and is not approximated here"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", default=str(BOOT))
    ap.add_argument("--out", default=str(D / "validation/r2"))
    a = ap.parse_args()

    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = [tree_name(c) for c in split["p1"]]
    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    boot = load_boot(pathlib.Path(a.boot), [r for b in ("pretrain", "anneal")
                                            for r, _ in branches(schedule)[b]]
                     + ["untrained", "untrained_seed2"])
    print(f"{len(boot)} reference files; P1 = {len(p1)}")

    report = {"n_p1": len(p1), "curves": {}, "tie_break": tie_break_table(boot)}
    tb = report["tie_break"]
    print(f"\nverdict-layer tie-break at {VERDICT_REF}: {tb['n']} constructs")
    print(f"  another layer above {TIE_FLOOR:.0%} of draws: {tb['n_tied']} of {tb['n']}")
    print(f"  modal layer IS the last layer: {tb['n_modal_is_last_layer']} of {tb['n']}")
    shares = [v["modal_share"] for v in tb["per_construct"].values()]
    print(f"  modal share: median {np.median(shares):.3f}, "
          f"min {min(shares):.3f}, max {max(shares):.3f}")

    for branch in ("pretrain", "anneal"):
        seq = [(r, t) for r, t in branches(schedule)[branch] if r in boot]
        refs, toks = [r for r, _ in seq], [t for _, t in seq]
        print(f"\n=== {branch}: {len(refs)} checkpoints ===")

        for name, field, form in (("curve2_norm", "norm_twin1", "share_of_last"),
                                  ("curve3_depth", "depth_twin1", "raw"),
                                  ("curve6_sparsity", "sparsity_twin1", "raw")):
            dates, series = {}, {}
            for c in p1:
                vals, units, got = pick(boot, refs, c, field)
                if len(vals) < 2:
                    dates[c] = {"outcome": "unreadable",
                                "why": "fewer than two checkpoints carry this construct"}
                    continue
                use = as_share_of_last(vals) if form == "share_of_last" else vals
                if use is None:
                    dates[c] = {"outcome": "unreadable",
                                "why": "the last checkpoint's value is zero or not finite, so "
                                       "the share form has no denominator"}
                    continue
                scale = vals[-1] if form == "share_of_last" else 1.0
                un = [None if u is None else u / scale for u in units]
                u0, _, _ = pick(boot, ["untrained"], c, field)
                untr = (u0[0] / scale) if u0 else None
                gt = [toks[refs.index(r)] for r in got]
                d = date(use, un, gt, branch, untr)
                if "index" in d and 0 <= d["index"] < len(got):
                    # the date is an index into the checkpoints this construct actually has,
                    # which is not the branch's full list when a reference is missing it
                    d["reference"] = got[d["index"]]
                    d["tokens"] = gt[d["index"]]
                dates[c] = d
                series[c] = {"references": got, "values": use, "units": un}
            s = summarise(dates)
            msg = f"  {name:16s} {s['counts']}"
            if "median_decades" in s:
                msg += (f"  median date {s['median_tokens']/1e9:.1f} G tokens "
                        f"(IQR {s['iqr_tokens'][0]/1e9:.1f}-{s['iqr_tokens'][1]/1e9:.1f})")
            print(msg)
            report["curves"][f"{branch}|{name}"] = {"form": form, "summary": s,
                                                    "dates": dates, "series": series}

    ex = {}
    for branch in ("pretrain", "anneal"):
        for name in ("curve2_norm", "curve3_depth", "curve6_sparsity"):
            k = f"{branch}|{name}"
            if k in report["curves"] and RUNNING_EXAMPLE in report["curves"][k]["series"]:
                ex[k] = {"series": report["curves"][k]["series"][RUNNING_EXAMPLE],
                         "date": report["curves"][k]["dates"][RUNNING_EXAMPLE]}
    report["running_example"] = {"construct": RUNNING_EXAMPLE, "curves": ex}
    print(f"\nrunning example `{RUNNING_EXAMPLE}`: {len(ex)} curves carried")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "R2_REPORT.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"-> {out / 'R2_REPORT.json'}")


if __name__ == "__main__":
    main()
