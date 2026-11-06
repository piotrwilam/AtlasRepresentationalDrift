"""A34: the exploratory agreement reading on the constructs whose norm, depth and sparsity date.

**Why it is exploratory and not part 4.** Part 4 asks whether a construct's **six** dates coincide,
on **P4** -- and step 0b measured **|P4| = 0**: no construct carries all six, nor all five of the
curves that are dated at all. The three **profile** curves (2 norm, 3 depth, 6 sparsity) do agree
on a population, and A34 reads the agreement question there instead. It is reported **beside** the
pre-registered reading, never in its place, and the three curves are named wherever it appears.

**The coincidence criterion is not the contract's, and this is the reason.** Part 3 says each date
"carries its bootstrap spread in decades of tokens". **That spread is not computable from what is
stored.** A date's spread requires the curve re-dated under each of the 1,000 draws; the bootstrap
artefact holds the per-checkpoint **mean and SE**, not the per-draw curves, so the draws that
would move a date are gone. Emitting per-draw dates is a small addition to the Colab pass and is
named as owed.

In its place, two units that **are** measurable, both labelled as what they are:

* **the instrument's resolution** -- the spacing of the recorded checkpoints. Two dates on the
  same checkpoint coincide trivially; two dates one 1.5x step apart are one resolution element
  apart and cannot be distinguished by this record whatever their spreads.
* **a date-stability sensitivity** -- each checkpoint perturbed by +/- 1.96 SE and the curve
  re-dated, giving the span over which the date moves. This is **not** a bootstrap spread: it
  moves every checkpoint together rather than resampling cores, so it is a worst-case envelope
  and is wider than the quantity part 3 asks for.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import label_curve
from experiments.r0b_gates import dated_set
from profilebattery.paths import data_root

D = data_root()
R2 = D / "validation/r2"
CURVES = {"curve2_norm": "norm", "curve3_depth": "depth", "curve6_sparsity": "sparsity"}


def population(rep: dict, branch: str) -> set:
    """The constructs dated on all three profile curves."""
    sets = [dated_set((rep["curves"].get(f"{branch}|{k}") or {}).get("dates") or {})
            for k in CURVES]
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out


def date_envelope(series: dict, branch: str) -> dict:
    """Re-date the curve with every checkpoint shifted by +/- 1.96 SE. A worst-case envelope.

    Not a bootstrap spread: the shift is applied to **all** checkpoints at once, in the same
    direction, so it answers "how far can this date move if the whole curve sits high or low?"
    and not "where does this date fall across resamplings of the cores?". Reported as
    `envelope_*` so the two are never confused.
    """
    v = np.asarray(series["values"], float)
    u = np.asarray([np.nan if x is None else x for x in series["units"]], float)
    toks = series.get("tokens")
    step = "consecutive" if branch == "anneal" else "factor"
    out = {}
    for name, shift in (("low", -1.96), ("high", +1.96)):
        vv = v + shift * np.nan_to_num(u, nan=0.0)
        d = label_curve(list(vv), list(u), toks, step=step)
        out[name] = (toks[d["index"]] if "index" in d and toks and
                     0 <= d["index"] < len(toks) else None)
        out[f"{name}_outcome"] = d.get("label") or d.get("outcome")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()
    br = a.branch

    rep = json.loads((R2 / "R2_REPORT.json").read_text())
    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    from experiments.construct_token_loss import branches
    seq = dict(branches(schedule)[br])

    pop = sorted(population(rep, br))
    print(f"A34 population on {br}: {len(pop)} constructs dated on norm, depth and sparsity")

    # the instrument's resolution: the spacing of the recorded checkpoints, in decades
    refs = [r for r in seq if (R2 / "R2_REPORT.json").exists()]
    toks = sorted(seq[r] for r in (rep["curves"][f"{br}|curve2_norm"]["series"]
                                   .get(pop[0], {}).get("references", []) or []))
    gaps = [np.log10(toks[i + 1] / toks[i]) for i in range(len(toks) - 1)] if len(toks) > 1 else []
    res = float(np.median(gaps)) if gaps else float("nan")
    print(f"  checkpoint spacing: median {res:.3f} decades "
          f"({10 ** res:.2f}x per step, {len(toks)} checkpoints)")

    rows = {}
    for c in pop:
        e = {"dates": {}, "envelope": {}}
        for key, short in CURVES.items():
            blk = rep["curves"][f"{br}|{key}"]
            d = blk["dates"][c]
            ser = dict(blk["series"][c])
            ser["tokens"] = [seq[r] for r in ser["references"]]
            e["dates"][short] = d.get("tokens")
            e["envelope"][short] = date_envelope(ser, br)
        t = {k: v for k, v in e["dates"].items() if v}
        if len(t) == 3:
            dec = {f"{x}-{y}": abs(np.log10(t[x] / t[y]))
                   for x, y in itertools.combinations(t, 2)}
            e["pairwise_decades"] = dec
            e["max_pairwise_decades"] = max(dec.values())
            e["same_checkpoint"] = len(set(t.values())) == 1
            e["within_one_step"] = max(dec.values()) <= res * 1.001
        rows[c] = e

    full = [c for c in pop if "max_pairwise_decades" in rows[c]]
    spread = [rows[c]["max_pairwise_decades"] for c in full]
    print(f"\n  constructs with all three dates in tokens: {len(full)}")
    print(f"  max pairwise separation: median {np.median(spread):.3f} decades, "
          f"min {min(spread):.3f}, max {max(spread):.3f}")
    print(f"  all three on the SAME checkpoint: {sum(rows[c]['same_checkpoint'] for c in full)}")
    print(f"  all three within ONE checkpoint step: "
          f"{sum(rows[c]['within_one_step'] for c in full)}")
    for k in (1, 2, 3):
        n = sum(1 for s in spread if s <= res * k)
        print(f"    within {k} step(s) ({res*k:.2f} decades): {n} of {len(full)}")

    # **Ties first, because they make an order meaningless.** `sorted`, `min` and `max` break a
    # tie by dictionary order, so a construct whose norm and sparsity date on the SAME checkpoint
    # gets an arbitrary ordering -- and two summaries computed that way disagreed: the orderings
    # put `norm` last for 3 constructs and `max()` for 9, the difference being nothing but tie
    # resolution. 12 of 26 constructs have at least one tie. So a strict order is reported only
    # where all three dates differ, and the ties are reported as ties.
    strict = [c for c in full if len(set(rows[c]["dates"].values())) == 3]
    tied = [c for c in full if c not in strict]
    for c in full:
        rows[c]["strict_order"] = c in strict
        t = rows[c]["dates"]
        rows[c]["tie_pattern"] = sorted(
            "=".join(sorted(k for k in t if t[k] == v)) for v in sorted(set(t.values())))
    print(f"\n  all three dates DISTINCT: {len(strict)} of {len(full)}; "
          f"at least one TIE: {len(tied)}")
    order = collections.Counter(" < ".join(sorted(rows[c]["dates"],
                                                  key=lambda k: rows[c]["dates"][k]))
                                for c in strict)
    print(f"  the order, on the {len(strict)} with distinct dates:")
    for k, n in order.most_common():
        print(f"    {k:34s} {n:3d}")
    pat = collections.Counter(" < ".join(rows[c]["tie_pattern"]) for c in tied)
    print(f"  the tie patterns, on the {len(tied)} with a tie:")
    for k, n in pat.most_common():
        print(f"    {k:34s} {n:3d}")

    # the marginals on the strict set only, where "first" and "last" mean something
    first = collections.Counter(min(rows[c]["dates"], key=lambda k: rows[c]["dates"][k])
                                for c in strict)
    last = collections.Counter(max(rows[c]["dates"], key=lambda k: rows[c]["dates"][k])
                               for c in strict)
    print(f"\n  on the {len(strict)} strict: dates FIRST {dict(first.most_common())}; "
          f"LAST {dict(last.most_common())}")

    # how far the worst-case envelope moves a date, as a check on whether the separations
    # above survive any plausible uncertainty at all
    moved = []
    for c in full:
        for short, e in rows[c]["envelope"].items():
            lo, hi = e.get("low"), e.get("high")
            if lo and hi:
                moved.append(abs(np.log10(hi / lo)))
    if moved:
        print(f"\n  worst-case envelope on a single date: median {np.median(moved):.3f} "
              f"decades, max {max(moved):.3f} (NOT a bootstrap spread -- see the docstring)")

    out = {"branch": br, "n_population": len(pop), "population": pop,
           "resolution_decades": res,
           "criterion": "NOT part 3's bootstrap spread, which is not computable from the "
                        "stored artefact (it needs the curve re-dated under each draw). Two "
                        "units are used instead and labelled: the recorded checkpoint spacing, "
                        "and a worst-case envelope from shifting every checkpoint by 1.96 SE.",
           "n_same_checkpoint": sum(rows[c]["same_checkpoint"] for c in full),
           "n_within_one_step": sum(rows[c]["within_one_step"] for c in full),
           "median_max_pairwise_decades": float(np.median(spread)),
           "n_strict_order": len(strict), "n_with_tie": len(tied),
           "order_counts_strict_only": dict(order), "tie_patterns": dict(pat),
           "dates_first_strict_only": dict(first), "dates_last_strict_only": dict(last),
           "ties_note": "an order is reported only where all three dates differ: sorted/min/max "
                        "break a tie by dictionary order, and two summaries computed that way "
                        "disagreed by 6 constructs on which curve dates last",
           "constructs": rows}
    p = pathlib.Path(a.out) / f"A34_AGREEMENT_{br}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
