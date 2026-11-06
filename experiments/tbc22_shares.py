r"""[TBC: 22] -- the six cuts' shares of their own ceiling, restated on the 169-construct ceilings.

The six cuts are the quantile cuts (top 1%, 2%, 5% of the standardised entries) and the
magnitude bands (|e| >= 1, 2, 3 sigma0). The reading is taken at the **first checkpoint** as the
start point (step 1,000, 4.19 B tokens) over the **20,000-30,000 step** gap window, which is the
window Figure 1(c) draws.

**The excess has to be un-normalised first.** `rate_readings.py` divides the six cuts' excesses
(and only theirs -- not `carrier5`, `cosine` or `spearman`) by a headroom taken from
B14_BANDS.json: one twin ceiling minus chance, measured at the **last** reference and applied to
every start point. Taken straight, `q5pct` therefore reads 0.913 where `carrier5`, the same set
undivided, reads 0.725, and dividing again by a ceiling put all six shares above 1.0 -- above
their own ceiling, which cannot happen. The B14 headroom is a scalar per band, so multiplying it
back commutes with the median and with the bootstrap quantiles, and the restatement is exactly
the substitution the item asks for: B14's single last-reference headroom out, the 169-construct
ceiling at this start point in.

Share = (median excess x B14 headroom) / ceiling, where

* the **excess** is the change beyond the twin turnover at the earlier reference of the pair
  (RATE_READINGS.json, the fit-free primary), and
* the **ceiling** is the twin agreement minus chance, with chance measured **across the pair's
  two references** -- between different constructs with one profile at each end of the pair,
  which is the quantity the excess itself is computed on (CEILINGS_PAIRWISE_169.json, all 169
  constructs, not the 16-construct sample).

The interval is the excess's own 95% interval divided by the ceiling. The ceiling is treated as
fixed, because it is a property of the pair of references and not of the construct sample being
resampled; the figure stated this way elsewhere, and stating it again here keeps the two
readings comparable.

    python experiments/tbc22_shares.py
"""
from __future__ import annotations

import json
import time

from experiments.atlas import provenance
from experiments.relatedness import OUT2

WINDOW = "20k-30k"
START = 1000
CUTS = [("q1pct", "top 1% of the standardised entries"),
        ("q2pct", "top 2%"),
        ("q5pct", "top 5% (the carrier set)"),
        ("abs1sd", "|e| >= 1 sigma0"),
        ("abs2sd", "|e| >= 2 sigma0"),
        ("abs3sd", "|e| >= 3 sigma0")]


def main() -> None:
    t0 = time.time()
    rr = json.loads((OUT2 / "RATE_READINGS.json").read_text())["primary"]
    cw = json.loads((OUT2 / "CEILINGS_PAIRWISE_169.json").read_text())["per_window"]
    b14 = json.loads((OUT2 / "B14_BANDS.json").read_text())["per_band"]
    head = {b: b14[b]["twin_ceiling_last_ref"] - b14[b]["median_chance_kept_fraction"]
            for b, _ in CUTS}

    doc = {"what": f"[TBC: 22] the six cuts' shares of their own ceiling at {WINDOW} steps "
                   f"from the first checkpoint",
           "definition": {
               "start_point": f"step {START}, the first checkpoint (4.19 B tokens)",
               "window": f"{WINDOW} steps of gap",
               "excess": "median over constructs of the change beyond the twin turnover at the "
                         "pair's earlier reference (RATE_READINGS.json, fit-free primary), "
                         "MULTIPLIED BACK by the B14 headroom that rate_readings.py divides "
                         "the six cuts by, so that the only normalisation left is the ceiling "
                         "below",
               "b14_headroom": "twin_ceiling_last_ref - median_chance_kept_fraction, per band, "
                               "from B14_BANDS.json; a scalar, so it commutes with the median "
                               "and with the bootstrap quantiles",
               "ceiling": "twin agreement minus chance, chance measured ACROSS the pair's two "
                          "references (CEILINGS_PAIRWISE_169.json, all 169 constructs)",
               "share": "median excess / ceiling",
               "interval": "the excess's own 95% interval divided by the same ceiling; the "
                           "ceiling is held fixed, being a property of the reference pair "
                           "rather than of the resampled construct sample",
               "also_given": "the within-checkpoint ceiling, for comparison only"},
           "fixed_before_computing": True,
           "start_iter": START, "window": WINDOW, "cuts": {}}

    ce = {r["start_iter"]: r for r in cw[WINDOW]}
    if START not in ce:
        raise SystemExit(f"no ceiling row at start {START} in window {WINDOW}")
    crow = ce[START]
    print(f"  the six cuts at start {START:,} (4.19 B), gap {WINDOW}, "
          f"{crow['n_pairs']} pair(s):\n")
    print(f"  {'cut':<9s} {'excess':>22s} {'ceiling':>9s} {'share of the ceiling':>26s}")
    for key, label in CUTS:
        rows = {r["start_iter"]: r for r in rr[key][WINDOW]}
        if START not in rows or key not in crow:
            raise SystemExit(f"missing {key} at start {START}")
        r, c = rows[START], crow[key]
        ceil = c["ceiling_cross"]
        if not ceil > 0:
            raise SystemExit(f"{key}: non-positive ceiling {ceil}")
        h = head[key]
        raw = r["median_excess"] * h
        share = raw / ceil
        lo, hi = (r["ci95"][0] * h / ceil, r["ci95"][1] * h / ceil)
        # a share of its OWN ceiling cannot exceed 1; a value above 1 means two
        # normalisations have been applied, which is how the first run of this was caught
        if not 0.0 <= share <= 1.0:
            raise SystemExit(f"REFUSED: {key}: share of its own ceiling is {share:.3f}, "
                             f"outside [0, 1] -- excess {raw:.4f}, ceiling {ceil:.4f}")
        doc["cuts"][key] = {
            "label": label, "n_pairs": r["n_pairs"], "n_constructs": r["n_constructs"],
            "median_excess_as_stored": r["median_excess"], "b14_headroom": h,
            "median_excess_raw": raw,
            "excess_ci95_raw": [r["ci95"][0] * h, r["ci95"][1] * h],
            "twin_agreement": c["twin"], "chance_cross": c["chance_cross"],
            "ceiling_cross": ceil, "share_of_ceiling": share, "share_ci95": [lo, hi],
            "ceiling_within": c["ceiling_within"],
            "share_of_within_ceiling": raw / c["ceiling_within"],
            "median_twin_floor": r.get("median_twin_floor")}
        print(f"  {key:<9s} {raw:.3f} "
              f"[{r['ci95'][0]*h:.3f}, {r['ci95'][1]*h:.3f}] {ceil:9.3f} "
              f"{share:12.3f} [{lo:.3f}, {hi:.3f}]")

    sh = [v["share_of_ceiling"] for v in doc["cuts"].values()]
    doc["range_of_shares"] = [min(sh), max(sh)]
    doc["spread"] = max(sh) - min(sh)
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["validation/paper2/RATE_READINGS.json",
                                    "validation/paper2/CEILINGS_PAIRWISE_169.json",
                                    "validation/paper2/B14_BANDS.json"])
    (OUT2 / "TBC22_SHARES.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"\n  shares run {min(sh):.3f} to {max(sh):.3f}, a spread of {max(sh)-min(sh):.3f}")
    print(f"-> {OUT2 / 'TBC22_SHARES.json'}")


if __name__ == "__main__":
    main()
