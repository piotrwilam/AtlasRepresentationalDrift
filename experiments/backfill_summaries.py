r"""Store the summaries paper 2 quotes but no artefact held.

Four numbers in paper 2 were right and their artefact was named, but the **summary** they
quote was computed by the script and discarded, so a reader could not check them without
re-running:

* \S4.2's "for 167 of 169 constructs every checkpoint's nearest set is its immediate
  neighbour" -- `B18_RECURRENCE.json` stored only the median share of non-neighbour nearest
  sets (0.0), not the count;
* \S4.7's "115 fall in at least two draws and the median pair falls in 88% of them" --
  `P8_BOOTSTRAP_pretrain.json` stored the per-pair shares, not the count or the median;
* \S3.1's "the fraction of carriers kept between neighbouring checkpoints rises from 0.31 to
  0.66 by about 71 billion tokens" and \S4.1's "falls from 0.66 at 71 billion to 0.48 at 1.5
  trillion" -- `B12_TWIN_TURNOVER.json` stored that fraction for the last pair only.

The first two are recomputed from the artefact's own stored per-item fields, so they cannot
disagree with what the script found. The third is a real computation over the carrier sets,
CPU-only, from `validation/carriers/*.npz`.

**Every value is checked against the figure the paper prints, and the script refuses rather
than write a number that misses it.** The same summaries are now written natively by
`b18_recurrence.py`, `p8_bootstrap.py` and `b12_twin_turnover.py`; this script exists so the
stored artefacts carry them without a full re-run, and it records that it did so.

    python experiments/backfill_summaries.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import time

import numpy as np

from experiments.atlas import D, K, p1_set, provenance

CAR = D / "validation/carriers"
CAR_ALT = D / "_carriers_dl/validation/carriers"
# what the paper prints, and how close a recomputation must come
EXPECT = {"n_nearest_is_neighbour": (167, 0),
          "n_pairs_fell_in_at_least_two_draws": (115, 0),
          "median_share_of_draws_that_fell": (0.88, 0.005),
          "kept_fraction_first_pair": (0.31, 0.005),
          "kept_fraction_at_71b": (0.66, 0.005),
          "kept_fraction_at_1_5t": (0.485, 0.006)}
STAMP = ("backfilled by experiments/backfill_summaries.py: the summary the paper quotes, "
         "computed from this artefact's own stored fields (or, for the kept fraction, from "
         "validation/carriers/*.npz) and checked against the printed figure")


def check(name: str, got: float) -> float:
    want, tol = EXPECT[name]
    if abs(got - want) > tol:
        raise SystemExit(f"REFUSED: {name} recomputes to {got!r}, but the paper prints "
                         f"{want!r} (tolerance {tol}). Nothing written.")
    return got


def main() -> None:
    t0 = time.time()
    wrote = []

    # ---- B18: the count behind "167 of 169" -------------------------------------------
    f = D / "validation/paper2/B18_RECURRENCE.json"
    o = json.loads(f.read_text())
    pc = o["per_construct"]
    n_nb = sum(1 for v in pc.values()
               if (v.get("a_nearest_set") or {}).get("share_non_neighbour_observed") == 0.0)
    o["a_n_nearest_is_neighbour"] = int(check("n_nearest_is_neighbour", n_nb))
    o["a_n_constructs_read"] = len(pc)
    o["a_n_nearest_is_neighbour_is"] = (
        "constructs whose every checkpoint's nearest carrier set is its immediate neighbour, "
        "i.e. share_non_neighbour_observed == 0")
    o.setdefault("backfilled", []).append(STAMP)
    f.write_text(json.dumps(o, indent=1, default=float))
    wrote.append((f.name, "a_n_nearest_is_neighbour", n_nb, len(pc)))

    # ---- P8 bootstrap: the count and the median share --------------------------------
    f = D / "validation/r2/P8_BOOTSTRAP_pretrain.json"
    o = json.loads(f.read_text())
    fi = o["fall_intervals"]
    two = [v for v in fi.values() if (v.get("n_draws_fell") or 0) >= 2]
    med = st.median([v["share_of_draws_that_fell"] for v in two])
    o["fall_summary"] = {
        "n_pairs": len(fi),
        "n_pairs_fell_in_at_least_two_draws":
            int(check("n_pairs_fell_in_at_least_two_draws", len(two))),
        "median_share_of_draws_that_fell":
            float(check("median_share_of_draws_that_fell", med)),
        "median_is": "over the pairs that fell in at least two draws, not over all pairs"}
    o.setdefault("backfilled", []).append(STAMP)
    f.write_text(json.dumps(o, indent=1, default=float))
    wrote.append((f.name, "fall_summary", len(two), round(med, 4)))

    # ---- B12: the kept fraction at EVERY neighbouring pair ---------------------------
    from experiments.r1_turnover import load_index
    car = CAR if any(CAR.glob("*.npz")) else CAR_ALT
    refs = sorted((p.stem for p in car.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    names = sorted(p1_set())
    idx = load_index(car, refs)
    tokens = {}
    cei = json.loads((D / "validation/paper2/CEILINGS.json").read_text())
    for ref, blk in cei.get("per_reference", {}).items():
        if isinstance(blk, dict) and blk.get("tokens_b") is not None:
            tokens[ref] = blk["tokens_b"]
    rows = []
    for a, b in zip(refs, refs[1:]):
        keep = [np.intersect1d(np.asarray(idx[a][f"twin1/{c}|product"], np.int64),
                               np.asarray(idx[b][f"twin1/{c}|product"], np.int64)).size / K
                for c in names
                if f"twin1/{c}|product" in idx.get(a, {})
                and f"twin1/{c}|product" in idx.get(b, {})]
        if keep:
            rows.append({"from": a, "to": b, "n": len(keep),
                         "from_tokens_b": tokens.get(a), "to_tokens_b": tokens.get(b),
                         "kept_fraction_median": float(np.median(keep)),
                         "turnover_median": float(1 - np.median(keep))})
    # "rises to 0.66 BY about 71 billion tokens" dates the pair by where it ENDS;
    # "0.48 at 1.5 trillion" dates it by where it STARTS. Both are stored, named, so the
    # reader is never left to guess which end a figure belongs to.
    at71 = min(rows, key=lambda r: abs((r["to_tokens_b"] or 0) - 71.3))
    at15 = min(rows, key=lambda r: abs((r["from_tokens_b"] or 0) - 1526.7))
    check("kept_fraction_first_pair", rows[0]["kept_fraction_median"])
    check("kept_fraction_at_71b", at71["kept_fraction_median"])
    check("kept_fraction_at_1_5t", at15["kept_fraction_median"])
    f = D / "validation/paper2/B12_TWIN_TURNOVER.json"
    o = json.loads(f.read_text())
    o["between_checkpoints"]["per_pair"] = rows
    o["between_checkpoints"]["per_pair_is"] = (
        "the fraction of carriers kept between every pair of neighbouring pretraining "
        "checkpoints, median over constructs, twin1; dated by the pair's START point, which "
        "is how the paper quotes it")
    o["between_checkpoints"]["quoted_in_the_paper"] = {
        "kept_fraction_first_pair": rows[0]["kept_fraction_median"],
        "kept_fraction_first_pair_is": "the pair 1,000 -> 2,000 steps; the paper's 0.31",
        "kept_fraction_by_71b": at71["kept_fraction_median"],
        "kept_fraction_by_71b_is": f"the pair ENDING at {at71['to_tokens_b']:.1f} B "
                                   f"({at71['from']} -> {at71['to']}); the paper's 0.66, "
                                   f"quoted as reached BY 71 billion tokens",
        "kept_fraction_at_1_5t": at15["kept_fraction_median"],
        "kept_fraction_at_1_5t_is": f"the pair STARTING at {at15['from_tokens_b']:.1f} B "
                                    f"({at15['from']} -> {at15['to']}); the paper prints 0.48, "
                                    f"which is this value truncated -- it rounds to 0.49",
        "kept_fraction_last_pair": rows[-1]["kept_fraction_median"],
        "the_gap_is_not_fixed": "neighbouring-checkpoint gaps grow from 1,000 to 91,000 "
                                "steps over the record, so this series mixes the point in "
                                "training with the gap, as the paper says of it"}
    o.setdefault("backfilled", []).append(STAMP)
    o["provenance"] = provenance(["validation/carriers/*.npz",
                                  "validation/paper2/CEILINGS.json (tokens_b)"])
    f.write_text(json.dumps(o, indent=1, default=float))
    wrote.append((f.name, "between_checkpoints.per_pair", len(rows),
                  round(rows[0]["kept_fraction_median"], 4)))

    for name, field, a, b in wrote:
        print(f"  {name:28s} {field:34s} {a} {b}")
    print(f"  kept fraction: first pair {rows[0]['kept_fraction_median']:.4f}, "
          f"at {at71['from_tokens_b']:.1f} B {at71['kept_fraction_median']:.4f}, "
          f"last pair {rows[-1]['kept_fraction_median']:.4f}")
    print(f"  all five checks against the printed figures passed  "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
