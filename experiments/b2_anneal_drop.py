r"""B2 — does the obliged-position loss fall on the annealing branch? (design item 14)

§5 of the draft leaves this open: *"Whether the pretraining floor is a learning-rate floor --
whether the obliged-position loss drops when the rate decays on the annealing branch -- is
[TBC: 14]"*. It matters because §4.3's "the loss stays flat" is the premise of the whole
three-timescale result, and if the loss falls as soon as the learning rate decays then what was
flat was a learning-rate floor rather than a competence floor.

**Reuse, not reimplementation.** This is `step4_flat.py`'s forced block run on the annealing
branch, through the same `series`, `drop_test` and `anchors` it uses for pretraining, so the
counts are comparable to `STEP4_FLAT_pretrain.json` line for line. Its two other blocks are
skipped: `form_a()` is about dates on the pretraining branch, and `window_is_turnover()` raises
on the annealing branch because `R2_INDEX_CURVES.json` carries no `shape` for `anneal|product|
ratio` -- a gap in that artefact, not in this reading, and nothing the drop test needs.

Added beyond step 4: each series' annealing level **against the pretraining floor**, since the
question is not only whether the loss moves within the annealing branch but whether it ends
below where pretraining left it, measured in that series' own core-bootstrap noise unit.

    python experiments/b2_anneal_drop.py
"""
from __future__ import annotations

import collections
import json
import time

import numpy as np

from experiments.atlas import D, provenance
from experiments.r2_curve4 import FORCED, read_positions, series
from experiments.relatedness import OUT2
from experiments.step4_flat import DL, piece_multiplicity, recorded, drop_test
from profilebattery.profiles.names import tree_name


def load(branch):
    seq = recorded(branch)
    per_ref = {}
    for r, _ in seq:
        p = DL / f"record/{r}/POSITIONS.json"
        if p.exists():
            per_ref[r] = read_positions(json.load(open(p)))
    refs = [r for r, _ in seq if r in per_ref]
    toks = [t for r, t in seq if r in per_ref]
    return per_ref, refs, toks


def main() -> None:
    t0 = time.time()
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = {tree_name(c) for c in split["p1"]}
    per_a, refs_a, toks_a = load("anneal")
    per_p, refs_p, toks_p = load("pretrain")
    print(f"anneal: {len(refs_a)} references; pretrain: {len(refs_p)}; |P1| = {len(p1)}",
          flush=True)
    # step 4's `anchors` is hardcoded to the PRETRAINING anchors (0001000 and the reference
    # nearest 84 B) and raises on this branch, correctly -- 84 B is not inside it. The annealing
    # anchors are the branch's own first, middle and last, and `drop_test` only needs the three
    # indices, so the same test applies unchanged with the branch's own span.
    idx_a = {"early": {"i": 0, "reference": refs_a[0], "tokens": toks_a[0]},
             "at_84b": {"i": len(refs_a) // 2, "reference": refs_a[len(refs_a) // 2],
                        "tokens": toks_a[len(refs_a) // 2],
                        "note": "the branch's MIDDLE, not 84 B -- 84 B is not on this branch"},
             "last": {"i": len(refs_a) - 1, "reference": refs_a[-1], "tokens": toks_a[-1]}}
    mult = piece_multiplicity(per_a, refs_a)
    keys = sorted({k for r in refs_a for k in per_a[r]["by_role"] if k[1] == FORCED})
    keys = [k for k in keys if k[0] in p1]

    out_series, strict = {}, {}
    against_floor = []
    for k in keys:
        vals, units, ok = series(per_a, refs_a, k)
        t = (drop_test(vals, units, idx_a) if ok and len(vals) == len(refs_a)
             else {"outcome": "unreadable",
                   "why": "the series does not cover every annealing reference, or its core "
                          "names differ across references so no core-bootstrap unit forms"})
        t["n_pieces_at_least_20"] = len(mult.get(k, ()))
        t["single_piece"] = t["n_pieces_at_least_20"] == 1
        # the annealing level against the pretraining floor, in this series' own noise unit
        pv, pu, pok = series(per_p, refs_p, k)
        if ok and pok and len(vals) and len(pv):
            floor = float(pv[-1])
            unit = float(np.nanmedian([u for u in units if u and np.isfinite(u)])
                         if any(units) else np.nan)
            last = float(vals[-1])
            t["pretrain_floor"] = floor
            t["anneal_last"] = last
            t["change_from_floor"] = last - floor
            t["change_in_noise_units"] = ((last - floor) / unit
                                          if unit and np.isfinite(unit) else None)
            if t["change_in_noise_units"] is not None:
                against_floor.append(t["change_in_noise_units"])
        out_series["|".join(k)] = t
        if t["single_piece"]:
            strict["|".join(k)] = t

    fc = collections.Counter(v["outcome"] for v in out_series.values())
    sc = collections.Counter(v["outcome"] for v in strict.values())
    below = [v for v in against_floor if v < -1]
    above = [v for v in against_floor if v > 1]
    out = {"what": "step 4's forced-position drop test on the annealing branch, plus each "
                   "series' annealing level against the pretraining floor",
           "reused_from": "experiments/step4_flat.py (series, drop_test, anchors, "
                          "piece_multiplicity) so the counts match STEP4_FLAT_pretrain.json",
           "skipped": {"form_a": "about dates on the pretraining branch",
                       "window_is_turnover": "raises on anneal -- R2_INDEX_CURVES.json has no "
                                             "shape for anneal|product|ratio; not needed here"},
           "n_anneal_references": len(refs_a), "n_pretrain_references": len(refs_p),
           "anchors": idx_a,
           "forced": {"n_series": len(out_series), "counts": dict(fc),
                      "n_single_piece": len(strict), "counts_single_piece": dict(sc),
                      "series": out_series},
           "against_the_pretraining_floor": {
               "n_series_compared": len(against_floor),
               "median_change_in_noise_units": (float(np.median(against_floor))
                                                if against_floor else None),
               "n_below_floor_by_more_than_one_unit": len(below),
               "n_above_floor_by_more_than_one_unit": len(above),
               # computed, NEVER hardcoded: the first draft of this script carried the string
               # "none, and §5's floor stands as a competence floor", written before the
               # result, and the result contradicted it.
               "reading": (f"{len(below)} of {len(against_floor)} series end more than one "
                           f"noise unit BELOW the pretraining floor and {len(above)} above; "
                           + ("no series drops, so §5's floor stands as a competence floor "
                              "on this branch" if not below else
                              "so the pretraining floor is not purely a competence floor -- "
                              "part of it lifts when the rate decays, though the data-mix "
                              "change forbids attributing that to the rate"))},
           "caveat": "the data mix changes at the start of this branch at the same moment as "
                     "the learning rate (§4.5), so a fall here could not be attributed to the "
                     "rate even if there were one",
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["record/<anneal ref>/POSITIONS.json",
                                     "record/<pretrain ref>/POSITIONS.json"])}
    (OUT2 / "B2_ANNEAL_DROP.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nFORCED on anneal: {len(out_series)} obliged roles in P1 "
          f"({len({k.split('|')[0] for k in out_series})} constructs): {dict(fc)}")
    print(f"  of which {len(strict)} realise one piece: {dict(sc)}")
    a = out["against_the_pretraining_floor"]
    print(f"against the pretraining floor: {a['n_series_compared']} series, median change "
          f"{a['median_change_in_noise_units']:+.3f} noise units; "
          f"{a['n_below_floor_by_more_than_one_unit']} below by >1 unit, "
          f"{a['n_above_floor_by_more_than_one_unit']} above")
    print(f"-> {OUT2 / 'B2_ANNEAL_DROP.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
