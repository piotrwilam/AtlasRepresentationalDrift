r"""Item 13 — the per-construct table of the six curves' dates, and the verdicts.

Appendix C owes a table of every construct's six dates under the corrected settling rule. This
assembles it **from the artefacts**, naming the artefact and the field behind every column, and
refuses to fill a column it cannot source.

The six curves and where their dates live:

  1 turnover  `R2_INDEX_CURVES.json` curve1, at BOTH partner spacings. The corrected rule needs
              agreement between the two, so the settling date is a date only where `ratio` and
              `fixed` agree. `ratio` has **0 dated** of 169, so no construct settles -- and this
              table shows the mechanism rather than asserting the conclusion.
  2 size      `R2_REPORT.json` curve2_norm -- **but its `form` is `share_of_last`, not `raw`**,
              and §4.6's third artefact makes raw primary. Its 75 dated is therefore NOT the
              draft's 92, and an exhaustive scan of every artefact finds no raw-form curve-2
              date block. The column is filled from the form that exists and marked; the raw
              form is owed.
  3 depth     `R2_REPORT.json` curve3_depth, `form: raw`. 59 dated.
  4 loss      `R2_CURVE4.json` pieces.pretrain.series, keyed per construct|class|role|piece --
              per ROLE, not per construct, so a construct's entry is the set of outcomes over
              its roles and the earliest dated one.
  5 sharing   no per-construct date exists: curve 5 is defined on the 14,196 construct PAIRS and
              `R2_CLOSE.json` records it as point estimates only. The column says so.
  6 sharpness `R2_REPORT.json` curve6_sparsity, `form: raw`. 97 dated.

    python experiments/dates_table.py
"""
from __future__ import annotations

import collections
import csv
import json
import time

from experiments.atlas import D, provenance
from experiments.relatedness import OUT2

R2 = D / "validation/r2"
OUT = D / "validation/paper2"
BRANCH = "pretrain"


def main() -> None:
    t0 = time.time()
    from experiments.atlas import p1_set
    names = sorted(p1_set())
    idxc = json.loads((R2 / "R2_INDEX_CURVES.json").read_text())
    rep = json.loads((R2 / "R2_REPORT.json").read_text())
    c4 = json.loads((R2 / "R2_CURVE4.json").read_text())
    close = json.loads((R2 / "R2_CLOSE.json").read_text())

    c1r = idxc["curve1"][f"{BRANCH}|product|ratio"]["dates"]
    c1f = idxc["curve1"][f"{BRANCH}|product|fixed"]["dates"]
    c2 = rep["curves"][f"{BRANCH}|curve2_norm"]
    c3 = rep["curves"][f"{BRANCH}|curve3_depth"]["dates"]
    c6 = rep["curves"][f"{BRANCH}|curve6_sparsity"]["dates"]
    c4s = c4["pieces"][BRANCH]["series"]
    # The RAW curve 2, re-run through the dataset-only resolver on 23 Oct. Added ALONGSIDE the
    # share_of_last column, not in place of it: the raw form is the paper's primary by §4.6's
    # third artefact, but its figures no longer match the ones the draft quotes (92/77 ->
    # 90/79), so promoting it is the Settle chat's call and not CC's.
    raw2 = None
    rp = R2 / "owed" / "CURVE2_FORMS_pretrain.json"
    if rp.exists():
        raw2 = json.loads(rp.read_text())["curves"]

    # curve 4 is per role; group it under its construct
    by_construct = collections.defaultdict(dict)
    for key, v in c4s.items():
        by_construct[key.split("|", 1)[0]][key] = v

    def tok(e):
        for k in ("tokens", "date_tokens", "at_tokens"):
            if isinstance(e, dict) and e.get(k) is not None:
                return e[k]
        return None

    rows, counts = [], collections.defaultdict(collections.Counter)
    for n in names:
        er, ef = c1r.get(n) or {}, c1f.get(n) or {}
        # the corrected rule: a date needs agreement between the two partner spacings
        agree = (er.get("outcome") == "dated" and ef.get("outcome") == "dated")
        c4e = by_construct.get(n, {})
        c4_outcomes = collections.Counter(v.get("outcome") for v in c4e.values())
        c4_dated = [v for v in c4e.values() if v.get("outcome") == "dated"]
        r = {
            "construct": n,
            "c1_turnover_ratio": er.get("outcome"),
            "c1_turnover_fixed": ef.get("outcome"),
            "c1_settling_date": ("agreed" if agree else None),
            "c1_why_no_date": (None if agree else
                               f"ratio {er.get('outcome')}, fixed {ef.get('outcome')}: "
                               f"the corrected rule needs both"),
            "c2raw_outcome": ((raw2.get(n, {}).get("raw") or {}).get("outcome")
                              if raw2 else None),
            "c2raw_index": ((raw2.get(n, {}).get("raw") or {}).get("index")
                            if raw2 else None),
            "c2raw_R": ((raw2.get(n, {}).get("raw") or {}).get("R") if raw2 else None),
            "c2_size_outcome": (c2["dates"].get(n) or {}).get("outcome"),
            "c2_size_form": c2["form"],
            "c2_size_tokens": tok(c2["dates"].get(n)),
            "c3_depth_outcome": (c3.get(n) or {}).get("outcome"),
            "c3_depth_tokens": tok(c3.get(n)),
            "c4_loss_n_roles": len(c4e),
            "c4_loss_outcomes": dict(c4_outcomes),
            "c4_loss_n_dated": len(c4_dated),
            "c5_sharing": "no per-construct date: curve 5 is defined on construct pairs",
            "c6_sharpness_outcome": (c6.get(n) or {}).get("outcome"),
            "c6_sharpness_tokens": tok(c6.get(n)),
        }
        r["n_of_six_dated"] = sum([
            bool(agree), r["c2_size_outcome"] == "dated", r["c3_depth_outcome"] == "dated",
            r["c4_loss_n_dated"] > 0, False, r["c6_sharpness_outcome"] == "dated"])
        rows.append(r)
        for k, v in (("c1_ratio", er.get("outcome")), ("c1_fixed", ef.get("outcome")),
                     ("c2", r["c2_size_outcome"]), ("c3", r["c3_depth_outcome"]),
                     ("c6", r["c6_sharpness_outcome"])):
            counts[k][str(v)] += 1

    n_all_six = sum(1 for r in rows if r["n_of_six_dated"] == 6)
    doc = {
        "what": "every construct's six curve dates under the corrected settling rule, "
                "assembled from the artefacts with each column's source named",
        "branch": BRANCH, "n_constructs": len(rows),
        "sources": {
            "1 turnover": "R2_INDEX_CURVES.json curve1.pretrain|product|{ratio,fixed}.dates",
            "2 size": "R2_REPORT.json curves.pretrain|curve2_norm.dates",
            "3 depth": "R2_REPORT.json curves.pretrain|curve3_depth.dates (form: raw)",
            "4 loss": "R2_CURVE4.json pieces.pretrain.series (per ROLE, grouped here)",
            "5 sharing": "none exists per construct; R2_CLOSE.json records point estimates "
                         "only, on the 14,196 pairs",
            "6 sharpness": "R2_REPORT.json curves.pretrain|curve6_sparsity.dates (form: raw)"},
        "outcome_counts": {k: dict(v) for k, v in counts.items()},
        "n_constructs_with_all_six_dated": n_all_six,
        "settling": {
            "n_dated_under_the_corrected_rule": sum(1 for r in rows
                                                    if r["c1_settling_date"] == "agreed"),
            "mechanism": "the corrected rule needs agreement between two partner spacings; "
                         "the ratio spacing dates 0 of 169, so no construct can settle",
            "ratio_outcomes": dict(counts["c1_ratio"]),
            "fixed_outcomes": dict(counts["c1_fixed"])},
        "OWED": {
            "curve_2_raw_form": {
                "problem": "the paper's primary form for the size is RAW (§4.6, third "
                           "artefact), and the only per-construct curve-2 dates in any "
                           "artefact are form `share_of_last`, which dates 75 of 169",
                "draft_quotes": "92 dated, 145 of 169 falling, 91 of 92 for P7",
                "scan": "every JSON under validation/ was scanned for a per-construct date "
                        "block; the three found are curve2_norm (share_of_last), curve3_depth "
                        "(raw) and curve6_sparsity (raw)",
                "consequence": "the size column of this table is the non-primary form and is "
                               "marked as such",
                "RESOLVED_23_OCT": {
                    "found": "owed/CURVE2_FORMS_pretrain.json, written 15 Oct by r2_owed.py, "
                             "whose reader was local-first and read the pilot tree",
                    "a52_did_not_check_it": "A52 names r2_owed as a local-first reader but "
                             "audits only FOR_EXAMPLE and P8_BOOTSTRAP",
                    "contamination_measured": "pilot references x pilot constructs: 12 of 12 "
                             "differ, median 17.9%; the other three cells of the 2x2 are 0 of "
                             "12 each",
                    "re_run": "the reader swapped to the dataset-only resolver, the 7 "
                              "contaminated per-reference caches quarantined, 169 constructs "
                              "refetched at each",
                    "figures": {"raw_dated": {"was": 92, "now": 90},
                                "raw_censored": {"was": 77, "now": 79},
                                "raw_norms_fall": {"was": "145 of 169", "now": "148 of 169"},
                                "P7": {"was": "91 falling of 92", "now": "90 of 90"}},
                    "constructs_that_moved": 13,
                    "outcome_changes": ["For: dated -> censored", "If: dated -> censored"],
                    "standardised_arm": "NOT recomputed -- r2_owed reads A_raw/B_raw only and "
                                        "takes the standardised curve from R2_REPORT.json, so "
                                        "its 0-of-169 change means not recomputed"}},
            "curve_5": close["curve5"].get("status")},
        "minutes": (time.time() - t0) / 60,
        "provenance": provenance(["validation/r2/R2_INDEX_CURVES.json",
                                  "validation/r2/R2_REPORT.json",
                                  "validation/r2/R2_CURVE4.json",
                                  "validation/r2/R2_CLOSE.json"])}
    doc["rows"] = rows
    (OUT / "DATES_TABLE.json").write_text(json.dumps(doc, indent=1, default=float))

    cols = ["construct", "c1_turnover_ratio", "c1_turnover_fixed", "c1_settling_date",
            "c2raw_outcome", "c2raw_index", "c2_size_outcome", "c2_size_tokens",
            "c3_depth_outcome", "c3_depth_tokens",
            "c4_loss_n_roles", "c4_loss_n_dated", "c6_sharpness_outcome",
            "c6_sharpness_tokens", "n_of_six_dated"]
    with open(OUT / "DATES_TABLE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    head = (f"**Table C.1 — the six curves' dates, all {len(rows)} constructs, pretraining "
            f"branch.** Curve 1's settling date requires agreement between the two partner "
            f"spacings (the corrected rule, §4.6); the ratio spacing dates 0 of 169, so the "
            f"column is empty by construction and both spacings are shown. Curve 2 is the "
            f"`{c2['form']}` form, NOT the raw form the paper makes primary — the raw per-"
            f"construct dates are in no artefact. Curve 4 is per role, so its columns count "
            f"roles. Curve 5 has no per-construct date: it is defined on construct pairs.")
    md = [head, "", "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c))
                                    for c in cols) + " |")
    (OUT / "DATES_TABLE.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} constructs")
    for k, v in counts.items():
        print(f"  {k}: {dict(v)}")
    print(f"  settling under the corrected rule: "
          f"{doc['settling']['n_dated_under_the_corrected_rule']} of {len(rows)}")
    print(f"  constructs with all six dated: {n_all_six}")
    print(f"  OWED: curve 2's raw form (draft quotes 92; the artefact's form dates 75)")
    print(f"-> {OUT / 'DATES_TABLE.json'}, .csv, .md  ({doc['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
