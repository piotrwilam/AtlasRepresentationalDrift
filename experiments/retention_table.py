r"""The old set's retained effect at 2,034 B, and the 16 constructs behind it.

The draft quotes "about 45% of what the current set costs, though only 14% of its neurons are
still in the set". The 14% reproduces (13.8%). The 45% does not, under any definition tried.

**The statistic, fixed here.** The *ratio of medians in control units*: the median over the 16
constructs of the old set's rise over the random control, divided by the median of the current
set's, both as `*_over_random_in_sd` in `FULL_pretrain_iter_0484865.json`, with a construct
bootstrap. The nats form -- `(drifted - random_mean) / (current - random_mean)`, again a ratio
of medians -- is reported as a check.

**Signed, not absolute.** `current_over_random_in_sd` is positive for 14 of 16 constructs and
negative for 2; `drifted_over_random_in_sd` is positive for 11 and negative for 5. Taking
absolute values flips the negatives and inflates the numerator of a quantity centred near zero:
it gave 40.2% where the signed ratio gives 28.4%. The signed form is the right one and the
absolute form should not be quoted.

    python experiments/retention_table.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from experiments.atlas import D, provenance
from experiments.relatedness import OUT2

ABL = D / "validation/ablation/FULL_pretrain_iter_0484865.json"
OUT = OUT2
DRAWS = 10_000


def main() -> None:
    t0 = time.time()
    cs = json.loads(ABL.read_text())["constructs"]
    rows = []
    for c, v in sorted(cs.items()):
        if v.get("current_over_random_in_sd") is None:
            continue
        rows.append({
            "construct": c,
            "baseline_nats": v["baseline_nats"],
            "current_nats": v["current"],
            "old_nats": v["drifted"],
            "random_mean_nats": v["random_mean"],
            "random_sd_nats": v["random_sd"],
            "current_over_random_sd": v["current_over_random_in_sd"],
            "old_over_random_sd": v["drifted_over_random_in_sd"],
            "ratio_sd": (v["drifted_over_random_in_sd"] / v["current_over_random_in_sd"]
                         if v["current_over_random_in_sd"] else None),
            "ratio_nats": ((v["drifted"] - v["random_mean"]) /
                           (v["current"] - v["random_mean"])
                           if (v["current"] - v["random_mean"]) else None),
            "overlap_share": v["overlap_share"],
            "drifted_from": v.get("drifted_from")})
    n = len(rows)
    cur_sd = np.array([r["current_over_random_sd"] for r in rows])
    old_sd = np.array([r["old_over_random_sd"] for r in rows])
    cur_n = np.array([r["current_nats"] - r["random_mean_nats"] for r in rows])
    old_n = np.array([r["old_nats"] - r["random_mean_nats"] for r in rows])
    ov = np.array([r["overlap_share"] for r in rows])
    rng = np.random.default_rng(20261023)

    def ratio_of_medians(num, den):
        pt = float(np.median(num) / np.median(den))
        b = []
        for _ in range(DRAWS):
            i = rng.integers(0, n, n)
            d = np.median(den[i])
            if d:
                b.append(np.median(num[i]) / d)
        return pt, [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

    r_sd, ci_sd = ratio_of_medians(old_sd, cur_sd)
    r_n, ci_n = ratio_of_medians(old_n, cur_n)
    bo = [np.median(ov[rng.integers(0, n, n)]) for _ in range(DRAWS)]

    head = (f"**Table D.1 — the ablation at 2,034 B: the current carrier set against the set of "
            f"84 B, for the {n} constructs whose positions can be read.** `current` and `old` "
            f"are rises in loss over the unablated baseline, in nats; `random` is the mean rise "
            f"of {json.loads(ABL.read_text())['n_random']} size-matched random sets. The two "
            f"ratios are each measure's old-over-current, per construct. The headline figure "
            f"is the ratio of MEDIANS, not the median of these ratios: the per-construct ratio "
            f"has a near-zero denominator for `NamedExpr` and `While` and is unusable.")
    cols = ["construct", "baseline (nats)", "current (nats)", "old (nats)", "random (nats)",
            "current (control SD)", "old (control SD)", "ratio (SD)", "ratio (nats)",
            "overlap"]

    def fmt(r):
        return [r["construct"], f"{r['baseline_nats']:.3f}", f"{r['current_nats']:+.4f}",
                f"{r['old_nats']:+.4f}", f"{r['random_mean_nats']:+.4f}",
                f"{r['current_over_random_sd']:+.2f}", f"{r['old_over_random_sd']:+.2f}",
                ("—" if r["ratio_sd"] is None else f"{r['ratio_sd']:+.1%}"),
                ("—" if r["ratio_nats"] is None else f"{r['ratio_nats']:+.1%}"),
                f"{r['overlap_share']:.1%}"]

    body = [fmt(r) for r in sorted(rows, key=lambda r: -(r["ratio_sd"] or -9))]
    md = [head, "", "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(b) + " |" for b in body]
    md += ["", f"**Ratio of medians, control units: {r_sd:.1%} "
               f"[{ci_sd[0]:.1%}, {ci_sd[1]:.1%}]. In nats: {r_n:.1%} "
               f"[{ci_n[0]:.1%}, {ci_n[1]:.1%}]. Overlap: {np.median(ov):.1%} "
               f"[{np.quantile(bo, .025):.1%}, {np.quantile(bo, .975):.1%}].**"]
    (OUT / "TABLE_D1_retention.md").write_text("\n".join(md) + "\n")
    with open(OUT / "TABLE_D1_retention.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(body)
    tex = [r"\begin{longtable}{l" + "r" * (len(cols) - 1) + "}",
           r"\caption{The ablation at 2,034 B: the current carrier set against the set of 84 B.}"
           r"\\", r"\toprule",
           " & ".join(c.replace("%", r"\%").replace("_", r"\_") for c in cols) + r" \\",
           r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for b in body:
        tex.append(" & ".join(x.replace("%", r"\%").replace("—", "---")
                              .replace("_", r"\_") for x in b) + r" \\")
    tex.append(r"\end{longtable}")
    (OUT / "TABLE_D1_retention.tex").write_text("\n".join(tex) + "\n")

    doc = {"what": "the old set's retained effect at 2,034 B, and the 16 constructs behind it",
           "reference": "pretrain_iter_0484865", "old_set_from": rows[0]["drifted_from"],
           "n_constructs": n,
           "statistic": "the ratio of MEDIANS, signed; not the median of per-construct ratios",
           "why_signed": "current_over_random_in_sd is positive for 14 of 16 and negative for "
                         "2; drifted for 11 and 5. Absolute values flip the negatives and "
                         "inflate the numerator: abs gives 40.2%, signed gives 28.4%.",
           "retention_control_units": {"ratio_of_medians": r_sd, "ci95": ci_sd,
                                       "median_old": float(np.median(old_sd)),
                                       "median_current": float(np.median(cur_sd))},
           "retention_nats": {"ratio_of_medians": r_n, "ci95": ci_n,
                              "median_old": float(np.median(old_n)),
                              "median_current": float(np.median(cur_n))},
           "overlap": {"median": float(np.median(ov)),
                       "ci95": [float(np.quantile(bo, .025)), float(np.quantile(bo, .975))]},
           "draft_says": "about 45% of the effect with 14% of the members",
           "verdict": "the 14% reproduces (13.8%); the 45% does not under any definition tried",
           "rows": rows, "minutes": (time.time() - t0) / 60,
           "provenance": provenance([str(ABL.relative_to(D))])}
    (OUT / "RETENTION.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"control units: {r_sd:.1%} [{ci_sd[0]:.1%}, {ci_sd[1]:.1%}]  "
          f"(old {np.median(old_sd):.3f} / current {np.median(cur_sd):.3f})")
    print(f"nats         : {r_n:.1%} [{ci_n[0]:.1%}, {ci_n[1]:.1%}]  "
          f"(old {np.median(old_n):.4f} / current {np.median(cur_n):.4f})")
    print(f"overlap      : {np.median(ov):.1%} [{np.quantile(bo,.025):.1%}, "
          f"{np.quantile(bo,.975):.1%}]")
    print(f"-> {OUT}/TABLE_D1_retention.{{md,csv,tex}}, RETENTION.json")


if __name__ == "__main__":
    main()
