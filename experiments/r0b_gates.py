"""Step 0b: |P3| and |P4|, the gates that are dates' outputs.

**The definitions** (`PLAN-Analysis` part 2, R6). **P3**: the constructs of **P2** whose
**stability date** -- curve 1, part 3 -- lies inside the record. **P4**: those with **all six
dates** inside the record, which is the population the agreement question of part 4 is asked on.
Both counts are results and are reported as gates beside P0-P2.

**What they gate** (part 5, step 0b). `|P3| >= 43` is C3's condition for a **verdict** on P2
FREQUENCY and `>= 84` for a reading with 80% power at the pilot's effect size; **below 43, P2 is
reported as the correlation with its interval and no verdict**.

**Three things this step has to decide rather than assume, and each is named in the output.**

1. **Which space curve 1's date comes from.** The product space is the plan's primary and the
   verdict layer is reported beside it; the two give very different counts, so both are computed
   and the primary is marked.
2. **Which of curve 4's classes counts as "the loss date".** Curve 4 is not one curve: the hashed
   forced, expected and dependent classes reach **16 of P1's 169** constructs, while A29's
   construct-token class reaches all 188. P4 is computed under each reading and the counts
   reported side by side, because "all six dates" means something different under each.
3. **Curve 5 has no dates at all**, by the Settle chat's ruling of 14 Oct -- point estimates only
   until R8 shows a pair worth bootstrapping. So **|P4| on the literal six is zero by
   construction**, and the useful quantity is |P4| on the **five** curves that are dated. Both
   are reported; the literal count is not quietly replaced by the five-curve one.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
R2 = D / "validation/r2"
P2_GATE = D / "validation/p2/P2_GATE_S15.json"
SPLIT = D / "validation/battery/SPLIT.json"
C3_VERDICT = 43
C3_POWERED = 84


def dated_set(dates: dict, labels=("dated",)) -> set:
    """The constructs whose entry carries one of `labels` as its outcome or sub-label."""
    out = set()
    for c, d in dates.items():
        k = d.get("flat") or d.get("label") or d.get("outcome")
        if k in labels:
            out.add(c)
    return out


def curve1_dated(report: dict, branch: str, space: str, theta_rule="dates") -> set:
    """Curve 1's dated set from `R2_INDEX_CURVES.json`, for one branch and space."""
    key = f"{branch}|{space}|ratio"
    block = report.get("curve1", {}).get(key) or {}
    return dated_set(block.get("dates") or {})


def curve4_dated(c4: dict, ctl: dict, branch: str, reading: str) -> set:
    """Curve 4's dated set under one reading of "the loss date".

    `reading="hashed"` takes the forced, governed and scope classes of `R2_CURVE4.json` -- a
    construct counts if any of its series is dated. `reading="construct_token"` takes A29's
    class from `CONSTRUCT_TOKEN_LOSS.json`, where `dated` means a **competence** date and
    `dated_never_learned` is excluded, since a date whose settled level is at or above untrained
    is not a competence date (A30 final form).
    """
    if reading == "hashed":
        out = set()
        for cls in ("obliged", "governed", "scope"):
            block = (c4.get("classes") or {}).get(f"{branch}|{cls}") or {}
            for key, d in (block.get("series") or {}).items():
                k = d.get("flat") or d.get("label") or d.get("outcome")
                if k == "dated":
                    out.add(key.split("|")[0])
        return out
    out = set()
    for c, e in (ctl.get("curves") or {}).items():
        b = (e.get("branch") or {}).get(branch)
        if not b:
            continue
        d = b.get("date") or {}
        if (d.get("label") or d.get("outcome")) == "dated":
            out.add(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()
    br = a.branch

    p1 = {tree_name(c) for c in json.loads(SPLIT.read_text())["p1"]}
    p2 = set(json.loads(P2_GATE.read_text())["p2"]) & p1
    idxc = json.loads((R2 / "R2_INDEX_CURVES.json").read_text())
    rep2 = json.loads((R2 / "R2_REPORT.json").read_text())
    c4 = json.loads((R2 / "R2_CURVE4.json").read_text())
    ctl = json.loads((D / "validation/loss_offset0/CONSTRUCT_TOKEN_LOSS.json").read_text())

    print(f"branch {br};  |P1| = {len(p1)},  |P2 n P1| = {len(p2)}")

    out = {"branch": br, "n_p1": len(p1), "n_p2_in_p1": len(p2),
           "thresholds": {"c3_verdict": C3_VERDICT, "c3_powered": C3_POWERED}, "p3": {}, "p4": {}}

    # ---- P3: P2 and curve 1 dated ----
    print("\n|P3| = constructs of P2 whose curve-1 stability date is inside the record")
    for space, mark in (("product", "PRIMARY"), ("layer", "beside it")):
        d1 = curve1_dated(idxc, br, space)
        p3 = p2 & d1
        verdict = ("a verdict on P2" if len(p3) >= C3_VERDICT else
                   "NO verdict on P2 -- the correlation with its interval only")
        power = "80% power" if len(p3) >= C3_POWERED else "below 80% power"
        print(f"  {space:8s} ({mark:9s}) curve 1 dated {len(d1):3d} -> |P3| = {len(p3):3d}"
              f"   {verdict}; {power}")
        out["p3"][space] = {"curve1_dated": len(d1), "n": len(p3), "members": sorted(p3),
                            "primary": space == "product",
                            "c3": verdict, "power": power}

    # ---- P4: all six dates inside the record ----
    print("\n|P4| = all six dates inside the record (the agreement question's population)")
    five = {"curve1": None, "curve2": None, "curve3": None, "curve4": None, "curve6": None}
    five["curve2"] = dated_set((rep2["curves"].get(f"{br}|curve2_norm") or {}).get("dates") or {})
    five["curve3"] = dated_set((rep2["curves"].get(f"{br}|curve3_depth") or {}).get("dates") or {})
    five["curve6"] = dated_set((rep2["curves"].get(f"{br}|curve6_sparsity") or {}).get("dates") or {})
    print(f"  curve 5 is NOT dated at all (point estimates by the ruling of 14 Oct), so the "
          f"LITERAL |P4| on six curves is 0 by construction")
    out["p4"]["literal_six"] = {"n": 0, "why": "curve 5 carries no dates: point estimates only, "
                                               "per the Settle chat's ruling of 14 Oct"}
    for space in ("product", "layer"):
        five["curve1"] = curve1_dated(idxc, br, space)
        for reading in ("hashed", "construct_token"):
            five["curve4"] = curve4_dated(c4, ctl, br, reading)
            inter = p2.copy()
            for k, v in five.items():
                inter &= v
            counts = {k: len(v) for k, v in five.items()}
            print(f"  curve 1 [{space:7s}] + curve 4 [{reading:15s}]: {counts} -> "
                  f"|P4 on five| = {len(inter)}")
            order = sorted(five, key=lambda k: -len(five[k]))
            cum, bits = None, []
            for k in order:
                cum = set(five[k]) if cum is None else cum & five[k]
                bits.append(f"{k}:{len(cum)}")
            print(f"      cumulative  {' -> '.join(bits)}")
            # **Where the intersection collapses, and in what order.** A bare zero says nothing
            # about why. The cumulative trace adds one curve at a time from the largest dated
            # set to the smallest, so the binding constraint is visible; the pairwise matrix
            # shows whether any two curves' dated sets overlap at all. Both are stored because
            # |P4| = 0 is a result about the record and not a computation that failed, and the
            # next reader will ask which curve emptied it.
            order = sorted(five, key=lambda k: -len(five[k]))
            cum, trace = None, []
            for k in order:
                cum = set(five[k]) if cum is None else cum & five[k]
                trace.append({"after": k, "n": len(cum)})
            pair = {f"{a}|{b}": len(set(five[a]) & set(five[b]))
                    for i, a in enumerate(order) for b in order[i + 1:]}
            out["p4"][f"{space}|{reading}"] = {
                "n_five_curves": len(inter), "members": sorted(inter),
                "per_curve_dated": counts, "cumulative": trace, "pairwise": pair,
                "note": "five curves, not six: curve 5 is undated"}

    p = pathlib.Path(a.out) / f"R0B_GATES_{br}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
