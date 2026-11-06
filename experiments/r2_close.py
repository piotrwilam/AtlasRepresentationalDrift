"""R2 closed: the twin ceiling and agreement as curves, and curve 1's interval.

**What the twin-2 pass gave, and what it did not.** `validation/bootstraps/<ref>_twin2.json`
holds, per construct, the core-bootstrap interval of the **twin ceiling** (twin 1's carriers
against twin 2's at the same checkpoint) and of the **twin agreement** (the cosine of the two
contrasts) -- 169 entries a reference, not the 14,196 construct **pairs** curve 5 would need. So:

* **curve 1 gains an interval**, because the ceiling is its denominator;
* **curve 5 does not**, and is closed with **point estimates, marked** (the Settle chat's ruling
  of 14 Oct). Its intervals are deferred to R8: if any pair's raw sharing falls by more than its
  point-to-point noise, the cost of a pairwise pass is measured *then* and reported as a
  decision. Not before.

**The interval curve 1 gets is partial, and is labelled so.** The ratio is
`overlap(twin1@t, twin2@partner) / sqrt(ceiling(t) * ceiling(partner))`. The pass bootstrapped
the **denominator** only; the numerator -- an overlap across two *different* checkpoints -- was
not drawn. So what is reported is the ratio's uncertainty **from the denominator alone**, which
is a **lower bound** on its width, under the field name `interval_denominator_only`. Calling it
the ratio's confidence interval would overstate it.

**And the plug-in ceiling stays the denominator**, per the bias measured before the pass ran:
these are intervals on the resampled quantity, which sits below the plug-in value, so
substituting the bootstrap mean would raise every ratio and move every date.

**Two curves that are new here, not re-expressions.** The ceiling and the agreement are
themselves developmental quantities and nobody had read them over training. Both are reported per
construct per branch with their intervals.

**And the first thing they say corrects a sentence that stood in this docstring.** CC wrote that
"the ceiling rises from about 0.21 at `untrained` to 0.87 at the end of pretraining". That is a
true two-point comparison and a false description of the curve. **Inside** the pretraining branch
the median ceiling is **flat -- 0.7373 at the first reference, 0.7400 at the last -- and
significantly falling for 118 of the 133 constructs whose trend clears its permutation interval,
rising for 15.** The rise from 0.21 happens entirely **before** the first recorded checkpoint, at
step 1,000 = 4.19 billion tokens, which is where the construct-token loss also arrives already at
its floor. On the **annealing** branch the ceiling instead rises (45 of 62 significant), the same
branch reversal curve 1's ratio shows.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import FACTOR_BAND, branches, partner_index
from experiments.r1_turnover import THETA, ceiling, load_index, overlap, ratio_shape
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
BOOT2 = D / "_boot2_dl/validation/bootstraps"
INDEX = D / "_carriers_dl/validation/carriers"


def load_twin2(root: pathlib.Path, refs) -> dict:
    out = {}
    for r in refs:
        p = root / f"{r}_twin2.json"
        if p.exists():
            out[r] = json.loads(p.read_text())
    return out


def series(boot: dict, refs, c: str, field: str):
    """`(means, ses, present)` for one construct's ceiling or agreement curve."""
    m, s, got = [], [], []
    for r in refs:
        e = (boot.get(r) or {}).get(c)
        if not e or "error" in e or field not in e:
            continue
        v = e[field]
        if v.get("mean") is None:
            continue
        m.append(float(v["mean"]))
        s.append(None if v.get("se") is None else float(v["se"]))
        got.append(r)
    return m, s, got


def ratio_interval(idx: dict, boot: dict, seq, c: str, space: str = "product") -> dict:
    """Curve 1 with the interval its denominator supports, and the plug-in ratio unchanged.

    Per point: the plug-in ratio (as A31 dates it), and the relative SE of the geometric mean of
    the two checkpoints' ceilings, propagated to the ratio. A geometric mean's relative SE is
    half the sum of its terms' relative SEs, so `rel = 0.5 * (s1/c1 + s2/c2)`.
    """
    refs = [r for r, _ in seq]
    toks = [t for _, t in seq]
    pts = []
    for i, (r, t) in enumerate(seq):
        j = partner_index(toks, i, band=FACTOR_BAND)
        if j is None:
            continue
        a = idx.get(r, {}).get(f"twin1/{c}|{space}")
        b = idx.get(refs[j], {}).get(f"twin2/{c}|{space}")
        if a is None or b is None:
            continue
        raw = overlap(a, b)
        c1, c2 = ceiling(idx, r, c, space), ceiling(idx, refs[j], c, space)
        cap = float(np.sqrt(c1 * c2)) if np.isfinite(c1) and np.isfinite(c2) else float("nan")
        ratio = raw / cap if cap and np.isfinite(cap) and cap > 0 else float("nan")
        e1 = (boot.get(r) or {}).get(c, {}).get("ceiling") or {}
        e2 = (boot.get(refs[j]) or {}).get(c, {}).get("ceiling") or {}
        s1, s2 = e1.get("se"), e2.get("se")
        rel = None
        if None not in (s1, s2) and c1 > 0 and c2 > 0:
            rel = 0.5 * (s1 / c1 + s2 / c2)
        p = {"reference": r, "tokens": t, "partner": refs[j], "raw": raw,
             "ceiling_plugin": cap, "ratio": ratio, "relative_se_denominator": rel}
        if rel is not None and np.isfinite(ratio):
            p["interval_denominator_only"] = [ratio * (1 - 1.96 * rel),
                                              ratio * (1 + 1.96 * rel)]
        pts.append(p)
    return {"construct": c, "space": space, "points": pts,
            "interval_is": "from the denominator alone; the numerator -- an overlap across two "
                           "different checkpoints -- was not drawn, so this is a LOWER BOUND "
                           "on the ratio's width and not a confidence interval for it"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(D / "validation/r2"))
    a = ap.parse_args()

    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = [tree_name(c) for c in split["p1"]]
    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    allrefs = [r for b in ("pretrain", "anneal") for r, _ in branches(schedule)[b]] + \
              ["untrained", "untrained_seed2"]
    boot = load_twin2(BOOT2, allrefs)
    print(f"{len(boot)} twin-2 reference files; P1 = {len(p1)}")

    rep = {"n_p1": len(p1), "curve5": {
        "status": "point estimates only, marked",
        "why": "the twin-2 pass bootstrapped the 169 same-construct ceilings, not the 14,196 "
               "construct pairs curve 5 is defined on",
        "ruling": "the Settle chat, 14 Oct: close R2 with point estimates; in R8 list every "
                  "pair whose raw sharing falls by more than its point-to-point noise, and "
                  "measure the pairwise pass only if that list is non-empty"}}

    for branch in ("pretrain", "anneal"):
        seq = [(r, t) for r, t in branches(schedule)[branch] if r in boot]
        refs = [r for r, _ in seq]
        idx = load_index(INDEX, refs)
        print(f"\n=== {branch}: {len(refs)} checkpoints ===")

        for field in ("ceiling", "agreement"):
            rows, first, last, rel = {}, [], [], []
            for c in p1:
                m, s, got = series(boot, refs, c, field)
                if len(m) < 2:
                    continue
                sh = ratio_shape({"points": [{"ratio": v, "tokens": dict(seq)[r]}
                                             for v, r in zip(m, got)]})
                rows[c] = {"references": got, "mean": m, "se": s, "shape": sh}
                first.append(m[0]); last.append(m[-1])
                good = [x / y for x, y in zip(s, m) if x is not None and y]
                if good:
                    rel.append(float(np.median(good)))
            sig = [v["shape"] for v in rows.values() if v["shape"].get("permutation_p", 1) < .05]
            print(f"  {field:10s} {len(rows):3d} curves  first {np.median(first):.4f} -> "
                  f"last {np.median(last):.4f}  median rel.SE {np.median(rel):.4f}  "
                  f"rising {sum(1 for x in sig if x['spearman_vs_log_tokens']>0)}, "
                  f"falling {sum(1 for x in sig if x['spearman_vs_log_tokens']<0)} "
                  f"of {len(sig)} significant")
            rep[f"{branch}|{field}"] = {
                "n": len(rows), "median_first": float(np.median(first)),
                "median_last": float(np.median(last)),
                "median_relative_se": float(np.median(rel)), "curves": rows}

        got_iv = 0
        c1 = {}
        for c in p1:
            r = ratio_interval(idx, boot, seq, c)
            c1[c] = r
            got_iv += sum(1 for p in r["points"] if "interval_denominator_only" in p)
        npts = sum(len(v["points"]) for v in c1.values())
        print(f"  curve 1    {len(c1)} curves, {npts} points, {got_iv} with a "
              f"denominator-only interval ({100*got_iv/max(npts,1):.0f}%)")
        rep[f"{branch}|curve1_interval"] = {"n_points": npts, "n_with_interval": got_iv,
                                            "curves": c1}

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "R2_CLOSE.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out / 'R2_CLOSE.json'}")


if __name__ == "__main__":
    main()
