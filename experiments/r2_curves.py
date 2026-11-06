"""R2, the part that runs off the carrier index: curves 1, 5 and 6 on all of P1.

**What is here and what is not.** Curves 1 (turnover), 5 (sharing) and 6 (sparsity) need only
carrier sets and the squared-mass shares, both of which the index carries, so they run locally in
seconds. Curves 2 (norm) and 3 (depth) need the per-layer norms at every reference, and **every
core-bootstrap interval** needs `A_cores`/`B_cores` -- about 390 GB of reads -- so those are the
Colab pass (`experiments/r2_bootstrap.py`), and this module **does not date curves 5 and 6**:
their rule is A30's, whose noise unit is the core-bootstrap SE. Dating them off a substitute unit
would put a number in the record that the contract does not define.

**Curve 1 is dated here**, because A31's theta = 0.70 is an absolute threshold on a ratio and
needs no interval. Both of A32's additions are computed: the **fixed-step** form beside the 1.5x
form, and **each branch separately**, each with its own trend and flatness label.

**What each curve is** (`PLAN-Analysis` part 3):

* **1, turnover** -- twin 1's carrier set at `t` against twin 2's at the nearest 1.5x later,
  over the geometric mean of the two checkpoints' twin ceilings. A32 adds the fixed-step form.
* **5, sharing** -- for every pair in P1 at the same checkpoint, twin 1 of one against twin 2 of
  the other, raw and over the geometric mean of the two constructs' twin ceilings. The whole
  matrix, never a chosen pair; 169 x 168 / 2 = 14,196 pairs.
* **6, sparsity** -- the share of the squared norm held by the top 5%, which the index stores as
  `product_share` and `layer_share`. The two twins are averaged, as gate P2 averages them, and
  both are kept.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import FACTOR_BAND, branches, partner_index,\
    partner_factors
from experiments.r1_turnover import (THETA, ceiling, load_index, overlap, ratio_shape,
                                     spearman, stability_date)
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
INDEX = D / "_carriers_dl/validation/carriers"
SPLIT = D / "validation/battery/SPLIT.json"
SCHEDULE = D / "validation/schedule/CHECKPOINTS.json"
FIXED_STEPS = 1_000             # A32: the dense block's spacing


def step_of(ref: str) -> int:
    """The training step in a reference's name, or a named error.

    `untrained` and `untrained_seed2` carry no step, and neither belongs on a branch -- so a
    caller that reaches here with one has a reference list it should not have, and an
    `IndexError` from a string split would not say that.
    """
    tail = ref.rsplit("_", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        raise ValueError(f"no training step in {ref!r}; the fixed-step form needs one")
    return int(tail[1])


def fixed_partner(refs, i: int, steps: int = FIXED_STEPS):
    """The reference closest to `steps` later in training steps, and the spacing it actually has.

    A32 holds the step fixed at 1,000 in the dense block; the tail is not evenly spaced, so the
    nearest available spacing is used and **returned**, so that a point from the tail is never
    read as though it were a 1,000-step step.
    """
    here = step_of(refs[i])
    later = [(j, step_of(refs[j]) - here) for j in range(len(refs))
             if step_of(refs[j]) > here]
    if not later:
        return None, None
    j, gap = min(later, key=lambda x: abs(x[1] - steps))
    return j, gap


def turnover(idx: dict, seq, c: str, space: str, form: str = "ratio") -> dict:
    """Curve 1 in either form. `form="ratio"` is the 1.5x reading; `"fixed"` is A32's."""
    refs = [r for r, _ in seq]
    toks = [t for _, t in seq]
    pts = []
    for i, (r, t) in enumerate(seq):
        if form == "ratio":
            j, gap = partner_index(toks, i, band=FACTOR_BAND), None
        else:
            j, gap = fixed_partner(refs, i)
        if j is None:
            continue
        a = idx.get(r, {}).get(f"twin1/{c}|{space}")
        b = idx.get(refs[j], {}).get(f"twin2/{c}|{space}")
        if a is None or b is None:
            continue
        raw = overlap(a, b)
        c1, c2 = ceiling(idx, r, c, space), ceiling(idx, refs[j], c, space)
        cap = float(np.sqrt(c1 * c2)) if np.isfinite(c1) and np.isfinite(c2) else float("nan")
        p = {"reference": r, "tokens": t, "partner": refs[j], "partner_tokens": toks[j],
             "raw": raw, "ceiling": cap,
             "ratio": raw / cap if cap and np.isfinite(cap) and cap > 0 else float("nan")}
        if gap is not None:
            p["step_gap"] = gap
        pts.append(p)
    return {"construct": c, "space": space, "form": form, "points": pts}


def sharing(idx: dict, refs, constructs, space: str) -> dict:
    """Curve 5: the whole pair matrix per checkpoint, raw and over the two constructs' ceilings.

    Stored as parallel arrays rather than nested dicts -- 14,196 pairs x 34 checkpoints is 483k
    numbers, and a dict of dicts of that size is slower to write than the computation itself.
    """
    pairs = list(itertools.combinations(constructs, 2))
    out = {"space": space, "pairs": [f"{a}|{b}" for a, b in pairs], "references": list(refs),
           "raw": [], "over_ceiling": []}
    for r in refs:
        caps = {c: ceiling(idx, r, c, space) for c in constructs}
        row_raw, row_cap = [], []
        for a, b in pairs:
            x = idx.get(r, {}).get(f"twin1/{a}|{space}")
            y = idx.get(r, {}).get(f"twin2/{b}|{space}")
            if x is None or y is None:
                row_raw.append(float("nan"))
                row_cap.append(float("nan"))
                continue
            v = overlap(x, y)
            row_raw.append(v)
            g = np.sqrt(caps[a] * caps[b])
            row_cap.append(v / g if np.isfinite(g) and g > 0 else float("nan"))
        out["raw"].append(row_raw)
        out["over_ceiling"].append(row_cap)
    return out


def sparsity_curve(meta: dict, seq, c: str, space: str) -> dict:
    """Curve 6 off the index's stored shares. Both twins kept, their mean the reading.

    `seq` is `[(reference, tokens)]` and not a bare reference list, because `ratio_shape` reads
    the trend against **log tokens** and a point without its token count cannot carry one.
    """
    key = f"{space}_share"
    pts = []
    for r, t in seq:
        m = meta.get(r, {})
        s1 = (m.get(f"twin1/{c}") or {}).get(key)
        s2 = (m.get(f"twin2/{c}") or {}).get(key)
        if s1 is None or s2 is None:
            continue
        pts.append({"reference": r, "tokens": t, "twin1": s1, "twin2": s2,
                    "ratio": (s1 + s2) / 2})
    return {"construct": c, "space": space, "points": pts}


def load_meta(root: pathlib.Path, refs) -> dict:
    out = {}
    for r in refs:
        p = root / f"{r}.json"
        if p.exists():
            out[r] = json.loads(p.read_text())
    return out


def summarise_turnover(curves: dict, theta: float = THETA) -> dict:
    """Counts by A31 outcome, and the shape distribution A31's additions ask for."""
    out = collections.Counter(stability_date(c, theta)["outcome"] for c in curves.values())
    shapes = {k: ratio_shape(v) for k, v in curves.items()}
    rho = [s["spearman_vs_log_tokens"] for s in shapes.values()
           if "spearman_vs_log_tokens" in s]
    sig = [s for s in shapes.values() if s.get("permutation_p", 1) < 0.05]
    med = [s["median"] for s in shapes.values() if "median" in s]
    return {"outcomes": dict(out), "shape": shapes,
            "flatness": dict(collections.Counter(s.get("flatness", "n/a")
                                                 for s in shapes.values())),
            "median_level": float(np.median(med)) if med else None,
            "median_rho": float(np.median(rho)) if rho else None,
            "significant": len(sig),
            "rising": sum(1 for s in sig if s["spearman_vs_log_tokens"] > 0),
            "falling": sum(1 for s in sig if s["spearman_vs_log_tokens"] < 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--out", default=str(D / "validation/r2"))
    a = ap.parse_args()

    split = json.loads(SPLIT.read_text())
    p1 = [tree_name(c) for c in split["p1"]]
    schedule = json.loads(SCHEDULE.read_text())
    root = pathlib.Path(a.index)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"P1 = {len(p1)} constructs")

    report = {"n_p1": len(p1), "theta": THETA, "fixed_steps": FIXED_STEPS, "curve1": {},
              "curve6": {}}

    for branch in ("pretrain", "anneal"):
        seq = branches(schedule)[branch]
        idx = load_index(root, [r for r, _ in seq])
        seq = [(r, t) for r, t in seq if r in idx]
        meta = load_meta(root, [r for r, _ in seq])
        refs = [r for r, _ in seq]
        pf = partner_factors([t for _, t in seq])
        print(f"\n=== {branch}: {len(seq)} checkpoints ===")
        print(f"  1.5x partner factor: n={pf['n']} median {pf.get('median', float('nan')):.3f} "
              f"range {pf.get('min', float('nan')):.3f}-{pf.get('max', float('nan')):.3f}; "
              f"IN BAND {pf.get('in_band')} of {pf['n']} (band {pf.get('band')})")
        if pf["n"] and pf["in_band"] == 0:
            print(f"  -> this branch has NO valid 1.5x step: the ratio form is not computable "
                  f"here and only A32's fixed-step form is read")
        report.setdefault("partner_factor", {})[branch] = pf

        for space in ("product", "layer"):
            for form in ("ratio", "fixed"):
                cur = {c: turnover(idx, seq, c, space, form) for c in p1}
                npt = max((len(v["points"]) for v in cur.values()), default=0)
                if npt == 0:
                    print(f"  curve 1 [{space:7s} {form:5s}] NOT COMPUTABLE on this branch: "
                          f"no checkpoint has a partner inside the factor band")
                    report["curve1"][f"{branch}|{space}|{form}"] = {
                        "not_computable": "no checkpoint has a 1.5x partner inside "
                                          f"the band {list(FACTOR_BAND)} on this branch"}
                    continue
                s = summarise_turnover(cur)
                gaps = [p["step_gap"] for v in cur.values()
                        for p in v["points"] if p.get("step_gap")]
                gs = ""
                if gaps:
                    g = np.array(sorted(set(gaps)))
                    gs = (f"  step gaps {g.min()}-{g.max()}, "
                          f"{int((np.array(gaps) == FIXED_STEPS).mean() * 100)}% at "
                          f"{FIXED_STEPS}")
                print(f"  curve 1 [{space:7s} {form:5s}] {s['outcomes']}  "
                      f"flat={s['flatness']}  level={s['median_level']:.3f}  "
                      f"rho={s['median_rho']:+.3f}  sig={s['significant']} "
                      f"(rising {s['rising']}, falling {s['falling']}){gs}")
                report["curve1"][f"{branch}|{space}|{form}"] = {
                    "step_gaps": sorted(set(gaps)) if gaps else None,
                    "share_at_fixed_step": (float((np.array(gaps) == FIXED_STEPS).mean())
                                            if gaps else None),
                    "summary": {k: v for k, v in s.items() if k != "shape"},
                    "shape": s["shape"],
                    "dates": {c: stability_date(v, THETA) for c, v in cur.items()}}
                np.savez_compressed(
                    out / f"curve1_{branch}_{space}_{form}.npz",
                    **{c: np.array([[p["tokens"], p["ratio"]] for p in v["points"]], float)
                       for c, v in cur.items()})

            cur6 = {c: sparsity_curve(meta, seq, c, space) for c in p1}
            sh6 = {c: ratio_shape(v) for c, v in cur6.items()}
            lev = [s["median"] for s in sh6.values() if "median" in s]
            rho6 = [s["spearman_vs_log_tokens"] for s in sh6.values()
                    if "spearman_vs_log_tokens" in s]
            sig6 = [s for s in sh6.values() if s.get("permutation_p", 1) < 0.05]
            print(f"  curve 6 [{space:7s}]       level={np.median(lev):.4f}  "
                  f"rho={np.median(rho6):+.3f}  sig={len(sig6)} "
                  f"(rising {sum(1 for s in sig6 if s['spearman_vs_log_tokens'] > 0)}, "
                  f"falling {sum(1 for s in sig6 if s['spearman_vs_log_tokens'] < 0)})"
                  f"  -- NOT DATED: A30's unit is the core bootstrap")
            report["curve6"][f"{branch}|{space}"] = {
                "median_level": float(np.median(lev)), "median_rho": float(np.median(rho6)),
                "significant": len(sig6),
                "rising": sum(1 for s in sig6 if s["spearman_vs_log_tokens"] > 0),
                "falling": sum(1 for s in sig6 if s["spearman_vs_log_tokens"] < 0),
                "shape": sh6, "not_dated": "A30's noise unit is the core-bootstrap SE, which "
                                           "needs the per-core arrays (the Colab pass)"}

        # curve 5: the product space only, which is where the plan reads it
        sh = sharing(idx, refs, p1, "product")
        raw = np.array(sh["raw"], float)
        print(f"  curve 5 [product] {raw.shape[1]} pairs x {raw.shape[0]} checkpoints; "
              f"raw overlap median {np.nanmedian(raw):.4f} "
              f"(min {np.nanmin(raw):.4f}, max {np.nanmax(raw):.4f})"
              f"  -- NOT DATED: same reason")
        np.savez_compressed(out / f"curve5_{branch}_product.npz",
                            raw=raw, over_ceiling=np.array(sh["over_ceiling"], float),
                            pairs=np.array(sh["pairs"]), references=np.array(sh["references"]))

    (out / "R2_INDEX_CURVES.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\n-> {out / 'R2_INDEX_CURVES.json'} and the npz beside it")


if __name__ == "__main__":
    main()
