"""R2's curve 4: the forced, expected and dependent position classes, on all of P1.

**Why this runs locally when curves 2 and 3 cannot.** Curve 4's input is
`record/<ref>/POSITIONS.json`, which is 2.2 MB a reference and already carries the per-core means:
every `by_class_role` entry has `mean`, `n`, `cores`, `se` and **`core_means`** keyed by core. So
the core bootstrap A30's noise unit requires is computable from 133 MB of JSON, with no per-core
arrays and no Colab.

**The three classes** (`PLAN-Analysis` part 3, curve 4, as amended by A27):

* **forced** -- the `obliged` class: 11 roles across the battery, `header_colon` (15 sets),
  `in` (8), `bare_colon` (7), `open_paren` (5), `assign_eq` (4), `comp_for` (4) and five more
  with one or two each. The mean loss per checkpoint with its untrained value and its
  core-bootstrap SE.
* **expected** -- one curve **per realised piece**, from `by_class_role_piece`, because a fixed
  position mixes pieces and a date on the mixture is a date on nothing in particular.
* **dependent** -- `governed` and `scope` separately, each with its **role gap**, and C0.14's
  condition that the gap exceed one noise unit for the position to count.

**The bootstrap honours C0.5.** One core-index set per draw, drawn once over the construct's core
names and applied at **every reference**, so a draw is a resampling of cores and not of
checkpoints. The core names are checked to be the same set at every reference before any draw is
made; a construct whose cores differ across references is refused rather than resampled by
position.

**Dating follows A30 and A33.** A30's R bands and its three flat sub-labels, the level test
against the untrained value at the same position, and the step rule the branch has: the 1.5x step
on the pretraining branch, **consecutive references on the annealing branch**, which A33 settled
after the annealing branch was measured to span x1.06 in tokens.
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
from profilebattery.paths import data_root, dataset_repo

REPO = dataset_repo()
D = data_root()
DRAWS = 1_000
SEED = 20261013
FORCED = "obliged"
DEPENDENT = ("governed", "scope")


def read_positions(doc: dict) -> dict:
    """`{(construct, class, role): entry}` and `{(construct, class, role, piece): entry}`.

    Keyed on the A set only: a `|CTL` set carries the control's loss, which enters as the role
    gap and never as the construct's own curve.
    """
    by_role, by_piece = {}, {}
    for key, e in (doc.get("sets") or {}).items():
        if not key.endswith("|A"):
            continue
        c = key[:-2]
        for cr, v in (e.get("by_class_role") or {}).items():
            cls, role = cr.split("|", 1)
            by_role[(c, cls, role)] = v
        for crp, v in (e.get("by_class_role_piece") or {}).items():
            parts = crp.split("|")
            if len(parts) == 3:
                by_piece[(c, parts[0], parts[1], parts[2])] = v
    return {"by_role": by_role, "by_piece": by_piece}


def core_draws(core_names, draws: int = DRAWS, seed: int = SEED):
    """One index set per draw over a construct's cores, reused at every reference (C0.5)."""
    rng = np.random.default_rng(seed)
    n = len(core_names)
    return [rng.integers(0, n, n) for _ in range(draws)]


def series(per_ref: dict, refs, key, which: str = "by_role"):
    """`(values, units, cores_ok)` for one position series across the references given.

    The unit at a checkpoint is the SD over draws of the resampled core mean -- the
    core-bootstrap SE, which is what A30 divides by. `cores_ok` is False when the construct's
    core names are not the same set at every reference, in which case no unit is produced: a
    bootstrap that resampled cores by position would be resampling different cores at different
    checkpoints.
    """
    entries = [(r, per_ref[r][which].get(key)) for r in refs]
    present = [(r, e) for r, e in entries if e and e.get("core_means")]
    if len(present) < 2:
        return [], [], True
    namesets = {frozenset(e["core_means"]) for _, e in present}
    if len(namesets) != 1:
        return ([e["mean"] for _, e in present], [None] * len(present), False)
    names = sorted(next(iter(namesets)))
    idx = core_draws(names)
    vals, units = [], []
    for _, e in present:
        cm = np.array([e["core_means"][n] for n in names], float)
        vals.append(float(e["mean"]))
        if cm.size < 2:
            units.append(None)
            continue
        units.append(float(np.std(np.array([cm[i].mean() for i in idx]), ddof=1)))
    return vals, units, True


def role_gaps(doc: dict) -> dict:
    """`{(construct, class): gap entry}` from a reference's `role_gap` block."""
    out = {}
    for c, e in (doc.get("role_gap") or {}).items():
        for cr, v in (e.get("by_class_role") or {}).items():
            out[(c, cr.split("|", 1)[0])] = v
    return out


def label(per_ref, refs, toks, key, untrained, branch: str, which="by_role") -> dict:
    """One position series, dated under A30 with A33's step rule for the branch."""
    vals, units, ok = series(per_ref, refs, key, which)
    if len(vals) < 2:
        return {"outcome": "unreadable", "why": "fewer than two checkpoints carry this position"}
    if not ok:
        return {"outcome": "unreadable",
                "why": "the construct's core names differ across references, so no "
                       "core-bootstrap unit can be formed"}
    n = len(vals)
    d = label_curve(vals, units, toks[:n], untrained=untrained,
                    step="consecutive" if branch == "anneal" else "factor")
    d["n_checkpoints"] = n
    d["step_rule"] = "consecutive" if branch == "anneal" else "factor"
    d["level_first"] = vals[0]
    d["level_last"] = vals[-1]
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(D / "validation/r2"))
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from experiments.run_s15 import references
    token = os.environ.get("HF_TOKEN")
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    p1 = {tree_name(c) for c in split["p1"]}
    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())

    per_ref, gaps = {}, {}
    for r in references(token):
        doc = json.load(open(hf_hub_download(REPO, f"record/{r}/POSITIONS.json",
                                             repo_type="dataset", token=token)))
        per_ref[r] = read_positions(doc)
        gaps[r] = role_gaps(doc)
    print(f"{len(per_ref)} references read")

    untrained = per_ref.get("untrained", {"by_role": {}, "by_piece": {}})
    report = {"n_p1": len(p1), "draws": a.draws, "classes": {}, "pieces": {}, "role_gap": {}}

    for branch in ("pretrain", "anneal"):
        seq = [(r, t) for r, t in branches(schedule)[branch] if r in per_ref]
        refs, toks = [r for r, _ in seq], [t for _, t in seq]
        print(f"\n=== {branch}: {len(refs)} checkpoints ===")

        for cls in (FORCED,) + DEPENDENT:
            keys = sorted({k for r in refs for k in per_ref[r]["by_role"] if k[1] == cls})
            inp1 = [k for k in keys if k[0] in p1]
            out = {}
            for k in inp1:
                u = (untrained["by_role"].get(k) or {}).get("mean")
                out["|".join(k)] = label(per_ref, refs, toks, k, u, branch)
            counts = collections.Counter(
                (v.get("flat") or v.get("label") or v["outcome"]) for v in out.values())
            print(f"  {cls:9s} {len(inp1):3d} series in P1 (of {len(keys)}): {dict(counts)}")
            report["classes"][f"{branch}|{cls}"] = {"n_series": len(inp1),
                                                    "n_all": len(keys),
                                                    "counts": dict(counts), "series": out}

        pkeys = sorted({k for r in refs for k in per_ref[r]["by_piece"] if k[0] in p1})
        pout = {}
        for k in pkeys:
            u = (untrained["by_piece"].get(k) or {}).get("mean")
            pout["|".join(k)] = label(per_ref, refs, toks, k, u, branch, which="by_piece")
        pc = collections.Counter((v.get("flat") or v.get("label") or v["outcome"])
                                 for v in pout.values())
        print(f"  pieces    {len(pkeys):3d} realised-piece curves in P1: {dict(pc)}")
        report["pieces"][branch] = {"n": len(pkeys), "counts": dict(pc), "series": pout}

        gk = sorted({k for r in refs for k in gaps[r] if k[0] in p1})
        gout = {}
        for c, cls in gk:
            vals = [gaps[r][(c, cls)]["mean"] for r in refs if (c, cls) in gaps[r]]
            ses = [gaps[r][(c, cls)].get("se") for r in refs if (c, cls) in gaps[r]]
            fin = [v for v in vals if np.isfinite(v)]
            su = [s for s in ses if s is not None and np.isfinite(s)]
            ncores = [gaps[r][(c, cls)].get("cores") for r in refs if (c, cls) in gaps[r]]
            unit = float(np.median(su)) if su else None
            e = {"n": len(fin), "cores": sorted({x for x in ncores if x}),
                 "mean": float(np.mean(fin)) if fin else None,
                 "median": float(np.median(fin)) if fin else None,
                 "noise_unit": unit}
            # **A zero unit is not a missing unit, and neither is a pass.** A series read on a
            # single core has no between-core spread, so its core-bootstrap SE is exactly 0 and
            # C0.14's "the gap exceeds one noise unit" is vacuous rather than satisfied. Both
            # cases are named instead of collapsing to a falsy `unit`, which is how `Import` and
            # `ImportFrom` first came back as `None` with no reason attached.
            if not fin:
                e["exceeds_one_unit"] = None
                e["why"] = "no finite gap at any reference"
            elif unit is None:
                e["exceeds_one_unit"] = None
                e["why"] = "no core-bootstrap SE is recorded for this series"
            elif unit == 0.0:
                e["exceeds_one_unit"] = None
                e["why"] = (f"the noise unit is exactly 0: this series is read on "
                            f"{e['cores']} core(s), so it has no between-core spread and "
                            f"C0.14 cannot be evaluated")
            else:
                e["exceeds_one_unit"] = bool(abs(np.median(fin)) > unit)
            gout[f"{c}|{cls}"] = e
        ok = sum(1 for v in gout.values() if v["exceeds_one_unit"] is True)
        no = sum(1 for v in gout.values() if v["exceeds_one_unit"] is False)
        un = sum(1 for v in gout.values() if v["exceeds_one_unit"] is None)
        print(f"  role gap  {len(gout):3d} dependent series: {ok} exceed one noise unit, "
              f"{no} do not, {un} cannot be evaluated (C0.14)")
        report["role_gap"][branch] = gout

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "R2_CURVE4.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\n-> {out / 'R2_CURVE4.json'}")


if __name__ == "__main__":
    main()
