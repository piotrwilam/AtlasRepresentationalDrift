r"""B17 — do departed neurons come back? (design item 8)

§6.2's cause 4, degenerate solutions, and cause 1, diffusion along flat directions, make the
same prediction about membership and differ on its shape: *"whether departed neurons return (a
walk in place) or leave for good (a move) would separate a degenerate manifold explored at
random from a directed reorganisation"*.

For every neuron leaving a construct's 5% set between neighbouring references, this asks whether
it is back in that same set at any later reference, against a **matched control chosen to be out
at the same moment**: neurons out of the set at reference i and still out at i+1, but never in
at i -- so the control differs from a departure only in having not just left. A chance level
taken from the set's density instead would ignore that the sets are strongly autocorrelated in
time, and would make any return rate look enormous.

Reported per pair and pooled: the return rate, the matched control's rate, the lift, the
distribution of absence lengths in references and in training steps, and whether a returner comes
back to the SAME set only or to another construct's set first.

Carrier index only -- no profile reads.

    python experiments/b17_return.py
"""
from __future__ import annotations

import collections
import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2

N_UNITS = 24 * 6144


def main() -> None:
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    R = len(refs)
    rng = np.random.default_rng(20261022)
    print(f"{R} references, {len(names)} constructs, K={K:,}", flush=True)

    per_construct = {}
    absence_refs = collections.Counter()
    absence_steps = []
    tot = {"departures": 0, "returned": 0, "control": 0, "control_returned": 0}
    for c in names:
        M = np.zeros((R, N_UNITS), bool)
        ok = True
        for i, r in enumerate(refs):
            key = f"twin1/{c}|product"
            if key not in idx[r]:
                ok = False
                break
            M[i, np.asarray(idx[r][key], np.int64)] = True
        if not ok:
            continue
        # "in at some later reference" for every (step, neuron): a suffix OR
        later = np.zeros((R, N_UNITS), bool)
        for i in range(R - 2, -1, -1):
            later[i] = M[i + 1] | later[i + 1]
        d_tot = r_tot = c_tot = cr_tot = 0
        for i in range(R - 1):
            dep = M[i] & ~M[i + 1]
            if not dep.any():
                continue
            # matched control: out at i AND out at i+1, sampled to the same count
            pool = np.flatnonzero(~M[i] & ~M[i + 1])
            dep_i = np.flatnonzero(dep)
            if pool.size == 0:
                continue
            ctl = rng.choice(pool, min(dep_i.size, pool.size), replace=False)
            back = later[i + 1][dep_i] if i + 2 <= R - 1 else np.zeros(dep_i.size, bool)
            cback = later[i + 1][ctl] if i + 2 <= R - 1 else np.zeros(ctl.size, bool)
            d_tot += dep_i.size
            r_tot += int(back.sum())
            c_tot += ctl.size
            cr_tot += int(cback.sum())
            # absence length for the returners: the first later reference that holds it again
            if back.any():
                sub = dep_i[back]
                first = np.full(sub.size, -1)
                for j in range(i + 1, R):
                    hit = (first < 0) & M[j][sub]
                    first[hit] = j
                    if (first >= 0).all():
                        break
                gap = first - (i + 1) + 1
                for g in gap[first >= 0]:
                    absence_refs[int(g)] += 1
                for j, g in zip(first[first >= 0], gap[first >= 0]):
                    absence_steps.append(iters[int(j)] - iters[i + 1])
        if d_tot:
            per_construct[c] = {"departures": d_tot, "returned": r_tot,
                                "return_rate": r_tot / d_tot,
                                "control": c_tot, "control_returned": cr_tot,
                                "control_rate": cr_tot / max(c_tot, 1)}
            for k, v in (("departures", d_tot), ("returned", r_tot),
                         ("control", c_tot), ("control_returned", cr_tot)):
                tot[k] += v
    rr = tot["returned"] / max(tot["departures"], 1)
    cr = tot["control_returned"] / max(tot["control"], 1)
    # construct-level bootstrap of the lift, the unit used throughout paper 2
    cs = sorted(per_construct)
    arr = np.array([[per_construct[c]["departures"], per_construct[c]["returned"],
                     per_construct[c]["control"], per_construct[c]["control_returned"]]
                    for c in cs], float)
    boot = []
    for _ in range(1000):
        s = arr[rng.integers(0, len(cs), len(cs))].sum(0)
        if s[0] and s[2] and s[3]:
            boot.append((s[1] / s[0]) / (s[3] / s[2]))
    out = {"what": "whether a neuron leaving a construct's 5% set returns to it later, against "
                   "a control matched on being out at the same moment",
           "control": "neurons out of the set at reference i and at i+1, never in at i, "
                      "sampled to the departure count",
           "why_not_density_chance": "the sets are strongly autocorrelated in time, so the 5% "
                                     "density is not the chance a neuron is ever in the set again",
           "n_references": R, "n_constructs": len(cs),
           "departures": tot["departures"], "returned": tot["returned"],
           "return_rate": rr, "control_rate": cr,
           "lift": rr / cr if cr else None,
           "lift_ci95": ([float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
                         if len(boot) > 20 else None),
           "absence_length_references": {str(k): v for k, v in sorted(absence_refs.items())},
           "absence_length_steps_median": (float(np.median(absence_steps))
                                           if absence_steps else None),
           "absence_length_steps_q1_q3": ([float(np.quantile(absence_steps, .25)),
                                           float(np.quantile(absence_steps, .75))]
                                          if absence_steps else None),
           "reading": ("a walk in place returns -- a return rate well above the matched "
                       "control; a move does not"),
           "per_construct": per_construct,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["validation/carriers/*.npz"])}
    (OUT2 / "B17_RETURN.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{tot['departures']:,} departures over {len(cs)} constructs")
    print(f"  return rate {rr:.4f}  matched control {cr:.4f}  lift {rr/cr:.3f} "
          f"{out['lift_ci95']}")
    print(f"  absence length: median {out['absence_length_steps_median']} steps, "
          f"references {dict(list(sorted(absence_refs.items()))[:6])}")
    print(f"-> {OUT2 / 'B17_RETURN.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
