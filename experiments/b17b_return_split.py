r"""B17b — B17's return rate split by whether the entry really fell.

B17 found 57.4% of departures returning against a matched control's 17.9%, lift 3.21. CC's own
caveat on it was that §4.4 reports a third of departures as neurons that have only slipped just
under the same set's threshold, so part of that rate is threshold jitter rather than
reorganisation. This splits it.

A departure at reference i is a neuron in twin1's 5% set at i and out at i+1. Its **entry fall**
is |e_i| - |e_{i+1}| on the standardised profile, and it counts as a real fall only if it exceeds
**one twin unit** -- |concept_twin1 - concept_twin2| for that construct on that neuron at
reference i, the same surface-noise unit test 6's displacement number uses. The control is
matched exactly as in B17 (out at i, out at i+1, never in at i, sampled to the departure count)
and is split by the same threshold on its own entry fall, so each arm has its own chance level
rather than sharing B17's pooled one.

Reading, fixed here before the result: if the return rate is high only in the within-noise arm,
B17's lift is threshold jitter; if it survives in the arm that fell by more than a twin unit,
neurons that genuinely left come back and §6.2's cause 4 keeps its support.

    python experiments/b17b_return_split.py [n_constructs]
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from experiments.concept_store import concept_of
from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2

N_UNITS = 24 * 6144
ARMS = ("fell_beyond_one_twin_unit", "fell_within_one_twin_unit")


def main() -> None:
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    if limit:
        names = names[:limit]
    idx = load_index(CARRIERS, refs)
    R = len(refs)
    rng = np.random.default_rng(20261023)
    print(f"{R} references, {len(names)} constructs, K={K:,}", flush=True)

    per_construct = {}
    tot = {a: {"dep": 0, "ret": 0, "ctl": 0, "ctl_ret": 0} for a in ARMS}
    ABS_STEPS = {a: [] for a in ARMS}
    ABS_REFS = {a: [] for a in ARMS}
    for c in names:
        t1 = time.time()
        key = f"twin1/{c}|product"
        if any(key not in idx[r] for r in refs):
            print(f"  {c}: SKIP, missing from the index", flush=True)
            continue
        E1 = np.empty((R, N_UNITS), np.float32)
        E2 = np.empty((R, N_UNITS), np.float32)
        bad = []

        def one(a):
            i, tw = a
            try:
                v = concept_of(refs[i], tw, c)
                (E1 if tw == "twin1" else E2)[i] = v
            except Exception as exc:                                # noqa: BLE001
                bad.append(type(exc).__name__)

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, [(i, tw) for i in range(R) for tw in ("twin1", "twin2")]))
        if bad:
            print(f"  {c}: SKIP, {len(bad)} unreadable ({bad[0]})", flush=True)
            continue
        M = np.zeros((R, N_UNITS), bool)
        for i, r in enumerate(refs):
            M[i, np.asarray(idx[r][key], np.int64)] = True
        later = np.zeros((R, N_UNITS), bool)
        for i in range(R - 2, -1, -1):
            later[i] = M[i + 1] | later[i + 1]
        A1, A2 = np.abs(E1), np.abs(E2)
        pc = {a: {"dep": 0, "ret": 0, "ctl": 0, "ctl_ret": 0} for a in ARMS}
        abs_steps = {a: [] for a in ARMS}
        abs_refs = {a: [] for a in ARMS}
        for i in range(R - 1):
            unit = np.abs(E1[i] - E2[i])              # one twin unit at this reference
            fall = A1[i] - A1[i + 1]
            real = fall > unit
            dep = np.flatnonzero(M[i] & ~M[i + 1])
            if dep.size == 0:
                continue
            pool = np.flatnonzero(~M[i] & ~M[i + 1])
            if pool.size == 0:
                continue
            ctl = rng.choice(pool, min(dep.size, pool.size), replace=False)
            back = later[i + 1][dep] if i + 2 <= R - 1 else np.zeros(dep.size, bool)
            cback = later[i + 1][ctl] if i + 2 <= R - 1 else np.zeros(ctl.size, bool)
            for a, mask_d, mask_c in ((ARMS[0], real[dep], real[ctl]),
                                      (ARMS[1], ~real[dep], ~real[ctl])):
                pc[a]["dep"] += int(mask_d.sum())
                pc[a]["ret"] += int((back & mask_d).sum())
                pc[a]["ctl"] += int(mask_c.sum())
                pc[a]["ctl_ret"] += int((cback & mask_c).sum())
                # absence length PER ARM: the first later reference that holds it again,
                # in references and in training steps
                sub = dep[back & mask_d]
                if sub.size:
                    first = np.full(sub.size, -1)
                    for j in range(i + 1, R):
                        hit = (first < 0) & M[j][sub]
                        first[hit] = j
                        if (first >= 0).all():
                            break
                    f = first[first >= 0]
                    abs_refs[a].extend((f - (i + 1) + 1).tolist())
                    abs_steps[a].extend([iters[int(j)] - iters[i + 1] for j in f])
        per_construct[c] = {a: dict(pc[a]) for a in ARMS}
        for a in ARMS:
            for k in ("dep", "ret", "ctl", "ctl_ret"):
                tot[a][k] += pc[a][k]
            ABS_STEPS[a].extend(abs_steps[a])
            ABS_REFS[a].extend(abs_refs[a])
        sh = tot[ARMS[0]]["dep"] / max(tot[ARMS[0]]["dep"] + tot[ARMS[1]]["dep"], 1)
        print(f"  {c:22s} beyond-unit share {sh:.3f}  "
              f"ret {pc[ARMS[0]]['ret']/max(pc[ARMS[0]]['dep'],1):.3f} / "
              f"{pc[ARMS[1]]['ret']/max(pc[ARMS[1]]['dep'],1):.3f}  "
              f"({time.time()-t1:.0f}s, {(time.time()-t0)/60:.1f} min)", flush=True)

    cs = sorted(per_construct)
    out = {"what": "B17's return rate split by whether the entry fell by more than one twin "
                   "unit, with the control matched the same way and split the same way",
           "twin_unit": "|concept_twin1 - concept_twin2| for that construct on that neuron at "
                        "the departure reference",
           "control": "out of the set at i and at i+1, never in at i, sampled to the departure "
                      "count, then split by its own entry fall",
           "reading": ("high return only within the noise -> B17's lift is threshold jitter; "
                       "surviving beyond one twin unit -> genuine departures return"),
           "n_references": R, "n_constructs": len(cs), "arms": {}}
    for a in ARMS:
        t = tot[a]
        rr = t["ret"] / max(t["dep"], 1)
        cr = t["ctl_ret"] / max(t["ctl"], 1)
        arr = np.array([[per_construct[c][a][k] for k in ("dep", "ret", "ctl", "ctl_ret")]
                        for c in cs], float)
        boot = []
        for _ in range(1000):
            s = arr[rng.integers(0, len(cs), len(cs))].sum(0)
            if s[0] and s[2] and s[3]:
                boot.append((s[1] / s[0]) / (s[3] / s[2]))
        out["arms"][a] = {
            "departures": t["dep"], "returned": t["ret"], "return_rate": rr,
            "control": t["ctl"], "control_rate": cr,
            "lift": rr / cr if cr else None,
            "lift_ci95": ([float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
                          if len(boot) > 20 else None),
            "absence_length_steps": (
                {"n": len(ABS_STEPS[a]),
                 "q1": float(np.quantile(ABS_STEPS[a], .25)),
                 "median": float(np.median(ABS_STEPS[a])),
                 "q3": float(np.quantile(ABS_STEPS[a], .75))}
                if ABS_STEPS[a] else None),
            "absence_length_histogram_references": (
                {str(k): int(v) for k, v in sorted(
                    __import__("collections").Counter(ABS_REFS[a]).items())}
                if ABS_REFS[a] else None),
            "absence_length_references": (
                {"q1": float(np.quantile(ABS_REFS[a], .25)),
                 "median": float(np.median(ABS_REFS[a])),
                 "q3": float(np.quantile(ABS_REFS[a], .75)),
                 "share_back_at_the_next_opportunity":
                     float(np.mean(np.asarray(ABS_REFS[a]) <= 2))}
                if ABS_REFS[a] else None)}
    d0 = out["arms"][ARMS[0]]["departures"]
    d1 = out["arms"][ARMS[1]]["departures"]
    out["share_of_departures_beyond_one_twin_unit"] = d0 / max(d0 + d1, 1)
    out["per_construct"] = per_construct
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["validation/carriers/*.npz",
                                    "profiles/<ref>/twin{1,2}/<c>.npz concept, via the "
                                    "dataset-only resolver and the local concept store"])
    (OUT2 / "B17B_RETURN_SPLIT.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{d0+d1:,} departures; {out['share_of_departures_beyond_one_twin_unit']:.3f} "
          f"fell by more than one twin unit")
    for a in ARMS:
        e = out["arms"][a]
        print(f"  {a:28s} n={e['departures']:12,d} return {e['return_rate']:.4f} "
              f"control {e['control_rate']:.4f} lift {(e['lift'] or 0):.3f} "
              f"{[round(x,3) for x in (e['lift_ci95'] or [0,0])]}")
        s_ = e.get("absence_length_steps") or {}
        r_ = e.get("absence_length_references") or {}
        if s_:
            print(f"    absence: steps q1 {s_['q1']:.0f} / median {s_['median']:.0f} / q3 "
                  f"{s_['q3']:.0f}; references q1 {r_['q1']:.0f} / median {r_['median']:.0f} "
                  f"/ q3 {r_['q3']:.0f}; back at the next opportunity "
                  f"{r_['share_back_at_the_next_opportunity']:.1%}")
    print(f"-> {OUT2 / 'B17B_RETURN_SPLIT.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
