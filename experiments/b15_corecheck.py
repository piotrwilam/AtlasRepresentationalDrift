r"""B15 — test 6's core-composition check (design item 10).

Test 6 found departing neurons rising towards constructs that **share a token**, at a lift of
1.41/1.56/1.77 over matched chance, and the lift sat in pairs sharing a token *in their cores*
rather than at the read site. §6.3 names the artefact that would produce exactly that without
any model fact: a token physically present in one construct's cores puts its own carriers inside
that construct's set, and they leave when that token's own set forms.

`CORE_TOKENS.json` now says which tokens the cores actually contain, and it confirms the shape:
`For`'s core spans carry `range`, `enumerate`, `zip`, `sorted`, `items`, `append`, and `else` is
present on the A side only. So `For`-`Try`, related by the grammar token `else`, is a pair whose
shared token the A cores render.

The check, as the design fixes it: recompute the token measure with the related class restricted
to pairs where **no shared token appears in either construct's cores**, and report

  * the restricted lift with its interval beside the original, computed in the same pass so the
    comparison is exact rather than against a remembered number;
  * the complement -- pairs whose shared token IS in a core -- so the two halves can be seen;
  * the share of the original observed count that the core-shared half carries.

Reading (fixed before the result): if the restricted lift stays above 1 with its interval clear,
the direction is the model's; if it falls to chance, it is the battery's composition.

    python experiments/b15_corecheck.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.atlas import D, K, provenance
from experiments.relatedness import OUT2, Relatedness
from experiments.test6 import carriers_of, entries

CORE_TOKENS = OUT2 / "CORE_TOKENS.json"
MEASURES = ("tokens", "tokens_outside_cores", "tokens_in_cores",
            "tokens_outside_cores_A_only")


class CoreAware(Relatedness):
    """`Relatedness` plus the three core-restricted token predicates."""

    def __init__(self):
        super().__init__()
        d = json.loads(CORE_TOKENS.read_text())["constructs"]
        self._core = {c: set(v["union"]) for c, v in d.items()}
        self._core_a = {c: set(v["A_only"]) for c, v in d.items()}

    def _shared(self, j, k):
        a = set(self._e(j).get("tokens")
                or ([self._e(j).get("token")] if self._e(j).get("token") else []))
        b = set(self._e(k).get("tokens")
                or ([self._e(k).get("token")] if self._e(k).get("token") else []))
        if not a or not b:
            return None
        return a & b

    def _restricted(self, j, k, cores):
        """True if they share a token and NO shared token is in either construct's cores."""
        s = self._shared(j, k)
        if s is None:
            return None
        if not s:
            return False
        cj, ck = cores.get(j, set()), cores.get(k, set())
        return not any((t in cj) or (t in ck) for t in s)

    def tokens_outside_cores(self, j, k):
        return self._restricted(j, k, self._core)

    def tokens_outside_cores_A_only(self, j, k):
        """The artefact's own shape: a token on A and absent on B is what biases the profile."""
        return self._restricted(j, k, self._core_a)

    def tokens_in_cores(self, j, k):
        """The complement of `tokens_outside_cores` within the token-sharing pairs."""
        s = self._shared(j, k)
        if s is None:
            return None
        if not s:
            return False
        cj, ck = self._core.get(j, set()), self._core.get(k, set())
        return any((t in cj) or (t in ck) for t in s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*",
                    default=["0001000:0002000", "0002000:0003000", "0019000:0020000"])
    ap.add_argument("--max-departures", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    from experiments.atlas import p1_set
    from profilebattery.profiles.names import contract_name
    names = sorted(p1_set())
    R = CoreAware()
    rates = json.loads((D / "calibration/construct_rates.json").read_text())
    brates = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    allr = {**rates, **brates}
    freq = np.array([np.log10(max((allr.get(contract_name(c)) or {}).get("per_file") or 1e-6,
                                  1e-6)) for c in names])
    # how many pairs each predicate admits, before any departure is read
    cov = {m: 0 for m in MEASURES}
    npair = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            npair += 1
            for m in MEASURES:
                v = getattr(R, m)(names[i], names[j])
                cov[m] += int(bool(v))
    print(f"{npair:,} construct pairs; related by: "
          + ", ".join(f"{m} {cov[m]:,} ({cov[m]/npair:.3%})" for m in MEASURES), flush=True)

    rng = np.random.default_rng(args.seed)
    out, t0 = {}, time.time()
    for spec in args.pairs:
        fp = OUT2 / "b15" / f"{spec.replace(':', '_')}.json"
        if fp.exists():
            out[spec] = json.loads(fp.read_text())
            print(f"  {spec}: from cache", flush=True)
            continue
        a, b = spec.split(":")
        ra, rb = f"pretrain_iter_{a}", f"pretrain_iter_{b}"
        Ea, Eb = entries(ra, names), entries(rb, names)
        rise = Eb - Ea
        car_a, car_b = carriers_of(Ea), carriers_of(Eb)
        setb = [set(x.tolist()) for x in car_b]
        ov = np.zeros((len(names), len(names)), np.float32)
        for i in range(len(names)):
            si = set(car_a[i].tolist())
            for j in range(i + 1, len(names)):
                v = len(si & set(car_a[j].tolist())) / K
                ov[i, j] = ov[j, i] = v
        tally = {m: {"observed": 0, "chance": 0, "n": 0} for m in MEASURES}
        per_construct = {m: {} for m in MEASURES}
        n_dep = 0
        for i, c in enumerate(names):
            dep = np.array([x for x in car_a[i] if x not in setb[i]], np.int64)
            if dep.size == 0:
                continue
            if dep.size > args.max_departures:
                dep = rng.choice(dep, args.max_departures, replace=False)
            n_dep += dep.size
            sub = rise[:, dep].copy()
            sub[i] = -np.inf
            top = np.argmax(sub, axis=0)
            for t_idx, k in enumerate(top):
                k = int(k)
                pool = np.where((np.abs(freq - freq[k]) < 0.5)
                                & (np.abs(ov[i] - ov[i, k]) < 0.05))[0]
                pool = pool[(pool != i) & (pool != k)]
                kc = int(rng.choice(pool)) if pool.size else None
                if kc is None:
                    continue
                for m in MEASURES:
                    o = getattr(R, m)(c, names[k])
                    ch = getattr(R, m)(c, names[kc])
                    if o is None or ch is None:
                        continue
                    tally[m]["n"] += 1
                    tally[m]["observed"] += int(o)
                    tally[m]["chance"] += int(ch)
                    d = per_construct[m].setdefault(c, [0, 0, 0])
                    d[0] += 1
                    d[1] += int(o)
                    d[2] += int(ch)
            if (i + 1) % 40 == 0:
                print(f"    {spec}: {i+1}/{len(names)} constructs, {n_dep:,} departures "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

        def boot_interval(m):
            """Construct-level bootstrap of the lift, the same unit test 6 used."""
            cs = sorted(per_construct[m])
            if len(cs) < 4:
                return None
            arr = np.array([per_construct[m][c] for c in cs], float)
            vals = []
            for _ in range(args.boot):
                pick = rng.integers(0, len(cs), len(cs))
                s = arr[pick].sum(0)
                if s[0] > 0 and s[2] > 0:
                    vals.append((s[1] / s[0]) / (s[2] / s[0]))
            return ([float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
                    if len(vals) > 20 else None)

        e = {"pair": [ra, rb], "n_departures": int(n_dep), "measures": {}}
        for m in MEASURES:
            t = tally[m]
            if not t["n"]:
                continue
            o, ch = t["observed"] / t["n"], t["chance"] / t["n"]
            e["measures"][m] = {
                "n": t["n"], "observed": o, "chance": ch,
                "lift": (o / ch) if ch else None,
                "ci95": boot_interval(m),
                "n_constructs": len(per_construct[m])}
        tk = e["measures"].get("tokens", {})
        ic = e["measures"].get("tokens_in_cores", {})
        if tk.get("observed") and ic.get("observed") is not None:
            e["share_of_original_observed_carried_by_core_shared"] = \
                ic["observed"] / tk["observed"]
        out[spec] = e
        # per-pair resume: a dropped connection costs one pair, not the pass
        (OUT2 / "b15").mkdir(parents=True, exist_ok=True)
        (OUT2 / "b15" / f"{spec.replace(':', '_')}.json").write_text(
            json.dumps(e, indent=1, default=float))
        print(f"  {spec}: {n_dep:,} departures", flush=True)
        for m, v in e["measures"].items():
            ci = v["ci95"]
            print(f"    {m:28s} obs {v['observed']:.4f} chance {v['chance']:.4f} "
                  f"lift {(v['lift'] or 0):.3f} "
                  f"{('[%.3f, %.3f]' % (ci[0], ci[1])) if ci else ''}", flush=True)

    doc = {"what": "test 6's token measure with the related class restricted to pairs whose "
                   "shared token is in neither construct's cores",
           "reading_rule": "restricted lift above 1 with its interval clear -> the direction "
                           "is the model's; falls to chance -> the battery's composition",
           "core_tokens_from": str(CORE_TOKENS.name),
           "pair_coverage": {m: {"n_pairs": cov[m], "share": cov[m] / npair}
                             for m in MEASURES},
           "pairs": out, "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["profiles/<ref>/twin1/<c>.npz via the dataset-only "
                                     "resolver", "validation/paper2/CORE_TOKENS.json"])}
    (OUT2 / "B15_TEST6_CORECHECK.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"-> {OUT2 / 'B15_TEST6_CORECHECK.json'}  ({doc['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
