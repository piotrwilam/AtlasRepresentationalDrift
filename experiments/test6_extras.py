r"""The two items owed on test 6: displacement with a threshold, and the token pairs behind the lift.

**(a) Displacement, redone.** The first form asked whether *any* other construct rose, and the
maximum over 168 is almost surely positive, so it answered yes by construction (1.0000 at every
pair). A rise now counts only if it exceeds **one twin unit** of that construct's entry on that
neuron -- |E_twin1 - E_twin2| for the same construct, neuron and reference -- and the same
threshold is applied to the matched chance construct.

**(b) The twenty pairs carrying the tokens lift**, with the shared token and whether it precedes
the read site.

    python experiments/test6_extras.py
"""
from __future__ import annotations

import collections
import json
import time

import numpy as np

from experiments.atlas import K, provenance
from experiments.relatedness import OUT2, Relatedness
from experiments.test6 import entries

PAIRS = ("0001000:0002000", "0002000:0003000", "0019000:0020000")


def main() -> None:
    from experiments.atlas import p1_set
    from profilebattery.profiles.names import contract_name
    t0 = time.time()
    names = sorted(p1_set())
    R = Relatedness()
    inv = R.inv
    rng = np.random.default_rng(20261022)
    out = {}
    for spec in PAIRS:
        a, b = spec.split(":")
        ra, rb = f"pretrain_iter_{a}", f"pretrain_iter_{b}"
        Ea, Eb = entries(ra, names), entries(rb, names)
        Ea2 = entries(ra, names, "twin2")
        rise = Eb - Ea
        twin_unit = np.abs(Ea - Ea2)              # the surface-noise unit, per construct, neuron
        car = [np.sort(np.argsort(-(Ea[i].astype(np.float64) ** 2), kind="stable")[:K])
               for i in range(len(names))]
        carb = [set(np.argsort(-(Eb[i].astype(np.float64) ** 2),
                               kind="stable")[:K].tolist()) for i in range(len(names))]
        disp = {"observed": 0, "chance": 0, "n": 0}
        tok_pairs = collections.Counter()
        for i, c in enumerate(names):
            dep = np.array([x for x in car[i] if x not in carb[i]], np.int64)
            if dep.size == 0:
                continue
            if dep.size > 20000:
                dep = rng.choice(dep, 20000, replace=False)
            sub = rise[:, dep].copy()
            sub[i] = -np.inf
            top = np.argmax(sub, axis=0)
            for n_, k in enumerate(top):
                k = int(k)
                thr_o = twin_unit[k, dep[n_]]
                kc = int(rng.integers(0, len(names)))
                while kc in (i, k):
                    kc = int(rng.integers(0, len(names)))
                disp["n"] += 1
                disp["observed"] += int(sub[k, n_] > thr_o)
                disp["chance"] += int(rise[kc, dep[n_]] > twin_unit[kc, dep[n_]])
                if R.tokens(c, names[k]):
                    tok_pairs[(c, names[k])] += 1
        top20 = []
        for (x, y), n in tok_pairs.most_common(20):
            tx = set(inv.get(contract_name(x), {}).get("tokens") or [])
            ty = set(inv.get(contract_name(y), {}).get("tokens") or [])
            shared = sorted(tx & ty)
            at_site = bool(R.tokens_at_read_site(x, y))
            top20.append({"left": x, "rose": y, "n_departures": n,
                          "shared_tokens": shared, "shared_at_read_site": at_site,
                          "precedes_read_site": (not at_site)})
        out[spec] = {
            "displacement_thresholded": {
                "n": disp["n"],
                "rate_observed": disp["observed"] / disp["n"] if disp["n"] else None,
                "rate_chance": disp["chance"] / disp["n"] if disp["n"] else None,
                "lift": ((disp["observed"] / disp["n"]) / (disp["chance"] / disp["n"])
                         if disp["n"] and disp["chance"] else None),
                "threshold": "one twin unit of that construct's entry on that neuron"},
            "top20_token_pairs": top20}
        d_ = out[spec]["displacement_thresholded"]
        print(f"{spec}: displacement observed {d_['rate_observed']:.4f} "
              f"chance {d_['rate_chance']:.4f} lift "
              f"{d_['lift'] if d_['lift'] is None else round(d_['lift'],3)}", flush=True)
        for e in top20[:5]:
            print(f"    {e['left']:14s} -> {e['rose']:14s} n={e['n_departures']:>7,} "
                  f"shared {e['shared_tokens']} "
                  f"{'at the read site' if e['shared_at_read_site'] else 'in the cores only'}",
                  flush=True)
    doc = {"what": "displacement with a twin-unit threshold, and the token pairs behind the lift",
           "pairs": out, "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["profiles/<ref>/twin{1,2}/<c>.npz via the resolver"])}
    (OUT2 / "TEST6_EXTRAS.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"-> {OUT2 / 'TEST6_EXTRAS.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
