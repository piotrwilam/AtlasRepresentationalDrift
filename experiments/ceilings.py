r"""The excess's ceiling AT EACH START POINT, for every measure figure 2 draws.

A single ceiling for the whole record was the first form, and it hides that both of its terms
move: the twin agreement rises over training and the chance level between unrelated constructs
is not constant either. The ceiling is therefore computed at each start point as

    ceiling(t) = twin agreement(t) - chance(t)

with both terms measured at t: for the 5% set, the twin carrier overlap against the 0.05 two
independent 5% sets would give; for the graded profile, the twin1-twin2 cosine (and the Spearman
analogue) against the median between DIFFERENT constructs at the same reference; for the six
bands, the band's own twin ceiling against its size-matched chance.

Local only -- the carrier index, the b14 caches and the concept store.

    python experiments/ceilings.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, D, K, provenance
from experiments.relatedness import OUT2

B14 = D / "validation/paper2/b14"
N_UNITS = 24 * 6144
BANDS = ("q1pct", "q2pct", "q5pct", "abs1sd", "abs2sd", "abs3sd")
N_CHANCE_PAIRS = 30            # constructs sampled for the between-construct chance level


def main() -> None:
    from scipy.stats import rankdata
    from experiments.concept_store import concept_of
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    rng = np.random.default_rng(20261023)
    sample = [names[i] for i in rng.choice(len(names), N_CHANCE_PAIRS, replace=False)]

    out = {"what": "the excess's ceiling at each start point: twin agreement minus chance, "
                   "both measured at that start point",
           "why": "a single ceiling hides that both terms move over training",
           "chance": {"5pct_set": "0.05, two independent 5% sets",
                      "graded": f"the median cosine (and Spearman) between {N_CHANCE_PAIRS} "
                                f"DIFFERENT constructs at the same reference",
                      "bands": "the band's own size over 147,456 at that reference"},
           "per_reference": {}}

    for i, r in enumerate(refs):
        e = {"iter": iters[i], "tokens_b": iters[i] * 4_194_304 / 1e9}
        # --- the 5% carrier set
        tw = []
        for c in names:
            a, b = f"twin1/{c}|product", f"twin2/{c}|product"
            if a in idx[r] and b in idx[r]:
                tw.append(np.intersect1d(np.asarray(idx[r][a], np.int64),
                                         np.asarray(idx[r][b], np.int64)).size / K)
        e["carrier5"] = {"twin_agreement": float(np.median(tw)), "chance": 0.05,
                         "ceiling": float(np.median(tw)) - 0.05, "n": len(tw)}
        # --- the graded profile
        V, Rk, tcos, tspe = [], [], [], []
        for c in sample:
            a = concept_of(r, "twin1", c).astype(np.float64)
            b = concept_of(r, "twin2", c).astype(np.float64)
            na, nb = a / np.linalg.norm(a), b / np.linalg.norm(b)
            tcos.append(float(na @ nb))
            ra, rb = rankdata(a), rankdata(b)
            ra -= ra.mean()
            rb -= rb.mean()
            ra, rb = ra / np.linalg.norm(ra), rb / np.linalg.norm(rb)
            tspe.append(float(ra @ rb))
            V.append(na)
            Rk.append(ra)
        cc = [float(V[x] @ V[y]) for x in range(len(V)) for y in range(x + 1, len(V))]
        rr = [float(Rk[x] @ Rk[y]) for x in range(len(Rk)) for y in range(x + 1, len(Rk))]
        for lab, tw_, ch_ in (("cosine", tcos, cc), ("spearman", tspe, rr)):
            e[lab] = {"twin_agreement": float(np.median(tw_)),
                      "chance": float(np.median(ch_)),
                      "ceiling": float(np.median(tw_)) - float(np.median(ch_)),
                      "n_constructs": len(tw_), "n_chance_pairs": len(ch_)}
        # --- the six bands
        for b in BANDS:
            cl, sz = [], []
            for c in names:
                f = B14 / f"{c}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                cl.append(float(z[f"ceil|{b}"][i]))
                sz.append(float(z[f"size|{b}"][i]))
            if not cl:
                continue
            ch = float(np.median(sz)) / N_UNITS
            e[b] = {"twin_agreement": float(np.nanmedian(cl)), "chance": ch,
                    "ceiling": float(np.nanmedian(cl)) - ch,
                    "median_size": float(np.median(sz))}
        out["per_reference"][refs[i]] = e
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(refs)} references ({(time.time()-t0)/60:.1f} min)", flush=True)

    ks = refs
    for m in ("carrier5", "cosine", "spearman"):
        v = [out["per_reference"][r][m]["ceiling"] for r in ks]
        tw = [out["per_reference"][r][m]["twin_agreement"] for r in ks]
        ch = [out["per_reference"][r][m]["chance"] for r in ks]
        out[f"{m}_summary"] = {
            "ceiling_first": v[0], "ceiling_last": v[-1],
            "ceiling_min": float(min(v)), "ceiling_max": float(max(v)),
            "twin_agreement_first": tw[0], "twin_agreement_last": tw[-1],
            "chance_first": ch[0], "chance_last": ch[-1]}
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["validation/carriers/*.npz",
                                    "validation/paper2/b14/<c>.npz",
                                    "the local concept store"])
    (OUT2 / "CEILINGS.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{'measure':10s} {'ceiling first':>14s} {'last':>8s} {'min':>8s} {'max':>8s}  "
          f"{'twin first->last':>18s} {'chance first->last':>20s}")
    for m in ("carrier5", "cosine", "spearman"):
        s = out[f"{m}_summary"]
        print(f"{m:10s} {s['ceiling_first']:14.4f} {s['ceiling_last']:8.4f} "
              f"{s['ceiling_min']:8.4f} {s['ceiling_max']:8.4f}  "
              f"{s['twin_agreement_first']:8.4f}->{s['twin_agreement_last']:<8.4f} "
              f"{s['chance_first']:9.4f}->{s['chance_last']:<9.4f}")
    print(f"-> {OUT2 / 'CEILINGS.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
