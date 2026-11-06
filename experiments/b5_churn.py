r"""B5 — churn against new learning (design item 7).

§6.2's cause 2, displacement by new learning, says other constructs need the neurons and push
the carriers aside. Its prediction is a positive relation between a construct's churn and how
much *other* sets are growing at the same step; cause 1, diffusion, predicts no relation beyond
the learning rate. The draft already records three things against cause 2 (churn about equal in
every layer while only the shallow layers fill; the most contested neurons churning least; the
rate holding a floor while new learning slows) and names this as the direct test.

Read over the **19 true 1,000-step pairs**, so the step is constant and the spacing confound of
§3.2 does not enter, and across the 24 layers. Three quantities a pair:

  * **churn** -- a construct's departed fraction of its own 5% set;
  * **growth of other sets** -- how many neurons enter the set of SOME OTHER construct between
    the two references, as a fraction of all units, computed per construct by excluding its own;
  * **filling** -- the change in the fraction of units belonging to at least one set.

Then the relation across pairs (is a pair with more new learning a pair with more churn?) and
across layers (is a layer that fills a layer that churns?). Carrier index only.

    python experiments/b5_churn.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2

N_UNITS = 24 * 6144
LAYERS, WIDTH = 24, 6144


def main() -> None:
    from scipy.stats import spearmanr
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    pairs = [(i, i + 1) for i in range(len(refs) - 1)
             if iters[i + 1] - iters[i] == 1000]
    print(f"{len(pairs)} true 1,000-step pairs, {len(names)} constructs", flush=True)

    def sets(r):
        return {c: np.asarray(idx[r][f"twin1/{c}|product"], np.int64)
                for c in names if f"twin1/{c}|product" in idx[r]}

    rows, layer_rows = [], []
    for (i, j) in pairs:
        A, B = sets(refs[i]), sets(refs[j])
        common = [c for c in names if c in A and c in B]
        # membership counts, for the filling rate and for "some other set"
        occ_a = np.zeros(N_UNITS, np.int16)
        occ_b = np.zeros(N_UNITS, np.int16)
        mem_a = {}
        for c in common:
            occ_a[A[c]] += 1
            occ_b[B[c]] += 1
            mem_a[c] = A[c]
        fill_a = float((occ_a > 0).mean())
        fill_b = float((occ_b > 0).mean())
        for c in common:
            own_a = np.zeros(N_UNITS, bool)
            own_a[A[c]] = True
            own_b = np.zeros(N_UNITS, bool)
            own_b[B[c]] = True
            churn = float((own_a & ~own_b).sum() / own_a.sum())
            # entries into some OTHER construct's set, excluding c's own
            other_a = (occ_a - own_a.astype(np.int16)) > 0
            other_b = (occ_b - own_b.astype(np.int16)) > 0
            growth_other = float((other_b & ~other_a).mean())
            rows.append({"pair": [refs[i], refs[j]], "start_iter": iters[i], "construct": c,
                         "churn": churn, "growth_of_other_sets": growth_other,
                         "filling_change": fill_b - fill_a})
        # per layer, pooled over constructs
        for L in range(LAYERS):
            lo, hi = L * WIDTH, (L + 1) * WIDTH
            ch, n = 0, 0
            for c in common:
                a = A[c][(A[c] >= lo) & (A[c] < hi)]
                if a.size == 0:
                    continue
                bs = set(B[c][(B[c] >= lo) & (B[c] < hi)].tolist())
                ch += sum(1 for x in a if x not in bs)
                n += a.size
            fa = float((occ_a[lo:hi] > 0).mean())
            fb = float((occ_b[lo:hi] > 0).mean())
            layer_rows.append({"start_iter": iters[i], "layer": L,
                               "churn": ch / max(n, 1), "filling_a": fa,
                               "filling_change": fb - fa})
        print(f"  {refs[i][-7:]}->{refs[j][-7:]}  filling {fill_a:.4f}->{fill_b:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    def rel(x, y):
        if len(x) < 6:
            return None
        s = spearmanr(x, y)
        return {"spearman": float(s.statistic), "p": float(s.pvalue), "n": len(x)}

    # (a) across pairs, pooled by taking each pair's median churn against its growth
    bypair = {}
    for r in rows:
        d = bypair.setdefault(r["start_iter"], {"churn": [], "growth": [], "fill": None})
        d["churn"].append(r["churn"])
        d["growth"].append(r["growth_of_other_sets"])
        d["fill"] = r["filling_change"]
    ts = sorted(bypair)
    pc = [float(np.median(bypair[t]["churn"])) for t in ts]
    pg = [float(np.median(bypair[t]["growth"])) for t in ts]
    pf = [bypair[t]["fill"] for t in ts]
    # (b) across constructs within each pair, then pooled: the per-construct test
    within = []
    for t in ts:
        sub = [r for r in rows if r["start_iter"] == t]
        v = rel([r["growth_of_other_sets"] for r in sub], [r["churn"] for r in sub])
        if v:
            within.append(v["spearman"])
    # (c) across layers
    lbyl = {}
    for r in layer_rows:
        d = lbyl.setdefault(r["layer"], {"churn": [], "fillchg": [], "filla": []})
        d["churn"].append(r["churn"])
        d["fillchg"].append(r["filling_change"])
        d["filla"].append(r["filling_a"])
    Ls = sorted(lbyl)
    lc = [float(np.median(lbyl[L]["churn"])) for L in Ls]
    lf = [float(np.median(lbyl[L]["fillchg"])) for L in Ls]
    la = [float(np.median(lbyl[L]["filla"])) for L in Ls]

    out = {"what": "churn against the growth of other sets and the filling rate, over the 19 "
                   "true 1,000-step pairs and across the 24 layers",
           "reading": "displacement predicts a positive relation; drift none beyond the rate",
           "n_pairs": len(pairs), "n_constructs": len({r["construct"] for r in rows}),
           "across_pairs": {"start_iters": ts, "median_churn": pc,
                            "median_growth_of_other_sets": pg, "filling_change": pf,
                            "churn_vs_growth": rel(pg, pc),
                            "churn_vs_filling_change": rel(pf, pc)},
           "within_pair_across_constructs": {
               "spearman_per_pair": within,
               "median": float(np.median(within)) if within else None,
               "n_pairs_positive": int(sum(1 for v in within if v > 0)),
               "n_pairs": len(within)},
           "across_layers": {"layers": Ls, "median_churn": lc,
                             "median_filling_change": lf, "median_filling_level": la,
                             "churn_vs_filling_change": rel(lf, lc),
                             "churn_vs_filling_level": rel(la, lc)},
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["validation/carriers/*.npz"])}
    (OUT2 / "B5_CHURN_VS_LEARNING.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nacross the {len(ts)} pairs: churn vs growth of other sets "
          f"{out['across_pairs']['churn_vs_growth']}")
    print(f"  churn vs filling change {out['across_pairs']['churn_vs_filling_change']}")
    print(f"within a pair, across constructs: median rho "
          f"{out['within_pair_across_constructs']['median']:.3f}, positive in "
          f"{out['within_pair_across_constructs']['n_pairs_positive']} of "
          f"{out['within_pair_across_constructs']['n_pairs']} pairs")
    print(f"across the 24 layers: churn vs filling change "
          f"{out['across_layers']['churn_vs_filling_change']}")
    print(f"  churn vs filling level {out['across_layers']['churn_vs_filling_level']}")
    print(f"-> {OUT2 / 'B5_CHURN_VS_LEARNING.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
