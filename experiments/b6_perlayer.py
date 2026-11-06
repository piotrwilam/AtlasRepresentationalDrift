r"""B6 — sharing between constructs, per layer (design item 11).

§6.4 names this as bearing on cause 2, capacity binding: *"sharing rising where a layer is full
supports capacity binding"*. The draft already reports that churn in the boundary band is about
equal in every layer (0.44-0.61) while only the shallow layers fill, which is against cause 2;
this asks the complementary question on sharing rather than churn.

Per reference and per layer: the median pairwise sharing between constructs, as the intersection
of two constructs' 5% sets **within that layer** over the smaller of the two layer-restricted
sets (so a layer where both constructs keep few carriers is not penalised for it), beside the
layer's own occupancy -- the fraction of its 6,144 units in at least one set. Then, per layer,
whether sharing rises over the record, and whether the layers where it rises are the layers that
filled or the layers that were full from the start.

All 169x169 pairs at all 24 layers at all 35 references, by packing each construct's set into a
dense (24 x 6,144) indicator and taking the intersections with one einsum -- 12 million
intersections in about a minute. Carrier index only.

    python experiments/b6_perlayer.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, provenance
from experiments.relatedness import OUT2

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
    n = len(names)
    iu = np.triu_indices(n, 1)
    print(f"{len(refs)} references, {n} constructs, {len(iu[0]):,} pairs x {LAYERS} layers",
          flush=True)

    per_ref = {}
    for r in refs:
        t1 = time.time()
        P = np.zeros((n, LAYERS, WIDTH), np.float32)
        ok = np.ones(n, bool)
        for i, c in enumerate(names):
            key = f"twin1/{c}|product"
            if key not in idx[r]:
                ok[i] = False
                continue
            v = np.asarray(idx[r][key], np.int64)
            P[i].reshape(-1)[v] = 1.0
        size = P.sum(2)                                   # [n, LAYERS]
        inter = np.einsum('ilk,jlk->ijl', P, P)           # [n, n, LAYERS]
        # share over the SMALLER layer-restricted set, so a thin layer is not penalised
        mn = np.minimum(size[:, None, :], size[None, :, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(mn > 0, inter / np.maximum(mn, 1), np.nan)
        occ = (P.sum(0) > 0).mean(1)                      # [LAYERS] occupancy
        med = []
        for L in range(LAYERS):
            v = share[:, :, L][iu]
            v = v[np.isfinite(v)]
            med.append(float(np.median(v)) if v.size else None)
        per_ref[r] = {"iter": iters[refs.index(r)],
                      "median_share_by_layer": med,
                      "occupancy_by_layer": [float(x) for x in occ],
                      "median_layer_set_size": [float(np.median(size[ok, L]))
                                                for L in range(LAYERS)],
                      "n_constructs": int(ok.sum())}
        print(f"  {r[-7:]}: layer-0 share {med[0]:.3f}, layer-23 {med[-1]:.3f}, "
              f"occupancy {occ[0]:.3f}..{occ[-1]:.3f}  ({time.time()-t1:.0f}s, "
              f"{(time.time()-t0)/60:.1f} min)", flush=True)

    ts = [per_ref[r]["iter"] for r in refs]
    out = {"what": "median pairwise sharing of 5% sets within each layer, and each layer's "
                   "occupancy, over the 35 pretraining references",
           "share_definition": "intersection within the layer over the smaller of the two "
                               "layer-restricted sets",
           "reading": "sharing rising where a layer is full supports capacity binding",
           "n_references": len(refs), "n_constructs": len(names),
           "per_reference": per_ref, "per_layer": {}}
    rises = 0
    for L in range(LAYERS):
        s = [per_ref[r]["median_share_by_layer"][L] for r in refs]
        o = [per_ref[r]["occupancy_by_layer"][L] for r in refs]
        tr = spearmanr(np.log(ts), s)
        out["per_layer"][L] = {
            "share_first": s[0], "share_last": s[-1], "share_change": s[-1] - s[0],
            "trend_spearman_vs_log_step": float(tr.statistic), "trend_p": float(tr.pvalue),
            "occupancy_first": o[0], "occupancy_last": o[-1],
            "occupancy_change": o[-1] - o[0],
            "full_from_the_start": bool(o[0] > 0.9)}
        rises += int(s[-1] > s[0])
    dsh = [out["per_layer"][L]["share_change"] for L in range(LAYERS)]
    docc = [out["per_layer"][L]["occupancy_change"] for L in range(LAYERS)]
    occ0 = [out["per_layer"][L]["occupancy_first"] for L in range(LAYERS)]
    r1 = spearmanr(docc, dsh)
    r2 = spearmanr(occ0, dsh)
    out["layers_where_sharing_rises"] = rises
    out["share_change_vs_occupancy_change"] = {"spearman": float(r1.statistic),
                                               "p": float(r1.pvalue), "n": LAYERS}
    out["share_change_vs_initial_occupancy"] = {"spearman": float(r2.statistic),
                                                "p": float(r2.pvalue), "n": LAYERS}
    out["n_layers_full_from_the_start"] = int(sum(1 for L in range(LAYERS)
                                                  if out["per_layer"][L]["full_from_the_start"]))
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["validation/carriers/*.npz"])
    (OUT2 / "B6_PERLAYER_SHARING.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsharing rises in {rises} of {LAYERS} layers; "
          f"{out['n_layers_full_from_the_start']} layers full from the start")
    print(f"  share change vs occupancy change: {out['share_change_vs_occupancy_change']}")
    print(f"  share change vs initial occupancy: {out['share_change_vs_initial_occupancy']}")
    print(f"-> {OUT2 / 'B6_PERLAYER_SHARING.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
