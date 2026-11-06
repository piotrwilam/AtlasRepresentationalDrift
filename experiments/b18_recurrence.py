r"""B18 — recurrence and coordination (design item 18).

B17 and B17b established that departed neurons come back: 57% return, lift 3.3 over a matched
control, and 92.5% of those departures were real falls rather than threshold jitter. That is
about single neurons. This asks the same question of the **set**: does a construct's carrier set
revisit configurations it has held before, and do its neurons move in coordinated patterns? If
so, §6.2's cause 4 becomes "switching among a few discrete wirings"; if not, it stays "a
continuous walk through a degenerate manifold".

Three readings, as the design fixes them.

**(a) Is the nearest set a neighbour?** Per construct, the 35x35 kept-fraction matrix. For each
checkpoint, the other checkpoint whose set it most resembles. Under B10's model -- overlap
falling monotonically with the gap -- that is always the immediate neighbour, so the share of
checkpoints whose nearest set is NOT a neighbour measures recurrence. The comparison is a
**parametric bootstrap of B10's own model**: the fitted overlap for every pair plus Gaussian
noise matched to this construct's residuals from that fit, 200 draws, giving the share the model
itself produces by noise alone.

**(b) How many configurations?** Complete-linkage clustering of the 35 sets on distance
1 - overlap, cut at one twin ceiling -- so two checkpoints share a cluster only when their sets
are as alike as two twins of the same checkpoint. One cluster means the set never leaves its own
noise; a handful means a repertoire; 35 means every checkpoint is its own configuration.

**(c) Do neurons move together?** Each neuron ever in the set has a 35-bit in/out series. The
share of neurons covered by the commonest patterns is compared with a shuffle that permutes
membership **across neurons independently at each checkpoint**, which destroys per-neuron
pattern structure while keeping every checkpoint's set size exactly. Coordination shows as real
patterns covering more neurons than the shuffle's.

Carrier index only -- no profile reads.

    python experiments/b18_recurrence.py [n_constructs]
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time

import numpy as np

from experiments.b10_fit import fit_exp
from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2

N_UNITS = 24 * 6144
TOP_PATTERNS = 20
N_DRAWS = 200


def clusters_at(dist: np.ndarray, cut: float) -> int:
    """Complete-linkage cluster count at a distance cut, without scipy's linkage overhead."""
    n = dist.shape[0]
    lab = list(range(n))
    groups = {i: [i] for i in range(n)}
    while True:
        best = None
        ks = sorted(groups)
        for a_i in range(len(ks)):
            for b_i in range(a_i + 1, len(ks)):
                a, b = ks[a_i], ks[b_i]
                d = max(dist[x, y] for x in groups[a] for y in groups[b])
                if d <= cut and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            return len(groups)
        _, a, b = best
        groups[a] = groups[a] + groups[b]
        del groups[b]


def main() -> None:
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = np.array([int(re.search(r"(\d+)$", r).group(1)) for r in refs])
    names = sorted(p1_set())
    if limit:
        names = names[:limit]
    idx = load_index(CARRIERS, refs)
    R = len(refs)
    rng = np.random.default_rng(20261023)
    print(f"{R} references, {len(names)} constructs, K={K:,}", flush=True)

    per = {}
    for c in names:
        t1 = time.time()
        key1, key2 = f"twin1/{c}|product", f"twin2/{c}|product"
        if any(key1 not in idx[r] for r in refs):
            continue
        M = np.zeros((R, N_UNITS), bool)
        for i, r in enumerate(refs):
            M[i, np.asarray(idx[r][key1], np.int64)] = True
        # the twin ceiling for this construct: median over references
        tc = []
        for r in refs:
            if key2 in idx[r]:
                a = np.asarray(idx[r][key1], np.int64)
                b = np.asarray(idx[r][key2], np.int64)
                tc.append(np.intersect1d(a, b).size / K)
        ceiling = float(np.median(tc)) if tc else None

        # ---- (a) the 35x35 kept-fraction matrix, and the nearest set ----
        Mf = M.astype(np.float32)
        inter = Mf @ Mf.T
        ov = inter / K
        np.fill_diagonal(ov, -1.0)
        nearest = np.argmax(ov, axis=1)
        non_neighbour = np.abs(nearest - np.arange(R)) > 1
        share_obs = float(non_neighbour.mean())

        # B10's model fitted on THIS construct's pairs, then the same statistic by noise alone
        iu = np.triu_indices(R, 1)
        gaps = (iters[iu[1]] - iters[iu[0]]).astype(float)
        turn = 1.0 - ov[iu]
        f = fit_exp(gaps, turn)
        share_null = None
        if f is not None:
            pred_t = f["A"] * (1 - np.exp(-f["r"] * gaps))
            resid = turn - pred_t
            # Per-GAP noise, not pooled. A single pooled sd is dominated by the wide pairs,
            # whose residuals are large, and applying it to the narrow pairs swamps the small
            # differences the nearest-set statistic turns on -- the first version reported a
            # null share of 0.755 against an observed 0.000, which measures the null's own
            # noise and nothing about the data.
            order = np.argsort(gaps)
            nb = max(4, len(gaps) // 10)
            sd_of = np.empty_like(gaps)
            for b0 in range(0, len(order), nb):
                sel = order[b0:b0 + nb]
                sd_of[sel] = float(np.std(resid[sel])) if sel.size > 2 else float(
                    np.std(resid))
            Pm = np.zeros((R, R))
            Pm[iu] = 1.0 - pred_t
            Pm = Pm + Pm.T
            SD = np.zeros((R, R))
            SD[iu] = sd_of
            SD = SD + SD.T
            shares = []
            for _ in range(N_DRAWS):
                Z = rng.normal(0.0, 1.0, size=(R, R)) * SD
                Z = np.triu(Z, 1)
                S = Pm + Z + Z.T
                np.fill_diagonal(S, -1.0)
                nn = np.argmax(S, axis=1)
                shares.append(float((np.abs(nn - np.arange(R)) > 1).mean()))
            # recurrence would also show as overlap EXCEEDING the model at wide gaps
            wide = gaps > np.median(gaps)
            share_null = {"mean": float(np.mean(shares)),
                          "ci95": [float(np.quantile(shares, .025)),
                                   float(np.quantile(shares, .975))],
                          "residual_sd_pooled": float(np.std(resid)),
                          "residual_sd_narrowest_decile": float(sd_of[order[:nb]].mean()),
                          "resid_vs_gap_corr": float(np.corrcoef(gaps, resid)[0, 1]),
                          "share_wide_pairs_above_model_by_one_twin_unit": (
                              float(np.mean(-resid[wide] > (1 - (ceiling or 1))))
                              if ceiling else None),
                          "A": f["A"], "r_per_step": f["r"]}

        # ---- (b) how many configurations within one twin ceiling ----
        np.fill_diagonal(ov, 1.0)
        nclus = clusters_at(1.0 - ov, 1.0 - ceiling) if ceiling is not None else None

        # ---- (c) in/out patterns against a per-checkpoint shuffle ----
        ever = np.flatnonzero(M.any(0))
        series = M[:, ever].T                       # [n_neurons, R] bool
        packed = np.packbits(series, axis=1)
        keys = [bytes(row) for row in packed]
        cnt = collections.Counter(keys)
        n_ever = len(keys)
        top_share = sum(v for _, v in cnt.most_common(TOP_PATTERNS)) / max(n_ever, 1)
        n_patterns = len(cnt)
        sh_top, sh_npat = [], []
        sizes = M.sum(1)
        for _ in range(20):                         # 20 shuffles: the statistic is stable
            S = np.zeros((R, n_ever), bool)
            for i in range(R):
                S[i, rng.choice(n_ever, int(min(sizes[i], n_ever)), replace=False)] = True
            pk = np.packbits(S.T, axis=1)
            cc = collections.Counter(bytes(r) for r in pk)
            sh_top.append(sum(v for _, v in cc.most_common(TOP_PATTERNS)) / max(n_ever, 1))
            sh_npat.append(len(cc))
        # A SECOND shuffle, because the design's one answers a different question. Permuting
        # membership across neurons at each checkpoint destroys each neuron's persistence as
        # well as any alignment between neurons, so beating it only shows that membership is
        # autocorrelated in time -- which the turnover curve already says. Shifting each
        # neuron's series circularly by its own random amount KEEPS its run structure and
        # breaks only the alignment across neurons, which is what "coordination" means.
        cs_top, cs_npat = [], []
        for _ in range(20):
            sh = rng.integers(0, R, n_ever)
            S2 = np.empty((n_ever, R), bool)
            for j in range(n_ever):
                S2[j] = np.roll(series[j], sh[j])
            pk = np.packbits(S2, axis=1)
            cc = collections.Counter(bytes(r) for r in pk)
            cs_top.append(sum(v for _, v in cc.most_common(TOP_PATTERNS)) / max(n_ever, 1))
            cs_npat.append(len(cc))
        per[c] = {
            "twin_ceiling": ceiling,
            "a_nearest_set": {
                "share_non_neighbour_observed": share_obs,
                "under_b10_model": share_null,
                "excess_over_model": (share_obs - share_null["mean"]
                                      if share_null else None)},
            "b_configurations": {"n_clusters_within_one_twin_ceiling": nclus,
                                 "cut_distance": (1.0 - ceiling) if ceiling else None},
            "c_patterns": {
                "n_neurons_ever_in_set": int(n_ever),
                "n_distinct_patterns": n_patterns,
                f"share_in_top_{TOP_PATTERNS}": top_share,
                "shuffled_n_distinct_patterns_mean": float(np.mean(sh_npat)),
                f"shuffled_share_in_top_{TOP_PATTERNS}_mean": float(np.mean(sh_top)),
                "coordination_vs_across_neuron_shuffle":
                    top_share / max(float(np.mean(sh_top)), 1e-12),
                "circular_shift_n_distinct_patterns_mean": float(np.mean(cs_npat)),
                f"circular_shift_share_in_top_{TOP_PATTERNS}_mean": float(np.mean(cs_top)),
                "coordination_vs_circular_shift":
                    top_share / max(float(np.mean(cs_top)), 1e-12),
                "which_control_is_which": (
                    "the across-neuron shuffle breaks persistence AND alignment, so beating it "
                    "shows only autocorrelation; the circular shift keeps each neuron's own "
                    "run structure and breaks alignment alone, so beating THAT is coordination")},
            "seconds": time.time() - t1}
        print(f"  {c:22s} non-neighbour {share_obs:.3f} vs model "
              f"{(share_null['mean'] if share_null else float('nan')):.3f}; clusters {nclus}; "
              f"top-{TOP_PATTERNS} {top_share:.3f} vs shuf {np.mean(sh_top):.3f} / shift "
              f"{np.mean(cs_top):.3f}  ({time.time()-t1:.0f}s, {(time.time()-t0)/60:.1f} min)",
              flush=True)

    def med(path):
        v = []
        for e in per.values():
            x = e
            for p in path:
                x = (x or {}).get(p) if isinstance(x, dict) else None
            if isinstance(x, (int, float)):
                v.append(x)
        return float(np.median(v)) if v else None

    out = {"what": "recurrence and coordination of the carrier set: is the nearest set a "
                   "neighbour, how many configurations within one twin ceiling, and do "
                   "neurons move in patterns",
           "reading": "recurrence and few patterns mean switching among discrete wirings; "
                      "neither means a continuous walk",
           "n_references": R, "n_constructs": len(per),
           "a_share_non_neighbour_median": med(["a_nearest_set",
                                                "share_non_neighbour_observed"]),
           # the COUNT, not only the median share: the draft quotes "167 of 169", and a
           # median of 0.0 does not let a reader check it
           "a_n_nearest_is_neighbour": sum(
               1 for v in per.values()
               if (v.get("a_nearest_set") or {}).get("share_non_neighbour_observed") == 0.0),
           "a_n_constructs_read": len(per),
           "a_n_nearest_is_neighbour_is":
               "constructs whose every checkpoint's nearest carrier set is its immediate "
               "neighbour, i.e. share_non_neighbour_observed == 0",
           "a_share_non_neighbour_under_model_median": med(["a_nearest_set", "under_b10_model",
                                                            "mean"]),
           "a_excess_median": med(["a_nearest_set", "excess_over_model"]),
           "b_n_clusters_median": med(["b_configurations",
                                       "n_clusters_within_one_twin_ceiling"]),
           "c_top_share_median": med(["c_patterns", f"share_in_top_{TOP_PATTERNS}"]),
           "c_shuffled_top_share_median": med(["c_patterns",
                                               f"shuffled_share_in_top_{TOP_PATTERNS}_mean"]),
           "c_circular_shift_top_share_median": med(
               ["c_patterns", f"circular_shift_share_in_top_{TOP_PATTERNS}_mean"]),
           "c_coordination_vs_across_neuron_median": med(
               ["c_patterns", "coordination_vs_across_neuron_shuffle"]),
           "c_coordination_vs_circular_shift_median": med(
               ["c_patterns", "coordination_vs_circular_shift"]),
           "per_construct": per,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["validation/carriers/*.npz"])}
    (OUT2 / "B18_RECURRENCE.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n(a) nearest set not a neighbour: {out['a_share_non_neighbour_median']:.3f} "
          f"observed vs {out['a_share_non_neighbour_under_model_median']:.3f} under B10's "
          f"model (excess {out['a_excess_median']:+.3f})")
    print(f"(b) configurations within one twin ceiling: median "
          f"{out['b_n_clusters_median']} of {R}")
    print(f"(c) share in the top {TOP_PATTERNS} patterns: {out['c_top_share_median']:.3f}; "
          f"across-neuron shuffle {out['c_shuffled_top_share_median']:.3f} "
          f"(x{out['c_coordination_vs_across_neuron_median']:.2f}); circular shift "
          f"{out['c_circular_shift_top_share_median']:.3f} "
          f"(x{out['c_coordination_vs_circular_shift_median']:.2f} <- coordination)")
    print(f"-> {OUT2 / 'B18_RECURRENCE.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
