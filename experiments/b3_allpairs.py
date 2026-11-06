r"""B3 all-pairs — B10's model applied to the **graded** profile.

B3 measured full-profile cosine and Spearman at two spacings only (51 pairs), and that design
confounds the start point with the gap: the x1.5 arm's gap grows from 1,000 to 105,000 steps as
the start point moves right, so the arm's falling cosine (corr -0.715 against log start) cannot
be read as the profile destabilising.

This removes the confound the way B10 did for turnover: **all pairs** of pretraining references,
then per start point t fit

    dissimilarity(t, D) = A(t) (1 - exp(-r(t) D))          [experiments.b10_fit.fit_exp]

with dissimilarity = 1 - cosine and 1 - Spearman of the full 24x6,144 profile. The per-step
figure is the slope at D -> 0; **stability** per step is its negative, so a falling loss rate
means the graded profile is settling. Intervals by construct-level bootstrap, 200 draws, the
same as B10.

Spearman is computed as Pearson on ranks, with each (construct, reference) vector ranked **once**
and reused across all 34 of its partners -- which is why all pairs cost little more than B3's 51.

Resumable per construct: a dropped connection costs one construct, not the pass.

    python experiments/b3_allpairs.py [n_constructs]       # argument times a subset
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from experiments.b10_fit import DENSE_END, fit_exp
from experiments.atlas import CARRIERS, D, provenance
from experiments.relatedness import OUT2

WORK = D / "validation/paper2/b3_allpairs"


def _pair_matrices(M):
    """(cosine, spearman) over all pairs of the rows of M, one construct's profile per reference."""
    from scipy.stats import rankdata
    X = M.astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    cos = (X / n) @ (X / n).T
    R = np.vstack([rankdata(row) for row in X])
    R -= R.mean(axis=1, keepdims=True)
    rn = np.linalg.norm(R, axis=1, keepdims=True)
    rn[rn == 0] = 1.0
    rho = (R / rn) @ (R / rn).T
    return cos, rho


def main() -> None:
    from experiments.atlas import p1_set, _tree
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = np.array([int(re.search(r"(\d+)$", r).group(1)) for r in refs])
    names = sorted(p1_set())
    if limit:
        names = names[:limit]
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"{len(refs)} references, {len(names)} constructs, "
          f"{len(refs)*(len(refs)-1)//2} pairs a construct", flush=True)

    rows = []                                   # (start, gap, 1-cos, 1-rho, construct)
    iu = np.triu_indices(len(refs), 1)
    done = 0
    for c in names:
        fp = WORK / f"{c}.npz"
        if fp.exists():
            z = np.load(fp)
            cos, rho = z["cos"], z["rho"]
        else:
            t1 = time.time()
            M = np.empty((len(refs), 24 * 6144), np.float32)
            bad = []

            def one(i_r):
                i, r = i_r
                try:
                    z = _tree(r, "twin1", c, ("concept", "A_mean", "B_mean"))
                    v = z.get("concept")
                    M[i] = np.asarray(v if v is not None else z["A_mean"] - z["B_mean"],
                                      np.float32).ravel()
                except Exception as exc:                            # noqa: BLE001
                    bad.append((r, type(exc).__name__))

            with ThreadPoolExecutor(8) as ex:
                list(ex.map(one, list(enumerate(refs))))
            if bad:
                print(f"  {c}: SKIP, {len(bad)} unreadable ({bad[0][1]})", flush=True)
                continue
            cos, rho = _pair_matrices(M)
            np.savez_compressed(fp, cos=cos, rho=rho)
            done += 1
            print(f"  {c:22s} cos median {np.median(cos[iu]):.4f}  spearman "
                  f"{np.median(rho[iu]):.4f}  ({time.time()-t1:.0f}s, "
                  f"{(time.time()-t0)/60:.1f} min, {done} new)", flush=True)
        for a, b in zip(*iu):
            rows.append((int(iters[a]), int(iters[b] - iters[a]),
                         1.0 - float(cos[a, b]), 1.0 - float(rho[a, b]), c))
    print(f"\n{len(rows):,} (start, gap, 1-cosine, 1-spearman) points", flush=True)

    rng = np.random.default_rng(20261022)
    starts = sorted({r[0] for r in rows})
    per_t = {}
    for t in starts:
        sub = [r for r in rows if r[0] == t]
        if len(sub) < 6:
            continue
        Dg = np.array([r[1] for r in sub], float)
        C = np.array([r[4] for r in sub])
        uc = np.unique(C)
        e = {"n_points": len(sub)}
        for k, col in (("cosine", 2), ("spearman", 3)):
            Y = np.array([r[col] for r in sub], float)
            f = fit_exp(Dg, Y)
            if f is None:
                continue
            boot = []
            for _ in range(200):
                pick = set(rng.choice(uc, len(uc), replace=True).tolist())
                m = np.array([x in pick for x in C])
                g = fit_exp(Dg[m], Y[m]) if m.sum() >= 4 else None
                if g:
                    boot.append(g["slope0"])
            e[k] = {"A": f["A"], "r_per_step": f["r"],
                    "loss_per_1000_steps": f["slope0"] * 1000,
                    "ci95": ([float(np.quantile(boot, .025)) * 1000,
                              float(np.quantile(boot, .975)) * 1000]
                             if len(boot) > 20 else None)}
        per_t[int(t)] = e

    out = {"what": "full-profile cosine and Spearman over all pairs of pretraining references, "
                   "fitted per start point with B10's model on (1 - similarity)",
           "model": "dissimilarity(t,D) = A(t)(1-exp(-r(t)D)); per-step loss = slope at D->0; "
                    "per-step STABILITY is its negative",
           "same_code_as": "experiments.b10_fit.fit_exp",
           "n_references": len(refs), "n_constructs_used": len({r[4] for r in rows}),
           "n_points": len(rows), "per_start": per_t,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["profiles/<ref>/twin1/<c>.npz via the dataset-only "
                                     "resolver", "validation/carriers/*.npz for the reference "
                                     "list"])}
    for k in ("cosine", "spearman"):
        ts = [t for t in sorted(per_t) if k in per_t[t]]
        lv = [per_t[t][k]["loss_per_1000_steps"] for t in ts]
        dense = [v for t, v in zip(ts, lv) if t <= DENSE_END]
        tail = [v for t, v in zip(ts, lv) if t > DENSE_END]
        rho = float(np.corrcoef(np.log(ts), lv)[0, 1]) if len(ts) > 2 else float("nan")
        # the verdict is about STABILITY, so a falling loss rate is rising stability
        verdict = ("rising" if rho < -0.3 else "falling" if rho > 0.3 else "plateau")
        out[k] = {"loss_median_dense": float(np.median(dense)) if dense else None,
                  "loss_median_tail": float(np.median(tail)) if tail else None,
                  "n_dense": len(dense), "n_tail": len(tail),
                  "loss_vs_log_start_corr": rho, "stability_verdict": verdict}
        print(f"\n{k}: per-1,000-step loss, start <= 84 B median "
              f"{out[k]['loss_median_dense']:.4f} (n={len(dense)}); > 84 B median "
              f"{out[k]['loss_median_tail']:.4f} (n={len(tail)})")
        print(f"  loss vs log start corr {rho:+.3f} -> stability {verdict} past 84 B")
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "B3_ALLPAIRS.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {OUT2 / 'B3_ALLPAIRS.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
