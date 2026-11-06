r"""The rate readings, redone fit-free (and a guarded refit beside them).

B10's `A(t)(1 - e^(-r(t)D))` does not fit: R^2 is **negative at every start point**, -175 at
iteration 1,000, and the statistic it reported is the slope at D -> 0, which is exactly where
its linearised estimator -- least squares on `-log(1 - y/A) = rD`, weighted by `d^2` -- has no
leverage. The observed turnover at a true 1,000-step pair is 0.6908 at the first checkpoint
against the 0.0107 the fit reported, a factor of 65.

**(1) The primary reading is fit-free.** Within four gap windows -- about 1,000 steps, 5,000 to
10,000, 20,000 to 30,000, and 50,000 to 100,000 -- each pair's change is taken **minus the
instrument's own**: the twin turnover at the EARLIER reference, `1 - twin overlap`, which is
what the same measurement gives when nothing has changed but the battery draw. The excess is the
part of the change that is not the instrument. Median over P1 with a construct bootstrap.

**(2) The secondary reading is a refit with that floor built in**: `turnover(t, D) = c(t) +
A(t)(1 - e^(-r(t)D))` with `c(t)` FIXED at the measured twin turnover, fitted on the original
scale by nonlinear least squares. **R^2 is reported beside every rate and a rate is reported
only where R^2 > 0.8.**

Both definitions are written to the artefact BEFORE anything is computed.

Every input is local: `b14/<c>.npz` (the six bands' kept matrices and twin ceilings),
`b3_allpairs/<c>.npz` (the graded cosine and Spearman matrices), the carrier index (the 5% set
and its twin overlap) and the concept store (the graded profile's twin agreement).

    python experiments/rate_readings.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, D, K, provenance
from experiments.relatedness import OUT2

B14 = D / "validation/paper2/b14"
B3 = D / "validation/paper2/b3_allpairs"
WINDOWS = (("~1,000", 900, 1_100), ("5k-10k", 5_000, 10_000),
           ("20k-30k", 20_000, 30_000), ("50k-100k", 50_000, 100_000))
BANDS = ("q1pct", "q2pct", "q5pct", "abs1sd", "abs2sd", "abs3sd")
R2_FLOOR = 0.8


def main() -> None:
    from scipy.optimize import curve_fit
    from experiments.concept_store import concept_of
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = np.array([int(re.search(r"(\d+)$", r).group(1)) for r in refs])
    names = sorted(p1_set())
    R = len(refs)

    doc = {
        "what": "the per-step rate readings, fit-free as primary and a floor-anchored refit as "
                "secondary",
        "why": "B10's single exponential has NEGATIVE R^2 at every start point (-175 at "
               "iteration 1,000) and its reported statistic is the slope at D->0, the region "
               "its linearised estimator ignores; observed 1,000-step turnover at the first "
               "checkpoint is 0.6908 against the 0.0107 it reported",
        "definition_1_primary": {
            "name": "excess over the instrument, fit-free",
            "quantity": "for a pair (i, j): the construct's change between i and j MINUS the "
                        "twin disagreement at the earlier reference i",
            "twin_floor": {"5pct_set": "1 - |twin1 n twin2| / K at reference i",
                           "bands": "1 - the band's own twin ceiling at reference i",
                           "graded": "1 - cosine(twin1, twin2) at reference i, and the "
                                     "Spearman analogue"},
            "windows_steps": [[lo, hi] for _, lo, hi in WINDOWS],
            "pooling": "median over P1's constructs; 1,000-draw construct bootstrap",
            "no_model": True},
        "definition_2_secondary": {
            "name": "floor-anchored refit",
            "model": "turnover(t, D) = c(t) + A(t)(1 - exp(-r(t) D))",
            "c_fixed_at": "the measured twin turnover at t, not fitted",
            "estimator": "nonlinear least squares on the ORIGINAL scale (scipy curve_fit), not "
                         "the linearised log form B10 used",
            "reporting_rule": f"R^2 beside every rate; a rate is reported ONLY where "
                              f"R^2 > {R2_FLOOR}"},
        "fixed_before_computing": True,
    }
    (OUT2 / "RATE_READINGS.json").write_text(json.dumps(doc, indent=1, default=float))
    print("definitions written before computing", flush=True)

    # ---- the per-construct matrices, all local -------------------------------------------
    idx = load_index(CARRIERS, refs)
    iu = np.triu_indices(R, 1)
    gap = (iters[iu[1]] - iters[iu[0]]).astype(float)
    start = iters[iu[0]].astype(float)

    series = {}                       # measure -> (n_constructs, n_pairs) change matrix
    floors = {}                       # measure -> (n_constructs, R) twin floor per reference
    ok = [c for c in names if (B14 / f"{c}.npz").exists() and (B3 / f"{c}.npz").exists()]
    print(f"{len(ok)} constructs with both caches", flush=True)

    chg = {k: [] for k in list(BANDS) + ["cosine", "spearman", "carrier5"]}
    flo = {k: [] for k in chg}
    for n, c in enumerate(ok, 1):
        z14 = np.load(B14 / f"{c}.npz")
        for b in BANDS:
            kept = z14[f"kept|{b}"]
            chg[b].append(1.0 - kept[iu])
            flo[b].append(1.0 - z14[f"ceil|{b}"])
        z3 = np.load(B3 / f"{c}.npz")
        chg["cosine"].append(1.0 - z3["cos"][iu])
        chg["spearman"].append(1.0 - z3["rho"][iu])
        # the graded twin floor: 1 - cosine(twin1, twin2) at each reference, from the store
        tc, ts = [], []
        from scipy.stats import rankdata
        for r in refs:
            a = concept_of(r, "twin1", c).astype(np.float64)
            b2 = concept_of(r, "twin2", c).astype(np.float64)
            tc.append(1.0 - float(a @ b2 / (np.linalg.norm(a) * np.linalg.norm(b2))))
            ra, rb = rankdata(a), rankdata(b2)
            ra -= ra.mean()
            rb -= rb.mean()
            ts.append(1.0 - float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb))))
        flo["cosine"].append(np.array(tc))
        flo["spearman"].append(np.array(ts))
        # the recorded 5% carrier set and its own twin overlap
        kp = np.full((R, R), np.nan)
        tw = np.full(R, np.nan)
        s1 = {r: np.asarray(idx[r][f"twin1/{c}|product"], np.int64) for r in refs
              if f"twin1/{c}|product" in idx[r]}
        for i, r in enumerate(refs):
            if r not in s1:
                continue
            k2 = f"twin2/{c}|product"
            if k2 in idx[r]:
                tw[i] = 1.0 - np.intersect1d(s1[r], np.asarray(idx[r][k2],
                                                               np.int64)).size / K
            for j in range(i + 1, R):
                if refs[j] in s1:
                    kp[i, j] = np.intersect1d(s1[r], s1[refs[j]]).size / K
        chg["carrier5"].append(1.0 - kp[iu])
        flo["carrier5"].append(tw)
        if n % 40 == 0:
            print(f"  {n}/{len(ok)} constructs ({(time.time()-t0)/60:.1f} min)", flush=True)
    for k in chg:
        series[k] = np.vstack(chg[k])
        floors[k] = np.vstack(flo[k])
    print(f"matrices built ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- (1) the fit-free reading --------------------------------------------------------
    rng = np.random.default_rng(20261023)
    headroom = {}
    b14doc = json.loads((OUT2 / "B14_BANDS.json").read_text())["per_band"]
    for b in BANDS:
        e = b14doc[b]
        headroom[b] = e["twin_ceiling_last_ref"] - e["median_chance_kept_fraction"]

    primary = {}
    for meas, M in series.items():
        F = floors[meas]
        # excess = change at (i,j) minus the twin floor at i, per construct
        fl = F[:, iu[0]]
        E = M - fl
        if meas in BANDS:
            E = E / headroom[meas]
        out = {}
        for lab, lo, hi in WINDOWS:
            w = (gap >= lo) & (gap <= hi)
            if not w.any():
                continue
            rows = []
            for t in sorted(set(start[w].tolist())):
                m = w & (start == t)
                if m.sum() == 0:
                    continue
                vals = E[:, m]
                per = np.nanmedian(vals, axis=1)        # one number a construct
                good = per[np.isfinite(per)]
                if good.size < 8:
                    continue
                boot = [np.median(good[rng.integers(0, good.size, good.size)])
                        for _ in range(1000)]
                rows.append({"start_iter": int(t), "n_pairs": int(m.sum()),
                             "n_constructs": int(good.size),
                             "median_excess": float(np.median(good)),
                             "ci95": [float(np.quantile(boot, .025)),
                                      float(np.quantile(boot, .975))],
                             "median_twin_floor": float(np.nanmedian(fl[:, m][:, 0]))})
            out[lab] = rows
        primary[meas] = out

    # ---- (2) the guarded refit -----------------------------------------------------------
    def model(Dg, A, r, c):
        return c + A * (1.0 - np.exp(-r * Dg))

    secondary = {}
    for meas, M in series.items():
        F = floors[meas]
        rows = {}
        for ti in range(R - 1):
            m = start == iters[ti]
            if m.sum() < 6 or np.unique(gap[m]).size < 4:
                continue
            y = np.nanmedian(M[:, m], axis=0)
            g = gap[m]
            c0 = float(np.nanmedian(F[:, ti]))
            if not np.isfinite(c0):
                continue
            try:
                p, _ = curve_fit(lambda d, A, r: model(d, A, r, c0), g, y,
                                 p0=[max(y.max() - c0, 1e-3), 1e-5],
                                 bounds=([1e-6, 1e-9], [2.0, 1e-2]), maxfev=20000)
            except Exception:                                       # noqa: BLE001
                continue
            pred = model(g, p[0], p[1], c0)
            ss = 1.0 - float(((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
            rows[int(iters[ti])] = {
                "A": float(p[0]), "r_per_step": float(p[1]), "c_fixed": c0,
                "r2": ss, "n_gaps": int(np.unique(g).size),
                "per_1000_steps": (float(p[0] * p[1] * 1000) if ss > R2_FLOOR else None),
                "withheld_because_r2_below_floor": ss <= R2_FLOOR}
        secondary[meas] = rows

    doc["primary"] = primary
    doc["secondary"] = secondary
    doc["secondary_summary"] = {
        m: {"n_start_points": len(v),
            "n_with_r2_above_floor": sum(1 for e in v.values() if e["r2"] > R2_FLOOR),
            "median_r2": float(np.median([e["r2"] for e in v.values()])) if v else None}
        for m, v in secondary.items()}
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["validation/paper2/b14/<c>.npz",
                                    "validation/paper2/b3_allpairs/<c>.npz",
                                    "validation/carriers/*.npz",
                                    "the local concept store"])
    (OUT2 / "RATE_READINGS.json").write_text(json.dumps(doc, indent=1, default=float))
    print("\n(1) fit-free excess over the instrument, median over P1:")
    for meas in ("carrier5", "cosine", "spearman", "q1pct", "q5pct", "abs3sd"):
        for lab, _, _ in WINDOWS:
            rows = primary[meas].get(lab) or []
            if not rows:
                continue
            f, l = rows[0], rows[-1]
            print(f"  {meas:9s} {lab:9s} n_start={len(rows):2d}  "
                  f"{f['median_excess']:+.4f} [{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}] "
                  f"@{f['start_iter']:>6d}  ->  {l['median_excess']:+.4f} "
                  f"[{l['ci95'][0]:+.4f},{l['ci95'][1]:+.4f}] @{l['start_iter']:>6d}")
    print("\n(2) the guarded refit:")
    for m, s in doc["secondary_summary"].items():
        print(f"  {m:9s} R^2 median {s['median_r2']:+.3f}; rates reported at "
              f"{s['n_with_r2_above_floor']} of {s['n_start_points']} start points")
    print(f"-> {OUT2 / 'RATE_READINGS.json'}  ({doc['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
