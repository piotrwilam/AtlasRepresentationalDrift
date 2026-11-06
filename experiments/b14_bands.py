r"""B14 — stability by band (design item 4).

§9 of the draft concedes that the carrier set is one cut of a graded profile at 5% and that a
third of its turnover is neurons crossing that line by a small margin. B3 answered the objection
at the other extreme, the whole profile with no cut at all. This answers it in between: does
churn fall as the band tightens, and **does any band settle where the 5% cut does not**?

Two band definitions, as the design fixes them:

  (a) **quantile** -- the top k of |entry| for k = 1%, 2%, 5% (1,474 / 2,949 / 7,373 of 147,456),
      a set of fixed size whose threshold moves;
  (b) **absolute** -- |entry| >= 1, 2, 3 sigma_0, a set of fixed threshold whose size moves. The
      stored `concept` is already yardstick-standardised (entries in sigma_0 units), so this is
      a direct threshold. Sizes are NOT constant and they grow over training for some
      constructs (`Try`: 17,056 -> 26,232 at >= 1 sigma_0), which is why the membership count
      per checkpoint and the size-matched chance level are reported beside every kept fraction.

For each band: the kept fraction under B10's all-pairs spacing model (the same `fit_exp`), the
band's own twin ceiling at every reference, a size-matched chance level per pair, and the
corrected settling rule. Plus the rank stability of entries by entry size, pooled over
constructs, at 1,000-step and x1.5 spacing.

One pass over the full profiles, twin1 and twin2 -- the carrier index holds indices only, at 5%,
so neither band definition can be cut from it. Resumable per construct.

    python experiments/b14_bands.py [n_constructs]
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

WORK = D / "validation/paper2/b14"
N_UNITS = 24 * 6144
QUANTILES = (0.01, 0.02, 0.05)
SIGMAS = (1.0, 2.0, 3.0)
BANDS = tuple(f"q{int(q*100)}pct" for q in QUANTILES) + tuple(f"abs{int(s)}sd" for s in SIGMAS)
THETA = 0.70                      # the settling threshold, as the corrected rule fixes it


def band_sets(v):
    """The six bands' index sets for one construct at one reference, from |entry|."""
    a = np.abs(v)
    order = np.argsort(-a)
    out = {}
    for q in QUANTILES:
        k = int(round(q * a.size))
        out[f"q{int(q*100)}pct"] = np.sort(order[:k])
    for s in SIGMAS:
        out[f"abs{int(s)}sd"] = np.flatnonzero(a >= s)
    return out


def rank_stability(v1, v2, sets1):
    """Spearman of entries between two references, within each band, plus over all units.

    "Rank stability against entry size" means: do the large entries hold their order better than
    the small ones? Computed inside each band rather than on binned deciles, so it is the same
    population every other number here is reported on.
    """
    from scipy.stats import rankdata
    out = {}
    for b, idx in sets1.items():
        if idx.size < 8:
            out[b] = None
            continue
        x, y = v1[idx].astype(np.float64), v2[idx].astype(np.float64)
        rx, ry = rankdata(x), rankdata(y)
        rx -= rx.mean()
        ry -= ry.mean()
        d = np.linalg.norm(rx) * np.linalg.norm(ry)
        out[b] = float(rx @ ry / d) if d else None
    return out


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
    R = len(refs)
    iu = np.triu_indices(R, 1)
    #: the two partner spacings the settling rule needs agreement between
    def partner(i, ratio):
        want = iters[i] * ratio
        j = int(np.argmin(np.abs(iters[i + 1:] - want))) + i + 1 if i + 1 < R else None
        return j if j is not None and abs(iters[j] - want) / want < 0.25 else None
    print(f"{R} references, {len(names)} constructs, {len(iu[0])} pairs a construct; "
          f"bands {BANDS}", flush=True)

    rows = []          # (band, start, gap, 1-kept, chance_kept, construct)
    ceil = {b: {r: [] for r in refs} for b in BANDS}
    sizes = {b: {r: [] for r in refs} for b in BANDS}
    rankst = {b: {"1000_steps": [], "x1.5": []} for b in BANDS}
    settle = {b: [] for b in BANDS}
    done = 0
    for c in names:
        fp = WORK / f"{c}.npz"
        if not fp.exists():
            t1 = time.time()
            V = {}
            bad = []

            def one(a):
                i, tw = a
                try:
                    z = _tree(refs[i], tw, c, ("concept",))
                    V[(i, tw)] = np.asarray(z["concept"], np.float32).ravel()
                except Exception as exc:                            # noqa: BLE001
                    bad.append(type(exc).__name__)

            with ThreadPoolExecutor(8) as ex:
                list(ex.map(one, [(i, tw) for i in range(R) for tw in ("twin1", "twin2")]))
            if bad:
                print(f"  {c}: SKIP, {len(bad)} unreadable ({bad[0]})", flush=True)
                continue
            S1 = [band_sets(V[(i, "twin1")]) for i in range(R)]
            S2 = [band_sets(V[(i, "twin2")]) for i in range(R)]
            kept = {b: np.full((R, R), np.nan, np.float64) for b in BANDS}
            chance = {b: np.full((R, R), np.nan, np.float64) for b in BANDS}
            tc = {b: np.full(R, np.nan) for b in BANDS}
            sz = {b: np.zeros(R, np.int64) for b in BANDS}
            for b in BANDS:
                for i in range(R):
                    a = S1[i][b]
                    sz[b][i] = a.size
                    o = S2[i][b]
                    if a.size and o.size:
                        tc[b][i] = np.intersect1d(a, o).size / a.size
                    for j in range(i + 1, R):
                        d2 = S1[j][b]
                        if a.size and d2.size:
                            kept[b][i, j] = np.intersect1d(a, d2).size / a.size
                            # two independent sets of these sizes: the kept fraction is the
                            # partner's density, which is what makes the absolute bands
                            # readable at all when their size moves
                            chance[b][i, j] = d2.size / N_UNITS
            rk = {b: {"1000_steps": [], "x1.5": []} for b in BANDS}
            for i in range(R):
                for lab, ratio in (("1000_steps", None), ("x1.5", 1.5)):
                    j = (i + 1 if ratio is None and i + 1 < R
                         and iters[i + 1] - iters[i] == 1000 else
                         (partner(i, ratio) if ratio else None))
                    if j is None:
                        continue
                    for b, val in rank_stability(V[(i, "twin1")], V[(j, "twin1")],
                                                 S1[i]).items():
                        if val is not None:
                            rk[b][lab].append(val)
            np.savez_compressed(
                fp, refs=np.array(refs),
                **{f"kept|{b}": kept[b] for b in BANDS},
                **{f"chance|{b}": chance[b] for b in BANDS},
                **{f"ceil|{b}": tc[b] for b in BANDS},
                **{f"size|{b}": sz[b] for b in BANDS},
                **{f"rank|{b}|{lab}": np.array(rk[b][lab]) for b in BANDS
                   for lab in ("1000_steps", "x1.5")})
            done += 1
            print(f"  {c:22s} 5% kept(med) {np.nanmedian(kept['q5pct'][iu]):.3f}  "
                  f"1% {np.nanmedian(kept['q1pct'][iu]):.3f}  3sd "
                  f"{np.nanmedian(kept['abs3sd'][iu]):.3f}  ({time.time()-t1:.0f}s, "
                  f"{(time.time()-t0)/60:.1f} min, {done} new)", flush=True)
        z = np.load(fp, allow_pickle=False)
        for b in BANDS:
            kp, ch = z[f"kept|{b}"], z[f"chance|{b}"]
            for a, bb in zip(*iu):
                if np.isfinite(kp[a, bb]):
                    rows.append((b, int(iters[a]), int(iters[bb] - iters[a]),
                                 1.0 - float(kp[a, bb]), float(ch[a, bb]), c))
            cl, sv = z[f"ceil|{b}"], z[f"size|{b}"]
            for i, r in enumerate(refs):
                if np.isfinite(cl[i]):
                    ceil[b][r].append(float(cl[i]))
                sizes[b][r].append(int(sv[i]))
            for lab in ("1000_steps", "x1.5"):
                rankst[b][lab].extend(z[f"rank|{b}|{lab}"].tolist())
            # the corrected settling rule, per band: an observed crossing of THETA with at
            # least one checkpoint below before, two after, and agreement between the two
            # partner spacings. Read on the kept fraction at each spacing.
            crossed = []
            for lab, ratio in (("x1.25", 1.25), ("x1.5", 1.5)):
                series = []
                for i in range(R):
                    j = partner(i, ratio)
                    if j is not None and np.isfinite(kp[i, j]):
                        series.append((int(iters[i]), float(kp[i, j])))
                date = None
                for n, (it_, v) in enumerate(series):
                    if v >= THETA and n >= 1 and series[n - 1][1] < THETA \
                            and all(w >= THETA for _, w in series[n + 1:n + 3]) \
                            and len(series) - n - 1 >= 2:
                        date = it_
                        break
                crossed.append(date)
            settle[b].append({"construct": c,
                              "date": (crossed[0] if crossed[0] is not None
                                       and crossed[1] is not None else None),
                              "per_spacing": crossed})
    print(f"\n{len(rows):,} (band, start, gap, turnover, chance) points", flush=True)

    rng = np.random.default_rng(20261022)
    out = {"what": "carrier-set stability by band: three quantile cuts and three absolute "
                   "sigma_0 thresholds, under B10's all-pairs spacing model",
           "same_code_as": "experiments.b10_fit.fit_exp",
           "bands": {"quantile": list(QUANTILES), "absolute_sigma0": list(SIGMAS)},
           "n_references": R, "n_constructs_used": len({r[5] for r in rows}),
           "settling_threshold": THETA, "per_band": {}}
    for b in BANDS:
        sub = [r for r in rows if r[0] == b]
        if not sub:
            continue
        Dg = np.array([r[2] for r in sub], float)
        Y = np.array([r[3] for r in sub], float)
        C = np.array([r[5] for r in sub])
        T = np.array([r[1] for r in sub])
        per_t = {}
        for t in sorted(set(T.tolist())):
            m = T == t
            if m.sum() < 6 or np.unique(Dg[m]).size < 4:
                continue
            f = fit_exp(Dg[m], Y[m])
            if f is None:
                continue
            uc = np.unique(C[m])
            boot = []
            for _ in range(200):
                pick = set(rng.choice(uc, len(uc), replace=True).tolist())
                mm = m & np.array([x in pick for x in C])
                g = fit_exp(Dg[mm], Y[mm]) if mm.sum() >= 4 else None
                if g:
                    boot.append(g["slope0"])
            per_t[int(t)] = {"n_points": int(m.sum()),
                             "per_1000_steps": f["slope0"] * 1000,
                             "ci95": ([float(np.quantile(boot, .025)) * 1000,
                                       float(np.quantile(boot, .975)) * 1000]
                                      if len(boot) > 20 else None)}
        ts = sorted(per_t)
        lv = [per_t[t]["per_1000_steps"] for t in ts]
        dense = [v for t, v in zip(ts, lv) if t <= DENSE_END]
        tail = [v for t, v in zip(ts, lv) if t > DENSE_END]
        rho = float(np.corrcoef(np.log(ts), lv)[0, 1]) if len(ts) > 2 else float("nan")
        dates = [e["date"] for e in settle[b] if e["date"] is not None]
        last = refs[-1]
        out["per_band"][b] = {
            "n_identifiable_start_points": len(ts),
            "per_start": per_t,
            "median_per_1000_steps_dense": float(np.median(dense)) if dense else None,
            "median_per_1000_steps_tail": float(np.median(tail)) if tail else None,
            "rate_vs_log_start_corr": rho,
            "median_kept_fraction_all_pairs": float(1 - np.median(Y)),
            "median_chance_kept_fraction": float(np.median([r[4] for r in sub])),
            "twin_ceiling_first_ref": (float(np.median(ceil[b][refs[0]]))
                                       if ceil[b][refs[0]] else None),
            "twin_ceiling_last_ref": (float(np.median(ceil[b][last]))
                                      if ceil[b][last] else None),
            "band_size_first_ref": int(np.median(sizes[b][refs[0]])),
            "band_size_last_ref": int(np.median(sizes[b][last])),
            "rank_stability_median_1000_steps": (float(np.median(rankst[b]["1000_steps"]))
                                                 if rankst[b]["1000_steps"] else None),
            "rank_stability_median_x1.5": (float(np.median(rankst[b]["x1.5"]))
                                           if rankst[b]["x1.5"] else None),
            "n_settled": len(dates), "n_constructs_tested": len(settle[b]),
            "settled_constructs": sorted({e["construct"] for e in settle[b]
                                          if e["date"] is not None})[:20]}
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["profiles/<ref>/twin{1,2}/<c>.npz via the dataset-only "
                                    "resolver", "validation/carriers/*.npz for the reference "
                                    "list"])
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "B14_BANDS.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{'band':10s} {'size':>7s} {'kept':>6s} {'chance':>7s} {'ceiling':>7s} "
          f"{'dense':>7s} {'tail':>7s} {'corr':>6s} {'rankS':>6s} {'settled':>8s}")
    for b, e in out["per_band"].items():
        print(f"{b:10s} {e['band_size_last_ref']:7d} "
              f"{e['median_kept_fraction_all_pairs']:6.3f} "
              f"{e['median_chance_kept_fraction']:7.4f} "
              f"{(e['twin_ceiling_last_ref'] or 0):7.4f} "
              f"{(e['median_per_1000_steps_dense'] or 0):7.4f} "
              f"{(e['median_per_1000_steps_tail'] or 0):7.4f} "
              f"{e['rate_vs_log_start_corr']:+6.3f} "
              f"{(e['rank_stability_median_1000_steps'] or 0):6.3f} "
              f"{e['n_settled']:4d}/{e['n_constructs_tested']:3d}")
    print(f"-> {OUT2 / 'B14_BANDS.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
