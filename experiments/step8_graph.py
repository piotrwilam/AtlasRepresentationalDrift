"""Step 8 -- the graph: R9, R11, the two exploratory rows, P6 and P8, from the derived tree.

Pretraining branch, product space, one artefact. Every figure comes from the **carrier index**
(the top 5% of the squared mass per construct per reference, built and verified in S3) and, for
R11's boundary band alone, from the **extended index** (the top 10% in rank order,
`experiments/extended_index.py`), which exists because a band just under the threshold cannot be
read from a set that stops at the threshold.

**Four validity statements, each of which changes a number if it is read differently.**

**1. Partner spacing is by ruling and is never "t to 1.5 t".** A38 measured carrier retention to
fall monotonically with the spacing it is read at -- Spearman **-0.813** over 1,404 pairs -- so a
rewiring figure read at a spacing the schedule happened to offer is a figure about the schedule.
The dense block is read at its true **fixed 1,000-step** partner (19 pairs); the tail is read in
A38's **four factor bands** (1.25-1.35, 1.35-1.50, 1.50-1.65, 1.65-2.00), one pair an index a band,
and every figure is reported per band. Pooling the two would mix two rulers.

**2. "Just under the threshold" is a band CC chose, and it is reported at two widths.** The primary
is the **5-10%** band -- a departing unit whose rank at the partner is inside the top 10% but
outside the top 5%, so it is still within twice the set's size. The sensitivity is the tighter
**5-6%** band. Neither is in the contract; both are reported, and the pooled retirement share
moves between them by the amount the tables show.

**3. The three outcomes are made disjoint in a stated order, and the overlap is published.** A unit
that has left construct c's set can be *both* just under c's threshold *and* inside another
construct's set. The order applied is **boundary churn, then reassignment, then retirement** --
boundary first because it is the weakest form of leaving and concerns the same construct -- and the
count of units satisfying both is reported, so the ordering's cost is visible rather than implied.

**4. Degree is over P1's 169 sets in the product space, and the independent-sets expectation is a
binomial.** Each set is 7,373 of 147,456 units, so independence gives degree ~ Binomial(169, 0.05),
mean 8.45, and a share in no set of 1.7e-4. The comparison is to that, not to a permutation of the
observed sets, which would preserve the concentration the comparison is about.

**The ablation result is the context the Settle chat named (A44).** The function of a carrier set
outlives most of its indices: the set measured at 83.9 B still delivers about 45% of its own effect
1,950 B later while sharing 13.8% of its indices. So **boundary churn is the figure that matters**
-- a unit that has slipped just under the threshold has not left the construct in any functional
sense -- and it is reported **per construct** as well as pooled.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.construct_token_loss import branches
from experiments.curve1_spacing import BANDS
from experiments.r2_curves import fixed_partner, step_of
from profilebattery.profiles.names import contract_name, tree_name
from profilebattery.paths import data_root

D = data_root()
CARRIERS = D / "_carriers_dl/validation/carriers"
EXTENDED = D / "validation/extended"
R2 = D / "validation/r2"
N_UNITS = 24 * 6144
K = 7373
DENSE_GAP = 1000
#: the boundary band, primary and sensitivity, as shares of the whole space
BANDS_BOUNDARY = {"5-10pct": 0.10, "5-6pct": 0.06}
SEED = 20261016


def recorded(branch: str = "pretrain") -> list:
    have = {p.stem for p in CARRIERS.glob("*.npz")}
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    return [(r, t) for r, t in branches(sched)[branch] if r in have]


def sets_at(ref: str, constructs, space: str = "product", twin: str = "twin1") -> dict:
    """`{construct: sorted index array}` from the carrier index."""
    out = {}
    with np.load(CARRIERS / f"{ref}.npz") as z:
        for c in constructs:
            k = f"{twin}/{c}|{space}"
            if k in z:
                out[c] = np.sort(np.asarray(z[k], np.int64))
    return out


def membership(sets: dict, constructs) -> np.ndarray:
    """A `(n_constructs, N_UNITS)` boolean matrix, the graph's incidence form."""
    m = np.zeros((len(constructs), N_UNITS), bool)
    for i, c in enumerate(constructs):
        if c in sets:
            m[i, sets[c]] = True
    return m


def degree_row(m: np.ndarray) -> np.ndarray:
    return m.sum(axis=0).astype(np.int32)


def partners(seq) -> list:
    """Every (i, j) pair the ruling allows, each labelled with its rule and spacing.

    Dense block: the true fixed 1,000-step partner. Tail: A38's four factor bands, the shortest
    partner inside each band at each index. No pair is formed by "t to 1.5 t".
    """
    refs = [r for r, _ in seq]
    toks = [t for _, t in seq]
    out = []
    for i in range(len(refs)):
        j, gap = fixed_partner(refs, i)
        if j is not None and gap == DENSE_GAP:
            out.append({"i": i, "j": j, "rule": "dense|1000_steps", "steps": int(gap),
                        "factor": float(toks[j] / toks[i])})
    for lo, hi in BANDS:
        for i in range(len(refs)):
            if any(p["i"] == i and p["rule"].startswith("dense") for p in out):
                continue
            cand = [j for j in range(i + 1, len(refs)) if lo <= toks[j] / toks[i] <= hi]
            if cand:
                j = cand[0]
                out.append({"i": i, "j": j, "rule": f"tail|{lo}-{hi}",
                            "steps": int(step_of(refs[j]) - step_of(refs[i])),
                            "factor": float(toks[j] / toks[i])})
    return out


def r9(seq, constructs) -> dict:
    """Unit degree per reference, against the independent-sets binomial."""
    from scipy.stats import binom
    n = len(constructs)
    p = K / N_UNITS
    exp = {"n_sets": n, "p": p, "mean": n * p, "variance": n * p * (1 - p),
           "share_degree_0": float(binom.pmf(0, n, p)),
           "share_degree_ge_5": float(1 - binom.cdf(4, n, p)),
           "basis": "Binomial(169, 7373/147456): each set an independent 5% draw"}
    rows = []
    for ref, tok in seq:
        d = degree_row(membership(sets_at(ref, constructs), constructs))
        hist = np.bincount(d, minlength=max(int(d.max()) + 1, 6))
        rows.append({"reference": ref, "tokens": tok,
                     "share_0": float((d == 0).mean()),
                     "share_ge_5": float((d >= 5).mean()),
                     "mean": float(d.mean()), "variance": float(d.var(ddof=1)),
                     "dispersion": float(d.var(ddof=1) / d.mean()) if d.mean() else None,
                     "max": int(d.max()), "p99": float(np.percentile(d, 99)),
                     "n_units_covered": int((d > 0).sum()),
                     "histogram_head": [int(x) for x in hist[:12]]})
    first, last = rows[0], rows[-1]
    return {"expectation_independent": exp, "per_reference": rows,
            "change_over_the_record": {
                "share_0": [first["share_0"], last["share_0"]],
                "share_ge_5": [first["share_ge_5"], last["share_ge_5"]],
                "dispersion": [first["dispersion"], last["dispersion"]],
                "mean_is_fixed_by_construction": "every set is exactly 5% of the space, so the "
                                                 "mean degree cannot move: only its spread can"}}


def extended_at(ref: str, constructs) -> dict:
    """`{construct: rank-ordered top-10% indices}` from the extended index, if it is built."""
    p = EXTENDED / f"{ref}.npz"
    if not p.exists():
        return {}
    out = {}
    with np.load(p) as z:
        for c in constructs:
            if c in z.files:
                out[c] = np.asarray(z[c], np.int64)
    return out


def r11(seq, constructs, pairs, log=print) -> dict:
    """Rewiring accounting: boundary churn, reassignment, retirement, per construct and pooled."""
    refs = [r for r, _ in seq]
    per_rule = collections.defaultdict(lambda: collections.Counter())
    per_construct = collections.defaultdict(lambda: collections.Counter())
    overlap_both = collections.Counter()
    missing_ext = set()
    for p in pairs:
        a, b = refs[p["i"]], refs[p["j"]]
        sa, sb = sets_at(a, constructs), sets_at(b, constructs)
        mb = membership(sb, constructs)
        other = mb.sum(axis=0).astype(np.int32)                 # degree at the partner
        ext = extended_at(b, constructs)
        if not ext:
            missing_ext.add(b)
        for ci, c in enumerate(constructs):
            if c not in sa or c not in sb:
                continue
            left = np.setdiff1d(sa[c], sb[c], assume_unique=True)
            if left.size == 0:
                continue
            # in another construct's set at the partner? (the unit is not in c's own, by
            # construction, so `other` counts only other constructs for these units)
            reassigned = other[left] > 0
            bands = {}
            for name, frac in BANDS_BOUNDARY.items():
                if c in ext:
                    kk = max(1, int(round(frac * N_UNITS)))
                    wide = ext[c][:kk]
                    bands[name] = np.isin(left, wide, assume_unique=False)
                else:
                    bands[name] = np.zeros(left.size, bool)
            prim = bands["5-10pct"]
            key = p["rule"]
            per_rule[key]["left"] += int(left.size)
            per_rule[key]["boundary"] += int(prim.sum())
            per_rule[key]["reassigned"] += int((~prim & reassigned).sum())
            per_rule[key]["retired"] += int((~prim & ~reassigned).sum())
            per_rule[key]["boundary_5_6"] += int(bands["5-6pct"].sum())
            per_rule[key]["boundary_and_reassigned"] += int((prim & reassigned).sum())
            per_construct[c]["left"] += int(left.size)
            per_construct[c]["boundary"] += int(prim.sum())
            per_construct[c]["reassigned"] += int((~prim & reassigned).sum())
            per_construct[c]["retired"] += int((~prim & ~reassigned).sum())
            overlap_both[key] += int((prim & reassigned).sum())
        log(f"  {p['rule']:20s} {a} -> {b}: left so far "
            f"{per_rule[p['rule']]['left']:,}", flush=True)
    # **The base rate, because 97% reassigned is mostly the graph's density.** With mean degree
    # 8.45 and nine units in ten inside at least one set, a departing unit is in *some* other
    # construct's set almost by default. So the share of units with degree >= 1 at the partner is
    # computed and reported beside the reassignment share: the reading is the difference, not the
    # level.
    base = []
    for p_ in pairs:
        mb = membership(sets_at(refs[p_["j"]], constructs), constructs)
        d = mb.sum(axis=0)
        base.append({"rule": p_["rule"], "to": refs[p_["j"]],
                     "share_units_in_any_set": float((d > 0).mean())})
    pooled = collections.Counter()
    for v in per_rule.values():
        pooled.update(v)

    def shares(cnt):
        n = cnt["left"] or 1
        return {"n_left": cnt["left"],
                "boundary_share": cnt["boundary"] / n,
                "reassigned_share": cnt["reassigned"] / n,
                "retired_share": cnt["retired"] / n,
                "boundary_share_5_6pct": cnt["boundary_5_6"] / n if "boundary_5_6" in cnt else None,
                "both_boundary_and_reassigned": cnt.get("boundary_and_reassigned", 0)}
    return {"base_rate_per_pair": base,
            "base_rate_pooled": float(np.mean([b["share_units_in_any_set"] for b in base])),
            "base_rate_is": "the share of all 147,456 units that sit in at least one carrier set "
                            "at the partner reference -- what 'reassigned' would be if leaving "
                            "one set told you nothing about the others",
            "order_applied": "boundary churn, then reassignment, then retirement",
            "boundary_band_primary": "top 5-10% of the squared mass",
            "boundary_band_sensitivity": "top 5-6%",
            "extended_index_missing_for": sorted(missing_ext),
            "pooled": shares(pooled),
            "per_rule": {k: shares(v) for k, v in per_rule.items()},
            "per_construct": {c: shares(v) for c, v in per_construct.items()}}


def entering(seq, constructs, pairs) -> dict:
    """Exploratory (i): where an entering unit comes from -- another set, or no set."""
    refs = [r for r, _ in seq]
    rows = []
    for p in pairs:
        a, b = refs[p["i"]], refs[p["j"]]
        sa, sb = sets_at(a, constructs), sets_at(b, constructs)
        ma = membership(sa, constructs)
        deg_a = ma.sum(axis=0).astype(np.int32)
        from_other = from_none = tot = 0
        for c in constructs:
            if c not in sa or c not in sb:
                continue
            came = np.setdiff1d(sb[c], sa[c], assume_unique=True)
            if came.size == 0:
                continue
            tot += came.size
            from_other += int((deg_a[came] > 0).sum())
            from_none += int((deg_a[came] == 0).sum())
        rows.append({"rule": p["rule"], "from": a, "to": b, "n_entered": tot,
                     "share_from_another_set": from_other / tot if tot else None,
                     "share_from_no_set": from_none / tot if tot else None})
    tot = sum(r["n_entered"] for r in rows)
    w = sum((r["share_from_another_set"] or 0) * r["n_entered"] for r in rows) / (tot or 1)
    bases = []
    for p in pairs:
        ma = membership(sets_at(refs[p["i"]], constructs), constructs)
        bases.append(float((ma.sum(axis=0) > 0).mean()))
    return {"per_pair": rows, "n_entered_total": tot,
            "pooled_share_from_another_set": w, "pooled_share_from_no_set": 1 - w,
            "base_rate_pooled": float(np.mean(bases)),
            "base_rate_is": "the share of units in at least one set at the SOURCE reference -- "
                            "what 'from another set' would be if entering told you nothing"}


def churn_predictors(seq, constructs, pairs, rates: dict, seed: int = SEED) -> dict:
    """Exploratory (ii) and P6: churn per unit against degree and against role frequency."""
    refs = [r for r, _ in seq]
    rng = np.random.default_rng(seed)
    logf = np.array([np.log(rates[c]) if rates.get(c) else np.nan for c in constructs])
    rows = []
    for p in pairs:
        a, b = refs[p["i"]], refs[p["j"]]
        sa, sb = sets_at(a, constructs), sets_at(b, constructs)
        ma, mb = membership(sa, constructs), membership(sb, constructs)
        deg = ma.sum(axis=0).astype(np.float64)
        left = (ma & ~mb).sum(axis=0).astype(np.float64)
        keep = deg > 0
        churn = left[keep] / deg[keep]
        dg = deg[keep]
        # the mean log role frequency of the sets the unit serves
        # **accumulated, not transposed.** `ma[good].T.astype(float64)` materialises a
        # 147,456 x 166 float64 array -- 196 MB a pair -- and the first run of this module stalled
        # on memory rather than compute with it. The same quantity is a sum over constructs.
        wsum = np.zeros(N_UNITS, np.float32)
        wcnt = np.zeros(N_UNITS, np.float32)
        for ci, c in enumerate(constructs):
            if not np.isfinite(logf[ci]):
                continue
            wsum[ma[ci]] += np.float32(logf[ci])
            wcnt[ma[ci]] += np.float32(1.0)
        wsum, wcnt = wsum[keep], wcnt[keep]
        mlf = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), np.nan)
        rows.append({"rule": p["rule"], "from": a, "to": b, "n_units": int(keep.sum()),
                     "churn": churn, "degree": dg, "mean_log_frequency": mlf})
    from scipy.stats import spearmanr
    out = {"per_pair": [], "pooled": {}}
    ch = np.concatenate([r["churn"] for r in rows])
    dg = np.concatenate([r["degree"] for r in rows])
    mlf = np.concatenate([r["mean_log_frequency"] for r in rows])
    ok = np.isfinite(mlf)
    for name, x in (("degree", dg), ("mean_log_frequency", mlf)):
        m = np.isfinite(x)
        rho, pv = spearmanr(x[m], ch[m])
        # a single-predictor least-squares fit, for the comparison of fits the row asks for
        A = np.column_stack([np.ones(m.sum()), x[m]])
        beta, *_ = np.linalg.lstsq(A, ch[m], rcond=None)
        pred = A @ beta
        ss = 1 - ((ch[m] - pred) ** 2).sum() / ((ch[m] - ch[m].mean()) ** 2).sum()
        # **the null is drawn on a 200,000-row subsample, and said so.** A Spearman permutation
        # on 2.8 million rows costs about 1.5 s a draw, so 200 draws is 10 minutes a predictor for
        # an interval that is +/-0.002 wide at that n. The subsample's interval is wider than the
        # full-sample one, which is the conservative direction.
        xs, cs_ = x[m], ch[m]
        sub = (rng.choice(xs.size, size=min(200_000, xs.size), replace=False)
               if xs.size > 200_000 else np.arange(xs.size))
        null = []
        idx = np.arange(sub.size)
        for _ in range(200):
            rng.shuffle(idx)
            null.append(spearmanr(xs[sub], cs_[sub][idx]).statistic)
        n_null = int(sub.size)
        out["pooled"][name] = {"n": int(m.sum()), "spearman": float(rho),
                               "p_scipy": float(pv), "r_squared": float(ss),
                               "slope": float(beta[1]),
                               "permutation_interval": [float(np.percentile(null, 2.5)),
                                                        float(np.percentile(null, 97.5))],
                               "permutation_p": float((np.abs(null) >= abs(rho)).mean()),
                               "permutation_n": n_null,
                               "permutation_is": "200 draws on a 200,000-row subsample; the "
                                                 "interval is therefore wider than the full "
                                                 "sample's, which is the conservative direction"}
    better = max(out["pooled"], key=lambda k: out["pooled"][k]["r_squared"])
    out["which_predicts_better"] = {
        "by_r_squared": better,
        "r_squared": {k: v["r_squared"] for k, v in out["pooled"].items()},
        "note": "single-predictor least squares on the pooled unit-pair rows; the two predictors "
                "are not nested, so this is a comparison of fits and not a test"}
    out["per_pair"] = [{k: v for k, v in r.items()
                        if k not in ("churn", "degree", "mean_log_frequency")} for r in rows]
    out["n_units_with_frequency"] = int(ok.sum())
    return out


def p8(seq, constructs) -> dict:
    """Curve 5's raw sharing per pair, and whether any pair's falls beyond its own noise."""
    refs = [r for r, _ in seq]
    pairs = list(itertools.combinations(constructs, 2))
    # **curve 5's own quantity, which is CROSS-twin.** `r2_curves.sharing` pairs `twin1/a` with
    # `twin2/b`: a pair's sharing is read across the two independently generated batteries, so
    # that a shared unit is not shared by construction of one battery. The symmetric twin-1 form
    # is reported beside it as a sensitivity, because CC computed that first and it answers a
    # different question.
    def stack(twin_b: str) -> np.ndarray:
        out = []
        for ref in refs:
            a = membership(sets_at(ref, constructs), constructs).astype(np.float32)
            b = (a if twin_b == "twin1"
                 else membership(sets_at(ref, constructs, space="product", twin="twin2"),
                                 constructs).astype(np.float32))
            out.append(a @ b.T / K)
        return np.stack(out)

    A = stack("twin2")
    iu = np.triu_indices(len(constructs), 1)
    series = A[:, iu[0], iu[1]]                                  # (refs, n_pairs)
    d = np.abs(np.diff(series, axis=0))
    noise = np.median(d, axis=0)
    change = series[-1] - series[0]
    fell = change < -noise
    return {"n_pairs": int(series.shape[1]), "n_references": len(refs),
            "quantity": "raw sharing = |S_a(twin1) n S_b(twin2)| / 7373, product space -- "
                        "curve 5's own cross-twin definition",
            "noise_unit_is": "median |point-to-point change| of that pair's own series",
            "n_fell_beyond_noise": int(fell.sum()),
            "median_change": float(np.median(change)),
            "median_noise": float(np.median(noise)),
            "n_rose_beyond_noise": int((change > noise).sum()),
            "pairs_that_fell": [f"{pairs[i][0]}|{pairs[i][1]}" for i in np.where(fell)[0][:50]],
            "outcome": ("no pair's raw sharing fell beyond its own noise; nothing to read"
                        if not fell.sum() else
                        f"{int(fell.sum())} pairs fell beyond their noise"),
            "pairs_checked": "every one of the 14,196 ordered-by-name pairs of P1, listed by "
                             "construction rather than enumerated here"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain")
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()

    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    seq = recorded(a.branch)
    cs = [c for c in p1 if c in sets_at(seq[0][0], p1)]
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {c: (({**ast, **bi}.get(contract_name(c)) or {}).get("per_file")) for c in cs}
    print(f"branch {a.branch}: {len(seq)} references, {len(cs)} constructs, "
          f"{N_UNITS:,} units, sets of {K:,}", flush=True)

    pr = partners(seq)
    by_rule = collections.Counter(p["rule"] for p in pr)
    print(f"partners by ruling: {dict(by_rule)}  (never t to 1.5t)")

    rep = {"branch": a.branch, "space": "product", "n_constructs": len(cs),
           "n_units": N_UNITS, "k": K, "partners": pr, "partners_by_rule": dict(by_rule)}

    print("\n=== R9: unit degree ===", flush=True)
    rep["r9"] = r9(seq, cs)
    e = rep["r9"]["expectation_independent"]
    f, l = rep["r9"]["per_reference"][0], rep["r9"]["per_reference"][-1]
    print(f"  independent sets would give: mean {e['mean']:.2f}, share 0 "
          f"{e['share_degree_0']:.2e}, share >=5 {e['share_degree_ge_5']:.3f}")
    print(f"  measured at {f['reference']}: mean {f['mean']:.2f}, share 0 {f['share_0']:.4f}, "
          f"share >=5 {f['share_ge_5']:.4f}, dispersion {f['dispersion']:.2f}, max {f['max']}")
    print(f"  measured at {l['reference']}: mean {l['mean']:.2f}, share 0 {l['share_0']:.4f}, "
          f"share >=5 {l['share_ge_5']:.4f}, dispersion {l['dispersion']:.2f}, max {l['max']}")

    print("\n=== R11: rewiring accounting ===", flush=True)
    rep["r11"] = r11(seq, cs, pr, log=lambda *x, **k: None)
    po = rep["r11"]["pooled"]
    print(f"  pooled over {po['n_left']:,} leaving events: boundary "
          f"{po['boundary_share']:.3f}, reassigned {po['reassigned_share']:.3f}, retired "
          f"{po['retired_share']:.3f}")
    print(f"  the tighter 5-6% band would call {po['boundary_share_5_6pct']:.3f} boundary; "
          f"{po['both_boundary_and_reassigned']:,} units are both boundary and reassigned")
    for k, v in sorted(rep["r11"]["per_rule"].items()):
        print(f"    {k:20s} n={v['n_left']:9,d}  boundary {v['boundary_share']:.3f}  "
              f"reassigned {v['reassigned_share']:.3f}  retired {v['retired_share']:.3f}")
    bc = sorted(rep["r11"]["per_construct"].items(), key=lambda x: -x[1]["boundary_share"])
    print(f"  per construct, boundary share: highest {bc[0][0]} {bc[0][1]['boundary_share']:.3f}, "
          f"lowest {bc[-1][0]} {bc[-1][1]['boundary_share']:.3f}")

    print("\n=== exploratory (i): where entering units come from ===", flush=True)
    rep["entering"] = entering(seq, cs, pr)
    en = rep["entering"]
    print(f"  {en['n_entered_total']:,} entering events: from another set "
          f"{en['pooled_share_from_another_set']:.3f}, from no set "
          f"{en['pooled_share_from_no_set']:.3f}")

    print("\n=== exploratory (ii) and P6: churn against degree and against frequency ===", flush=True)
    rep["churn"] = churn_predictors(seq, cs, pr, rates)
    for k, v in rep["churn"]["pooled"].items():
        print(f"  {k:20s} n={v['n']:,} spearman {v['spearman']:+.4f} "
              f"(perm p={v['permutation_p']:.3f}, interval "
              f"[{v['permutation_interval'][0]:+.4f},{v['permutation_interval'][1]:+.4f}])  "
              f"R2 {v['r_squared']:.4f}")
    print(f"  predicts better by R2: {rep['churn']['which_predicts_better']['by_r_squared']}")

    print("\n=== P8: curve 5's raw sharing ===", flush=True)
    rep["p8"] = p8(seq, cs)
    q = rep["p8"]
    print(f"  {q['n_pairs']:,} pairs over {q['n_references']} references: "
          f"{q['n_fell_beyond_noise']} fell beyond their noise, "
          f"{q['n_rose_beyond_noise']} rose; median change {q['median_change']:+.4f}")
    print(f"  -> {q['outcome']}")

    out = pathlib.Path(a.out) / f"STEP8_GRAPH_{a.branch}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
