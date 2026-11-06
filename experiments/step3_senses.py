"""Step 3's second and third senses, the first being empty.

**Why these two carry the question.** Part 4 asks whether a construct's six dates coincide, in
three senses. The **first** -- in time, within a construct, on P4 -- is reported **empty**: step 0b
measured `|P4| = 0`, and the plan's own clause says that below 20 "the first sense is reported as
empty and the second and third senses carry the question". A34 reads the within-construct question
exploratorily on the three profile curves instead; this module is the other two senses.

* **Second sense, across constructs, on P1.** The Spearman rank correlation between each pair of
  date columns, **censored dates tied at the last rank**, with a permutation interval. Because
  censoring is handled by tying, a construct with no date inside the record still enters -- which
  is what makes this sense readable where the first is not.
* **Third sense, in predictors.** The same regression fitted to **each date column separately**,
  and whether the coefficients agree. The norm, depth and sparsity columns are where the
  regression has events (75, 59 and 97 dated); the curve-1 column's *no verdict* is reported
  **beside** them and not in their place.

**Five columns, not six.** Curve 5 carries no dates (the ruling of 14 Oct), so the pairwise matrix
is 5 x 4 / 2 = **10 pairs**, not 15.

**Every reading is reported with and without the pilot's 30 constructs**, as the plan specifies
for step 3 onward. The 30 are derived as P0 minus the pool, not listed, so they cannot drift from
`SPLIT.json`'s own arithmetic.

**Tie handling is explicit, because it has already produced one wrong figure.** On A34, `sorted`,
`min` and `max` each broke a tie by dictionary order and two summaries disagreed by six
constructs. Here the ties are the point rather than an accident: curve 1's product-space column
has **0 dated and 159 censored**, so 159 of 169 constructs share one rank. The **effective number
of distinct ranks** is therefore reported beside every correlation, because a near-zero Spearman
against a nearly constant column is an absence of information and not a measured independence.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import pathlib

import numpy as np

from experiments.r0b_gates import curve1_dated, dated_set
from profilebattery.paths import data_root

D = data_root()
R2 = D / "validation/r2"
DRAWS = 10_000
SEED = 20261014
#: a curve-1 outcome that is not a date inside the record: all are tied at the last rank
NOT_DATED = ("censored", "transient")


def pilot_30() -> set:
    """The 30 held-out pilot constructs: P0 minus the pool, derived rather than listed."""
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    pool = set(split["first_half"]) | set(split["second_half"])
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    p0 = {c for c, v in {**ast, **bi}.items() if (v.get("occurrences") or 0) > 0}
    from profilebattery.profiles.names import tree_name
    return {tree_name(c) for c in p0 - pool}


def date_columns(branch: str, space: str = "product", curve4: str = "construct_token") -> dict:
    """`{column: {construct: tokens | None}}` -- `None` meaning no date inside the record.

    A construct absent from a column is **unreadable** on that curve and is not in the dict at
    all; a construct present with `None` is **censored or transient**, which is a date beyond the
    record and ranks last. The two are different and are never merged: an unreadable curve says
    nothing about when the construct settled, a censored one says it had not settled by 2.03 T
    tokens.
    """
    rep2 = json.loads((R2 / "R2_REPORT.json").read_text())
    idxc = json.loads((R2 / "R2_INDEX_CURVES.json").read_text())
    c4 = json.loads((R2 / "R2_CURVE4.json").read_text())
    ctl = json.loads((D / "validation/loss_offset0/CONSTRUCT_TOKEN_LOSS.json").read_text())
    cols: dict[str, dict] = {}

    blk = (idxc["curve1"].get(f"{branch}|{space}|ratio") or {}).get("dates") or {}
    cols["curve1_turnover"] = {
        c: d.get("tokens") if (d.get("label") or d["outcome"]) == "dated" else None
        for c, d in blk.items() if (d.get("label") or d["outcome"]) in ("dated",) + NOT_DATED}

    for key, name in (("curve2_norm", "curve2_norm"), ("curve3_depth", "curve3_depth"),
                      ("curve6_sparsity", "curve6_sparsity")):
        b = (rep2["curves"].get(f"{branch}|{key}") or {}).get("dates") or {}
        cols[name] = {}
        for c, d in b.items():
            k = d.get("flat") or d.get("label") or d["outcome"]
            if k == "dated":
                cols[name][c] = d.get("tokens")
            elif k == "censored":
                cols[name][c] = None

    if curve4 == "construct_token":
        col = {}
        for c, e in (ctl.get("curves") or {}).items():
            b = (e.get("branch") or {}).get(branch)
            if not b:
                continue
            d = b.get("date") or {}
            k = d.get("label") or d.get("outcome")
            if k == "dated":
                col[c] = d.get("tokens")
            elif k in ("censored",):
                col[c] = None
        cols["curve4_loss"] = col
    else:
        col = {}
        for cls in ("obliged", "governed", "scope"):
            blk4 = (c4.get("classes") or {}).get(f"{branch}|{cls}") or {}
            for key, d in (blk4.get("series") or {}).items():
                c = key.split("|")[0]
                k = d.get("flat") or d.get("label") or d["outcome"]
                if k == "dated":
                    col[c] = min(d.get("tokens") or np.inf, col.get(c) or np.inf)
                elif k == "censored":
                    col.setdefault(c, None)
        cols["curve4_loss"] = {c: (None if v in (None, np.inf) else v) for c, v in col.items()}
    return cols


def ranks_tied_last(col: dict, members) -> np.ndarray:
    """Ranks over `members`, censored (`None`) tied at the last rank, as the plan requires.

    Average ranks within ties, so the correlation is the standard tie-corrected Spearman and a
    block of 159 censored constructs does not bias it by the order they happen to be listed in.
    """
    vals = [col[c] for c in members]
    dated = [v for v in vals if v is not None]
    order = {v: i for i, v in enumerate(sorted(set(dated)))}
    # dated constructs rank by token count; every censored one takes the next rank up
    raw = np.array([order[v] if v is not None else len(order) for v in vals], float)
    out = np.empty_like(raw)
    for v in np.unique(raw):
        m = raw == v
        out[m] = np.flatnonzero(np.sort(raw) == v).mean()
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def pair(colA: dict, colB: dict, members, draws: int = DRAWS, seed: int = SEED) -> dict:
    """One pair's tie-corrected Spearman with a permutation interval and its rank resolution."""
    both = [c for c in members if c in colA and c in colB]
    if len(both) < 10:
        return {"n": len(both), "spearman": None, "why": "fewer than ten constructs in both"}
    ra, rb = ranks_tied_last(colA, both), ranks_tied_last(colB, both)
    na, nb = int(len(np.unique(ra))), int(len(np.unique(rb)))
    base = {"n": len(both), "distinct_ranks_a": na, "distinct_ranks_b": nb,
            "n_censored_a": sum(1 for c in both if colA[c] is None),
            "n_censored_b": sum(1 for c in both if colB[c] is None)}
    # **A column with one rank is not independent of anything -- it is empty of information.**
    # Curve 1's product-space column is every construct censored, so all 169 share one rank and
    # the Spearman denominator is zero. Reporting that as rho = 0 would read as "no relation
    # measured"; it is "no relation measurable", and the two must not look alike.
    if na < 2 or nb < 2:
        side = "a" if na < 2 else "b"
        return {**base, "spearman": None, "permutation_p": None,
                "why": f"column {side} has a single rank ({na if side == 'a' else nb}): every "
                       f"construct is censored or they all date together, so the correlation is "
                       f"undefined rather than zero"}
    rho = spearman(ra, rb)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(rng.permutation(ra), rb) for _ in range(draws)])
    return {**base, "spearman": rho,
            "permutation_interval": [float(np.quantile(null, 0.025)),
                                     float(np.quantile(null, 0.975))],
            "permutation_p": float((np.abs(null) >= abs(rho)).mean())}


def second_sense(branch: str, space: str, curve4: str, members, draws: int = DRAWS) -> dict:
    """The pairwise matrix on `members`, with each column summarised **on `members` too**.

    The summary is restricted to `members` and not taken over the whole column, because the
    without-pilot variants otherwise carry the with-pilot counts beside correlations computed on
    151 constructs -- a label that does not describe the reading next to it. (It did: the first
    run reported 75 dated on norm in both variants while the correlations differed.)
    """
    cols = date_columns(branch, space, curve4)
    keep = set(members)
    out = {"space": space, "curve4": curve4, "n_members": len(keep),
           "columns": {k: {"n": sum(1 for c in v if c in keep),
                           "n_dated": sum(1 for c, x in v.items()
                                          if c in keep and x is not None),
                           "n_censored": sum(1 for c, x in v.items()
                                             if c in keep and x is None)}
                       for k, v in cols.items()},
           "pairs": {}}
    for a, b in itertools.combinations(sorted(cols), 2):
        out["pairs"][f"{a} x {b}"] = pair(cols[a], cols[b], members, draws)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--aft-draws", type=int, default=200,
                    help="bootstrap and permutation draws for the regression; each is a full "
                         "likelihood fit, so this is not the 10,000 the Spearman uses")
    ap.add_argument("--senses", default="both", choices=("both", "second", "third"),
                    help="running one sense MERGES into the existing artefact rather than "
                         "overwriting it, so a cheap re-run of the Spearman does not discard "
                         "hours of regression fits")
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()

    from profilebattery.profiles.names import tree_name
    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    pilot = pilot_30()
    sets = {"with_pilot": p1, "without_pilot": [c for c in p1 if c not in pilot]}
    print(f"P1 = {len(p1)};  pilot's 30 derived = {len(pilot)};  "
          f"P1 without them = {len(sets['without_pilot'])}")

    rep = {"branch": a.branch, "first_sense": {
        "status": "EMPTY", "why": "|P4| = 0, and part 4's clause reports the first sense empty "
                                  "below 20; A34 reads it exploratorily on the three profile "
                                  "curves instead"},
        "pilot_30": sorted(pilot), "second_sense": {}}

    out = pathlib.Path(a.out) / f"STEP3_SENSES_{a.branch}.json"
    if a.senses != "both" and out.exists():
        rep = json.loads(out.read_text()) | rep
        rep["second_sense"] = {} if a.senses == "second" else rep.get("second_sense", {})
        print(f"merging the {a.senses} sense into {out.name}; the other sense is kept as it "
              f"stands in the artefact")

    for which, members in (sets.items() if a.senses in ("both", "second") else ()):
        for space in ("product", "layer"):
            for c4 in ("construct_token", "hashed"):
                key = f"{which}|{space}|{c4}"
                r = second_sense(a.branch, space, c4, members, a.draws)
                rep["second_sense"][key] = r
                sig = [(k, v) for k, v in r["pairs"].items()
                       if v.get("permutation_p") is not None and v["permutation_p"] < 0.05]
                nd = sum(1 for v in r["pairs"].values() if v.get("spearman") is None)
                print(f"\n{key}")
                for k, v in sorted(r["pairs"].items()):
                    if v.get("spearman") is None:
                        print(f"   {k:40s} n={v['n']:3d}  ranks "
                              f"{v.get('distinct_ranks_a','?')}/{v.get('distinct_ranks_b','?')}"
                              f"  UNDEFINED: {v['why'][:60]}")
                        continue
                    star = "*" if v["permutation_p"] < 0.05 else " "
                    print(f"   {k:40s} n={v['n']:3d}  rho {v['spearman']:+.3f}{star} "
                          f"p={v['permutation_p']:.4f}  ranks {v['distinct_ranks_a']}/"
                          f"{v['distinct_ranks_b']}  censored {v['n_censored_a']}/"
                          f"{v['n_censored_b']}")
                print(f"   -> {len(sig)} of {len(r['pairs'])} pairs outside their interval; "
                      f"{nd} undefined")

    # ---- third sense: the regression, per date column ----
    schedule = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    from experiments.construct_token_loss import branches
    toks = sorted(t for _, t in branches(schedule)[a.branch])
    if a.senses in ("both", "third"):
        rep["third_sense"] = {}
    for which, members in (sets.items() if a.senses in ("both", "third") else ()):
        pred = predictors(members)
        print(f"\n=== third sense, {which}: {len(pred['rows'])} of {len(members)} have "
              f"predictors; frame levels {pred['frame_levels']}, pooled "
              f"{pred['pooled_frame_levels']} ===")
        cols = date_columns(a.branch, "product", "construct_token")
        for name, col in sorted(cols.items()):
            for dist in ("lognormal", "weibull"):
                r = aft_column(col, members, pred, toks, draws=a.aft_draws, dist=dist)
                rep["third_sense"][f"{which}|{name}|{dist}"] = r
                if r.get("fit") is None:
                    print(f"  {name:18s} {dist:9s} n={r['n']:3d} events={r['n_events']:3d}  "
                          f"NO FIT: {r['why'][:70]}")
                    continue
                f = r["coefficients"].get("log_frequency", {})
                star = "*" if f.get("excludes_zero") else " "
                print(f"  {name:18s} {dist:9s} n={r['n']:3d} events={r['n_events']:3d}  "
                      f"log_freq {f.get('beta', float('nan')):+.3f}{star} "
                      f"[{f.get('lo', float('nan')):+.3f},{f.get('hi', float('nan')):+.3f}] "
                      f"perm p={r['permutation_p_log_frequency']}  "
                      f"composite {'in' if r['composite_entered'] else 'OUT'} "
                      f"({r['n_composite']})  conv={r['fit']['converged']}")

    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {out}")



# ----------------------------------------------------------------------------------------------
# Third sense: the predictor regression, fitted to each date column
# ----------------------------------------------------------------------------------------------

def predictors(members) -> dict:
    """`{construct: {log_frequency, composite, frame_class}}`, the model's right-hand side.

    `log_frequency` is `log(per_file)` from the two rate tables, the only rate field both the 63
    AST constructs and the 143 builtins carry -- `per_ktoken` exists for the AST side alone.
    `composite` is the plan's composite indicator, true where `inventory_v2` records components.
    `frame_class` is the covariate; levels with fewer than five members are pooled into `other`
    and the pooling is reported.
    """
    ast = json.loads((D / "calibration/construct_rates.json").read_text())
    bi = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    rates = {**ast, **bi}
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]
    from profilebattery.profiles.names import contract_name
    out = {}
    for c in members:
        base = contract_name(c)
        r, e = rates.get(base), inv.get(base)
        if not r or not e or not (r.get("per_file") or 0) > 0:
            continue
        out[c] = {"log_frequency": float(np.log(r["per_file"])),
                  "composite": bool(e.get("components")),
                  "frame_class": e.get("frame_class") or "unknown"}
    levels = collections.Counter(v["frame_class"] for v in out.values())
    pooled = {k for k, n in levels.items() if n < 5}
    for v in out.values():
        if v["frame_class"] in pooled:
            v["frame_class"] = "other"
    return {"rows": out, "pooled_frame_levels": sorted(pooled),
            "frame_levels": sorted({v["frame_class"] for v in out.values()})}


def design(members, pred: dict, use_composite: bool) -> tuple:
    """`(X, names)` -- intercept, log frequency, optionally the composite, and frame dummies."""
    rows = pred["rows"]
    levels = [l for l in pred["frame_levels"] if l != "other"]
    names = ["intercept", "log_frequency"] + (["composite"] if use_composite else []) + \
            [f"frame:{l}" for l in levels]
    X = []
    for c in members:
        p = rows[c]
        r = [1.0, p["log_frequency"]] + ([1.0 if p["composite"] else 0.0] if use_composite else [])
        r += [1.0 if p["frame_class"] == l else 0.0 for l in levels]
        X.append(r)
    return np.array(X, float), names


def aft_fit(X: np.ndarray, lo: np.ndarray, hi: np.ndarray, dist: str = "lognormal") -> dict:
    """Interval-censored accelerated-failure-time fit by direct likelihood, `scipy` only.

    `lifelines` and `statsmodels` are not installed, so the likelihood is written out: for each
    construct the date lies in `(lo, hi]` in **log tokens**, with `hi = inf` for a right-censored
    one. The contribution is `Phi((hi-mu)/s) - Phi((lo-mu)/s)` for the log-normal and the
    corresponding difference of Weibull survivals for the check, with `mu = X @ beta`.

    An interval of `(lo, hi]` and not a point is the whole reason this is interval-censored: a
    date is a **checkpoint**, so the true settling time lies between that checkpoint and the one
    before it, and treating it as exact would understate every interval.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    k = X.shape[1]

    def nll(theta):
        beta, ls = theta[:k], theta[k]
        s = np.exp(ls)
        mu = X @ beta
        if dist == "lognormal":
            a = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0)
            b = norm.cdf((lo - mu) / s)
        else:                                    # Weibull on log time: Gumbel-min
            z_hi = np.where(np.isfinite(hi), (hi - mu) / s, np.inf)
            z_lo = (lo - mu) / s
            a = np.where(np.isfinite(hi), 1 - np.exp(-np.exp(z_hi)), 1.0)
            b = 1 - np.exp(-np.exp(z_lo))
        p = np.clip(a - b, 1e-12, None)
        return -np.log(p).sum()

    t0 = np.zeros(k + 1)
    t0[0] = np.nanmedian(lo[np.isfinite(lo)]) if np.any(np.isfinite(lo)) else 0.0
    res = minimize(nll, t0, method="Nelder-Mead",
                   options={"maxiter": 20000, "fatol": 1e-8, "xatol": 1e-8})
    return {"converged": bool(res.success), "nll": float(res.fun),
            "beta": [float(x) for x in res.x[:k]], "log_scale": float(res.x[k]),
            "dist": dist, "n": int(X.shape[0])}


def aft_column(col: dict, members, pred: dict, toks_sorted, draws: int = 1000,
               seed: int = SEED, dist: str = "lognormal") -> dict:
    """One date column's regression, with a core-free bootstrap and a permutation null.

    The bootstrap resamples **constructs**, which is what an across-construct regression's
    uncertainty is over; the core bootstrap belongs to a single construct's curve and has already
    been spent making the date. The permutation null shuffles `log_frequency` against the dates.
    """
    rows = pred["rows"]
    use = [c for c in members if c in col and c in rows]
    ev = [c for c in use if col[c] is not None]
    if len(ev) < 10:
        return {"n": len(use), "n_events": len(ev), "fit": None,
                "why": "fewer than ten events: a regression on this column would be an interval "
                       "and not an estimate"}
    comp = sum(1 for c in use if rows[c]["composite"])
    use_comp = comp >= 10
    X, names = design(use, pred, use_comp)
    # interval bounds in log tokens: a date at checkpoint t means (previous checkpoint, t]
    prev = {t: (toks_sorted[i - 1] if i else t / 1.5) for i, t in enumerate(toks_sorted)}
    lo = np.array([np.log(prev[col[c]]) if col[c] is not None else np.log(toks_sorted[-1])
                   for c in use])
    hi = np.array([np.log(col[c]) if col[c] is not None else np.inf for c in use])
    fit = aft_fit(X, lo, hi, dist)
    rng = np.random.default_rng(seed)
    bidx = [rng.integers(0, len(use), len(use)) for _ in range(draws)]
    boots = []
    for idx in bidx[:draws]:
        try:
            boots.append(aft_fit(X[idx], lo[idx], hi[idx], dist)["beta"])
        except Exception:                                    # noqa: BLE001
            continue
    B = np.array(boots) if boots else np.zeros((0, len(names)))
    ci = {}
    for i, nm in enumerate(names):
        if B.size:
            ci[nm] = {"beta": fit["beta"][i],
                      "lo": float(np.quantile(B[:, i], 0.025)),
                      "hi": float(np.quantile(B[:, i], 0.975))}
            ci[nm]["excludes_zero"] = bool(ci[nm]["lo"] > 0 or ci[nm]["hi"] < 0)
    jf = names.index("log_frequency")
    null = []
    for _ in range(draws):
        Xp = X.copy()
        Xp[:, jf] = rng.permutation(Xp[:, jf])
        try:
            null.append(aft_fit(Xp, lo, hi, dist)["beta"][jf])
        except Exception:                                    # noqa: BLE001
            continue
    p = (float(np.mean(np.abs(null) >= abs(fit["beta"][jf]))) if null else None)
    return {"n": len(use), "n_events": len(ev), "n_composite": comp,
            "composite_entered": use_comp, "names": names, "fit": fit,
            "coefficients": ci, "permutation_p_log_frequency": p,
            "n_bootstrap": int(B.shape[0]), "n_permutations": len(null)}


if __name__ == "__main__":
    main()
