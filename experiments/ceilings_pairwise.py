r"""The ceiling computed on the PAIR, not on one checkpoint.

`CEILINGS.json` took chance within a single checkpoint: the median similarity between different
constructs both measured at the start point. But the excess it bounds is computed on a PAIR --
construct c at reference i against the same construct at reference j -- so the matching chance
level is the similarity between DIFFERENT constructs with one at i and one at j. Two things
could move it: constructs drift apart over training, and the profiles at two checkpoints are
themselves less alike than two profiles at one.

Both versions are computed and reported side by side, per start point and per gap window:

    ceiling(t, w) = twin agreement(t) - chance(t, w)

with `chance(t, w)` the median over the pairs in that window of the between-construct similarity
across the pair's two references. Local only.

    python experiments/ceilings_pairwise.py
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
WINDOWS = (("~1,000", 900, 1_100), ("5k-10k", 5_000, 10_000),
           ("20k-30k", 20_000, 30_000), ("50k-100k", 50_000, 100_000))
N_SAMPLE = 169          # A1.7: all of P1, not the 16-construct sample


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
    sample = (names if N_SAMPLE >= len(names)
              else [names[i] for i in rng.choice(len(names), N_SAMPLE, replace=False)])
    R = len(refs)

    # cache the sample's unit-normalised profiles and rank profiles, and their band sets
    V, Rk, S = {}, {}, {}
    for i, r in enumerate(refs):
        for c in sample:
            v = concept_of(r, "twin1", c).astype(np.float64)
            V[(i, c)] = v / np.linalg.norm(v)
            rk = rankdata(v)
            rk -= rk.mean()
            Rk[(i, c)] = rk / np.linalg.norm(rk)
        for c in sample:
            kk = f"twin1/{c}|product"
            if kk in idx[r]:
                S[(i, c)] = np.sort(np.asarray(idx[r][kk], np.int64))
    # the band index sets, rebuilt from the concept store: B14's cache keeps kept fractions
    # and ceilings, not the sets themselves
    from experiments.b14_bands import band_sets
    BS, BT = {}, {}
    for i, r in enumerate(refs):
        for c in sample:
            BS[(i, c)] = band_sets(concept_of(r, "twin1", c).astype(np.float64))
    for i, r in enumerate(refs):
        for b in BANDS:
            o = []
            for c in names:
                f = B14 / f"{c}.npz"
                if f.exists():
                    v = np.load(f)[f"ceil|{b}"][i]
                    if np.isfinite(v):
                        o.append(float(v))
            BT[(i, b)] = float(np.median(o)) if o else None
    print(f"sample of {len(sample)} constructs cached ({(time.time()-t0)/60:.1f} min)",
          flush=True)

    IND = {}

    def _ind(i, band):
        """(n_sample, N_UNITS) float32 indicator rows, cached for two references at a time."""
        key = (i, band)
        if key not in IND:
            M = np.zeros((len(sample), N_UNITS), np.float32)
            for r_, c_ in enumerate(sample):
                v = S[(i, c_)] if band == "product" else BS[(i, c_)][band]
                if v is not None and len(v):
                    M[r_, v] = 1.0
            IND[key] = M
            while len(IND) > 14:          # 2 references x 7 set kinds
                IND.pop(next(iter(IND)))
        return IND[key]

    def cross(i, j):
        """Between-construct chance with one profile at i and one at j.

        Vectorised: at 169 constructs the pairwise loops are 28,392 dot products of 147,456
        each, which as one matrix product is milliseconds and as a list comprehension is not.
        """
        A = np.stack([V[(i, a)] for a in sample])
        B = np.stack([V[(j, b)] for b in sample])
        M = A @ B.T
        off = ~np.eye(len(sample), dtype=bool)
        cc = M[off]
        A = np.stack([Rk[(i, a)] for a in sample])
        B = np.stack([Rk[(j, b)] for b in sample])
        rr = (A @ B.T)[off]
        # the set overlaps as ONE matrix product, not 28,392 intersect1d calls: at 169
        # constructs the list comprehension did not finish, and |A n B| for indicator rows is
        # exactly their inner product
        Ia = _ind(i, "product")
        Ib = _ind(j, "product")
        ovm = (Ia @ Ib.T) / K
        ov = ovm[off]
        bb = {}
        for b in BANDS:
            Ja, Jb = _ind(i, b), _ind(j, b)
            sz = Ja.sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(sz > 0, (Ja @ Jb.T) / np.maximum(sz, 1), np.nan)
            v = m[off]
            v = v[np.isfinite(v)]
            bb[b] = float(np.median(v)) if v.size else None
        return float(np.median(cc)), float(np.median(rr)), float(np.median(ov)), bb

    # twin agreement at each reference (the ceiling's first term, unchanged)
    tw = {}
    for i, r in enumerate(refs):
        o = []
        for c in names:
            a, b = f"twin1/{c}|product", f"twin2/{c}|product"
            if a in idx[r] and b in idx[r]:
                o.append(np.intersect1d(np.asarray(idx[r][a], np.int64),
                                        np.asarray(idx[r][b], np.int64)).size / K)
        tc, ts = [], []
        for c in sample:
            a = concept_of(r, "twin1", c).astype(np.float64)
            b = concept_of(r, "twin2", c).astype(np.float64)
            tc.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            ra, rb = rankdata(a), rankdata(b)
            ra -= ra.mean()
            rb -= rb.mean()
            ts.append(float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb))))
        tw[i] = {"carrier5": float(np.median(o)), "cosine": float(np.median(tc)),
                 "spearman": float(np.median(ts))}

    out = {"what": "the excess's ceiling with chance measured ACROSS the pair's two "
                   "references, beside the within-checkpoint version",
           "why": "the excess is computed on a pair, so the matching chance level is between "
                  "different constructs with one profile at each end of the pair",
           "n_sample_constructs": N_SAMPLE, "per_window": {}}
    for lab, lo, hi in WINDOWS:
        rows = []
        for i in range(R):
            js = [j for j in range(i + 1, R) if lo <= iters[j] - iters[i] <= hi]
            if not js:
                continue
            cs, rs, os_, bs = [], [], [], {b: [] for b in BANDS}
            for j in js:
                a, b_, c, bb = cross(i, j)
                cs.append(a)
                rs.append(b_)
                os_.append(c)
                for k2 in BANDS:
                    if bb[k2] is not None:
                        bs[k2].append(bb[k2])
            within = cross(i, i)
            rows.append({
                "start_iter": iters[i], "tokens_b": iters[i] * 4_194_304 / 1e9,
                "n_pairs": len(js),
                "cosine": {"twin": tw[i]["cosine"],
                           "chance_cross": float(np.median(cs)),
                           "chance_within": within[0],
                           "ceiling_cross": tw[i]["cosine"] - float(np.median(cs)),
                           "ceiling_within": tw[i]["cosine"] - within[0]},
                "spearman": {"twin": tw[i]["spearman"],
                             "chance_cross": float(np.median(rs)),
                             "chance_within": within[1],
                             "ceiling_cross": tw[i]["spearman"] - float(np.median(rs)),
                             "ceiling_within": tw[i]["spearman"] - within[1]},
                "carrier5": {"twin": tw[i]["carrier5"],
                             "chance_cross": float(np.median(os_)),
                             "chance_within": within[2],
                             "ceiling_cross": tw[i]["carrier5"] - float(np.median(os_)),
                             "ceiling_within": tw[i]["carrier5"] - within[2]},
                **{b: {"twin": BT[(i, b)],
                       "chance_cross": (float(np.median(bs[b])) if bs[b] else None),
                       "chance_within": within[3][b],
                       "ceiling_cross": ((BT[(i, b)] - float(np.median(bs[b])))
                                         if (bs[b] and BT[(i, b)] is not None) else None),
                       "ceiling_within": ((BT[(i, b)] - within[3][b])
                                          if (within[3][b] is not None
                                              and BT[(i, b)] is not None) else None)}
                   for b in BANDS}})
        out["per_window"][lab] = rows
        print(f"  window {lab}: {len(rows)} start points "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # does the cross version differ enough to matter?
    diffs = {}
    for m in ("carrier5", "cosine", "spearman") + BANDS:
        d = [r[m]["ceiling_cross"] - r[m]["ceiling_within"]
             for w in out["per_window"].values() for r in w
             if r[m]["ceiling_cross"] is not None and r[m]["ceiling_within"] is not None]
        _ = [r[m]["ceiling_cross"] - r[m]["ceiling_within"]
             for w in out["per_window"].values() for r in w
             if r[m]["ceiling_cross"] is not None and r[m]["ceiling_within"] is not None]
        diffs[m] = {"median_difference": float(np.median(d)),
                    "max_abs_difference": float(np.max(np.abs(d))),
                    "cross_is_lower_in": int(sum(1 for x in d if x < 0)), "n": len(d)}
    out["cross_minus_within"] = diffs
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["validation/carriers/*.npz", "the local concept store"])
    (OUT2 / "CEILINGS_PAIRWISE_169.json").write_text(json.dumps(out, indent=1, default=float))
    print("\ncross-checkpoint ceiling minus within-checkpoint:")
    for m, v in diffs.items():
        print(f"  {m:9s} median {v['median_difference']:+.4f}  max |diff| "
              f"{v['max_abs_difference']:.4f}  cross lower in {v['cross_is_lower_in']}/{v['n']}")
    print(f"-> {OUT2 / 'CEILINGS_PAIRWISE_169.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
