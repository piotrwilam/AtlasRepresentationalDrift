r"""B3 — the graded stability reading.

The reviewer's objection to "identity never settles" is that turnover is an artefact of the 5%
cut. This answers it on the graded profile: the rank correlation and the cosine of the **full**
profile between the same checkpoint pairs the turnover curve uses, at 1,000-step and x1.5
spacing, pooled over constructs.

**Load-bearing, not a robustness check.** If the full profile plateaus while the thresholded set
does not, the headline becomes "the graded profile settles, the thresholded set does not".

Resumable per pair: a dropped connection costs one pair, not the pass. A line per pair as it
lands, so progress is visible without waiting for the end.

    python experiments/b3_graded.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from experiments.atlas import CARRIERS, D, provenance
from experiments.relatedness import OUT2

WORK = D / "validation/paper2/b3"


def spacings(iters):
    """(i, j) pairs at a 1,000-step gap and at the nearest x1.5 ratio."""
    out = {"1000_steps": [], "x1.5": []}
    for i, a in enumerate(iters):
        for j in range(i + 1, len(iters)):
            if iters[j] - a == 1000:
                out["1000_steps"].append((i, j))
                break
        target = a * 1.5
        best = min(range(i + 1, len(iters)), key=lambda j: abs(iters[j] - target),
                   default=None)
        if best is not None and abs(iters[best] - target) / target < 0.2:
            out["x1.5"].append((i, best))
    return out


def main() -> None:
    from scipy.stats import spearmanr
    from experiments.atlas import p1_set, _tree
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    pairs = spacings(iters)
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"{len(refs)} references; {len(pairs['1000_steps'])} pairs at 1,000 steps, "
          f"{len(pairs['x1.5'])} at x1.5", flush=True)
    cache = {}

    def entries(ref):
        """One reference's 169 profiles, loaded with eight threads.

        The read is **latency-bound, not bandwidth-bound**: a profile npz is 99 MB of which
        `concept` is 0.6 MB, and the time per file is ~3.2 s whatever the block size, because it
        is round-trips to the Hub. Eight workers bring the effective cost to 0.57 s a file;
        sixteen is worse (0.94 s), past the server's concurrency sweet spot. Single-threaded
        this pass is 14.3 h; at eight it is 2.3 h.
        """
        if ref in cache:
            return cache[ref]
        m = np.empty((len(names), 24 * 6144), np.float32)

        def one(i_c):
            i, c = i_c
            z = _tree(ref, "twin1", c, ("concept", "A_mean", "B_mean"))
            v = z.get("concept")
            m[i] = np.asarray(v if v is not None else z["A_mean"] - z["B_mean"],
                              np.float32).ravel()

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, list(enumerate(names))))
        # keep at most two references: consecutive pairs share an endpoint, so a 2-slot cache
        # halves the loading, and two entry matrices are ~200 MB -- well inside the 17 GB this
        # machine has, which is the constraint A52's thrashing taught.
        while len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[ref] = m
        return m

    out = {}
    # order the pairs so consecutive ones share an endpoint, which is what the 2-slot cache needs
    for lab, ps in pairs.items():
        for (i, j) in sorted(ps):
            a, b = refs[i], refs[j]
            fp = WORK / f"{lab}__{a}__{b}.json"
            if fp.exists():
                out[f"{lab}|{a}|{b}"] = json.loads(fp.read_text())
                continue
            t1 = time.time()
            try:
                Ea = entries(a).copy()
                Eb = entries(b)
            except Exception as exc:                                   # noqa: BLE001
                print(f"  {lab} {a}->{b}: SKIP {type(exc).__name__}", flush=True)
                continue
            cos, rho = [], []
            for ci in range(len(names)):
                x, y = Ea[ci].astype(np.float64), Eb[ci].astype(np.float64)
                nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                if nx == 0 or ny == 0:
                    continue
                cos.append(float(x @ y / (nx * ny)))
                rho.append(float(spearmanr(x, y).statistic))
            e = {"pair": [a, b], "spacing": lab, "n_constructs": len(cos),
                 "cosine_median": float(np.median(cos)),
                 "cosine_q1": float(np.quantile(cos, .25)),
                 "cosine_q3": float(np.quantile(cos, .75)),
                 "spearman_median": float(np.median(rho)),
                 "start_iter": iters[i], "gap_steps": iters[j] - iters[i],
                 "seconds": time.time() - t1}
            fp.write_text(json.dumps(e, indent=1, default=float))
            out[f"{lab}|{a}|{b}"] = e
            print(f"  {lab:11s} {a[-7:]}->{b[-7:]}  cosine {e['cosine_median']:.4f}  "
                  f"spearman {e['spearman_median']:.4f}  ({e['seconds']:.0f}s, "
                  f"{(time.time()-t0)/60:.0f} min)", flush=True)
    doc = {"what": "rank correlation and cosine of the full profile between the checkpoint "
                   "pairs the turnover curve uses, at two spacings",
           "why": "whether 'identity never settles' holds beyond the 5% cut",
           "pairs": out, "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["profiles/<ref>/twin1/<c>.npz via the dataset-only "
                                     "resolver"])}
    (OUT2 / "B3_GRADED.json").write_text(json.dumps(doc, indent=1, default=float))
    for lab in pairs:
        v = [e for e in out.values() if e["spacing"] == lab]
        if v:
            print(f"\n{lab}: {len(v)} pairs, cosine median "
                  f"{np.median([e['cosine_median'] for e in v]):.4f}, spearman median "
                  f"{np.median([e['spearman_median'] for e in v]):.4f}")
    print(f"-> {OUT2 / 'B3_GRADED.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
