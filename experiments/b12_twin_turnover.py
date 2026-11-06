r"""B12 — the twin turnover row for paper 2 §3.1.

Carrier-set turnover **between the twins at one checkpoint**, beside turnover between
checkpoints. The point of the row: twin agreement is 0.74 while the late kept-fraction is 0.48,
so a reader can see how much of the between-checkpoint change is larger than the battery's own
surface noise and how much is not.

    python experiments/b12_twin_turnover.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2


def main() -> None:
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    # the untrained reference goes in the SAME artefact, because §3.1 quotes 0.84 and the
    # untrained figure in one sentence and should read them from one file. It is not a
    # checkpoint, so it stays out of `refs` and out of the between-checkpoint pair.
    untrained = [p.stem for p in CARRIERS.glob("untrained.npz")]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs + untrained)
    rows = {}
    for r in refs + untrained:
        d = idx.get(r, {})
        tw = []
        for c in names:
            a, b = f"twin1/{c}|product", f"twin2/{c}|product"
            if a in d and b in d:
                ov = np.intersect1d(np.asarray(d[a], np.int64),
                                    np.asarray(d[b], np.int64)).size / K
                tw.append(ov)
        if tw:
            rows[r] = {"n": len(tw), "twin_overlap_median": float(np.median(tw)),
                       "twin_overlap_q1": float(np.quantile(tw, 0.25)),
                       "twin_overlap_q3": float(np.quantile(tw, 0.75)),
                       "twin_turnover_median": float(1 - np.median(tw)),
                       "trained": r != "untrained"}
    # between checkpoints: the last neighbour pair, for the comparison the section makes
    last_two = refs[-2], refs[-1]
    keep = []
    for c in names:
        k1 = f"twin1/{c}|product"
        if k1 in idx[last_two[0]] and k1 in idx[last_two[1]]:
            keep.append(np.intersect1d(np.asarray(idx[last_two[0]][k1], np.int64),
                                       np.asarray(idx[last_two[1]][k1], np.int64)).size / K)
    # ...and at EVERY neighbouring pair, not only the last: section 3.1 quotes the fraction
    # rising "from 0.31 to 0.66 by about 71 billion tokens" and section 4.1 its fall to the
    # end, which one pair cannot support. Dated at both ends, because the draft quotes one
    # figure by where its pair ENDS ("by 71 billion") and another by where it STARTS
    # ("at 1.5 trillion").
    tokens = {}
    try:
        _cei = json.loads((OUT2 / "CEILINGS.json").read_text())
        for _r, _b in _cei.get("per_reference", {}).items():
            if isinstance(_b, dict) and _b.get("tokens_b") is not None:
                tokens[_r] = _b["tokens_b"]
    except Exception:
        pass
    per_pair = []
    for a, b in zip(refs, refs[1:]):
        kp = [np.intersect1d(np.asarray(idx[a][f"twin1/{c}|product"], np.int64),
                             np.asarray(idx[b][f"twin1/{c}|product"], np.int64)).size / K
              for c in names
              if f"twin1/{c}|product" in idx.get(a, {})
              and f"twin1/{c}|product" in idx.get(b, {})]
        if kp:
            per_pair.append({"from": a, "to": b, "n": len(kp),
                             "from_tokens_b": tokens.get(a), "to_tokens_b": tokens.get(b),
                             "kept_fraction_median": float(np.median(kp)),
                             "turnover_median": float(1 - np.median(kp))})

    out = {"what": "carrier-set turnover between the twins at one checkpoint, beside turnover "
                   "between checkpoints",
           "per_reference": rows,
           "between_checkpoints": {"pair": list(last_two), "n": len(keep),
                                   "kept_fraction_median": float(np.median(keep)),
                                   "turnover_median": float(1 - np.median(keep)),
                                   "per_pair": per_pair,
                                   "per_pair_is":
                                       "the fraction of carriers kept between every pair of "
                                       "neighbouring pretraining checkpoints, median over "
                                       "constructs, twin1, with both ends' token counts",
                                   "the_gap_is_not_fixed":
                                       "neighbouring-checkpoint gaps grow from 1,000 to "
                                       "91,000 steps over the record, so this series mixes "
                                       "the point in training with the gap"},
           "untrained_reference": ({"n": rows["untrained"]["n"],
                                    "twin_overlap_median": rows["untrained"][
                                        "twin_overlap_median"],
                                    "q1": rows["untrained"]["twin_overlap_q1"],
                                    "q3": rows["untrained"]["twin_overlap_q3"],
                                    "two_independent_5pct_sets_would_give": 0.05}
                                   if "untrained" in rows else None),
           "reading": "the twins differ by their own surface noise at a single checkpoint; "
                      "between-checkpoint change beyond that is what the turnover curve adds",
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["validation/carriers/*.npz"])}
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "B12_TWIN_TURNOVER.json").write_text(json.dumps(out, indent=1, default=float))
    last = rows[refs[-1]]
    print(f"B12: twin overlap at the last checkpoint {last['twin_overlap_median']:.4f} "
          f"(turnover {last['twin_turnover_median']:.4f})")
    print(f"     between the last two checkpoints: kept "
          f"{out['between_checkpoints']['kept_fraction_median']:.4f}")
    print(f"-> {OUT2 / 'B12_TWIN_TURNOVER.json'}")


if __name__ == "__main__":
    main()
