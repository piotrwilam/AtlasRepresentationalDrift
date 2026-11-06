r"""Co-occurrence within a function, for every pair of P1's constructs.

Amendment of 21 Oct, fixed before any departure data is read: replace nu's companion rates --
which cover 42 constructs and 90 of P1's 14,196 pairs -- with a measure computed from the
reference corpus for all of them.

  unit            one **function** of a corpus file (`ast.FunctionDef` / `AsyncFunctionDef`),
                  not a file: two constructs in one 2,000-line module are not neighbours.
  observed        the share of functions holding both j and k.
  expected        the product of their marginal shares over the same functions, which is what
                  independence would give.
  ratio           observed / expected, reported graded.
  related         the ratio exceeds 2 **and at least 20 functions hold both**. Below 20 the
                  pair is *not measured*, not 'unrelated': a ratio resting on a handful of
                  functions is noise, and counting it as evidence either way would let the
                  rarest pairs drive the result. Pairs below support are counted separately.

Both the graded value and the boolean go into the artefact, with the count of functions holding
each construct, so a ratio resting on a handful of functions can be told from one resting on
thousands.

    python experiments/cooccurrence.py --shards 1 18 36
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import time

import numpy as np

from experiments.atlas import D, provenance
from experiments.relatedness import MIN_COOC_SUPPORT, OUT2

#: the node classes a construct name maps to; builtins are matched by name use instead
def construct_of(node) -> str:
    return type(node).__name__


def functions(text: str):
    """Every function body in one file, as a set of the construct names inside it."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = set()
        for n in ast.walk(fn):
            names.add(construct_of(n))
            if isinstance(n, ast.Name):
                names.add(n.id)               # builtins are read by the name used
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        yield names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="files per shard, 0 for all")
    args = ap.parse_args()
    from profilebattery.calibration.corpus import CALIBRATION_SHARDS, fetch_shard, read
    from profilebattery.profiles.names import contract_name, tree_name
    from experiments.atlas import p1_set
    names = sorted(p1_set())
    want = {contract_name(c): c for c in names}
    shards = args.shards if args.shards is not None else list(CALIBRATION_SHARDS)
    have = collections.Counter()
    pair = collections.Counter()
    n_fn, t0 = 0, time.time()
    for s in shards:
        p = fetch_shard(s)
        for i, doc in enumerate(read(p, limit=args.limit or None)):
            for fnames in functions(doc.content):
                present = sorted({want[x] for x in fnames if x in want})
                if not present:
                    continue
                n_fn += 1
                for a in present:
                    have[a] += 1
                for ii, a in enumerate(present):
                    for b in present[ii + 1:]:
                        pair[(a, b)] += 1
        print(f"  shard {s}: {n_fn:,} functions with at least one construct "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    rows, ratios = {}, []
    for ii, a in enumerate(names):
        for b in names[ii + 1:]:
            na, nb = have.get(a, 0), have.get(b, 0)
            if na == 0 or nb == 0 or n_fn == 0:
                continue
            obs = pair.get((a, b), 0) / n_fn
            exp = (na / n_fn) * (nb / n_fn)
            r = obs / exp if exp > 0 else None
            if r is None:
                continue
            rows[f"{a}|{b}"] = {"functions_with_both": pair.get((a, b), 0),
                                "functions_with_a": na, "functions_with_b": nb,
                                "observed": obs, "expected": exp, "ratio": r,
                                "support": pair.get((a, b), 0),
                                "measured": bool(pair.get((a, b), 0) >= MIN_COOC_SUPPORT),
                                "related": bool(pair.get((a, b), 0) >= MIN_COOC_SUPPORT
                                                and r > 2.0)}
            ratios.append(r)
    OUT2.mkdir(parents=True, exist_ok=True)
    doc = {"what": "co-occurrence within one function, observed against independence, for every "
                   "pair of P1's constructs",
           "unit": "one function of a corpus file (FunctionDef / AsyncFunctionDef)",
           "related_when": f"ratio > 2 and at least {MIN_COOC_SUPPORT} functions hold both",
           "min_support": MIN_COOC_SUPPORT, "shards": shards,
           "functions_with_at_least_one_construct": n_fn,
           "pairs_with_data": len(rows), "pairs_total": len(names) * (len(names) - 1) // 2,
           "coverage": len(rows) / (len(names) * (len(names) - 1) / 2),
           "median_ratio": float(np.median(ratios)) if ratios else None,
           "share_related": float(np.mean([r > 2.0 for r in ratios])) if ratios else None,
           "rows": rows,
           "provenance": provenance(["codeparrot-clean train shards "
                                     + ",".join(map(str, shards))])}
    (OUT2 / "COOCCURRENCE.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"\n{n_fn:,} functions; {len(rows):,} of "
          f"{len(names)*(len(names)-1)//2:,} pairs have data ({doc['coverage']:.1%})")
    print(f"median ratio {doc['median_ratio']:.3f}; related (ratio > 2) "
          f"{doc['share_related']:.1%}")
    print(f"-> {OUT2 / 'COOCCURRENCE.json'}")


if __name__ == "__main__":
    main()
