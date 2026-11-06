"""Step 8 by layer: the degree distribution, the R11 partition and the memberships each layer holds.

One table a row per layer, from the carrier index and the extended top-10% index, pretraining
branch, product space. The partition is boundary-first, as `step8_graph` applies it, and is given
at the **dense** 1,000-step spacing and **pooled** over every pair the ruling allows -- never at
"t to 1.5 t".

**The unit that ends in all 169 sets** is located and described here too: its layer, its position in
that layer, its unstandardised mean activation (`A_raw`) at the construct-token read site and its
concept-profile value, pooled over the constructs read. The question that motivates it -- whether it
is a massive-activation unit -- is answered by `A_raw`'s size against the layer's own distribution,
which is why the raw and not the standardised form is read.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

import numpy as np

from experiments.step8_graph import (BANDS_BOUNDARY, N_UNITS, extended_at, membership, partners,
                                     recorded, sets_at)
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root

D = data_root()
R2 = D / "validation/r2"
WIDTH = 6144
N_LAYERS = 24
VERDICT_REF = "pretrain_iter_0484865"


def layer_of(idx: np.ndarray) -> np.ndarray:
    return np.asarray(idx, np.int64) // WIDTH


def degree_by_layer(ref: str, constructs) -> list:
    """Per layer: the share in no set, in five or more, the dispersion, and the memberships held."""
    m = membership(sets_at(ref, constructs), constructs)
    d = m.sum(axis=0).astype(np.int32)
    total = int(d.sum())
    rows = []
    for L in range(N_LAYERS):
        seg = d[L * WIDTH:(L + 1) * WIDTH]
        rows.append({"layer": L, "share_0": float((seg == 0).mean()),
                     "share_ge_5": float((seg >= 5).mean()),
                     "mean": float(seg.mean()),
                     "dispersion": float(seg.var(ddof=1) / seg.mean()) if seg.mean() else None,
                     "max": int(seg.max()),
                     "memberships_held": int(seg.sum()),
                     "share_of_all_memberships": float(seg.sum() / total) if total else None})
    return rows


def partition_by_layer(seq, constructs, pairs) -> dict:
    """The R11 partition, boundary-first, bucketed by the departing unit's layer."""
    refs = [r for r, _ in seq]
    acc = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for p in pairs:
        a, b = refs[p["i"]], refs[p["j"]]
        sa, sb = sets_at(a, constructs), sets_at(b, constructs)
        other = membership(sb, constructs).sum(axis=0).astype(np.int32)
        ext = extended_at(b, constructs)
        kk = max(1, int(round(BANDS_BOUNDARY["5-10pct"] * N_UNITS)))
        scope = "dense" if p["rule"].startswith("dense") else "tail"
        for c in constructs:
            if c not in sa or c not in sb or c not in ext:
                continue
            left = np.setdiff1d(sa[c], sb[c], assume_unique=True)
            if left.size == 0:
                continue
            prim = np.isin(left, ext[c][:kk])
            reas = other[left] > 0
            lay = layer_of(left)
            for L in np.unique(lay):
                sel = lay == L
                for key in (scope, "pooled"):
                    t = acc[int(L)][key]
                    t["left"] += int(sel.sum())
                    t["boundary"] += int((prim & sel).sum())
                    t["reassigned"] += int((~prim & reas & sel).sum())
                    t["retired"] += int((~prim & ~reas & sel).sum())
    out = {}
    for L, byscope in acc.items():
        out[L] = {}
        for key, t in byscope.items():
            n = t["left"] or 1
            out[L][key] = {"n_left": t["left"], "boundary": t["boundary"] / n,
                           "reassigned": t["reassigned"] / n, "retired": t["retired"] / n}
    return out


def the_full_degree_unit(ref: str, constructs, token=None, n_sample: int = 0) -> dict:
    """The unit in every set at `ref`: its layer, its raw activation and its profile values."""
    m = membership(sets_at(ref, constructs), constructs)
    d = m.sum(axis=0)
    top = int(d.max())
    units = np.where(d == top)[0]
    out = {"reference": ref, "max_degree": top, "n_units_at_max": int(units.size),
           "units": [{"index": int(u), "layer": int(u // WIDTH),
                      "within_layer": int(u % WIDTH)} for u in units[:10]]}
    if n_sample == 0:
        return out
    from concurrent.futures import ThreadPoolExecutor
    from experiments.r2_owed import reader
    read = reader(token)
    pick = constructs if n_sample >= len(constructs) else constructs[:n_sample]

    def one(c):
        try:
            z = read(ref, "twin1", c, ("A_raw", "concept"))
            return c, (np.asarray(z["A_raw"], np.float64).ravel(),
                       np.asarray(z["concept"], np.float64).ravel())
        except Exception:                                        # noqa: BLE001
            return c, None

    with ThreadPoolExecutor(max_workers=12) as ex:
        got = {c: v for c, v in ex.map(one, pick) if v is not None}
    out["n_constructs_read"] = len(got)
    for e in out["units"]:
        u, L = e["index"], e["layer"]
        raws = np.array([v[0][u] for v in got.values()])
        cons = np.array([v[1][u] for v in got.values()])
        # the layer's own distribution of raw means, so "massive" has a scale
        layraw = np.array([np.abs(v[0][L * WIDTH:(L + 1) * WIDTH]).mean() for v in got.values()])
        e.update({
            "A_raw_mean_over_constructs": float(raws.mean()),
            "A_raw_abs_median": float(np.median(np.abs(raws))),
            "layer_mean_abs_A_raw": float(layraw.mean()),
            "ratio_to_layer_mean_abs": float(np.median(np.abs(raws)) / layraw.mean())
                                       if layraw.mean() else None,
            "concept_median": float(np.median(cons)),
            "concept_abs_median": float(np.median(np.abs(cons))),
            "rank_of_abs_A_raw_in_its_layer": int(np.median([
                (np.abs(v[0][L * WIDTH:(L + 1) * WIDTH]) > abs(v[0][u])).sum() for v in got.values()
            ])),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=169,
                    help="constructs to read for the full-degree unit's activation")
    ap.add_argument("--out", default=str(R2))
    a = ap.parse_args()

    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    seq = recorded("pretrain")
    refs = [r for r, _ in seq]
    cs = [c for c in p1 if c in sets_at(refs[0], p1)]
    pr = partners(seq)
    print(f"{len(refs)} references, {len(cs)} constructs, {N_LAYERS} layers x {WIDTH}; "
          f"{len(pr)} partner pairs", flush=True)

    deg = {r: degree_by_layer(r, cs) for r in refs}
    print("degree by layer: done", flush=True)
    part = partition_by_layer(seq, cs, pr)
    print("partition by layer: done", flush=True)
    token = os.environ.get("HF_TOKEN")
    if token is None:
        from huggingface_hub import get_token
        token = get_token()
    full = the_full_degree_unit(VERDICT_REF, cs, token, a.sample)
    print("the full-degree unit: done", flush=True)

    first, last = refs[0], refs[-1]
    print(f"\n{'L':>3} | {'share0':>7} {'ge5':>6} {'disp':>6} {'%memb':>6} |"
          f" {'share0':>7} {'ge5':>6} {'disp':>6} {'%memb':>6} |"
          f" {'bnd':>5} {'reas':>5} {'ret':>5} | {'bnd':>5} {'reas':>5} {'ret':>5}")
    print(f"{'':>3} | {'--- 4.19 B ---':^29} | {'--- 2,034 B ---':^29} |"
          f" {'-- dense --':^17} | {'-- pooled --':^17}")
    for L in range(N_LAYERS):
        f, l = deg[first][L], deg[last][L]
        pd = (part.get(L) or {}).get("dense") or {}
        pp = (part.get(L) or {}).get("pooled") or {}
        print(f"{L:3d} | {f['share_0']:7.4f} {f['share_ge_5']:6.3f} "
              f"{(f['dispersion'] or 0):6.2f} {100*f['share_of_all_memberships']:6.2f} |"
              f" {l['share_0']:7.4f} {l['share_ge_5']:6.3f} {(l['dispersion'] or 0):6.2f} "
              f"{100*l['share_of_all_memberships']:6.2f} |"
              f" {pd.get('boundary', float('nan')):5.3f} {pd.get('reassigned', float('nan')):5.3f} "
              f"{pd.get('retired', float('nan')):5.3f} |"
              f" {pp.get('boundary', float('nan')):5.3f} {pp.get('reassigned', float('nan')):5.3f} "
              f"{pp.get('retired', float('nan')):5.3f}")

    print(f"\nthe unit(s) of maximum degree at {VERDICT_REF}: degree {full['max_degree']} "
          f"({full['n_units_at_max']} unit(s))")
    for e in full["units"]:
        print(f"   index {e['index']:,} = layer {e['layer']}, unit {e['within_layer']}")
        if "A_raw_mean_over_constructs" in e:
            print(f"      A_raw over {full['n_constructs_read']} constructs: mean "
                  f"{e['A_raw_mean_over_constructs']:+.4f}, |median| "
                  f"{e['A_raw_abs_median']:.4f}")
            print(f"      its layer's mean |A_raw|: {e['layer_mean_abs_A_raw']:.4f}  -> ratio "
                  f"{e['ratio_to_layer_mean_abs']:.2f}x; median rank in its layer by |A_raw|: "
                  f"{e['rank_of_abs_A_raw_in_its_layer']} of {WIDTH}")
            print(f"      concept profile: median {e['concept_median']:+.4f}, |median| "
                  f"{e['concept_abs_median']:.4f}")

    out = pathlib.Path(a.out) / "STEP8_BY_LAYER_pretrain.json"
    out.write_text(json.dumps({"references": refs, "n_constructs": len(cs),
                               "degree_by_layer": deg, "partition_by_layer": part,
                               "full_degree_unit": full,
                               "partition_order": "boundary-first, as step8_graph applies it",
                               "spacing": "dense = the true 1,000-step partner; pooled = every "
                                          "pair the ruling allows; never t to 1.5t"},
                              indent=1, default=float))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
