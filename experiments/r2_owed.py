"""The two owed R2 items -- and the finding that curve 2 was read in the wrong form.

**What was owed** (the Settle chat, 14 Oct; `REPORT-R2` §3 and the owed list): curve 2's
**standardised form beside the raw**, and **every verdict-layer figure also with the last layer
excluded**, which needs the argmax recomputed per draw and so could not be approximated from the
stored argmax shares.

**What the tree actually holds, measured before building either.** A twin-1 npz carries
`A_mean`/`B_mean` (standardised), `A_raw`/`B_raw` (unstandardised) **and** `A_cores`/`B_cores`.
The per-core arrays match the **standardised** mean to 4e-7 and differ from the raw mean by up to
70 -- checked on **32 npz across 8 references, 32 of 32**. So the R2 bootstrap pass, which reads
only `A_cores`/`B_cores`, computed curve 2 on the **standardised** profile.

**That inverts the owed item, and it matters, because C0.12 makes the raw form primary.** C10a
dates curve 2 **on the raw curve**, with the standardised beside it, precisely because the pilot's
Fact 3 was a growth that existed in the standardised form only: sigma-0 shrinks as ordinary code
becomes more predictable, so a rising z-score can be a falling denominator. What was owed is
therefore the **raw** form -- the primary -- and what R2 reported is the one that belongs beside
it. The two are not close. For `For` over the eight local references:

| reference | standardised | raw |
|---|---|---|
| 4.19 B | 194.95 | 156.14 |
| 12.58 B | 237.96 | **200.38** |
| 41.9 B | 234.08 | 145.86 |
| 83.9 B | 230.22 | **125.77** |

The standardised norm rises and flattens; the raw norm peaks at 12.6 B and **falls 37%** by 84 B.
Every reading taken from curve 2 -- its 75 dates, A34's agreement, step 3's frequency effect on
"the norm's date", step 6's group ordering and step 7's P7 -- was taken on the standardised form.
They are reported again here on the raw form, and the pair is what C0.12 asks for.

**One limit, stated rather than worked around.** The raw form has **no per-core arrays**: `A_raw`
is a mean. So the raw curve carries **point estimates** and a point-to-point noise unit, named as
such, and not a core bootstrap -- the same limitation curve 5 has, for the same reason. Its
intervals would need the record's accumulators.

**And neither item needs Colab, which is a measurement and not an assumption.** A ranged read of
just the two members takes **6.5 s** a construct (`A_raw` + `B_raw`, out of a 95 MB npz) and
**9.1 s** for `A_cores` + `B_cores`. The raw curve is 169 constructs x 53 remaining references
(8 are already local) at 8 workers; the last-layer tie-break is 169 constructs at **one**
reference, the verdict reference. Both are a few hours on this laptop, against the 390 GB the full
per-core pass needed.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import time

import numpy as np

from experiments.construct_token_loss import branches, label_curve
from experiments.r2_bootstrap import (DEPTH_LAYERS, DRAWS, K, SEED, VERDICT_REF, draw_indices,
                                      tie_break)
from experiments.step4_flat import noise_unit
from profilebattery.profiles.names import tree_name
from profilebattery.paths import data_root, dataset_repo

REPO = dataset_repo()
D = data_root()
LOCAL_TREE = D / "profiles"
OUT = D / "validation/r2/owed"


def reader(token):
    """Read named members of one npz, locally if present and by ranged read if not.

    The local tree holds 8 references, **but only 30 constructs each** -- the pilot's set, not
    P1's 169 (measured 16 Oct, after this docstring's first version implied whole references were
    local and CC's read-cost estimate inherited the error). So a local hit saves a read for 30 of
    169 constructs at those references and every other read is a ranged read of the two members
    wanted, not of the 95 MB file.
    """
    # **The filesystem is built on first remote use, not on construction.** 7 of the 61
    # references are on disk, and a pass over only those should need no hub client at all --
    # which is also what makes the local path testable without the network.
    state = {}

    def remote():
        if "fs" not in state:
            from huggingface_hub import HfFileSystem
            state["fs"] = HfFileSystem(token=token)
        return state["fs"]

    def read(ref, twin, c, keys):
        # DATASET ONLY (A52). The local-first branch that used to stand here read
        # `$ATLAS_DATA/profiles/`, which held the PILOT recording at the phase
        # recording's path: 8 references x 30 constructs. Measured 23 Oct, the damage is exactly
        # at that intersection -- 12 of 12 pilot constructs at a pilot reference differ by a
        # median 17.9%, and 0 of 12 differ in each of the other three cells of the 2x2. The
        # resolver refuses any local tree by signature, so this cannot recur.
        from profilebattery.profiles import resolve
        return resolve.profile(ref, twin, c, tuple(keys))
    return read


def raw_and_standardised(z: dict) -> dict:
    """Both norms of one construct's concept profile at one reference, over the product space."""
    out = {}
    if "A_raw" in z and "B_raw" in z:
        out["raw"] = float(np.linalg.norm(np.asarray(z["A_raw"], np.float64)
                                          - np.asarray(z["B_raw"], np.float64)))
    if "A_mean" in z and "B_mean" in z:
        out["standardised"] = float(np.linalg.norm(np.asarray(z["A_mean"], np.float64)
                                                   - np.asarray(z["B_mean"], np.float64)))
    elif "concept" in z:
        out["standardised"] = float(np.linalg.norm(np.asarray(z["concept"], np.float64)))
    return out


def one_reference_norms(ref: str, constructs, read, workers: int = 8, log=print) -> dict:
    """Both norm forms for every construct at one reference, pooled over threads."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()

    def one(c):
        for attempt in range(4):
            try:
                # **only the raw members are read.** The standardised curve already exists in
                # `R2_REPORT.json` -- that is what the bootstrap pass computed -- so re-reading
                # `A_mean`/`B_mean` doubles the bytes for a series already in hand. Measured: the
                # four-member read ran at about 11 minutes a reference, the two-member one at
                # about 6.
                return c, raw_and_standardised(read(ref, "twin1", c, ("A_raw", "B_raw")))
            except Exception as e:                                # noqa: BLE001
                if attempt == 3:
                    return c, {"error": f"{type(e).__name__}: {e}"}
                time.sleep(1.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = dict(ex.map(one, constructs))
    bad = sum(1 for v in rows.values() if "error" in v)
    log(f"  {ref}: {len(rows)} constructs in {time.time() - t0:.0f}s ({bad} failed)")
    return rows


def curve(series: dict, refs, toks, form: str, branch: str) -> dict:
    """One construct's norm curve in one form, as a share of the branch's last value, dated.

    The share convention is C10a's: the norm per checkpoint **as a share of its value at the last
    checkpoint of the branch**. The unit is the point-to-point one and is labelled, because the raw
    form has no per-core arrays to bootstrap -- `A_raw` is a mean.
    """
    vals, got = [], []
    for r in refs:
        e = series.get(r) or {}
        if form in e:
            vals.append(e[form])
            got.append(r)
    if len(vals) < 2:
        return {"outcome": "unreadable", "why": "fewer than two references carry this form"}
    last = vals[-1]
    share = [v / last for v in vals] if last else vals
    nu = noise_unit(share, [None] * len(share))
    d = label_curve(share, [nu["unit"]] * len(share) if nu["unit"] else [None] * len(share),
                    [dict(zip(refs, toks))[r] for r in got],
                    step="consecutive" if branch == "anneal" else "factor")
    d.update({"form": form, "n_checkpoints": len(vals), "references": got,
              "values": vals, "share_of_last": share,
              "level_first": share[0], "level_last": share[-1],
              "unit": nu["unit"], "unit_is": nu["unit_is"],
              "interval": "POINT ESTIMATES only: the raw form has no per-core arrays, so no core "
                          "bootstrap is possible from the tree -- the same limit curve 5 has"})
    return d


def last_layer_excluded(z: dict, draws: int = DRAWS, seed: int = SEED) -> dict:
    """The verdict-layer tie-break with the last layer excluded, recomputed per draw.

    The stored artefact keeps only the argmax **shares**, so excluding a layer could not be done
    from it -- the argmax has to be retaken per draw over the reduced layer range. Both forms are
    returned so the sensitivity sits beside the rule as it stands, which is what the R1 ruling
    asked for.
    """
    a = np.asarray(z["A_cores"], np.float64)
    b = np.asarray(z["B_cores"], np.float64)
    nlay, nunit = a.shape[1], a.shape[2]
    af, bf = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    pairs = draw_indices(a.shape[0], b.shape[0], draws, seed)
    from experiments.r2_bootstrap import _weights
    wa, wb = _weights(pairs, a.shape[0], 0), _weights(pairs, b.shape[0], 1)
    full, cut = [], []
    for s0 in range(0, len(pairs), 100):
        s1 = min(s0 + 100, len(pairs))
        v = wa[s0:s1] @ af - wb[s0:s1] @ bf
        ln = np.linalg.norm(v.reshape(-1, nlay, nunit), axis=2)
        full.append(np.argmax(ln, axis=1))
        cut.append(np.argmax(ln[:, :nlay - 1], axis=1))
    return {"as_the_rule_stands": tie_break(np.concatenate(full)),
            "last_layer_excluded": tie_break(np.concatenate(cut)),
            "n_layers": int(nlay), "n_draws": int(len(pairs))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="pretrain", choices=("pretrain", "anneal"))
    ap.add_argument("--what", default="both", choices=("both", "norms", "tiebreak"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--limit", type=int, default=0, help="first N constructs, for a trial")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p1 = [tree_name(c) for c in json.loads((D / "validation/battery/SPLIT.json").read_text())["p1"]]
    if args.limit:
        p1 = p1[:args.limit]
    have = {p.stem for p in (D / "_carriers_dl/validation/carriers").glob("*.npz")}
    sched = json.loads((D / "validation/schedule/CHECKPOINTS.json").read_text())
    seq = [(r, t) for r, t in branches(sched)[args.branch] if r in have]
    refs, toks = [r for r, _ in seq], [t for _, t in seq]
    token = os.environ.get("HF_TOKEN")
    if token is None:
        from huggingface_hub import get_token
        token = get_token()
    read = reader(token)
    nlocal = sum(1 for r in refs if (LOCAL_TREE / r / "twin1").exists())
    print(f"branch {args.branch}: {len(refs)} references ({nlocal} already local), "
          f"{len(p1)} constructs")

    if args.what in ("both", "norms"):
        print("\n=== curve 2 in both forms (raw PRIMARY by C0.12) ===")
        series = collections.defaultdict(dict)
        for r in refs:
            p = out / f"NORMS_{r}.json"
            if p.exists():
                rows = json.loads(p.read_text())
            else:
                rows = one_reference_norms(r, p1, read, args.workers)
                p.write_text(json.dumps(rows, indent=1, default=float))
            for c, e in rows.items():
                if "error" not in e:
                    series[c][r] = e
        # the standardised series from R2's own artefact, which is what the bootstrap computed
        rep2 = json.loads((D / "validation/r2/R2_REPORT.json").read_text())
        std = (rep2["curves"].get(f"{args.branch}|curve2_norm") or {}).get("series") or {}
        for c, e in std.items():
            for r, v in zip(e["references"], e["values"]):
                series[c].setdefault(r, {})["standardised_share"] = v
        curves = {}
        for c in p1:
            curves[c] = {"raw": curve(series.get(c, {}), refs, toks, "raw", args.branch),
                         "standardised": curve(series.get(c, {}), refs, toks,
                                               "standardised_share", args.branch)}
            if "raw" in curves[c]["standardised"]:
                curves[c]["standardised"]["form"] = "standardised"
        for f in ("raw", "standardised"):
            lab = collections.Counter(
                (v[f].get("flat") or v[f].get("label") or v[f]["outcome"]) for v in curves.values())
            _ = lab
            lv = [v[f]["level_first"] for v in curves.values() if "level_first" in v[f]]
            print(f"  {f:13s} {dict(lab)}")
            print(f"{'':15s} median share of the final value at the first checkpoint: "
                  f"{np.median(lv):.3f}")
        # the two forms' directions over the record, which is what C0.12 exists to expose
        agree = dis = 0
        rows = {}
        for c, v in curves.items():
            dr = {f: (None if "level_first" not in v[f] else
                      "rose" if v[f]["level_last"] > v[f]["level_first"] else "fell")
                  for f in ("raw", "standardised")}
            rows[c] = dr
            if None not in dr.values():
                agree += dr["raw"] == dr["standardised"]
                dis += dr["raw"] != dr["standardised"]
        print(f"  direction over the record: the two forms AGREE for {agree} constructs and "
              f"DISAGREE for {dis}")
        dirs = collections.Counter((v["raw"], v["standardised"]) for v in rows.values())
        for k, n in dirs.most_common():
            print(f"      raw {k[0]}, standardised {k[1]}: {n}")
        (out / f"CURVE2_FORMS_{args.branch}.json").write_text(json.dumps(
            {"branch": args.branch, "primary": "raw, by C0.12 and C10a",
             "r2_used": "standardised -- the per-core arrays are standardised, measured on 32 of "
                        "32 npz across 8 references",
             "n_agree_direction": agree, "n_disagree_direction": dis,
             "directions": rows, "curves": curves}, indent=1, default=float))
        print(f"  -> {out / f'CURVE2_FORMS_{args.branch}.json'}")

    if args.what in ("both", "tiebreak") and args.branch == "pretrain":
        print(f"\n=== the verdict layer with the last layer excluded, at {VERDICT_REF} ===")
        from concurrent.futures import ThreadPoolExecutor
        p = out / "TIEBREAK_NO_LAST.json"
        rows = json.loads(p.read_text()) if p.exists() else {}

        def one(c):
            for attempt in range(4):
                try:
                    z = read(VERDICT_REF, "twin1", c, ("A_cores", "B_cores"))
                    return c, last_layer_excluded(z, args.draws)
                except Exception as e:                            # noqa: BLE001
                    if attempt == 3:
                        return c, {"error": f"{type(e).__name__}: {e}"}
                    time.sleep(1.5 * (attempt + 1))

        todo = [c for c in p1 if c not in rows]
        if todo:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for i, (c, e) in enumerate(ex.map(one, todo), 1):
                    rows[c] = e
                    if i % 25 == 0:
                        print(f"    {i}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
                        p.write_text(json.dumps(rows, indent=1, default=float))
            p.write_text(json.dumps(rows, indent=1, default=float))
        ok = {c: e for c, e in rows.items() if "error" not in e}
        moved = [c for c, e in ok.items()
                 if e["as_the_rule_stands"]["modal_layer"] != e["last_layer_excluded"]["modal_layer"]]
        was_last = [c for c, e in ok.items()
                    if e["as_the_rule_stands"]["modal_layer"] == e["n_layers"] - 1]
        tied_before = [c for c, e in ok.items() if e["as_the_rule_stands"]["tied"]]
        tied_after = [c for c, e in ok.items() if e["last_layer_excluded"]["tied"]]
        print(f"  {len(ok)} constructs read ({len(rows) - len(ok)} failed)")
        print(f"  modal layer IS the last layer: {len(was_last)}")
        print(f"  modal layer MOVES when the last layer is excluded: {len(moved)}")
        print(f"  tie-break fires (a second layer above 5%): {len(tied_before)} as the rule "
              f"stands, {len(tied_after)} with the last layer excluded")
        if ok:
            new = collections.Counter(e["last_layer_excluded"]["modal_layer"] for e in ok.values())
            print(f"  the new modal layers: {dict(new.most_common(8))}")
        (out / f"TIEBREAK_SENSITIVITY.json").write_text(json.dumps(
            {"reference": VERDICT_REF, "n_read": len(ok),
             "n_modal_is_last_layer": len(was_last), "n_modal_moves": len(moved),
             "constructs_whose_modal_moves": sorted(moved),
             "n_tied_as_the_rule_stands": len(tied_before),
             "n_tied_last_layer_excluded": len(tied_after),
             "why_it_could_not_be_approximated": "the stored artefact keeps the argmax SHARES "
                                                 "only; excluding a layer needs the argmax "
                                                 "retaken per draw",
             "per_construct": rows}, indent=1, default=float))
        print(f"  -> {out / 'TIEBREAK_SENSITIVITY.json'}")


if __name__ == "__main__":
    main()
