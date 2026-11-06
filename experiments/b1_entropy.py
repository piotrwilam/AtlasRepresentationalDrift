r"""B1 — the battery's own continuation entropy at the dependent positions (design item 15).

§5 states the floor at two of the three position classes and leaves the third open: at forced
positions the loss is about 0.09 nats; at expected positions the model's 0.55-0.59 nats after
`for x` matches the battery's own continuation entropy of 0.571; *"at dependent positions the
floor is [TBC: 15]"*. Without it, §4.2's "`for`: 5.6 to 2.5 nats" at a governed position has no
floor to be read against, and a loss of 2.5 nats could be anything.

The floor is the conditional entropy of **the battery's own** distribution at that position --
the generator declares it, so the entropy is computable without the model, which is why this item
is generator-only. `floor_table.py` already does this, but on the **pilot** battery
(`FLOORS.json` says so in its own `source` field, and A52 marks the pilot artefacts). This runs
the same `floors_for_set` and `slot_report` over the **phase** battery and reports the
`governed` and `scope` roles, with the obliged roles beside them as the check that the phase
numbers reproduce the pilot's where both exist.

Where a role has no estimable floor the reason is reported rather than a number invented: a role
whose slot pieces carry the end of the expression in front of them has no floor at all, which is
nine of twenty roles on the pilot.

    python experiments/b1_entropy.py
"""
from __future__ import annotations

import gzip
import json
import time
import types

import numpy as np
from transformers import AutoTokenizer

from experiments.atlas import D, provenance
from experiments.relatedness import OUT2
from experiments.split_p1 import BATTERY, TOK_REV
from profilebattery.generate.floors import UNCLEAN_TOLERANCE, floors_for_set, slot_report
from profilebattery.generate.positions import positions_for
from profilebattery.record.sweep import REPO

DEPENDENT = ("governed", "scope")


def main() -> None:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    cache: dict[str, list[str]] = {}

    def pieces_of(text):
        if text not in cache:
            cache[text] = [tok.convert_ids_to_tokens(i)
                           for i in tok(text, add_special_tokens=False)["input_ids"]]
        return cache[text]

    out = {"what": "the battery's own conditional continuation entropy at each role, on the "
                   "PHASE battery, for the dependent classes §5 leaves open",
           "tokeniser_revision": TOK_REV, "unclean_tolerance": UNCLEAN_TOLERANCE,
           "source": "phase_battery/twin1 A sets",
           "not_the_pilot": "validation/positions/FLOORS.json is the pilot battery (its own "
                            "`source` field); A52 marks the pilot artefacts",
           "constructs": {}}
    files = sorted((BATTERY / "twin1").glob("*_A.jsonl.gz"))
    for n, f in enumerate(files, 1):
        c = f.name.rsplit("_", 1)[0]
        recs, skipped = [], 0
        for r in (json.loads(l) for l in gzip.open(f, "rt")):
            ct = r.get("site_construct_token")
            if ct is None:
                continue
            got = positions_for(r["text"], c, pieces_of(r["text"]), ct)
            if not got:
                skipped += 1
                continue
            recs.append(types.SimpleNamespace(text=r["text"],
                                              positions=[p.as_dict() for p in got]))
        floors = floors_for_set(recs, pieces_of)
        report = slot_report(recs, pieces_of)
        cache.clear()
        if not floors:
            continue
        out["constructs"][c] = {
            "prompts": len(recs), "unparsed": skipped,
            "roles": {role: {"floor_nats": floors[role],
                             "estimable": floors[role] is not None,
                             **report.get(role, {})} for role in sorted(floors)}}
        if n % 25 == 0:
            print(f"  {n}/{len(files)} sets, {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- the dependent slots, which `floors_for_set` does not cover ----
    # Every role it returns is an OBLIGED one (`in`, `header_colon`, `close_paren`, ...), so the
    # governed and scope floors cannot come from it. The same estimator -- the entropy of the
    # pieces the generator actually placed at that slot, over the set's prompts -- applies
    # directly at a dependent position, which is what `positions_for` marks.
    from profilebattery.generate.floors import conditional_entropy
    import collections as _c
    dep_counts: dict = {}
    for n, f in enumerate(files, 1):
        c = f.name.rsplit("_", 1)[0]
        for r in (json.loads(l) for l in gzip.open(f, "rt")):
            ct = r.get("site_construct_token")
            if ct is None:
                continue
            pcs = pieces_of(r["text"])
            for p in positions_for(r["text"], c, pcs, ct) or ():
                d = p.as_dict()
                if d["class"] not in DEPENDENT:
                    continue
                i = d["token_index"]
                if i is None or i >= len(pcs):
                    continue
                key = (c, d["class"], d["role"])
                dep_counts.setdefault(key, _c.Counter())[pcs[i]] += 1
        cache.clear()
        if n % 50 == 0:
            print(f"  dependent slots: {n}/{len(files)} sets, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    out["dependent_slots"] = {}
    for (c, cls, role), cnt in sorted(dep_counts.items()):
        tot = sum(cnt.values())
        out["dependent_slots"][f"{c}|{cls}|{role}"] = {
            "class": cls, "role": role, "n_positions": tot,
            "n_distinct_pieces": len(cnt),
            "entropy_nats": conditional_entropy(cnt),
            "top_pieces": cnt.most_common(5)}
    roles = [(c, r, v) for c, d in out["constructs"].items() for r, v in d["roles"].items()]

    def block(pred):
        sub = [(c, r, v) for c, r, v in roles if pred(r)]
        est = [v["floor_nats"] for _, _, v in sub if v["estimable"]]
        return {"n_roles": len(sub), "n_estimable": len(est),
                "n_unavailable": len(sub) - len(est),
                "floor_nats_median": float(np.median(est)) if est else None,
                "floor_nats_min": float(min(est)) if est else None,
                "floor_nats_max": float(max(est)) if est else None,
                "non_zero": {f"{c}.{r}": round(v["floor_nats"], 4) for c, r, v in sub
                             if v["estimable"] and v["floor_nats"] and v["floor_nats"] > 1e-9},
                "unavailable_reasons": sorted({
                    str(v.get("why") or v.get("reason")
                        or "slot pieces carry the end of the expression in front of them")
                    for c, r, v in sub if not v["estimable"]})[:6]}

    out["dependent"] = {k: block(lambda r, k=k: r.startswith(k) or k in r)
                        for k in DEPENDENT}
    out["dependent_pooled"] = block(lambda r: any(k in r for k in DEPENDENT))
    out["obliged_for_comparison"] = block(lambda r: not any(k in r for k in DEPENDENT))
    out["summary"] = {"roles": len(roles),
                      "estimable": sum(1 for _, _, v in roles if v["estimable"]),
                      "constructs": len(out["constructs"])}
    for cls in DEPENDENT:
        vs = [v for v in out["dependent_slots"].values() if v["class"] == cls]
        ents = [v["entropy_nats"] for v in vs if v["n_positions"] >= 20]
        out[f"{cls}_slot_entropy"] = {
            "n_slots": len(vs), "n_slots_with_at_least_20_positions": len(ents),
            "entropy_nats_median": float(np.median(ents)) if ents else None,
            "entropy_nats_q1_q3": ([float(np.quantile(ents, .25)),
                                    float(np.quantile(ents, .75))] if ents else None),
            "entropy_nats_min_max": ([float(min(ents)), float(max(ents))] if ents else None),
            "note": ("`floors_for_set` covers the obliged slot only, so this is the same "
                     "estimator applied at the position `positions_for` marks; the scope class "
                     "has almost no population in the battery (RESULTS.md #52)")}
    # ---- against the model's loss at the same positions ----
    # This is the point of the item, and it decides how the number may be quoted. At an OBLIGED
    # position the generator's distribution IS the floor: nothing in the prompt determines the
    # continuation beyond the grammar. At a DEPENDENT position it is not -- the continuation is a
    # use of a name the prompt has already bound, so a model that reads the prompt can and should
    # beat the generator's marginal. The marginal is therefore an UPPER BOUND on the floor, not
    # the floor, and quoting it as a floor in §5 would be wrong.
    try:
        st = json.loads((D / "validation/r2/STEP4_FLAT_pretrain.json").read_text())
        dep = ((st.get("dependent") or {}).get("governed") or {}).get("series") or {}
        lv = [(k, v.get("level_early"), v.get("level_84b"), v.get("level_last"))
              for k, v in dep.items()]
        ge = out["governed_slot_entropy"]["entropy_nats_median"]
        early = [x[1] for x in lv if x[1] is not None]
        last = [x[3] for x in lv if x[3] is not None]
        out["against_the_model"] = {
            "generator_marginal_entropy_governed_median_nats": ge,
            "model_loss_governed_early_median_nats": (float(np.median(early))
                                                      if early else None),
            "model_loss_governed_last_median_nats": (float(np.median(last)) if last else None),
            "n_governed_series": len(lv),
            "reading": (
                "the model's governed loss starts at about the generator's marginal entropy and "
                "ends well below it, so the marginal is an UPPER BOUND on the floor and not the "
                "floor: at a dependent position the prompt has already bound the name, so the "
                "conditional entropy of a correct use is near zero. §5 must not quote this "
                "number as 'the dependent floor'. What it does establish is that the model is "
                "still far above the true floor at dependent positions, unlike the forced ones "
                "which sit at theirs -- so learning has not finished where the construct "
                "governs, only where it forces.")}
        print(f"\nagainst the model: generator marginal {ge:.3f} nats; model governed loss "
              f"{np.median(early):.3f} early -> {np.median(last):.3f} last "
              f"({len(lv)} series)")
    except Exception as exc:                                        # noqa: BLE001
        out["against_the_model"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"\nagainst the model: unavailable ({type(exc).__name__})")
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["phase_battery/twin1/*_A.jsonl.gz",
                                    "profilebattery.generate.floors (the generator's own "
                                    "declared distribution)"])
    (OUT2 / "B1_ENTROPY.json").write_text(json.dumps(out, indent=1, default=float))
    for k in ("governed", "scope"):
        b = out["dependent"][k]
        print(f"\n{k}: {b['n_roles']} roles, {b['n_estimable']} estimable; floor median "
              f"{b['floor_nats_median']}, range {b['floor_nats_min']}-{b['floor_nats_max']}")
        if b["non_zero"]:
            print(f"  non-zero: {dict(list(b['non_zero'].items())[:8])}")
        if b["n_unavailable"]:
            print(f"  {b['n_unavailable']} unavailable: {b['unavailable_reasons'][:2]}")
    for cls in DEPENDENT:
        b = out[f"{cls}_slot_entropy"]
        print(f"\n{cls} SLOTS: {b['n_slots']} slots, {b['n_slots_with_at_least_20_positions']} "
              f"with >=20 positions; entropy median {b['entropy_nats_median']}, range "
              f"{b['entropy_nats_min_max']}")
    o = out["obliged_for_comparison"]
    print(f"\nobliged, for comparison: {o['n_estimable']} of {o['n_roles']} estimable, "
          f"median {o['floor_nats_median']}")
    print(f"-> {OUT2 / 'B1_ENTROPY.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
