r"""The token strings that actually appear in each construct's battery cores (design item 10).

B15 needs this and nothing else does yet. The draft's candidate artefact for test 6's token lift
is physical: *"`for`'s A prompts carry `range` in their cores and its B prompts do not, so
`for`'s profile contains the model's response to `range`"*. Deciding whether a pair's shared
token is in the cores therefore needs the cores' actual contents, which the inventory does not
hold -- `inventory_v2.json` gives a construct's **grammar** tokens (`For`: for, in, else), not
what its rendered cores contain.

The core span runs from the **start of the line** holding `site_construct_token` through
`site_last_token`, and covers the construct including its body: `For/c2` renders
`for iterator in range(3):\n    if ligand_ok:\n ...`, so `range` is inside the span and `if` is
too. The span begins at the line and not at the anchor because the anchor is not always the
construct's first token -- `AsyncWith`'s anchor is `with`, with `async` one token to its left --
and a span starting at the anchor reported `async` as absent from AsyncWith's cores. Tokens are collected per construct over **both sides and both twins**; the A-only set is
kept beside the union because the artefact's shape is a token present on A and absent on B.

    python experiments/core_tokens.py        ->  validation/paper2/CORE_TOKENS.json
"""
from __future__ import annotations

import collections
import gzip
import json
import re
import time

from transformers import AutoTokenizer

from experiments.b10v import piece_offsets
from experiments.build_battery import construct_of_stem
from experiments.atlas import provenance
from experiments.relatedness import OUT2
from experiments.split_p1 import BATTERY, TOK_REV
from profilebattery.record.sweep import REPO

OUT = OUT2 / "CORE_TOKENS.json"
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def main() -> None:
    t0 = time.time()
    oc = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    per = {}
    for twin in ("twin1", "twin2"):
        for f in sorted((BATTERY / twin).glob("*_A.jsonl.gz")):
            stem = f.name.rsplit("_", 1)[0]
            c = construct_of_stem(stem)
            e = per.setdefault(c, {"A": collections.Counter(), "B": collections.Counter(),
                                   "n_rows": 0})
            for side in ("A", "B"):
                g = BATTERY / twin / f"{stem}_{side}.jsonl.gz"
                if not g.exists():
                    continue
                for line in gzip.open(g, "rt"):
                    r = json.loads(line)
                    text = r["text"]
                    i, k = r.get("site_construct_token"), r.get("site_last_token")
                    if i is None or k is None:
                        continue
                    offs = piece_offsets(oc.tokenize(text))
                    if i >= len(offs):
                        continue
                    # from the START OF THE LINE holding the anchor, not from the anchor.
                    # `Core.render` returns whole source lines, and the anchor is not always
                    # the construct's first token: `AsyncWith`'s anchor is `with`, so a span
                    # beginning at the anchor drops the `async` one token to its left -- which
                    # made `async` look absent from AsyncWith's cores and left five pairs
                    # "related outside the cores" that are nothing of the kind.
                    a = text.rfind("\n", 0, offs[i]) + 1
                    b = offs[k + 1] if k + 1 < len(offs) else len(text)
                    for w in WORD.findall(text[a:b]):
                        e[side][w] += 1
                    e["n_rows"] += 1
            print(f"  {c:22s} A {len(e['A']):4d} distinct, B {len(e['B']):4d}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    #: a token counts as "in the cores" if it appears in at least this share of that side's core
    #: spans. A token in one prompt of 1,920 is not what the artefact is about; a token the core
    #: template always renders is. Fixed here, before B15 reads it.
    SHARE = 0.05
    out = {"what": "the identifier and keyword tokens appearing in each construct's core span "
                   "in the battery, per side",
           "core_span": "site_construct_token .. site_last_token of each prompt",
           "in_cores_rule": f"appears in >= {SHARE:.0%} of that side's core spans",
           "constructs": {}}
    for c, e in sorted(per.items()):
        n = max(e["n_rows"] // 2, 1)
        A = {w for w, v in e["A"].items() if v >= SHARE * n}
        B = {w for w, v in e["B"].items() if v >= SHARE * n}
        out["constructs"][c] = {"n_core_spans_per_side": n,
                                "A": sorted(A), "B": sorted(B),
                                "union": sorted(A | B),
                                "A_only": sorted(A - B)}
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["phase_battery/twin{1,2}/*_{A,B}.jsonl.gz"])
    OUT.write_text(json.dumps(out, indent=1, default=float))
    ex = out["constructs"].get("For", {})
    print(f"\nFor: union {ex.get('union')}")
    print(f"     A_only {ex.get('A_only')}")
    print(f"-> {OUT}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
