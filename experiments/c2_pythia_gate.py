r"""C2 — the Pythia readability check, part 1: does the read-site gate survive the tokeniser?

P1 is a **tokeniser** property, not a model property: the gate is "the modal anchor piece covers
>= 0.80 of the indented-scope prompts, on both twins and both sides" (`SPLIT.json`'s `p1_rule`,
counted by `experiments/split_p1.py`). It passed 169 of 188 constructs on OpenCoder's
SentencePiece vocabulary. Pythia uses GPT-NeoX BPE, a different vocabulary with a different
word-boundary convention, so the gate must be recounted before any Pythia record is planned --
a construct whose read site is unstable under the new tokeniser has no read site to record.

The anchor is located by **character offset, not token index**: the OpenCoder piece offsets give
the character at which the construct's surface token starts, and the Pythia piece covering that
character is the new anchor. Same place in the same prompt, different vocabulary.

    python experiments/c2_pythia_gate.py
"""
from __future__ import annotations

import collections
import gzip
import json
import time

from transformers import AutoTokenizer

from experiments.b10v import piece_offsets
from experiments.build_battery import construct_of_stem
from experiments.atlas import D, provenance
from experiments.relatedness import OUT2
from experiments.split_p1 import BATTERY, TOK_REV, read_piece
from profilebattery.generate.twouse import INDENTED_SCOPES
from profilebattery.record.sweep import REPO

PYTHIA = "EleutherAI/pythia-1.4b"
PYTHIA_REV = "step143000"           # the last checkpoint; the gate is revision-independent
GATE = 0.80


def main() -> None:
    t0 = time.time()
    oc = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    py = AutoTokenizer.from_pretrained(PYTHIA, revision=PYTHIA_REV)
    print(f"OpenCoder vocab {oc.vocab_size:,}; Pythia vocab {py.vocab_size:,} "
          f"({'fast' if py.is_fast else 'slow'})", flush=True)

    shares_oc: dict[str, list[float]] = {}
    shares_py: dict[str, list[float]] = {}
    unresolved = collections.Counter()
    n_rows = 0
    for twin in ("twin1", "twin2"):
        for f in sorted((BATTERY / twin).glob("*_A.jsonl.gz")):
            stem = f.name.rsplit("_", 1)[0]
            c = construct_of_stem(stem)
            for side in ("A", "B"):
                g = BATTERY / twin / f"{stem}_{side}.jsonl.gz"
                if not g.exists():
                    continue
                rows = [json.loads(l) for l in gzip.open(g, "rt")]
                rows = [r for r in rows if r.get("scope") in INDENTED_SCOPES]
                if not rows:
                    shares_oc.setdefault(c, []).append(0.0)
                    shares_py.setdefault(c, []).append(0.0)
                    continue
                n_rows += len(rows)
                c_oc = collections.Counter(read_piece(r.get("anchor_piece")) for r in rows)
                shares_oc.setdefault(c, []).append(c_oc.most_common(1)[0][1] / len(rows))
                c_py = collections.Counter()
                for r in rows:
                    text = r["text"]
                    ocp = oc.tokenize(text)
                    offs = piece_offsets(ocp)
                    i = r.get("site_construct_token")
                    if i is None or i >= len(offs):
                        unresolved[c] += 1
                        c_py["<unresolved>"] += 1
                        continue
                    # SentencePiece's `\u2581for` piece BEGINS on the preceding space, and under
                    # GPT-NeoX BPE that space belongs to the indentation RUN token, so taking
                    # offs[i] literally made the anchor the indentation and scored every
                    # statement keyword at exactly 0.500 -- four spaces at function scope, eight
                    # at method scope. The anchor is the surface token's first non-space
                    # character, which is the same character under either vocabulary.
                    char = offs[i]
                    while char < len(text) and text[char] in " \t":
                        char += 1
                    enc = py(text, add_special_tokens=False, return_offsets_mapping=True)
                    piece = "<unresolved>"
                    for (a, b), tid in zip(enc["offset_mapping"], enc["input_ids"]):
                        if a <= char < b:
                            piece = py.convert_ids_to_tokens(tid)
                            break
                    if piece == "<unresolved>":
                        unresolved[c] += 1
                    c_py[piece] += 1
                shares_py.setdefault(c, []).append(c_py.most_common(1)[0][1] / len(rows))
            if len(shares_py) % 25 == 0:
                print(f"  {len(shares_py)} constructs, {(time.time()-t0)/60:.1f} min", flush=True)

    def passes(sh):
        return sorted(c for c, s in sh.items() if len(s) == 4 and min(s) >= GATE)

    p_oc, p_py = passes(shares_oc), passes(shares_py)
    lost = [c for c in p_oc if c not in set(p_py)]
    gained = [c for c in p_py if c not in set(p_oc)]
    split = json.loads((D / "validation/battery/SPLIT.json").read_text())
    out = {"what": "the P1 read-site gate recounted on Pythia's GPT-NeoX vocabulary",
           "rule": split["p1_rule"], "gate": GATE,
           "model": PYTHIA, "revision": PYTHIA_REV,
           "anchor_located_by": "character offset of the OpenCoder piece at "
                                "site_construct_token, advanced past leading space; the "
                                "Pythia piece covering that character",
           "n_constructs_scored": len(shares_py), "n_indented_rows": n_rows,
           "pass_count_opencoder_recounted": len(p_oc),
           "pass_count_opencoder_in_SPLIT": split["p1_count"],
           "pass_count_pythia": len(p_py),
           "lost_under_pythia": lost, "gained_under_pythia": gained,
           "unresolved_rows_by_construct": dict(unresolved),
           "shares_pythia": {c: s for c, s in sorted(shares_py.items())},
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["phase_battery/twin{1,2}/*_{A,B}.jsonl.gz",
                                     f"{PYTHIA} @ {PYTHIA_REV} tokeniser",
                                     f"{REPO} @ {TOK_REV} tokeniser"])}
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "C2_PYTHIA_GATE.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nOpenCoder recounted: {len(p_oc)} pass (SPLIT.json says {split['p1_count']})")
    print(f"Pythia:              {len(p_py)} pass of {len(shares_py)} scored")
    print(f"  lost {len(lost)}: {lost[:12]}")
    print(f"  gained {len(gained)}: {gained[:12]}")
    print(f"-> {OUT2 / 'C2_PYTHIA_GATE.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
