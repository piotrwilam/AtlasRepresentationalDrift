r"""C2 — the Pythia readability check, part 2: the twin unit at one trained checkpoint.

The gate (part 1, `c2_pythia_gate.py`) says a read site exists under Pythia's vocabulary. This
says whether anything read there is **above the instrument's own noise**: the twin unit is the
disagreement between two independently drawn twins of the same battery set, and on OpenCoder it
is what every curve's range is quoted against -- a curve whose range is under five twin units is
`unreadable` (`construct_token_loss.py`).

One trained checkpoint, both twins, both sides. Per construct the concept profile is
pi(A) - pi(B) over the 24 x 8,192 MLP hidden-unit product space, and the twin unit is reported
three ways, each the analogue of a figure already quoted for OpenCoder:

  * **carrier-set overlap** between the twins' top-5% sets -- OpenCoder's 0.8436 (B12);
  * **profile cosine** between the twins' full concept profiles;
  * **the per-neuron twin unit** |pi_twin1 - pi_twin2|, as a fraction of the profile's own
    scale, which is the quantity the five-twin-unit readability rule divides by.

The site is the construct token, located the way part 1 locates the anchor: the character offset
of OpenCoder's piece at `site_construct_token`, advanced past leading space, then the Pythia
token covering that character. Nothing here reuses OpenCoder's token index directly -- the
vocabularies differ, so the index does not transfer.

    python experiments/c2_pythia_twin.py [n_prompts_per_set] [n_constructs]
"""
from __future__ import annotations

import gzip
import json
import sys
import time

import numpy as np
import torch

from experiments.b10v import piece_offsets
from experiments.atlas import provenance
from experiments.relatedness import OUT2
from experiments.split_p1 import BATTERY, PILOT, TOK_REV
from profilebattery.record.sweep import REPO

from experiments.c2_pythia_gate import PYTHIA, PYTHIA_REV

CARRIER_FRACTION = 0.05


class Gather:
    """Pre-hooks on every `mlp.dense_4h_to_h`: the 8,192-unit activation at one site a prompt."""

    def __init__(self, model):
        self.n = len(model.layers)
        self.sites: torch.Tensor | None = None
        self.buf: list[torch.Tensor] = [None] * self.n
        self.handles = [l.mlp.dense_4h_to_h.register_forward_pre_hook(self._pre(i))
                        for i, l in enumerate(model.layers)]

    def _pre(self, i):
        def hook(_mod, args):
            t = args[0]                                   # [batch, seq, 8192]
            idx = self.sites.view(-1, 1, 1).expand(-1, 1, t.shape[-1])
            self.buf[i] = t.gather(1, idx).squeeze(1).float()
            return None
        return hook

    def stack(self):
        return torch.stack(self.buf, 1)                   # [batch, layers, 8192]

    def close(self):
        for h in self.handles:
            h.remove()


def site_index(text, i_oc, offs, enc_off):
    """The Pythia token covering the construct token's first non-space character."""
    ch = offs[i_oc]
    while ch < len(text) and text[ch] in " \t":
        ch += 1
    for k, (a, b) in enumerate(enc_off):
        if a <= ch < b:
            return k
    return None


def main() -> None:
    from transformers import AutoModel, AutoTokenizer
    n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    n_constructs = int(sys.argv[2]) if len(sys.argv) > 2 else 10**6
    t0 = time.time()
    oc = AutoTokenizer.from_pretrained(REPO, revision=TOK_REV, trust_remote_code=True)
    py = AutoTokenizer.from_pretrained(PYTHIA, revision=PYTHIA_REV)
    # GPT-NeoX ships no pad token, and padding must be on the RIGHT: the site index is a
    # position in the unpadded sequence, so left padding would shift every read by the pad
    # length -- the same class of fault as the byte-fallback offset shift in B10 (iv).
    py.pad_token = py.eos_token
    py.padding_side = "right"
    model = AutoModel.from_pretrained(PYTHIA, revision=PYTHIA_REV,
                                      dtype=torch.float16).to("mps").eval()
    g = Gather(model)
    L, H = len(model.layers), model.config.intermediate_size
    K = int(round(CARRIER_FRACTION * L * H))
    # the population is the constructs that PASS part 1's gate: a construct with no stable read
    # site under this vocabulary has no twin unit worth measuring. PILOT is the fallback if the
    # gate artefact is not there yet.
    gf = OUT2 / "C2_PYTHIA_GATE.json"
    if gf.exists():
        sh = json.loads(gf.read_text())["shares_pythia"]
        names = sorted(c for c, v in sh.items() if len(v) == 4 and min(v) >= 0.80)
        src = f"the {len(names)} constructs passing part 1's gate"
    else:
        names, src = PILOT, "the 30 pilot constructs (gate artefact absent)"
    names = names[:n_constructs] if n_constructs < len(names) else names
    print(f"{PYTHIA} @ {PYTHIA_REV}: {L} layers x {H} units, carrier K={K:,}; "
          f"{len(names)} constructs ({src}) x 2 twins x 2 sides x {n_prompts} prompts",
          flush=True)

    def mean_profile(construct, twin, side):
        f = BATTERY / twin / f"{construct}_{side}.jsonl.gz"
        if not f.exists():
            return None, 0
        rows = [json.loads(l) for l in gzip.open(f, "rt")][:n_prompts]
        acc = np.zeros((L, H), np.float64)
        kept = 0
        for b in range(0, len(rows), 32):
            chunk = rows[b:b + 32]
            texts, sites = [], []
            for r in chunk:
                offs = piece_offsets(oc.tokenize(r["text"]))
                i = r.get("site_construct_token")
                if i is None or i >= len(offs):
                    continue
                e = py(r["text"], add_special_tokens=False, return_offsets_mapping=True)
                k = site_index(r["text"], i, offs, e["offset_mapping"])
                if k is None:
                    continue
                texts.append(r["text"])
                sites.append(k)
            if not texts:
                continue
            enc = py(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=512, add_special_tokens=False)
            g.sites = torch.tensor(sites, device="mps")
            with torch.no_grad():
                model(enc["input_ids"].to("mps"),
                      attention_mask=enc["attention_mask"].to("mps"))
            acc += g.stack().cpu().numpy().astype(np.float64).sum(0)
            kept += len(texts)
        return (acc / kept if kept else None), kept

    per = {}
    for c in names:
        t1 = time.time()
        pi = {}
        for twin in ("twin1", "twin2"):
            a, na = mean_profile(c, twin, "A")
            b, nb = mean_profile(c, twin, "B")
            if a is None or b is None:
                pi[twin] = None
                continue
            pi[twin] = ((a - b).ravel(), na, nb)
        if pi.get("twin1") is None or pi.get("twin2") is None:
            print(f"  {c:16s} SKIP (a side is missing)", flush=True)
            continue
        v1, n1a, n1b = pi["twin1"]
        v2, n2a, n2b = pi["twin2"]
        s1 = np.sort(np.argsort(-np.abs(v1))[:K])
        s2 = np.sort(np.argsort(-np.abs(v2))[:K])
        overlap = np.intersect1d(s1, s2).size / K
        cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        unit = np.abs(v1 - v2)
        scale = np.abs((v1 + v2) / 2)
        on = np.union1d(s1, s2)
        per[c] = {"carrier_overlap": float(overlap), "profile_cosine": cos,
                  "twin_unit_median": float(np.median(unit)),
                  "twin_unit_median_on_carriers": float(np.median(unit[on])),
                  "profile_scale_median_on_carriers": float(np.median(scale[on])),
                  "twin_unit_over_scale_on_carriers":
                      float(np.median(unit[on]) / max(np.median(scale[on]), 1e-12)),
                  "n_prompts": [n1a, n1b, n2a, n2b], "seconds": time.time() - t1}
        print(f"  {c:16s} overlap {overlap:.4f}  cosine {cos:.4f}  twin unit / scale "
              f"{per[c]['twin_unit_over_scale_on_carriers']:.3f}  ({time.time()-t1:.0f}s, "
              f"{(time.time()-t0)/60:.1f} min)", flush=True)
    g.close()

    def med(k):
        v = [e[k] for e in per.values()]
        return float(np.median(v)) if v else None

    out = {"what": "the twin unit on Pythia at one trained checkpoint, beside the gate count",
           "model": PYTHIA, "revision": PYTHIA_REV,
           "space": f"{L} layers x {H} MLP hidden units = {L*H:,}; carrier K={K:,} "
                    f"(top {CARRIER_FRACTION:.0%})",
           "site": "the Pythia token covering the construct token's first non-space character",
           "n_prompts_per_set": n_prompts, "population": src,
           "constructs": sorted(per),
           "per_construct": per,
           "median_carrier_overlap": med("carrier_overlap"),
           "median_profile_cosine": med("profile_cosine"),
           "median_twin_unit_over_scale_on_carriers": med("twin_unit_over_scale_on_carriers"),
           "opencoder_comparison": {
               "carrier_overlap_last_checkpoint": 0.8436,
               "source": "validation/paper2/B12_TWIN_TURNOVER.json, twin1 vs twin2 at "
                         "pretrain_iter_0484865, median over 169 constructs"},
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["phase_battery/twin{1,2}/<c>_{A,B}.jsonl.gz",
                                     f"{PYTHIA} @ {PYTHIA_REV}"])}
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "C2_PYTHIA_TWIN.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{len(per)} constructs: median carrier overlap {med('carrier_overlap'):.4f}, "
          f"profile cosine {med('profile_cosine'):.4f}, twin unit / scale "
          f"{med('twin_unit_over_scale_on_carriers'):.3f}")
    print(f"-> {OUT2 / 'C2_PYTHIA_TWIN.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
