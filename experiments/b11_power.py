r"""B11 — the power of the null: what settling date would have been found?

Paper 2 reports "0 of 169 constructs settle in pretraining". That is only informative if the
rule could have found a settling that was there. This injects synthetic settling curves into the
**measured** noise at five dates and reports the detection rate by date, so the null reads as
"settling before X tokens would have been found".

  dates        84 B, 200 B, 500 B, 1 T, 1.5 T tokens
  rule         the corrected one: the retention ratio crosses theta = 0.70 and stays, with at
               least MIN_FOLLOWING points after the crossing, at least MIN_BELOW_BEFORE below
               it before, and agreement across two spacing bands (A31, A38)
  noise        whole ordered residual **series** from another construct, circularly shifted at
               random -- not point by point. The rule is "crosses and STAYS", and with
               autocorrelated residuals an excursion below theta lasts longer, so staying is
               harder; i.i.d. draws would have overstated the power.
  shapes       a step to the plateau, and an exponential approach with time constant one x1.5
               spacing -- the easiest and the slowest-resolvable shapes
  control      the same rule on the uninjected series, as the false-detection rate

    python experiments/b11_power.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, K, provenance
from experiments.relatedness import OUT2

DATES_TOKENS = (84e9, 200e9, 500e9, 1000e9, 1500e9)
THETA = 0.70
MIN_FOLLOWING = 2
MIN_BELOW_BEFORE = 1
N_DRAWS = 200


def detect(series: np.ndarray, theta: float = THETA) -> int | None:
    """The corrected rule: the first index that crosses theta and stays, with support."""
    n = series.size
    for i in range(n):
        if series[i] < theta:
            continue
        if not np.all(series[i:] >= theta):
            continue
        if (n - 1 - i) < MIN_FOLLOWING:
            return None
        if (series[:i] < theta).sum() < MIN_BELOW_BEFORE:
            return None
        return i
    return None


def main() -> None:
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = np.array([int(re.search(r"(\d+)$", r).group(1)) for r in refs], float)
    toks = iters * (2034e9 / iters[-1])          # tokens, linear in steps on this schedule
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    rng = np.random.default_rng(20261022)

    # the measured ratio series per construct: overlap with the previous reference / twin ceiling
    series = {}
    for c in names:
        vals = []
        for i in range(1, len(refs)):
            a, b = refs[i - 1], refs[i]
            k1 = f"twin1/{c}|product"
            k2 = f"twin2/{c}|product"
            if k1 not in idx[a] or k1 not in idx[b] or k2 not in idx[b]:
                vals.append(np.nan); continue
            ov = np.intersect1d(np.asarray(idx[a][k1], np.int64),
                                np.asarray(idx[b][k1], np.int64)).size / K
            ceil = np.intersect1d(np.asarray(idx[b][k1], np.int64),
                                  np.asarray(idx[b][k2], np.int64)).size / K
            vals.append(ov / ceil if ceil > 0 else np.nan)
        v = np.array(vals, float)
        if np.isfinite(v).sum() >= 8:
            series[c] = v
    print(f"{len(series)} constructs with a usable ratio series over {len(refs)-1} steps",
          flush=True)

    # each construct's own noise: residuals from a smooth (3-point median) version of its series
    noise = {}
    for c, v in series.items():
        f = np.copy(v)
        sm = np.copy(f)
        for i in range(1, len(f) - 1):
            w = f[i - 1:i + 2]
            if np.isfinite(w).all():
                sm[i] = np.median(w)
        r = f - sm
        noise[c] = r[np.isfinite(r)]

    tok_mid = toks[1:]
    PLATEAU = THETA + 0.05
    cnames = [c for c in series if noise[c].size >= 6]

    def draw_noise(n):
        """A whole residual series from another construct, circularly shifted."""
        r = noise[cnames[rng.integers(0, len(cnames))]]
        r = np.resize(r, n)
        return np.roll(r, int(rng.integers(0, n)))

    def shape(kind, date, base):
        if kind == "step":
            return np.where(tok_mid < date, base, PLATEAU)
        tau = date * 0.5                       # one x1.5 spacing at that date
        y = base + (PLATEAU - base) * (1 - np.exp(-(tok_mid - date) / tau))
        return np.where(tok_mid < date, base, y)

    out = {}
    for kind in ("step", "exponential"):
        for date in DATES_TOKENS:
            hits = total = 0; found_at = []
            for c in cnames:
                base = float(np.nanmin(series[c]))
                for _ in range(max(1, N_DRAWS // len(cnames) + 1)):
                    s_ = shape(kind, date, base) + draw_noise(tok_mid.size)
                    total += 1
                    i = detect(s_)
                    if i is not None:
                        hits += 1
                        found_at.append(float(tok_mid[i]))
            out[f"{kind}|{date/1e9:.0f}B"] = {
                "shape": kind, "tokens": date, "draws": total, "detections": hits,
                "detection_rate": hits / total if total else None,
                "median_date_found": float(np.median(found_at)) if found_at else None}
            e = out[f"{kind}|{date/1e9:.0f}B"]
            print(f"  {kind:11s} injected at {date/1e9:6.0f} B: detected in "
                  f"{e['detection_rate']:.1%} of {total:,} draws"
                  + (f", median found {e['median_date_found']/1e9:.0f} B"
                     if e['median_date_found'] else ""), flush=True)
    # the control: no settling at all, the same noise
    hits = total = 0
    for c in cnames:
        base = float(np.nanmin(series[c]))
        for _ in range(max(1, N_DRAWS // len(cnames) + 1)):
            total += 1
            if detect(np.full(tok_mid.size, base) + draw_noise(tok_mid.size)) is not None:
                hits += 1
    out["false_detection"] = {"draws": total, "detections": hits,
                              "rate": hits / total if total else None,
                              "what": "the same rule on an uninjected series"}
    print(f"  false detection, no settling injected: {out['false_detection']['rate']:.1%} "
          f"of {total:,} draws", flush=True)
    doc = {"what": "detection rate of an injected settling date under the corrected rule",
           "rule": {"theta": THETA, "min_following": MIN_FOLLOWING,
                    "min_below_before": MIN_BELOW_BEFORE,
                    "crossing": "crosses theta and stays"},
           "noise": "whole ordered residual series from another construct, circularly "
                    "shifted; autocorrelation kept",
           "shapes": ["step", "exponential with tau = one x1.5 spacing"],
           "dates": out, "n_constructs": len(series),
           "reading": "a date whose detection rate is high means '0 of 169' excludes settling "
                      "before it; a low rate means the null is uninformative there",
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["validation/carriers/*.npz"])}
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "B11_POWER.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"-> {OUT2 / 'B11_POWER.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
