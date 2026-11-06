r"""Item 3(b) -- how much of layer 12's weight movement decoupled decay could account for.

Item 3(a) could not be answered because OpenCoder states neither its optimiser's decay nor its
schedule: the paper gives eta = 3e-4, constant after a 2,000-step warm-up (arXiv:2411.04905v3
section 3.2), and nothing else. So this is 3(b): the comparison made **without** claiming a
setting, by naming a default and saying it is one.

**lambda = 0.1 is Megatron-LM's default weight decay. It is NOT a published OpenCoder setting,
and nothing here should be read as saying it is.** It is used because it is the value a reader
would assume if they assumed anything, which makes the comparison worth stating.

Under AdamW the decay is decoupled, so in the absence of any gradient a weight is multiplied by
(1 - eta*lambda) each step. Over N steps W -> (1 - eta*lambda)^N W, so

    dW = -(1 - (1 - eta*lambda)^N) W   and   ||dW|| / ||W|| = 1 - (1 - eta*lambda)^N

exactly -- the same quantity B16C measures as `relative_step`, which is why the two can be
divided. At eta = 3e-4 and lambda = 0.1 that is 1 - (1 - 3e-5)^1000 = 2.96% per 1,000 steps.

The measured side is B16C's ~1,000-step window on layer 12's gate_proj, up_proj and down_proj,
on each construct's carrier rows and on the random rows beside them. The ratio is taken **per
row with that row's own gap**, so a window that is 900 or 1,100 steps rather than exactly 1,000
is charged its own decay, and then pooled by median.

    python experiments/b16d_decay_term.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np

from experiments.atlas import provenance
from experiments.relatedness import OUT2

ETA = 3e-4            # OpenCoder's stated peak LR, constant after a 2,000-step warm-up
LAMBDA = 0.1          # Megatron-LM's DEFAULT weight decay -- not an OpenCoder setting
WINDOW = "1000"
MATRICES = ("gate_proj", "up_proj", "down_proj")


def shrink(n_steps: float, eta: float = ETA, lam: float = LAMBDA) -> float:
    """Relative size of the step decoupled decay alone would take over n_steps."""
    return float(1.0 - (1.0 - eta * lam) ** n_steps)


def main() -> None:
    t0 = time.time()
    src = json.loads((OUT2 / "B16C_WEIGHT_STEPS.json").read_text())
    rows = [r for r in src["rows"] if r["window"] == WINDOW]
    if not rows:
        raise SystemExit(f"no rows in window {WINDOW}")

    doc = {
        "what": "item 3(b): layer 12's measured weight step per 1,000 steps against the step "
                "decoupled weight decay alone would take, at a NAMED DEFAULT decay",
        "why_not_3a": "OpenCoder publishes eta = 3e-4 constant after a 2,000-step warm-up "
                      "(arXiv:2411.04905v3 s3.2) and states neither the optimiser's weight "
                      "decay nor any schedule, so no comparison can be made at OpenCoder's "
                      "own setting. 3(b) names a default instead of assuming one.",
        "definition": {
            "eta": ETA,
            "eta_source": "OpenCoder, arXiv:2411.04905v3 s3.2 -- 3e-4, constant after a "
                          "2,000-step warm-up",
            "lambda": LAMBDA,
            "lambda_source": "Megatron-LM's DEFAULT weight decay. NOT a published OpenCoder "
                             "setting and not presented as one.",
            "decay_step": "under AdamW the decay is decoupled, so with no gradient W -> "
                          "(1 - eta*lambda)^N W over N steps and ||dW||/||W|| = "
                          "1 - (1 - eta*lambda)^N exactly -- the same quantity B16C stores as "
                          "relative_step, which is what makes the ratio meaningful",
            "measured": f"B16C_WEIGHT_STEPS.json, layer {src['layer']}, window {WINDOW} "
                        f"(~1,000 steps of gap), relative_step on each construct's rows",
            "ratio": "measured relative_step / decay shrink, taken PER ROW with that row's own "
                     "gap and then pooled by median over constructs and start points",
        },
        "fixed_before_computing": True,
        "layer": src["layer"], "window": WINDOW,
        "decay_shrink_per_1000_steps": shrink(1000),
        "per_matrix": {},
    }

    print(f"  eta = {ETA:g}, lambda = {LAMBDA:g} (Megatron's DEFAULT, not OpenCoder's setting)")
    print(f"  decoupled decay alone moves a weight by "
          f"{doc['decay_shrink_per_1000_steps']:.4%} per 1,000 steps\n")
    print(f"  {'matrix':<11s} {'rows':<9s} {'n':>5s} {'median relative step':>21s} "
          f"{'ratio to decay':>15s}")
    agg = defaultdict(list)
    for mat in MATRICES:
        doc["per_matrix"][mat] = {}
        for kind in ("carriers", "random"):
            sel = [r for r in rows if r["matrix"] == mat and r["rows"] == kind]
            if not sel:
                continue
            step = np.array([r["relative_step"] for r in sel], float)
            gap = np.array([r["gap"] for r in sel], float)
            dec = np.array([shrink(g) for g in gap])
            ratio = step / dec
            if not np.all(np.isfinite(ratio)):
                raise SystemExit(f"{mat}/{kind}: non-finite ratio")
            starts = sorted({r["start_iter"] for r in sel})
            by_start = {int(s): float(np.median(
                [r["relative_step"] for r in sel if r["start_iter"] == s])) for s in starts}
            doc["per_matrix"][mat][kind] = {
                "n_rows": len(sel), "n_start_points": len(starts),
                "median_relative_step": float(np.median(step)),
                "median_decay_shrink": float(np.median(dec)),
                "median_ratio": float(np.median(ratio)),
                "ratio_iqr": [float(np.percentile(ratio, 25)),
                              float(np.percentile(ratio, 75))],
                "relative_step_at_first_start": by_start[min(by_start)],
                "relative_step_at_last_start": by_start[max(by_start)],
                "ratio_at_first_start": by_start[min(by_start)] / shrink(1000),
                "ratio_at_last_start": by_start[max(by_start)] / shrink(1000),
                "first_start_iter": min(by_start), "last_start_iter": max(by_start)}
            agg[kind].append(float(np.median(ratio)))
            e = doc["per_matrix"][mat][kind]
            print(f"  {mat:<11s} {kind:<9s} {len(sel):5d} {e['median_relative_step']:21.4f} "
                  f"{e['median_ratio']:14.1f}x")

    lo = min(agg["carriers"])
    doc["carrier_ratio_range"] = [min(agg["carriers"]), max(agg["carriers"])]
    doc["summary"] = (
        f"at lambda = {LAMBDA:g} and eta = {ETA:g}, decoupled decay alone would move a weight by "
        f"{doc['decay_shrink_per_1000_steps']:.2%} per 1,000 steps, while layer "
        f"{src['layer']}'s carrier rows move by {min(agg['carriers'])*shrink(1000):.1%} to "
        f"{max(agg['carriers'])*shrink(1000):.1%} over the same gap -- "
        f"{min(agg['carriers']):.1f} to {max(agg['carriers']):.1f} times as much. The decay "
        f"term is too small to be what the measured step is made of, at this default; "
        f"accounting for even the smallest of them would need lambda = "
        f"{LAMBDA * lo:.2f}, {lo:.1f}x Megatron's default. The carrier rows and the random "
        f"rows move alike, so this says nothing about the carriers in particular -- it bounds "
        f"the decay term for the layer as a whole.")
    doc["lambda_that_would_account_for_the_smallest"] = LAMBDA * lo
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(["validation/paper2/B16C_WEIGHT_STEPS.json"])
    (OUT2 / "B16D_DECAY_TERM.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"\n  {doc['summary']}")
    print(f"-> {OUT2 / 'B16D_DECAY_TERM.json'}")


if __name__ == "__main__":
    main()
