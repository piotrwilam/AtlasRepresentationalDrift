r"""A1.4 — the per-parameter step against what the optimiser's own step size would give.

**The optimiser's form, as far as the record states it.** Nothing names the update rule:
`B16D_DECAY_TERM.json` records that the two released `config.json` files, the model card, the
training repository and the full text of arXiv:2411.04905v3 contain **one** optimiser word
between them, and it is "distributed optimization" -- Megatron-LM's optimiser *sharding*. What
IS stated, in section 3.2, is the schedule: a WSD schedule with a 2,000-step warm-up over 8 B
tokens, and **"The peak learning rate was 3e-4, which remained constant after the warm-up"**.

**Why the learning rate is the right yardstick even so.** Megatron-LM's default is AdamW, and
for any Adam-family rule the update is the gradient divided by its own running RMS, so a single
step moves a parameter by about **eta**, almost regardless of the gradient's size. That makes
eta a natural unit: over D steps, a parameter whose gradient points the same way throughout
moves about **eta*D**, and one whose gradients cancel moves about **eta*sqrt(D)** -- a random
walk. The measured displacement between two checkpoints therefore reads as a statement about
**gradient consistency**, not about step size:

    consistency = ||W(t+D) - W(t)||_RMS / (eta * D)          1 = fully consistent
                                                             ~1/sqrt(D) = a random walk

Reported for layer 12's three MLP matrices, over the dense block's true 1,000-step pairs and the
20,000-30,000 step window, against both reference lines. Local only: the weights are cached.

    python experiments/b16c_steps_vs_lr.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

import numpy as np

from experiments.atlas import D, provenance
from experiments.relatedness import OUT2

WCACHE = D / "validation/paper2/layer12_weights"
ETA = 3e-4          # arXiv:2411.04905v3 §3.2, constant after the 2,000-step warm-up
MATS = ("gate_proj", "up_proj", "down_proj")


def main() -> None:
    t0 = time.time()
    files = {int(re.search(r"(\d+)", os.path.basename(f)).group(1)): f
             for f in glob.glob(str(WCACHE / "*.npz"))}
    iters = sorted(files)
    print(f"{len(iters)} cached references, layer 12, eta = {ETA}", flush=True)
    W = {}

    def get(i):
        if i not in W:
            z = np.load(files[i])
            W[i] = {m: np.asarray(z[m], np.float64) for m in MATS}
            while len(W) > 6:
                W.pop(next(k for k in W if k != i))
        return W[i]

    rows = []
    for a in iters:
        for b in iters:
            if b <= a:
                continue
            Dsteps = b - a
            if not (Dsteps == 1000 or 20_000 <= Dsteps <= 30_000):
                continue
            Wa, Wb = get(a), get(b)
            for m in MATS:
                d = Wb[m] - Wa[m]
                n = d.size
                rms = float(np.sqrt((d * d).sum() / n))
                rows.append({
                    "start_iter": a, "gap": Dsteps,
                    "window": "1000" if Dsteps == 1000 else "20k-30k",
                    "matrix": m, "n_params": int(n),
                    "rms_step": rms,
                    "base_rms": float(np.sqrt((Wa[m] ** 2).sum() / n)),
                    "if_fully_consistent": ETA * Dsteps,
                    "if_random_walk": ETA * np.sqrt(Dsteps),
                    "consistency": rms / (ETA * Dsteps),
                    "over_random_walk": rms / (ETA * np.sqrt(Dsteps))})
        if a % 8000 == 0:
            print(f"  through iteration {a} ({(time.time()-t0)/60:.1f} min)", flush=True)

    out = {"what": "layer 12's per-parameter step against the learning rate's own scale",
           "optimiser": {
               "update_rule": "NOT STATED anywhere in the release, the training repository or "
                              "the paper (B16D_DECAY_TERM.json)",
               "framework": "Megatron-LM, whose default is AdamW",
               "learning_rate": ETA,
               "learning_rate_source": "arXiv:2411.04905v3 section 3.2: 'The peak learning "
                                       "rate was 3e-4, which remained constant after the "
                                       "warm-up and subsequently decayed exponentially to 1e-5 "
                                       "during the annealing phase.'",
               "why_eta_is_the_yardstick": "for any Adam-family rule the update is the gradient "
                                           "over its own running RMS, so one step moves a "
                                           "parameter by about eta whatever the gradient's size"},
           "reference_lines": {"fully_consistent": "eta * D", "random_walk": "eta * sqrt(D)"},
           "n_rows": len(rows), "per_window": {}}
    for w in ("1000", "20k-30k"):
        for m in MATS:
            sub = [r for r in rows if r["window"] == w and r["matrix"] == m]
            if not sub:
                continue
            ts = sorted({r["start_iter"] for r in sub})
            con = [float(np.median([r["consistency"] for r in sub if r["start_iter"] == t]))
                   for t in ts]
            orw = [float(np.median([r["over_random_walk"] for r in sub if r["start_iter"] == t]))
                   for t in ts]
            rho = float(np.corrcoef(np.log(ts), con)[0, 1]) if len(ts) > 2 else float("nan")
            out["per_window"][f"{w}|{m}"] = {
                "n_start_points": len(ts), "start_iters": ts,
                "consistency_first": con[0], "consistency_last": con[-1],
                "consistency_median": float(np.median(con)),
                "over_random_walk_first": orw[0], "over_random_walk_last": orw[-1],
                "over_random_walk_median": float(np.median(orw)),
                "consistency_vs_log_start_corr": rho,
                "direction": ("falling" if rho < -0.3 else "rising" if rho > 0.3 else "flat")}
    out["rows"] = rows
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance(["validation/paper2/layer12_weights/*.npz (cached from the "
                                    "checkpoints' pytorch_model.bin, layer 12 MLP)",
                                    "arXiv:2411.04905v3 section 3.2 for eta"])
    (OUT2 / "B16C_STEPS_VS_LR.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{'window|matrix':22s} {'n':>3s} {'consistency':>22s} {'vs random walk':>18s} "
          f"{'trend':>8s}")
    for k, e in out["per_window"].items():
        print(f"  {k:20s} {e['n_start_points']:3d} "
              f"{e['consistency_first']:.4f} -> {e['consistency_last']:.4f} "
              f"{e['over_random_walk_first']:8.2f} -> {e['over_random_walk_last']:5.2f} "
              f"{e['direction']:>9s}")
    print(f"-> {OUT2 / 'B16C_STEPS_VS_LR.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
