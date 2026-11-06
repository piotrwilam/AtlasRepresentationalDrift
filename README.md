# AtlasRepresentationalDrift — v1.0

The scripts of **"Representational Drift in a Code Model: Behaviour Settles, Which Neurons
Carry It Does Not"** (Wilam, 2026; arXiv:[id]).

The instrument they use is described in **"Calibrated Concept Profiles: Making the Reading of
a Model's Neurons a Measurement"** (Wilam, 2026; arXiv:[id]), whose code is
`AtlasCalibratedConceptProfiles`.

These are the scripts for one paper, not a reusable tool. They read the dataset and the
artefacts, and produce the paper's four figures and its tables. If you want the instrument
rather than this paper's analysis, use `AtlasCalibratedConceptProfiles`.

## The dataset this reads

Every script here reads **`AtlasOpenCoder` at tag `v1`** — the 20-cell yardstick. Point
`$ATLAS_DATA` at a checkout of that tag:

```
hf download piotrwilam/AtlasOpenCoder --repo-type dataset --revision v1 \
    --local-dir ./AtlasOpenCoder
export ATLAS_DATA=./AtlasOpenCoder
```

A later tag is a different yardstick and names different neurons for the same
constructs, so the numbers in the paper belong to `v1` and only to `v1`.

This repository is code only: every artefact it writes and reads lives in
the dataset above.

## Install

```sh
pip install -e ../AtlasCalibratedConceptProfiles      # the tool, v1.0
export PYTHONPATH=.:../AtlasCalibratedConceptProfiles  # this paper's scripts import
                                                       # five helpers from the tool's
                                                       # experiments/, which pip does
                                                       # not put on the path
export ATLAS_DATA=/path/to/AtlasOpenCoder             # the dataset, v1.0
python experiments/paper2_figures.py
```

This repository names the versions it used: **tool v1.0, dataset v1.0 (`AtlasOpenCoder`,
OpenCoder-1.5B).** It imports from the tool and from nothing in the method paper's own scripts.

## The scripts

50 scripts in `experiments/`. They read the dataset and write the artefacts this paper's
manifest names; `REBUILD.md` in the dataset says which run from the release alone.

| | |
|---|---|
| figures | `paper2_figures.py` |
| the rate of change, the spacing model | `rate_readings.py`, `b10_fit.py`, `b3_allpairs.py`, `b3_graded.py`, `curve1_spacing.py` |
| the settling rule and the dates | `dates_table.py`, `r2_report.py`, `r2_curves.py`, `r2_curve4.py`, `r2_close.py`, `r2_owed.py`, `r0b_gates.py` |
| the position classes and the loss legs | `step3_senses.py`, `step4_flat.py`, `step5_frequency.py`, `step6_token.py`, `step7.py`, `step8_by_layer.py`, `step8_graph.py`, `raw_norm_reread.py`, `a34_agreement.py` |
| the power of the null | `b11_power.py`, `p8_bootstrap.py`, `p8_cheap.py` |
| ceilings and bands | `ceilings.py`, `ceilings_pairwise.py`, `b14_bands.py`, `tbc22_shares.py`, `b12_twin_turnover.py` |
| departure, return, recurrence | `b17_return.py`, `b17b_return_split.py`, `b18_recurrence.py`, `test6.py`, `test6_extras.py` |
| the weights | `b16a_decay.py`, `b16b_weight_norms.py`, `b16c_steps_vs_lr.py`, `b16c_weight_steps.py`, `b16d_decay_term.py` |
| the ablation and retention | `ablation_splits.py`, `retention_table.py`, `b5_churn.py`, `b6_perlayer.py` |
| entropy, annealing, co-occurrence, tokens | `b1_entropy.py`, `b2_anneal_drop.py`, `cooccurrence.py`, `core_tokens.py`, `b15_corecheck.py` |
| the second model | `c2_pythia_gate.py`, `c2_pythia_twin.py` |
| the stored summaries | `backfill_summaries.py` |

## Licence

Apache-2.0 for this code. The dataset is CC BY 4.0 for its own content; its padding
snippets keep their original licences, listed in its own `ATTRIBUTION.md`.
