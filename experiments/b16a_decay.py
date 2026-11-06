r"""B16, part a — OpenCoder's weight-decay setting (design item 9a).

§6.2's cause 5 is mechanical and must be excluded before any of the others is credited: with
decay and a constant learning rate, weights move even where the gradient is zero. Part a asks
only what the release says, at no compute cost, because it gates part b: if decay is zero the
cause closes and the ~100 GB of checkpoint weights part b needs are never fetched.

Everything the release exposes is searched -- the checkpoints repo's own files, the base model's
files, and any training configuration among them. A setting that is NOT in the release is
reported as absent rather than guessed: that is itself the answer that 9b cannot be gated on the
release and would need the paper or the cookbook.

    python experiments/b16a_decay.py
"""
from __future__ import annotations

import json
import time

from experiments.atlas import provenance
from experiments.relatedness import OUT2
from profilebattery.record.sweep import REPO

BASE = "infly/OpenCoder-1.5B-Base"
KEYS = ("weight_decay", "adam_weight_decay", "wd", "decay", "optimizer", "adam_beta1",
        "adam_beta2", "adam_epsilon", "learning_rate", "lr", "max_grad_norm")


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    t0 = time.time()
    api = HfApi()
    found, listings, errors = {}, {}, {}
    for repo, kind, rev in ((REPO, "model", "pretrain_iter_0008000"),
                            (BASE, "model", None)):
        try:
            fs = api.list_repo_files(repo, repo_type=kind, revision=rev)
            listings[f"{repo}@{rev or 'main'}"] = sorted(fs)
        except Exception as exc:                                    # noqa: BLE001
            errors[repo] = f"{type(exc).__name__}: {exc}"
            continue
        for f in fs:
            if not (f.endswith(".json") or f.endswith(".yaml") or f.endswith(".yml")
                    or "config" in f.lower() or "args" in f.lower()):
                continue
            try:
                p = hf_hub_download(repo, f, repo_type=kind, revision=rev)
                txt = open(p, encoding="utf-8", errors="replace").read()
            except Exception:                                       # noqa: BLE001
                continue
            try:
                d = json.loads(txt)
            except Exception:                                       # noqa: BLE001
                d = None
            hits = {}
            if isinstance(d, dict):
                for k in KEYS:
                    if k in d:
                        hits[k] = d[k]
            else:
                for k in KEYS:
                    if k in txt:
                        hits[k] = "present in a non-JSON file; see the file"
            if hits:
                found[f"{repo}:{f}"] = hits
    decay = None
    for src, hits in found.items():
        for k in ("weight_decay", "adam_weight_decay", "wd"):
            if k in hits and isinstance(hits[k], (int, float)):
                decay = {"value": hits[k], "key": k, "source": src}
    out = {"what": "OpenCoder's weight-decay setting as the release exposes it",
           "why": "§6.2 cause 5 is mechanical and gates design item 9b (~100 GB of weights)",
           "repos_searched": [f"{REPO}@pretrain_iter_0008000", BASE],
           "files_listed": listings, "errors": errors,
           "hyperparameters_found": found,
           "weight_decay": decay,
           "verdict": ("the release does not state a weight decay; 9b cannot be gated on it "
                       "and would need the OpenCoder paper or cookbook"
                       if decay is None else
                       ("decay is zero: cause 5 is excluded and 9b does not run"
                        if decay["value"] == 0 else
                        "decay is non-zero: cause 5 is live and 9b is warranted")),
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["the model repos' own config files"])}
    (OUT2 / "B16A_DECAY.json").write_text(json.dumps(out, indent=1, default=float))
    print("hyperparameters found in the release:")
    for s, h in found.items():
        print(f"  {s}: {h}")
    if not found:
        print("  none")
    print(f"\nweight_decay: {decay}")
    print(f"verdict: {out['verdict']}")
    print(f"-> {OUT2 / 'B16A_DECAY.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
