r"""9c — is the slowing the growth of the weights? (design item 9c)

9b found layer 12's MLP weights growing monotonically, +81% to +88% over pretraining, which
excluded weight decay. That growth suggests a different reading of the deceleration the rate
readings show: if the **absolute** step a weight takes per 1,000 steps is roughly constant while
the weights themselves get longer, then the **relative** step falls for arithmetic reasons, and
so would anything measured on the direction of the profile.

Per construct, layer 12's carrier rows -- `gate_proj` and `up_proj` are (6144, 2240) so a hidden
unit is a ROW; `down_proj` is (2240, 6144) so a hidden unit is a COLUMN, and it is transposed
before the same statistic is taken. For each pair (t, t+D):

    relative step = || W_rows(t+D) - W_rows(t) || / || W_rows(t) ||
    absolute step = || W_rows(t+D) - W_rows(t) ||

at the **19 true 1,000-step dense pairs** and in the **20k-30k window**, against the start
point, and the same for a size-matched random row set drawn once per construct.

Reading, fixed before the result: a falling relative step with a roughly constant absolute step
means the slowing is the growth of the weights; both falling means the weights themselves are
moving less.

    python experiments/b16c_weight_steps.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.b16b_weight_norms import LAYER, TARGETS, read_layer
from experiments.atlas import CARRIERS, D, provenance
from experiments.relatedness import OUT2

WCACHE = D / "validation/paper2/layer12_weights"
WIDTH = 6144


def weights(fs, ref):
    """Layer 12's three matrices at one reference, cached locally after the first read."""
    p = WCACHE / f"{ref}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    import io, pickle, zipfile                                      # noqa: E401
    from experiments.b16b_weight_norms import _Meta, _bf16_to_f32, DT
    out = {}
    with fs.open(f"{__import__('profilebattery.record.sweep', fromlist=['REPO']).REPO}"
                 f"@{ref}/pytorch_model.bin", "rb") as f:
        z = zipfile.ZipFile(f)
        dp = next(n for n in z.namelist() if n.endswith("data.pkl"))
        meta = _Meta(io.BytesIO(z.read(dp))).load()
        root = dp.rsplit("/", 1)[0]
        for name in TARGETS:
            t = meta[name]
            _, cls, key, _loc, _n = t["storage"]
            raw = z.read(f"{root}/data/{key}")
            cname = getattr(cls, "__name__", str(cls))
            arr = (_bf16_to_f32(raw) if cname == "BFloat16Storage"
                   else np.frombuffer(raw, DT[cname]).astype(np.float32))
            W = arr[t["offset"]:t["offset"] + int(np.prod(t["size"]))].reshape(t["size"])
            # down_proj is (hidden_out, 6144): transpose so a hidden unit is always a row
            out[name.split(".")[-2]] = (W.T if W.shape[0] != WIDTH else W).copy()
    WCACHE.mkdir(parents=True, exist_ok=True)
    tmp = WCACHE / f".{ref}.tmp.npz"
    np.savez(tmp, **out)
    tmp.replace(p)
    return out


def main() -> None:
    from huggingface_hub import HfFileSystem, get_token
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    fs = HfFileSystem(token=get_token())
    rng = np.random.default_rng(20261023)
    lo, hi = LAYER * WIDTH, (LAYER + 1) * WIDTH
    pairs = [(i, j) for i in range(len(refs)) for j in range(i + 1, len(refs))
             if (iters[j] - iters[i] == 1000) or (20_000 <= iters[j] - iters[i] <= 30_000)]
    print(f"{len(refs)} references, {len(pairs)} pairs "
          f"({sum(1 for i, j in pairs if iters[j]-iters[i] == 1000)} dense)", flush=True)

    rows = []
    for n, (i, j) in enumerate(pairs, 1):
        Wi, Wj = weights(fs, refs[i]), weights(fs, refs[j])
        for c in names:
            key = f"twin1/{c}|product"
            if key not in idx[refs[i]]:
                continue
            p = np.asarray(idx[refs[i]][key], np.int64)
            u = p[(p >= lo) & (p < hi)] - lo
            if u.size < 8:
                continue
            rnd = rng.choice(WIDTH, u.size, replace=False)
            for mat in ("gate_proj", "up_proj", "down_proj"):
                A, B = Wi[mat], Wj[mat]
                for lab, sel in (("carriers", u), ("random", rnd)):
                    a, b = A[sel].astype(np.float64), B[sel].astype(np.float64)
                    absd = float(np.linalg.norm(b - a))
                    base = float(np.linalg.norm(a))
                    rows.append({"start_iter": iters[i], "gap": iters[j] - iters[i],
                                 "window": "1000" if iters[j] - iters[i] == 1000 else "20k-30k",
                                 "construct": c, "matrix": mat, "rows": lab,
                                 "n_rows": int(u.size),
                                 "absolute_step": absd, "base_norm": base,
                                 "relative_step": absd / base if base else None})
        print(f"  [{n}/{len(pairs)}] {refs[i][-7:]}->{refs[j][-7:]} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    def agg(window, mat, which, field):
        by = {}
        for r in rows:
            if r["window"] != window or r["matrix"] != mat or r["rows"] != which:
                continue
            by.setdefault(r["start_iter"], []).append(r[field])
        ts = sorted(by)
        return ts, [float(np.median(by[t])) for t in ts]

    summary = {}
    for window in ("1000", "20k-30k"):
        for mat in ("gate_proj", "up_proj", "down_proj"):
            for which in ("carriers", "random"):
                for field in ("relative_step", "absolute_step"):
                    ts, v = agg(window, mat, which, field)
                    if len(ts) < 4:
                        continue
                    rho = float(np.corrcoef(np.log(ts), v)[0, 1])
                    summary[f"{window}|{mat}|{which}|{field}"] = {
                        "n_start_points": len(ts), "first": v[0], "last": v[-1],
                        "change_fraction": (v[-1] - v[0]) / v[0] if v[0] else None,
                        "spearman_vs_log_start": rho,
                        "direction": ("falling" if rho < -0.3 else
                                      "rising" if rho > 0.3 else "flat")}
    out = {"what": "layer 12's MLP weight step on each construct's carrier rows, absolute and "
                   "relative, against the start point",
           "reading": "a falling RELATIVE step with a roughly constant ABSOLUTE step means the "
                      "slowing is the growth of the weights; both falling means the weights "
                      "are themselves moving less",
           "layer": LAYER, "n_pairs": len(pairs), "n_rows": len(rows),
           "down_proj_note": "transposed so a hidden unit is a row in all three matrices",
           "summary": summary, "rows": rows,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance([f"OpenCoder checkpoints @ <ref>/pytorch_model.bin, "
                                     f"layer {LAYER} MLP", "validation/carriers/*.npz"])}
    (OUT2 / "B16C_WEIGHT_STEPS.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nkey rows:")
    for k in sorted(summary):
        if "gate_proj" in k:
            e = summary[k]
            print(f"  {k:46s} {e['first']:.5f} -> {e['last']:.5f} "
                  f"({e['change_fraction']:+.1%}) rho {e['spearman_vs_log_start']:+.3f} "
                  f"{e['direction']}")
    print(f"-> {OUT2 / 'B16C_WEIGHT_STEPS.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
