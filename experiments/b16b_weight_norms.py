r"""B16, part b (cheap form) — layer 12's MLP weight norms across pretraining (design item 9b).

Part a found that **the release states no weight decay at all** -- no `weight_decay` in any
config of the checkpoints repo or of `infly/OpenCoder-1.5B-Base` -- so cause 5 cannot be gated
on the release. The cheap form of part b asks the question of the weights directly: if decay is
acting, the carriers' weight norms shrink over training while the loss is flat. One layer
answers it; the full 24 does not change the logic and costs 24 times as much.

**Range reads, not downloads.** A checkpoint is `pytorch_model.bin`, 3.81 GB -- a torch pickle,
not safetensors, so there is no safetensors header to read. But a torch `.bin` is a ZIP: its
central directory (176 members) and its 28 KB `data.pkl` are readable in a few seconds over the
network, and `data.pkl` gives each tensor's storage key, dtype and shape. Three members then
carry layer 12's `gate_proj`, `up_proj` and `down_proj`, about 75 MB a checkpoint against 3.81 GB
for the file -- 2.6 GB for all 35 rather than 133 GB.

Reported per checkpoint: the Frobenius norm of each of the three matrices, their row-norm
quantiles, and the same for a fixed random subset of rows as a scale reference; then the trend
against log steps, beside the obliged-position loss over the same span where it is available.

    python experiments/b16b_weight_norms.py [n_checkpoints]
"""
from __future__ import annotations

import io
import json
import pickle
import re
import sys
import time
import zipfile

import numpy as np

from experiments.atlas import CARRIERS, provenance
from experiments.relatedness import OUT2
from profilebattery.record.sweep import REPO

LAYER = 12
TARGETS = (f"model.layers.{LAYER}.mlp.gate_proj.weight",
           f"model.layers.{LAYER}.mlp.up_proj.weight",
           f"model.layers.{LAYER}.mlp.down_proj.weight")
#: torch storage class -> numpy dtype. Only the ones a released checkpoint can use.
DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": None,
      "DoubleStorage": np.float64, "LongStorage": np.int64}


class _Meta(pickle.Unpickler):
    """Reads a torch `data.pkl` without torch: every tensor's storage key, dtype and shape."""

    def persistent_load(self, pid):
        return pid                      # ('storage', storage_cls, key, location, numel)

    def find_class(self, module, name):
        if name == "OrderedDict":
            return dict
        if name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, *a):
                return {"storage": storage, "offset": offset, "size": tuple(size),
                        "stride": tuple(stride)}
            return rebuild
        if name.endswith("Storage"):
            class S:
                pass
            S.__name__ = name
            return S
        def anything(*a, **k):
            return None
        return anything


def _bf16_to_f32(buf: bytes) -> np.ndarray:
    """bfloat16 has no numpy dtype: widen each 2-byte value into the top half of a float32."""
    u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def read_layer(fs, rev: str) -> dict:
    path = f"{REPO}@{rev}/pytorch_model.bin"
    out = {}
    with fs.open(path, "rb") as f:
        z = zipfile.ZipFile(f)
        names = z.namelist()
        dp = next(n for n in names if n.endswith("data.pkl"))
        meta = _Meta(io.BytesIO(z.read(dp))).load()
        root = dp.rsplit("/", 1)[0]
        for name in TARGETS:
            t = meta.get(name)
            if t is None:
                out[name] = {"error": "not in data.pkl"}
                continue
            _, cls, key, _loc, numel = t["storage"]
            cname = getattr(cls, "__name__", str(cls))
            raw = z.read(f"{root}/data/{key}")
            if cname == "BFloat16Storage":
                arr = _bf16_to_f32(raw)
            else:
                dt = DT.get(cname)
                if dt is None:
                    out[name] = {"error": f"unhandled storage {cname}"}
                    continue
                arr = np.frombuffer(raw, dt).astype(np.float32)
            arr = arr[t["offset"]:t["offset"] + int(np.prod(t["size"]))]
            W = arr.reshape(t["size"])
            rn = np.linalg.norm(W, axis=1)
            out[name] = {"dtype": cname, "shape": list(t["size"]),
                         "frobenius": float(np.linalg.norm(W)),
                         "row_norm_median": float(np.median(rn)),
                         "row_norm_q1": float(np.quantile(rn, .25)),
                         "row_norm_q3": float(np.quantile(rn, .75)),
                         "mb_read": len(raw) / 1e6}
    return out


def main() -> None:
    from scipy.stats import spearmanr
    from huggingface_hub import HfFileSystem, get_token
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    if limit:
        refs = refs[:limit]
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    fs = HfFileSystem(token=get_token())
    print(f"layer {LAYER}, {len(refs)} checkpoints, 3 matrices each", flush=True)
    per, mb = {}, 0.0
    for r, it in zip(refs, iters):
        t1 = time.time()
        try:
            e = read_layer(fs, r)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  {r}: FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        per[r] = {"iter": it, "tensors": e}
        got = [v for v in e.values() if "frobenius" in v]
        mb += sum(v["mb_read"] for v in got)
        print(f"  {r[-7:]}  " + "  ".join(f"{k.split('.')[-2]} {v['frobenius']:.1f}"
                                          for k, v in e.items() if "frobenius" in v)
              + f"  ({time.time()-t1:.0f}s, {mb/1000:.2f} GB read)", flush=True)
    out = {"what": f"Frobenius and row norms of layer {LAYER}'s MLP weights across the "
                   f"pretraining checkpoints, by ZIP range reads of pytorch_model.bin",
           "why": "design item 9b, cheap form: part a found no weight decay in the release, so "
                  "cause 5 is tested on the weights directly",
           "checkpoint_format": "pytorch_model.bin, 3.81 GB, a torch ZIP -- not safetensors",
           "bytes_read_gb": mb / 1000,
           "bytes_if_downloaded_gb": 3.811 * len(per),
           "layer": LAYER, "tensors": list(TARGETS),
           "per_checkpoint": per}
    for name in TARGETS:
        ts = [per[r]["iter"] for r in per if "frobenius" in per[r]["tensors"].get(name, {})]
        vs = [per[r]["tensors"][name]["frobenius"] for r in per
              if "frobenius" in per[r]["tensors"].get(name, {})]
        if len(ts) > 2:
            s = spearmanr(np.log(ts), vs)
            out[name] = {"first": vs[0], "last": vs[-1],
                         "change_fraction": (vs[-1] - vs[0]) / vs[0] if vs[0] else None,
                         "spearman_vs_log_steps": float(s.statistic), "p": float(s.pvalue),
                         "n": len(ts)}
    falls = [out[n]["change_fraction"] for n in TARGETS
             if n in out and out[n]["change_fraction"] is not None]
    out["verdict"] = ("no matrix falls: decay is not visibly acting on these weights and "
                      "cause 5 is excluded for layer 12"
                      if falls and all(v >= -0.01 for v in falls) else
                      "at least one matrix falls over training: cause 5 is live and the full "
                      "form of 9b is warranted" if falls else "not enough checkpoints read")
    out["minutes"] = (time.time() - t0) / 60
    out["provenance"] = provenance([f"{REPO}@<ref>/pytorch_model.bin, ZIP members for layer "
                                    f"{LAYER}'s MLP"])
    (OUT2 / "B16B_WEIGHT_NORMS.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nread {mb/1000:.2f} GB instead of {3.811*len(per):.0f} GB")
    for n in TARGETS:
        if n in out:
            e = out[n]
            print(f"  {n.split('.')[-2]:10s} {e['first']:.2f} -> {e['last']:.2f} "
                  f"({e['change_fraction']:+.2%})  rho vs log steps {e['spearman_vs_log_steps']:+.3f} "
                  f"p={e['p']:.2g}")
    print(f"verdict: {out['verdict']}")
    print(f"-> {OUT2 / 'B16B_WEIGHT_NORMS.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
