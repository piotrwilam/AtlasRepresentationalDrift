r"""Paper 2, test 6 — the directionality of departures, with the relatedness column.

For each neuron that leaves a construct's **carrier set** between adjacent checkpoints
(amendment of 21 Oct: the carrier set, the top 5%, not the core), the largest rise among the
other 168 constructs' entries for that neuron. The question is what the neuron goes to:

  drift          no rise beyond chance in any construct
  displacement   a rise, in any construct
  branching      a rise concentrated in constructs **related** to the one left

Relatedness is the three measures of `experiments/relatedness.py`, fixed before this ran and
reported separately with the number of related pairs beside every rate.

**The chance level is matched** (amendment of 21 Oct) on two things, not one: the comparison
construct has a similar corpus frequency to the one that actually rose, and a similar
carrier-set overlap with the construct left, measured at the earlier checkpoint. Matching on
frequency alone would let a construct that shares many carriers with j stand in for one that
shares few, and overlap is the stronger confound -- a neuron already near two constructs' sets
will look 'related' under any measure that tracks sharing.

    python experiments/test6.py --pairs 0001000:0002000 0002000:0003000 0019000:0020000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.atlas import D, K, provenance
from experiments.relatedness import OUT2, Relatedness

MEASURES = ("family", "tokens", "cooccurrence",
            "tokens_at_read_site", "tokens_in_cores_only")


def entries(ref: str, names, twin: str = "twin1") -> np.ndarray:
    """(n_constructs, 147456) concept-profile entries at one reference, for one twin.

    **Cached per construct, not per reference.** The first version of this cache wrote one file
    once all 169 reads had succeeded, so a single read exhausting its retries threw the whole
    reference away: on a link that drops DNS every few minutes the pass ran for two hours and
    cached nothing. Progress is now monotone -- every file that lands is kept, and a restart
    reads only what is missing.
    """
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from experiments.atlas import D, _tree
    cdir = D / "validation/paper2/entries_cache" / f"{ref}__{twin}"
    cdir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256("|".join(names).encode()).hexdigest()[:12]
    whole = cdir.parent / f"{ref}__{twin}__{len(names)}_{tag}.npy"
    if whole.exists():
        return np.load(whole)
    out = np.empty((len(names), 24 * 6144), np.float32)
    missing = []
    for i, c in enumerate(names):
        f = cdir / f"{c}.npy"
        if f.exists():
            try:
                out[i] = np.load(f)
                continue
            except Exception:                                       # noqa: BLE001
                f.unlink(missing_ok=True)                           # truncated: refetch
        missing.append((i, c))
    if missing:
        print(f"    {ref} {twin}: {len(names)-len(missing)} cached, {len(missing)} to fetch",
              flush=True)

    def one(i_c):
        i, c = i_c
        last = None
        # 20 attempts, backoff capped at 30 s: about ten minutes a file. The outages here last
        # minutes, so five attempts over 30 s were exhausted routinely.
        for attempt in range(20):
            try:
                # ONE member, not three. Measured on the wire: a member read costs ~11.4 MB
                # however small the member is (`concept` is 0.6 MB of data), because the Xet
                # CDN serves whole dedup chunks -- so asking for concept+A_mean+B_mean cost
                # 22.3 MB a file and asking for concept alone costs 11.4 MB. The whole file is
                # 49.6 MB. `concept` is present in every profile of these references (B14 read
                # it from 11,830 of them), and the difference is fetched only if it is absent.
                z = _tree(ref, twin, c, ("concept",))
                v = z.get("concept")
                if v is None:
                    z = _tree(ref, twin, c, ("A_mean", "B_mean"))
                    v = z["A_mean"] - z["B_mean"]
                arr = np.asarray(v, np.float32).ravel()
                out[i] = arr
                # the temp name MUST end in .npy: np.save appends the suffix otherwise,
                # and the rename then targets a file that was never written
                t = cdir / f".{c}.tmp.npy"
                np.save(t, arr)
                t.replace(cdir / f"{c}.npy")
                return
            except Exception as exc:                                # noqa: BLE001
                last = exc
                time.sleep(min(30, 2 * (attempt + 1)))
        raise last

    if missing:
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, missing))
    t = whole.parent / f".{whole.stem}.tmp.npy"
    np.save(t, out)
    t.replace(whole)
    for f in cdir.glob("*.npy"):
        f.unlink(missing_ok=True)          # the whole-matrix file supersedes the per-construct
    return out


def carriers_of(mat: np.ndarray) -> list[np.ndarray]:
    return [np.sort(np.argsort(-(row.astype(np.float64) ** 2), kind="stable")[:K])
            for row in mat]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*",
                    default=["0001000:0002000", "0002000:0003000", "0019000:0020000"])
    ap.add_argument("--max-departures", type=int, default=40000,
                    help="per construct per pair, sampled if exceeded")
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    from experiments.atlas import p1_set
    from profilebattery.profiles.names import contract_name
    names = sorted(p1_set())
    R = Relatedness()
    rates = json.loads((D / "calibration/construct_rates.json").read_text())
    brates = json.loads((D / "calibration/builtin_rates.json").read_text())["rates"]
    allr = {**rates, **brates}
    freq = np.array([np.log10(max((allr.get(contract_name(c)) or {}).get("per_file") or 1e-6,
                                  1e-6)) for c in names])
    rng = np.random.default_rng(args.seed)
    out, t0 = {}, time.time()
    for spec in args.pairs:
        a, b = spec.split(":")
        ra, rb = f"pretrain_iter_{a}", f"pretrain_iter_{b}"
        Ea, Eb = entries(ra, names), entries(rb, names)
        rise = Eb - Ea
        # cross-twin: the departure is twin 1's, the rise twin 2's. A lift that survives the
        # twins is not an artefact of one battery draw.
        Ea2, Eb2 = entries(ra, names, "twin2"), entries(rb, names, "twin2")
        rise2 = Eb2 - Ea2
        car_a, car_b = carriers_of(Ea), carriers_of(Eb)
        setb = [set(x.tolist()) for x in car_b]
        ov = np.zeros((len(names), len(names)), np.float32)
        for i in range(len(names)):
            si = set(car_a[i].tolist())
            for j in range(i + 1, len(names)):
                v = len(si & set(car_a[j].tolist())) / K
                ov[i, j] = ov[j, i] = v
        tally = {m: {"observed": 0, "chance": 0, "n": 0} for m in MEASURES}
        tally_x = {m: {"observed": 0, "chance": 0, "n": 0} for m in ("family", "tokens")}
        per_construct_x = {m: {} for m in ("family", "tokens")}
        disp = {"observed": 0, "chance": 0, "n": 0}
        disp_pc = {}
        per_construct = {m: {} for m in MEASURES}      # for the bootstrap over constructs
        n_dep = 0
        for i, c in enumerate(names):
            dep = np.array([x for x in car_a[i] if x not in setb[i]], np.int64)
            if dep.size == 0:
                continue
            if dep.size > args.max_departures:
                dep = rng.choice(dep, args.max_departures, replace=False)
            n_dep += dep.size
            sub = rise[:, dep].copy()
            sub[i] = -np.inf                       # the construct left is not a candidate
            top = np.argmax(sub, axis=0)
            sub2 = rise2[:, dep].copy()
            sub2[i] = -np.inf
            top2 = np.argmax(sub2, axis=0)
            for t_idx, k in zip(range(dep.size), top):
                k = int(k)
                kx = int(top2[t_idx])
                pool = np.where((np.abs(freq - freq[k]) < 0.5)
                                & (np.abs(ov[i] - ov[i, k]) < 0.05))[0]
                pool = pool[(pool != i) & (pool != k)]
                kc = int(rng.choice(pool)) if pool.size else None
                for m in MEASURES:
                    o = getattr(R, m)(c, names[k])
                    ch = getattr(R, m)(c, names[kc]) if kc is not None else None
                    if o is None or ch is None:
                        continue
                    tally[m]["n"] += 1
                    tally[m]["observed"] += int(o)
                    tally[m]["chance"] += int(ch)
                    d = per_construct[m].setdefault(c, [0, 0, 0])
                    d[0] += 1
                    d[1] += int(o)
                    d[2] += int(ch)
                # the displacement number: did ANY other construct rise, against the matched one
                if kc is not None:
                    disp["n"] += 1
                    disp["observed"] += int(sub[k, t_idx] > 0)
                    disp["chance"] += int(rise[kc, t_idx] > 0)
                    dd = disp_pc.setdefault(c, [0, 0, 0])
                    dd[0] += 1
                    dd[1] += int(sub[k, t_idx] > 0)
                    dd[2] += int(rise[kc, t_idx] > 0)
                for m in ("family", "tokens"):
                    o = getattr(R, m)(c, names[kx])
                    ch = getattr(R, m)(c, names[kc]) if kc is not None else None
                    if o is None or ch is None:
                        continue
                    tally_x[m]["n"] += 1
                    tally_x[m]["observed"] += int(o)
                    tally_x[m]["chance"] += int(ch)
                    dx = per_construct_x[m].setdefault(c, [0, 0, 0])
                    dx[0] += 1
                    dx[1] += int(o)
                    dx[2] += int(ch)
        e = {"departures": n_dep}
        for m in MEASURES:
            t = tally[m]
            # the interval resamples **constructs**: every departure of one construct shares
            # its relatedness profile, so departures are not exchangeable and constructs are
            pc = list(per_construct[m].values())
            boot = []
            if pc:
                arr = np.asarray(pc, np.float64)
                for _ in range(500):
                    pick = arr[rng.integers(0, len(arr), len(arr))]
                    n_, o_, c_ = pick.sum(axis=0)
                    if n_ > 0 and c_ > 0:
                        boot.append((o_ / n_) / (c_ / n_))
            e[m] = {"n_with_both_measurable": t["n"],
                    "related_observed": t["observed"], "related_chance": t["chance"],
                    "rate_observed": t["observed"] / t["n"] if t["n"] else None,
                    "rate_chance": t["chance"] / t["n"] if t["n"] else None,
                    "lift": ((t["observed"] / t["n"]) / (t["chance"] / t["n"])
                             if t["n"] and t["chance"] else None),
                    "lift_ci95": ([float(np.quantile(boot, 0.025)),
                                   float(np.quantile(boot, 0.975))] if len(boot) > 20 else None),
                    "bootstrap_unit": "construct", "n_constructs": len(pc)}
        def lift_block(tal, pcs, label):
            r = {}
            for m, tt in tal.items():
                pc = list(pcs[m].values())
                boot = []
                if pc:
                    arr = np.asarray(pc, np.float64)
                    for _ in range(500):
                        n_, o_, c_ = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
                        if n_ > 0 and c_ > 0:
                            boot.append((o_ / n_) / (c_ / n_))
                r[m] = {"n": tt["n"], "related_observed": tt["observed"],
                        "related_chance": tt["chance"],
                        "rate_observed": tt["observed"] / tt["n"] if tt["n"] else None,
                        "rate_chance": tt["chance"] / tt["n"] if tt["n"] else None,
                        "lift": ((tt["observed"] / tt["n"]) / (tt["chance"] / tt["n"])
                                 if tt["n"] and tt["chance"] else None),
                        "lift_ci95": ([float(np.quantile(boot, 0.025)),
                                       float(np.quantile(boot, 0.975))]
                                      if len(boot) > 20 else None)}
            return r
        e["cross_twin"] = lift_block(tally_x, per_construct_x, "cross-twin")
        pc = list(disp_pc.values())
        boot = []
        if pc:
            arr = np.asarray(pc, np.float64)
            for _ in range(500):
                n_, o_, c_ = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
                if n_ > 0 and c_ > 0:
                    boot.append((o_ / n_) / (c_ / n_))
        e["displacement"] = {
            "n": disp["n"], "rate_observed": disp["observed"] / disp["n"] if disp["n"] else None,
            "rate_chance": disp["chance"] / disp["n"] if disp["n"] else None,
            "lift": ((disp["observed"] / disp["n"]) / (disp["chance"] / disp["n"])
                     if disp["n"] and disp["chance"] else None),
            "lift_ci95": ([float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
                          if len(boot) > 20 else None),
            "what": "share of departures followed by a rise in ANY other construct, against "
                    "the matched chance construct"}
        out[spec] = e
        print(f"{spec}: {n_dep:,} departures", flush=True)
        d_ = e["displacement"]
        print(f"   displacement   observed {d_['rate_observed']:.4f} chance "
              f"{d_['rate_chance']:.4f}  lift {d_['lift']:.3f}"
              + (f" [{d_['lift_ci95'][0]:.3f}, {d_['lift_ci95'][1]:.3f}]"
                 if d_.get('lift_ci95') else ""), flush=True)
        for m, x in e["cross_twin"].items():
            if x["n"]:
                print(f"   x-twin {m:9s} n={x['n']:>9,}  lift "
                      f"{x['lift'] if x['lift'] is None else round(x['lift'],2)}"
                      + (f" [{x['lift_ci95'][0]:.2f}, {x['lift_ci95'][1]:.2f}]"
                         if x.get('lift_ci95') else ""), flush=True)
        for m in MEASURES:
            x = e[m]
            if x["n_with_both_measurable"]:
                print(f"   {m:13s} n={x['n_with_both_measurable']:>9,}  "
                      f"observed {x['rate_observed']:.4f} ({x['related_observed']:,} pairs)  "
                      f"chance {x['rate_chance']:.4f} ({x['related_chance']:,})  "
                      f"lift {x['lift'] if x['lift'] is None else round(x['lift'],2)}"
                      + (f" [{x['lift_ci95'][0]:.2f}, {x['lift_ci95'][1]:.2f}]"
                         if x.get("lift_ci95") else ""), flush=True)
            else:
                print(f"   {m:13s} no pair measurable under this measure", flush=True)
    OUT2.mkdir(parents=True, exist_ok=True)
    doc = {"what": "paper 2 test 6: where a neuron goes when it leaves a construct's carrier set",
           "departure_from": "the carrier set (top 5%), not the core (21 Oct)",
           "chance_matched_on": "corpus frequency (log per-file rate, within 0.5) and the pair's "
                                "carrier-set overlap with the construct left at the earlier "
                                "checkpoint (within 0.05)",
           "readings": {"drift": "no rise beyond chance in any construct",
                        "displacement": "a rise in any construct",
                        "branching": "a rise concentrated in related constructs"},
           "measures": MEASURES, "pairs": out, "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["profiles/<ref>/twin1/<c>.npz",
                                     "validation/paper2/RELATEDNESS.json",
                                     "validation/paper2/COOCCURRENCE.json"])}
    (OUT2 / "TEST6.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"\n-> {OUT2 / 'TEST6.json'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
