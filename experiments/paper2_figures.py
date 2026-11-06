r"""Paper 2's figures, per Design §5.

Produced here: **1** (the three timescales), **2** (the spacing model), **3** (the power of the
null) and **4** (return and convergence). Figure 1's panel (b) reads the 200-mask ablation run
in nats rather than control units, because the control's spread has not converged at the two
later references; see `ablation_nats.py` for the measurement that decided it.

Figure 2's left panel needs the raw (start, gap, turnover) points, which `B10_SPACING_ADJUSTED`
summarises but does not store, so they are recomputed from the carrier index here -- index only,
no profile reads, and the fitted curves come from the artefact's own A and r.

    python experiments/paper2_figures.py
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from experiments.atlas import CARRIERS, D, K, provenance
from experiments.relatedness import OUT2

OUT = OUT2 / "figures"
TPS = 4_194_304            # tokens a step: 2,033,671,208,960 / 484,865


def _points():
    """(start, gap, turnover) over all pairs, from the carrier index."""
    from experiments.r1_turnover import load_index
    from experiments.atlas import p1_set
    refs = sorted((p.stem for p in CARRIERS.glob("*.npz") if p.stem.startswith("pretrain")),
                  key=lambda r: int(re.search(r"(\d+)$", r).group(1)))
    iters = [int(re.search(r"(\d+)$", r).group(1)) for r in refs]
    names = sorted(p1_set())
    idx = load_index(CARRIERS, refs)
    sets = {r: {c: np.sort(np.asarray(idx[r][f"twin1/{c}|product"], np.int64))
                for c in names if f"twin1/{c}|product" in idx[r]} for r in refs}
    rows = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            for c in names:
                if c in sets[refs[i]] and c in sets[refs[j]]:
                    keep = np.intersect1d(sets[refs[i]][c], sets[refs[j]][c]).size / K
                    rows.append((iters[i], iters[j] - iters[i], 1.0 - keep))
    return np.array(rows, float)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P = OUT2
    b10 = json.loads((P / "B10_SPACING_ADJUSTED.json").read_text())
    b3 = json.loads((P / "B3_ALLPAIRS.json").read_text())
    b11 = json.loads((P / "B11_POWER.json").read_text())
    b14 = json.loads((P / "B14_BANDS.json").read_text())
    DROP = {291000, 364000, 455000}
    made = []

    rr = json.loads((P / "RATE_READINGS.json").read_text())
    prim = rr["primary"]
    # the CROSS-checkpoint ceiling: chance measured between different constructs with one
    # profile at each end of the pair, which is the quantity the excess is computed on. The
    # within-checkpoint version is violated by up to 2 points a series; the cross version is
    # violated by none, which is what decided it.
    # A1.7: computed on ALL 169 constructs, not the 16-construct sample. The two agree to
    # under 0.01 for the quantile cuts and the 5% set, but the |e| >= 1 sigma0 band moves by
    # up to 0.018 at 27 of 32 start points in the widest window, so the figure uses the 169.
    ceilw = json.loads((P / "CEILINGS_PAIRWISE_169.json").read_text())["per_window"]
    b17b = json.loads((P / "B17B_RETURN_SPLIT.json").read_text())
    scope = json.loads((D / "validation/paper1/SCOPE_OVERLAP.json").read_text())["rows"]

    # ================= Figure 1: three stacked panels, one token axis =================
    import glob as _glob
    _rng = np.random.default_rng(20261023)

    def _med_ci(x):
        x = np.asarray(x, float)
        if x.size == 0:
            return None, None
        b = [np.median(x[_rng.integers(0, x.size, x.size)]) for _ in range(10000)]
        return float(np.median(x)), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

    abl = {}
    for f in sorted(_glob.glob(str(D / "validation/ablation/FULL_*.json"))):
        a = json.loads(open(f).read())
        cs = a["constructs"]
        # panel (b) is in NATS, not control units: at 83.9 B and 2,034 B the control's SD is
        # ~2.2x its own mean and still rising with the mask count, so the unit's denominator
        # has not converged (ablation_nats.py). The quantity drawn is the carriers' rise minus
        # the random sets' MEAN rise, signed, which needs no converged spread.
        cur = np.array([v["current"] - v["random_mean"] for v in cs.values()
                        if v.get("current") is not None
                        and v.get("random_mean") is not None])
        if cur.size == 0:
            continue
        dri = np.array([v["drifted"] - v["random_mean"] for v in cs.values()
                        if v.get("drifted") is not None and v.get("random_mean") is not None])
        it = int(re.search(r"(\d+)$", a["reference"]).group(1))
        m, c = _med_ci(cur)
        md, cd = _med_ci(dri) if dri.size else (None, None)
        abl[it] = {"cur": m, "cur_ci": c, "old": md, "old_ci": cd, "n": int(cur.size),
                   "n_old": int(dri.size) if dri.size else None,
                   "n_old_pos": int((dri > 0).sum()) if dri.size else None}
    st4 = json.loads((D / "validation/r2/STEP4_FLAT_pretrain.json").read_text())
    ser, anch = st4["forced"]["series"], st4["anchors"]
    loss = {anch[k]["tokens"] / 1e9: float(np.median(
        [v[f] for v in ser.values() if v.get(f) is not None]))
        for k, f in (("early", "level_early"), ("at_84b", "level_84b"),
                     ("last", "level_last"))}

    # The three panels in ONE ROW, at the full text width like figures 2 to 4. Each panel
    # is then a third as wide and the figure is short instead of tall.
    #
    # **Every point size below is multiplied by 1.44, and that is not cosmetic.** Stacked,
    # the figure was 5.52in wide set at 0.6 of a 6.75in column -- a scale of 0.73. In a row
    # at the full width it is 13.2in set at 6.75 -- a scale of 0.51. Keeping the numbers
    # would have shrunk every label on the page by a third; 1.44 is 0.73/0.51, which lands
    # them at the size they are now.
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.1),
                            gridspec_kw={"wspace": 0.30, "top": 0.965, "bottom": 0.185,
                                         "left": 0.105, "right": 0.995})
    a1, a2, a3 = axs
    lt = sorted(loss)
    a1.plot(lt, [loss[t] for t in lt], "-o", color="#117a65", lw=2, ms=7)
    a1.set_ylabel("loss at the forced and expected\npositions (nats)", fontsize=13.7)
    a1.text(0.012, 0.9, "(a)  behaviour", transform=a1.transAxes, fontsize=15.1,
            fontweight="bold")
    a1.set_ylim(0, max(loss.values()) * 1.25)

    at = sorted(abl)
    tt = np.array([t * TPS / 1e9 for t in at])
    vv = np.array([abl[t]["cur"] for t in at])
    a2.errorbar(tt, vv,
                yerr=[vv - np.array([abl[t]["cur_ci"][0] for t in at]),
                      np.array([abl[t]["cur_ci"][1] for t in at]) - vv],
                fmt="-s", color="#922b21", lw=2, ms=7, capsize=3, elinewidth=1.2,
                label="the current set")
    last = max(at)
    e = abl[last]
    if e["old"] is not None:
        # offset right: at the same x the hollow marker's bar overlapped the filled square's
        a2.errorbar([last * TPS / 1e9 * 1.18], [e["old"]],
                    yerr=[[e["old"] - e["old_ci"][0]], [e["old_ci"][1] - e["old"]]],
                    fmt="o", mfc="none", mec="#922b21", color="#922b21", ms=9, mew=1.8,
                    capsize=3, elinewidth=1.2,
                    label="the 84 B set, at the end")
    a2.axhline(0, color="0.7", lw=0.8)
    a2.set_ylabel("effect of removal\n(nats over random)", fontsize=13.7)
    a2.text(0.012, 0.9, "(b)  reliance", transform=a2.transAxes, fontsize=15.1,
            fontweight="bold")
    # below the "(b) function" label, which sits at y=0.9 in axes coordinates
    a2.legend(fontsize=11.5, frameon=False, loc="upper left", bbox_to_anchor=(0.012, 0.83))

    rows = prim["carrier5"]["20k-30k"]
    rt = np.array([r["start_iter"] for r in rows]) * TPS / 1e9
    rv = np.array([r["median_excess"] for r in rows])
    a3.fill_between(rt, [r["ci95"][0] for r in rows], [r["ci95"][1] for r in rows],
                    color="#1b4f72", alpha=0.2, linewidth=0)
    a3.plot(rt, rv, "-", color="#1b4f72", lw=2)
    a3.set_ylabel("turnover beyond\nthe twin noise", fontsize=13.7)
    a3.text(0.012, 0.9, "(c)  membership", transform=a3.transAxes, fontsize=15.1,
            fontweight="bold")
    a3.set_ylim(0, 0.95)
    a3.set_xscale("log")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("tokens (billions, log)", fontsize=13.7)
        ax.tick_params(labelsize=11.5)
        ax.axvline(84, color="0.45", ls=":", lw=1.1)
        ax.grid(alpha=0.2, lw=0.5)
        # the 84 B marker sat at a tenth of the axis height, where the curves run; on the
        # axis line below the panel it is clear of every series in all three
        ax.annotate("84 B", xy=(84, 0), xycoords=("data", "axes fraction"),
                    xytext=(2, 3), textcoords="offset points",
                    fontsize=11.5, color="0.35", ha="left", va="bottom")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"FIG1_three_timescales.{ext}", dpi=300)
    plt.close(fig)
    made.append("FIG1_three_timescales")

    # ================= Figure 2: the rate, fit-free, with moving ceilings =============
    WINS = ["~1,000", "5k-10k", "20k-30k", "50k-100k"]
    wcol = ["#1b4f72", "#2e86c1", "#d35400", "#922b21"]
    BANDS6 = (("q1pct", "top 1%"), ("q2pct", "top 2%"), ("q5pct", "top 5%"),
              ("abs1sd", "|e| \u2265 1\u03c3\u2080"), ("abs2sd", "|e| \u2265 2\u03c3\u2080"),
              ("abs3sd", "|e| \u2265 3\u03c3\u2080"))
    from matplotlib.gridspec import GridSpec
    # the six bands get their own strip under panel (a): they are a RATIO (share of headroom)
    # while the four window series are an absolute excess, so sharing one axis read as though
    # the bands sat above the windows.
    # One row of three. The band was a strip under panel A sharing its x axis; as a column
    # of its own it keeps that axis but stands full height, so it is given a narrower share.
    fig = plt.figure(figsize=(13.2, 4.6))
    gs = GridSpec(1, 3, width_ratios=[1.15, 0.92, 1.15], wspace=0.30, figure=fig,
                  left=0.072, right=0.99, top=0.92, bottom=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axBand = fig.add_subplot(gs[0, 1], sharex=axA)
    axB = fig.add_subplot(gs[0, 2], sharey=axA)
    for ax, meas, nice in ((axA, "carrier5", "the 5% carrier set"),
                           (axB, "cosine", "the graded profile (1 \u2212 cosine)")):
        for w, col in zip(WINS, wcol):
            rws = prim[meas].get(w) or []
            if not rws:
                continue
            t = np.array([r["start_iter"] for r in rws]) * TPS / 1e9
            ax.fill_between(t, [r["ci95"][0] for r in rws], [r["ci95"][1] for r in rws],
                            color=col, alpha=0.18, linewidth=0, zorder=2)
            ax.plot(t, [r["median_excess"] for r in rws], "-o", ms=3.2, lw=1.5, color=col,
                    label=f"gap {w} steps", zorder=3)
        for w, col in zip(WINS, wcol):          # one ceiling a window: chance depends on the gap
            cr = [r for r in ceilw[w] if r[meas]["ceiling_cross"] is not None]
            if cr:
                ax.plot([r["tokens_b"] for r in cr],
                        [r[meas]["ceiling_cross"] for r in cr], "--", color=col, lw=1.0,
                        alpha=0.85, zorder=4)
        ax.plot([], [], "--", color="0.35", lw=1.0,
                label="ceiling for that gap and start point\n(twin agreement \u2212 chance "
                      "across the pair)")
        ax.axvline(84, color="0.45", ls=":", lw=1.1)
        ax.set_xscale("log")
        ax.set_title(f"({'ab'[meas != 'carrier5']})  {nice}", fontsize=10.5, loc="left")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.2, lw=0.5)
    # shortened: the full phrase collided with the band panel's label in the left margin,
    # and the caption carries it
    axA.set_ylabel("change beyond the twin noise", fontsize=13.7)
    # panel (a) hid its x labels when the band sat beneath it and carried the axis for
    # both; side by side it needs its own
    axA.set_xlabel("start point (billions of tokens, log scale)")
    # ONE shared legend, below both panels and outside the axes: in the axes it covered
    # series wherever it was put -- lower left sat on the ~1,000-step run at 10-20 B, lower
    # right on the 1,000 and 5k-10k runs at 70-160 B in (a) and the 20k-30k run near 300 B
    # in (b)
    h, l = axA.get_legend_handles_labels()
    # The figure is drawn 13.2in wide and set at a text width of about 6.75in, so every
    # point size here reaches the page at roughly half its value: 8pt arrived as 4pt and
    # could not be read. 12 lands at about 6, which is the size of the panel titles.
    fig.legend(h, l, fontsize=12, frameon=False, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.4, handlelength=1.5)
    axB.set_xlabel("start point (billions of tokens, log scale)")

    cmap = plt.get_cmap("Greys")
    b14doc = json.loads((P / "B14_BANDS.json").read_text())["per_band"]
    cmapw = {r["start_iter"]: r for r in ceilw["20k-30k"]}
    for i, (bk, blab) in enumerate(BANDS6):
        rws = prim[bk].get("20k-30k") or []
        if not rws:
            continue
        # RATE_READINGS normalised by a STATIC headroom; un-normalise and divide by the
        # cross-checkpoint ceiling at that start point instead
        stat = (b14doc[bk]["twin_ceiling_last_ref"]
                - b14doc[bk]["median_chance_kept_fraction"])
        t, y = [], []
        for r in rws:
            c = cmapw.get(r["start_iter"])
            cc = (c or {}).get(bk, {}).get("ceiling_cross")
            if not cc:
                continue
            t.append(r["start_iter"] * TPS / 1e9)
            y.append(r["median_excess"] * stat / cc)
        axBand.plot(t, y, "-", lw=0.95, color=cmap(0.35 + 0.1 * i), label=blab)
    axBand.axvline(84, color="0.45", ls=":", lw=1.1)
    axBand.set_xscale("log")
    axBand.set_ylabel("share of headroom\n(change \u2212 twin turnover)\n/ ceiling", fontsize=7.5)
    axBand.set_xlabel("start point (billions of tokens, log scale)")
    axBand.set_title("the six bands, gap 20k\u201330k steps", fontsize=9, loc="left")
    axBand.grid(alpha=0.2, lw=0.5)
    # raised from 6.6pt, which reached the page at about 3.3; at three columns the widest
    # entries then ran past the panel, so the six go in two columns of three
    axBand.legend(fontsize=9.5, frameon=False, ncol=2, loc="lower left",
                  columnspacing=1.0, handlelength=1.4)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"FIG2_rate_fit_free.{ext}", dpi=300)
    plt.close(fig)
    made.append("FIG2_rate_fit_free")

    # ---------------- Figure 3: the power of the null ----------------------------------
    # a third lower (4.2 -> 2.81); the width is held so the page scale, and with it
    # the type size, does not move
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 2.81),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    # `dates` also carries a `false_detection` entry with no shape or tokens -- the uninjected
    # control. Filter on the presence of a shape rather than on the key's spelling.
    dates = {k: v for k, v in b11["dates"].items() if "shape" in v}
    fd_entry = b11["dates"].get("false_detection") or {}
    shapes = sorted({v["shape"] for v in dates.values()})
    xs = sorted({v["tokens"] for v in dates.values()})
    mk = {"step": ("o-", "#1b4f72"), }
    for sh in shapes:
        ys = [next((v["detection_rate"] for v in dates.values()
                    if v["shape"] == sh and v["tokens"] == x), np.nan) for x in xs]
        style, col = mk.get(sh, ("s--", "#d35400"))
        axL.plot(np.array(xs) / 1e9, ys, style, color=col, ms=5,
                 label=sh if len(sh) < 20 else "exponential approach")
    fd = fd_entry.get("rate", 0.0)
    axL.axhline(fd, color="#117a65", lw=1.2, ls="-.",
                label=f"false detection on uninjected series ({fd:.1%})")
    axL.set_xscale("log")
    axL.set_ylim(-0.04, 1.06)
    axL.set_xlabel("settling date injected (billions of tokens, log scale)")
    axL.set_ylabel("detection rate")
    axL.set_title("(a)  the rule finds an injected settling date — by date and by shape",
                  fontsize=10, loc="left")
    axL.legend(fontsize=8, frameon=False, loc="center left")
    axL.grid(alpha=0.25, lw=0.5)
    # the right panel: the found date against the injected date, which says the rule is late
    for sh in shapes:
        inj, fnd = [], []
        for v in dates.values():
            if v["shape"] == sh and v.get("median_date_found"):
                inj.append(v["tokens"] / 1e9)
                fnd.append(v["median_date_found"] / 1e9)
        o = np.argsort(inj)
        style, col = mk.get(sh, ("s--", "#d35400"))
        axR.plot(np.array(inj)[o], np.array(fnd)[o], style, color=col, ms=5)
    lim = [60, 2200]
    axR.plot(lim, lim, "-", color="0.6", lw=1)
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlim(*lim)
    axR.set_ylim(*lim)
    axR.set_xlabel("injected (B tokens)")
    axR.set_ylabel("median found (B tokens)")
    axR.set_title("(b)  the rule dates late:\nevery point is on or above the diagonal",
                  fontsize=10, loc="left")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(OUT / f"FIG3_power_of_the_null.{e}", dpi=300)
    plt.close(fig)
    made.append("FIG3_power_of_the_null")

    # ================= Figure 4: recurrence and convergence ==========================
    # a quarter lower (4.9 -> 3.68); width held, so the type size on the page is the
    # size it was
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 3.68))
    # ONE colour for both arms: the legend has a single "return rate" entry
    arms = [("fell_beyond_one_twin_unit", "fell by more than\none twin unit", "#1b4f72"),
            ("fell_within_one_twin_unit", "fell within\none twin unit", "#1b4f72")]
    xs = np.arange(len(arms))
    for k, (arm, lab, col) in enumerate(arms):
        e = b17b["arms"][arm]
        axA.bar(k - 0.19, e["return_rate"], 0.36, color=col,
                label="return rate" if k == 0 else None)
        axA.bar(k + 0.19, e["control_rate"], 0.36, color="0.75",
                label="matched control" if k == 0 else None)
        axA.text(k - 0.19, e["return_rate"] + 0.025,
                 f"lift {e['lift']:.2f}\n[{e['lift_ci95'][0]:.2f}, {e['lift_ci95'][1]:.2f}]",
                 ha="center", va="bottom", fontsize=8)
        axA.text(k - 0.19, 0.02, f"n={e['departures']/1e6:.1f} M", ha="center", fontsize=7,
                 color="white", rotation=90, va="bottom")
    # widen the axis so the upper right is genuinely empty: at xlim (-0.5, 1.5) the inset had
    # to sit over a bar whatever its position, and it hid the second arm's interval line
    axA.set_xlim(-0.55, 2.25)
    axA.set_xticks(xs)
    axA.set_xticklabels([l for _, l, _ in arms], fontsize=9)
    axA.set_ylabel("share of departed neurons that return")
    axA.set_ylim(0, 0.92)
    axA.set_title("(a)  departed neurons return, including those that fell beyond one twin "
                  "unit", fontsize=10.5, loc="left")
    axA.legend(fontsize=8, frameon=False, loc="upper right")
    # upper-LEFT: at [0.42, 0.42] the inset covered the second arm's bars and its lift label
    ins = axA.inset_axes([0.63, 0.42, 0.33, 0.30])
    for (arm, lab, _), icol, ilab in zip(arms, ("#1b4f72", "#d35400"),
                                         ("beyond one twin unit", "within one twin unit")):
        h = b17b["arms"][arm].get("absence_length_histogram_references") or {}
        kk = sorted(int(x) for x in h)[:12]
        tot = sum(h.values())
        ins.plot(kk, [h[str(x)] / tot for x in kk], "-o", ms=2.6, lw=1.2, color=icol,
                 label=ilab)
    ins.legend(fontsize=5.6, frameon=False, loc="upper right")
    ins.set_xlabel("absence (references)", fontsize=7)
    ins.set_ylabel("share of returners", fontsize=7)
    ins.tick_params(labelsize=6.5)
    ins.grid(alpha=0.25, lw=0.4)

    first, lastr = "pretrain_iter_0001000", "pretrain_iter_0484865"
    cons = sorted({k.split("|")[1] for k in scope})
    for c in cons:
        a = scope.get(f"{first}|{c}")
        b = scope.get(f"{lastr}|{c}")
        if not a or not b:
            continue
        y0, y1 = a["share_of_ceiling"]["twin1"], b["share_of_ceiling"]["twin1"]
        hi = c == "For"
        axB.plot([0, 1], [y0, y1], "-o", ms=5 if hi else 3.4,
                 lw=2.4 if hi else 1.0, color="#922b21" if hi else "0.62",
                 zorder=3 if hi else 1)
        # nudge the two that collide
        dy = {"Import": 7, "If": -7}.get(c, 0)
        axB.annotate(c, xy=(1, y1), xytext=(5, dy), textcoords="offset points",
                     fontsize=8.5 if hi else 7, va="center",
                     color="#922b21" if hi else "0.45",
                     fontweight="bold" if hi else "normal")
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["4 B tokens", "2,034 B tokens"])
    axB.set_xlim(-0.08, 1.30)
    axB.set_ylabel("shared carriers of the two tokens,\nas a share of the twin ceiling")
    axB.set_title("(b)  one role, two tokens: the sets converge, for all twelve",
                  fontsize=10.5, loc="left")
    axB.grid(alpha=0.2, lw=0.5, axis="y")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"FIG4_return_and_convergence.{ext}", dpi=300)
    plt.close(fig)
    made.append("FIG4_return_and_convergence")

    meta = {"what": "paper 2's figures 2, 3 and 4 per Design §5",
            "figure_1_source": "validation/ablation/FULL_*.json, the 200-mask run, "
                               "16 constructs at 3 references. Panel (b) is now in NATS: the "
                               "carriers' loss rise minus the random sets' MEAN rise, signed, "
                               "with construct-bootstrap intervals (ABLATION_NATS.json). "
                               "Control units were dropped from the panel because at 83.9 B "
                               "and 2,034 B the control SD is ~2.2x its own mean and still "
                               "rising with the mask count, so the unit's denominator has not "
                               "converged; they remain in Table D.1 with a per-cell stability "
                               "flag. The draft's 1.0 / 3.3 / 5.2 control units came from a "
                               "20-mask control and fall to 0.79 / 2.27 / 3.97 at 200 masks. "
                               "No retention ratio is quoted (Piotr, 23 Oct).",
            "figure_4_withdrawn": "the stability-by-band figure plotted the withdrawn fit and "
                                  "is replaced by return (B17b) and convergence (SCOPE_OVERLAP)",
            "figures_5_and_6": "not produced; the Settle chat withdrew them",
            "ceilings": "computed at EACH start point (CEILINGS.json), not once",
            "b11_shape_note": "detection is 0% at 1 T and 1.5 T for the EXPONENTIAL shape "
                              "only; the step shape is detected at 99.4% there. The draft's "
                              "§3.2 says 'under both shapes' and is wrong.",
            "files": [f"{m}.{e}" for m in made for e in ("png", "svg")],
            "figure_2_source": "RATE_READINGS.json primary (fit-free); B10's fitted curves "
                               "are deliberately not drawn (negative R^2)",
            "minutes": (time.time() - t0) / 60,
            "provenance": provenance(["validation/paper2/B10_SPACING_ADJUSTED.json",
                                      "validation/paper2/B3_ALLPAIRS.json",
                                      "validation/paper2/B11_POWER.json",
                                      "validation/paper2/B14_BANDS.json",
                                      "validation/carriers/*.npz for figure 2(a)'s points"])}
    (OUT2 / "PAPER2_FIGURES.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"{len(made)} figures -> {OUT}")
    for m in made:
        print(f"  {m}.png / .svg")
    print(f"({meta['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
