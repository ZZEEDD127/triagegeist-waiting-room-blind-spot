"""
Figures for the notebook and write-up, drawn only from artifacts/results.json
and the saved out-of-fold predictions, so every plotted number matches the
reported one.

  python src/figures.py   ->  artifacts/fig_*.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
NAVY, RED, BLUE, GREEN, AMBER, GREY = "#0b1f33", "#d0453a", "#2b6cb0", "#2b8a3e", "#e08a2c", "#9aa5b1"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 140, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def _err(ci):
    return [[ci["est"] - ci["lo"]], [ci["hi"] - ci["est"]]]


def fig_generator(R):
    """Two signatures of generated labels: vitals track acuity far more tightly
    than in real triage, and the label shows no rater effect."""
    f = R["m1_forensics"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    names = ["PULSE", "RESPR", "POPCT", "BPSYS", "TEMPF"]
    pretty = ["heart rate", "resp. rate", "SpO2", "systolic BP", "temperature"]
    syn = [abs(f["corr_with_acuity_synthetic_same_vitals"][n]) for n in names]
    real = [abs(f["corr_with_acuity_nhamcs"][n]) for n in names]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, syn, 0.4, color=RED, label="synthetic")
    ax[0].bar(x + 0.2, real, 0.4, color=GREEN, label="real (NHAMCS)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(pretty, rotation=20)
    ax[0].set_ylabel("|correlation| with triage level"); ax[0].set_ylim(0, 0.85)
    ax[0].legend(fontsize=8, ncol=2, loc="upper center")
    ax[0].set_title("Vitals track the label far too tightly")
    r = f["rater_effect"]
    vals = [r["synthetic_nurse_cramers_v"], r["nhamcs_hospital_cramers_v"]]
    ax[1].bar([f"synthetic\n{r['synthetic_nurses']} nurses", f"real\n{r['nhamcs_hospitals']} hospitals"],
              vals, color=[RED, GREEN], width=0.5)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.005, f"V = {v:.3f}", ha="center")
    ax[1].set_ylabel("Cramér's V (rater × triage level)")
    ax[1].set_title("No rater effect in the synthetic labels")
    fig.tight_layout()
    return fig


def fig_acuity(R):
    m = R["m2_acuity"]; t = R["m1_forensics"]["text_model"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
    rows = [("synthetic\nstructured", m["structured_stratified"]["qwk"], RED),
            ("synthetic\n10 shared vitals", m["harmonised_synthetic"]["qwk"], AMBER),
            ("real NHAMCS\n10 shared vitals", m["harmonised_nhamcs"]["qwk"], GREEN)]
    for i, (lab, ci, c) in enumerate(rows):
        ax[0].bar(i, ci["est"], color=c, yerr=_err(ci), capsize=4, width=0.6)
        ax[0].text(i, ci["hi"] + 0.02, f"{ci['est']:.2f}", ha="center")
    ax[0].axhspan(0.6, 0.8, color=GREY, alpha=0.25, label="published nurse–nurse agreement")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([r[0] for r in rows])
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("quadratic-weighted κ (95% CI)")
    ax[0].set_title("Same model, same features: synthetic vs real"); ax[0].legend(fontsize=8, loc="upper right")
    labs = ["random folds", "unseen exact\ncomplaint", "unseen complaint\nbase phrase"]
    keys = ["stratified", "group_exact_complaint", "group_complaint_base"]
    vals = [t[k]["qwk"] for k in keys]
    ax[1].bar(labs, vals, color=[RED, AMBER, BLUE], width=0.6)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("κ, complaint text only")
    ax[1].set_title("The text shortcut is template memorisation")
    fig.tight_layout()
    return fig


def fig_equity(R):
    e = R["m3_equity"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col in zip(axes, ["language", "insurance_type"]):
        g = e[col]["groups"]
        items = sorted(g.items(), key=lambda kv: kv[1]["oe_ratio"]["est"])
        for i, (name, v) in enumerate(items):
            ci = v["oe_ratio"]
            ax.errorbar(ci["est"], i, xerr=[[ci["est"] - ci["lo"]], [ci["hi"] - ci["est"]]],
                        fmt="o", color=NAVY, capsize=3)
        ax.axvline(1, color=RED, ls="--", lw=1)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"{k} (n={v['n']:,})" for k, v in items])
        ax.set_xlabel("observed / expected undertriage (95% CI)")
        ax.set_title(f"{col.replace('_', ' ')} · Holm p = {e[col]['holm_p']:.2f}")
    fig.tight_layout()
    return fig


def _decile_rates(y, p, n=10):
    d = pd.qcut(pd.Series(p).rank(method="first"), n, labels=False)
    return pd.Series(y).groupby(d).mean().to_numpy()


def fig_watch(R, oof_s, oof_n):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    low_s = oof_s["acuity"].isin([3, 4, 5]).to_numpy()
    rs = _decile_rates(oof_s["y"].to_numpy()[low_s], oof_s["p_cal"].to_numpy()[low_s])
    low_n = oof_n["low"].to_numpy().astype(bool)
    rn = _decile_rates(oof_n["y"].to_numpy()[low_n], oof_n["p_cal"].to_numpy()[low_n])
    s4 = R["m4_waiting_room"]["auc_low"]; s5 = R["m5_nhamcs"]["admit"]["auc_low"]
    t_syn = (f"Synthetic ESI 3–5: steps = ESI 5/4/3\n"
             f"AUC {s4['full']['est']:.2f} vs acuity alone {s4['acuity_only']['est']:.2f}")
    t_real = (f"Real NHAMCS IMMEDR 3–5\n"
              f"AUC {s5['intake_plus_immedr']['est']:.2f} vs triage level alone {s5['immedr_only']['est']:.2f}")
    for a, r, base, title, c in [(ax[0], rs, R["m4_waiting_room"]["prevalence_low"], t_syn, BLUE),
                                 (ax[1], rn, R["m5_nhamcs"]["admit"]["prevalence_low"], t_real, GREEN)]:
        a.bar(range(1, 11), r, color=c)
        a.axhline(base, color="k", ls="--", lw=1, label=f"queue base rate {base:.1%}")
        a.set_xlabel("predicted-risk decile (1 = lowest)"); a.set_ylabel("observed escalation / admission")
        a.set_title(title); a.legend(fontsize=8); a.set_xticks(range(1, 11))
    fig.suptitle("Risk inside the 'can wait' queue: re-sorted acuity (synthetic) vs new signal (real)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def fig_decision_curve(R):
    dc = R["m5_nhamcs"]["admit"]["decision_curve_low"]
    th = np.array(dc["thresholds"])
    model = np.array(dc["model"]); allv = np.array(dc["treat_all"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(th, model, color=GREEN, lw=2.2, label="waiting-room watch")
    ax[0].plot(th, allv, color=GREY, ls="--", label="review everyone")
    ax[0].axhline(0, color="k", lw=1, label="review no-one")
    ax[0].set_ylim(min(-0.01, model.min() * 1.1), max(model.max(), allv.max()) * 1.15)
    ax[0].set_xlabel("threshold probability"); ax[0].set_ylabel("net benefit")
    ax[0].set_title("Decision curve, real low-acuity queue"); ax[0].legend(fontsize=8)
    adv = model - np.maximum(allv, 0)
    ax[1].fill_between(th, dc["adv_lo"], dc["adv_hi"], color=GREEN, alpha=0.25, label="95% hospital-cluster CI")
    ax[1].plot(th, adv, color=GREEN, lw=2)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("threshold probability"); ax[1].set_ylabel("net benefit over best default")
    ax[1].set_title("Advantage over the better default policy"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_real_equity(R):
    eq = R["m5_nhamcs"]["equity_low_acuity_watch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    overall = R["m5_nhamcs"]["admit"]["operating_points_low"]["top_20pct"]["recall"]
    for ax, col in zip(axes, ["race_ethnicity", "payer"]):
        items = sorted(eq[col].items(), key=lambda kv: kv[1]["recall_of_watch"]["est"])
        for i, (name, v) in enumerate(items):
            ci = v["recall_of_watch"]
            ax.errorbar(ci["est"], i, xerr=[[ci["est"] - ci["lo"]], [ci["hi"] - ci["est"]]],
                        fmt="o", color=NAVY, capsize=3)
        ax.axvline(overall, color=GREEN, ls="--", lw=1, label=f"overall {overall:.0%}")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"{k} ({v['admissions']} adm.)" for k, v in items])
        ax.set_xlabel("share of admissions caught by top-20% watch")
        ax.set_title(col.replace("_", " ")); ax.legend(fontsize=8)
    fig.suptitle("Real-data equity check of the watch", fontweight="bold")
    fig.tight_layout()
    return fig


def all_figures():
    R = json.loads((ART / "results.json").read_text())
    oof_s = pd.read_parquet(ART / "oof_synthetic.parquet")
    oof_n = pd.read_parquet(ART / "oof_nhamcs.parquet")
    return {"generator": fig_generator(R), "acuity": fig_acuity(R), "equity": fig_equity(R),
            "watch": fig_watch(R, oof_s, oof_n), "decision_curve": fig_decision_curve(R),
            "real_equity": fig_real_equity(R)}


if __name__ == "__main__":
    matplotlib.use("Agg")
    for name, fig in all_figures().items():
        fig.savefig(ART / f"fig_{name}.png", bbox_inches="tight")
        print("saved", ART / f"fig_{name}.png")
