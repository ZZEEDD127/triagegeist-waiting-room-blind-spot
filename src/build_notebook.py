"""
Generate notebook/triagegeist.ipynb.

The notebook does not duplicate analysis code. Its first cells write this
repository's modules (stats.py, load_nhamcs.py, analysis.py, figures.py) into
the Kaggle working directory with %%writefile, and later cells call them, so
the notebook and `python src/analysis.py` always produce the same numbers.
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s):
    cells.append(("code", s.strip("\n")))


md(r"""
# Triagegeist — The Waiting-Room Blind Spot
### What a synthetic triage benchmark can and cannot tell us, and a second-read watch validated on real ED data

**Summary.**
1. The synthetic labels were generated, not assigned by people: the 50 nurses show no rater effect (Cramér's V 0.02 vs 0.35 across real hospitals), and the vitals encode the label (|r| 0.46–0.65 vs ≤ 0.08 in real triage).
2. With the same 10 vitals and the same model, acuity agreement is κ 0.90 on the synthetic data and **0.24 on real NHAMCS triage**.
3. On the synthetic data, intake fields add nothing beyond the acuity level for predicting escalation (ΔAUC +0.001), so the benchmark cannot evaluate a waiting-room tool.
4. On **real** NHAMCS data, among patients triaged as lower acuity, intake data predicts admission far better than the triage level alone: **AUC 0.80 vs 0.64** (hospital-grouped validation). Flagging the top 20% catches **58% [55, 61]** of later admissions at 2.9× the base rate, with positive net benefit for thresholds from 2% to 35%.

All intervals are 95% bootstrap (hospital-cluster on NHAMCS). Full write-up and code: linked repository.
""")

md(r"""
## 1. Setup
The cells below write the project's four modules into the working directory. They are the same files as `src/` in the repository.
""")
for name in ["stats.py", "load_nhamcs.py", "analysis.py", "figures.py"]:
    code("%%writefile src/" + name + "\n" + (SRC / name).read_text(encoding="utf-8"))
cells.insert(2, ("code", "import os\nos.makedirs('src', exist_ok=True)\nos.makedirs('data', exist_ok=True)"))

code(r"""
import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import stats as S, load_nhamcs, analysis as A, figures as F

nd = load_nhamcs.build()            # downloads ed2021.zip / ED2022.zip from the CDC
df, hx = A.load_synthetic()
R = {"meta": {"seed": A.SEED, "n_train": len(df)}}
print("synthetic:", df.shape, "| NHAMCS:", nd.shape)
""")

md(r"""
## 2. Clinical problem
Triage assigns an acuity level (ESI 1–5) that sets how long a patient may wait. The dangerous error is undertriage. The level is a snapshot: it does not say which patients in the lower-acuity queue will need a hospital bed. We ask (1) whether this synthetic benchmark can judge triage AI, and (2) whether triage-time data identifies future admissions among lower-acuity patients **beyond the triage level itself**.

**Data.** Triagegeist synthetic ED data (80k visits, 5 sites, 50 nurses) and NHAMCS ED 2021–2022 (CDC/NCHS, public domain, 32,232 real visits). The NHAMCS fixed-width files are parsed with codebook byte positions; the disposition block moves 2 bytes between 2021 and 2022. Only triage-time information is used as a predictor.
""")

md(r"""
## 3. Module 1 — How the synthetic labels were made
""")
code(r"""
R["m1_forensics"] = A.m1_forensics(df, nd)
m1 = R["m1_forensics"]
print(json.dumps({k: m1[k] for k in ["rater_effect", "news2_recomputed", "complaints", "text_model"]}, indent=1))
F.fig_generator(R); plt.show()
""")
md(r"""
- **No rater effect.** Fifty nurses with ~1,600 patients each produce indistinguishable acuity distributions (p ≈ 0.99). Real triage differs strongly between hospitals. Labels without rater variation were not assigned by people.
- **Vitals encode the label.** Recomputing NEWS2 from the vitals reproduces the provided score (r = 0.95), so NEWS2 is not separately leaked; the vitals themselves were sampled from acuity.
- **Text is a lookup table.** 99.7% of complaints map to one level. Holding out exact strings does not help because templates share modifiers; holding out base phrases does.
""")

md(r"""
## 4. Module 2 — Acuity prediction does not transfer
""")
code(r"""
R["m2_acuity"], oof_acuity = A.m2_acuity(df, hx, nd)
for k, v in R["m2_acuity"].items():
    q = v["qwk"]; print(f"{k:38s} acc={v['accuracy']:.3f}  kappa={q['est']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}]")
F.fig_acuity(R); plt.show()
""")
md(r"""
With identical features and model, κ falls from 0.90 (synthetic) to 0.24 (real). On the synthetic data the vitals alone give κ 0.895; without them κ is 0.69. Acuity accuracy on this benchmark measures recovery of the generator.
""")

md(r"""
## 5. Module 3 — Equity audit, adjusted for acuity mix
Crude undertriage rates mislead because undertriage is impossible at ESI 5. We compare observed with expected undertriage given each group's true acuity mix, test heterogeneity by permutation within acuity levels, and correct for four attributes (Holm).
""")
code(r"""
R["m3_equity"] = A.m3_equity(df, oof_acuity)
for col in ["language", "insurance_type", "sex", "age_group"]:
    print(f"{col:15s} Holm-adjusted p = {R['m3_equity'][col]['holm_p']}")
F.fig_equity(R); plt.show()
""")
md(r"""
No attribute shows significant heterogeneity after adjustment. The audit method carries over to real data; on this synthetic data it finds no disparity.
""")

md(r"""
## 6. Module 4 — The synthetic data cannot evaluate a waiting-room tool
Escalation = admitted, transferred, observation or died. The key comparison is against a model that sees **only the acuity level**.
""")
code(r"""
R["m4_waiting_room"], oof_syn = A.m4_waiting_room(df, hx)
m4 = R["m4_waiting_room"]
for k in ["acuity_only", "full", "delta_full_minus_acuity"]:
    c = m4["auc_low"][k]; print(f"ESI 3-5  {k:26s} {c['est']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]")
print("LWBS AUC", m4["lwbs"]["auc"])
""")
md(r"""
Inside the ESI 3–5 queue the full model's AUC (0.76) equals the acuity-only model's: the synthetic disposition depends on acuity alone. Any apparent within-queue signal is just ESI 3 ranked above ESI 4 and 5. A waiting-room tool can only be judged on real outcomes.
""")

md(r"""
## 7. Module 5 — Real NHAMCS validation
Folds are grouped by hospital; intervals resample hospitals; weighted results use the survey weights `PATWT`.
""")
code(r"""
R["m5_nhamcs"], oof_real = A.m5_nhamcs(nd)
a = R["m5_nhamcs"]["admit"]
for k, c in a["auc_low"].items():
    print(f"IMMEDR 3-5  {k:34s} {c['est']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]")
op = a["operating_points_low"]["top_20pct"]
print(f"\nFlag top 20%: recall {op['recall']:.1%} {op['recall_ci']}, lift {op['lift']}x, precision {op['precision']:.1%}")
print("calibration:", a["calibration"])
print("decision curve CI above zero for thresholds:", a["decision_curve_low"]["thresholds_with_ci_above_zero"])
print("left early AUC:", R["m5_nhamcs"]["left_early"]["auc"])

A.ART.mkdir(exist_ok=True)
(A.ART / "results.json").write_text(json.dumps(R, indent=2, default=str))
oof_syn.to_parquet(A.ART / "oof_synthetic.parquet", index=False)
pd.DataFrame(oof_real).to_parquet(A.ART / "oof_nhamcs.parquet", index=False)
F.fig_watch(R, oof_syn, pd.DataFrame(oof_real)); plt.show()
F.fig_decision_curve(R); plt.show()
""")
md(r"""
Among real patients triaged as lower acuity, triage-time data ranks admission risk far better than the triage level (AUC 0.80 vs 0.64, a gain of about +0.16 with an interval far from zero; exact values printed above). Hospital-grouped validation is slightly lower than a random split (about 0.81), as expected. Survey-weighted AUC is about 0.82. After cross-fitted calibration the slope is about 0.96. Net benefit beats both "review everyone" and "review no-one" from about 2% to 35% thresholds.

*Exact figures can shift in the third decimal between library versions, because fold assignment by hospital depends on the scikit-learn version. The repository's `results.json` was produced with lightgbm 4.6.0.*
""")

md(r"""
### Equity on real data
""")
code(r"""
F.fig_real_equity(R); plt.show()
for col, rows in R["m5_nhamcs"]["equity_low_acuity_watch"].items():
    for g, v in rows.items():
        print(f"{col:15s} {g:18s} caught {v['recall_of_watch']['est']:.0%}  observed {v['observed_rate']:.1%}  predicted {v['mean_predicted']:.1%}")
""")
md(r"""
With one threshold, the share of admissions caught follows baseline risk (Medicare 79%, self-pay 26%; NH White 62%, NH Black 52%). Predicted risk is not lower than observed for any of these groups, so the gap comes from the threshold policy, not miscalibration. Choosing the threshold policy is an equity decision to make before deployment.
""")

md(r"""
## 8. Test-set predictions
The competition asked for acuity predictions; here is the structured, leak-free acuity model fitted on all training data. Given Module 2, these predictions reproduce the generator and should not be read as clinical skill.
""")
code(r"""
test = pd.read_csv(A.find("test.csv"))
hist = pd.read_csv(A.find("patient_history.csv"))
test = test.merge(hist[["patient_id"] + hx], on="patient_id", how="left")
test["pain_score"] = test["pain_score"].replace(-1, np.nan)
test["hx_burden"] = test[hx].sum(axis=1)
model = A.clf().fit(A.make_X(df, hx), df["triage_acuity"].values)
sub = pd.DataFrame({"patient_id": test["patient_id"], "triage_acuity": model.predict(A.make_X(test, hx)).astype(int)})
sub.to_csv("submission.csv", index=False)
print(sub.shape, sub["triage_acuity"].value_counts(normalize=True).round(3).sort_index().to_dict())
""")

md(r"""
## 9. Limitations
1. NHAMCS is cross-sectional: admission stands in for deterioration, and timing during the wait is unobserved.
2. Only 12 public-use variables are used; local data with full complaint text and repeat vitals should do better.
3. Hospital-grouped validation tests transfer across US hospitals, not to Nordic EDs. A silent-deployment pilot is the next step.
4. The bootstrap resamples hospitals but does not reproduce the full stratified survey design.
5. Decision support only; not a medical device.

## References
Gilboy N et al., *ESI v4*, AHRQ 2012 · Mistry B et al., *Ann Emerg Med* 2018;71:581–587 · Levin S et al., *Ann Emerg Med* 2018;71:565–574 · Hong WS et al., *PLoS One* 2018;13:e0201016 · Vickers AJ, Elkin EB, *Med Decis Making* 2006;26:565–574 · RCP, *NEWS2* 2017 · Obermeyer Z et al., *Science* 2019;366:447–453 · NCHS, *NHAMCS 2021–2022*, CDC.
""")

nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, (typ, src) in enumerate(cells):
    c = {"cell_type": typ, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
    if typ == "code":
        c["outputs"] = []; c["execution_count"] = None
    nb["cells"].append(c)
out = SRC.parent / "notebook" / "triagegeist.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
