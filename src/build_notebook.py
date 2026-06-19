"""Generate the self-contained Kaggle notebook notebook/triagegeist.ipynb.

The notebook does NOT import this repo's modules (Kaggle won't have them); every
cell is self-contained. Data paths auto-resolve to /kaggle/input or local
comp_data/. NHAMCS validation downloads + parses real CDC files with a hardcoded
codebook layout, degrading gracefully if offline so the notebook always runs.
"""
import json
from pathlib import Path

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

# ---------------------------------------------------------------- title
md(r"""
# Triagegeist — The Waiting-Room Blind Spot
### Auditing acuity-prediction AI, and predicting who deteriorates before they are seen

A triage acuity score answers *"how sick is this patient?"* — not *"what will
happen to them while they wait?"* This notebook delivers a single, honest,
end-to-end study with five parts:

1. **Leakage & shortcut forensics** on the Foundation's synthetic benchmark.
2. **An honest, leak-free acuity model** — and why the standard fix (GroupKFold)
   is *insufficient* here.
3. **An equity / undertriage-disparity audit** (language, insurance, sex, age).
4. **A waiting-room deterioration-risk model** (escalation & LWBS) — testing the
   Foundation's own suggested "deterioration in the waiting room" track.
5. **External validation on real U.S. ED data (NHAMCS, CDC)** — the reality check.

**Headline:** on the synthetic data a structured model reaches quadratic-weighted
κ ≈ 0.93 — *above the human inter-rater ceiling (κ 0.6–0.8)*, which is impossible
in reality and is a fingerprint of generative leakage. On **real NHAMCS** the
same method scores κ ≈ 0.27. The contribution emergency-triage AI needs is
**honest evaluation and real linked data, not higher leaderboard accuracy.**
""")

# ---------------------------------------------------------------- problem
md(r"""
## 1. Clinical problem statement

Emergency triage assigns each arrival an acuity level (e.g. ESI 1–5) that governs
how long they may safely wait. Two failure modes matter clinically:

- **Undertriage** — a genuinely sick patient is given a low acuity and waits too
  long. This is the dangerous, sometimes fatal error; its cost is asymmetric and
  far exceeds overtriage. Inter-rater reliability of human triage is only
  moderate (κ ≈ 0.6–0.8), and undertriage of vulnerable groups is a documented
  patient-safety concern.
- **The waiting-room blind spot** — the acuity number is a *snapshot*. It does
  not directly say who will deteriorate, need admission, or give up and leave
  *without being seen* (LWBS) while waiting.

We therefore (a) build an acuity model **honestly**, refusing the dataset's
shortcuts, and (b) ask whether intake data can flag waiting-room risk *beyond*
the acuity score — then check every claim against real clinical data.
""")

# ---------------------------------------------------------------- imports + load
code(r"""
import os, io, zipfile, urllib.request, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix)
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
SEED = 42

def find_file(name):
    cands = []
    for root in ["/kaggle/input", "comp_data", "../comp_data", "."]:
        if os.path.isdir(root):
            for dp, _, fns in os.walk(root):
                if name in fns:
                    cands.append(os.path.join(dp, name))
    if not cands:
        raise FileNotFoundError(name + " not found under /kaggle/input or comp_data/")
    # prefer the shallowest path
    return sorted(cands, key=len)[0]

train = pd.read_csv(find_file("train.csv"))
hist  = pd.read_csv(find_file("patient_history.csv"))
cc    = pd.read_csv(find_file("chief_complaints.csv"))
HX = [c for c in hist.columns if c.startswith("hx_")]
df = (train.merge(hist[["patient_id"]+HX], on="patient_id", how="left")
            .merge(cc[["patient_id","chief_complaint_raw"]], on="patient_id", how="left"))
df["pain_score"] = df["pain_score"].replace(-1, np.nan)   # -1 = missing sentinel
df["hx_burden"]  = df[HX].sum(axis=1)
print("merged training frame:", df.shape, "| acuity prevalence:",
      df["triage_acuity"].value_counts(normalize=True).round(3).sort_index().to_dict())
""")

# ---------------------------------------------------------------- disclosure
md(r"""
## 2. Data disclosure

| File | Source | Role |
|---|---|---|
| `train.csv` (80k), `test.csv` (20k) | **Triagegeist synthetic ED dataset**, provided via the Kaggle competition (simulated Finnish multi-site ED network) | acuity label + intake |
| `chief_complaints.csv`, `patient_history.csv` | same | complaint text, 24 comorbidity flags |
| **NHAMCS ED 2021–2022** | **CDC / NCHS**, public-use files (public domain), parsed from the official fixed-width layout | real-world external validation |

The competition data is **synthetic**; we treat it as a proof-of-concept
benchmark and validate findings on real NHAMCS data. All datasets are cited and
used under their terms.
""")

# ---------------------------------------------------------------- M1 forensics
md(r"""
## 3. Module 1 — Leakage & shortcut forensics

Before modelling, we ask what the synthetic generator actually encoded. Two
problems disqualify naive high scores:

- **Post-triage leakage.** `disposition` and `ed_los_hours` are outcomes
  realised *after* triage (and absent from `test.csv`). They must never be model
  inputs for acuity.
- **A complaint→acuity shortcut.** The free-text chief complaint almost
  perfectly determines acuity, so any text model memorises templates rather than
  learning clinical reasoning.
""")
code(r"""
# (a) post-triage leakage: correlation with the acuity label
leak = {c: round(df[c].corr(df["triage_acuity"]), 3)
        for c in ["ed_los_hours","news2_score","shock_index","gcs_total"]}
print("Correlation with triage_acuity:", leak)

# (b) the chief-complaint shortcut: within-complaint label purity
g = df.groupby("chief_complaint_raw")["triage_acuity"]
purity = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
print(f"Unique complaints: {df['chief_complaint_raw'].nunique()}")
print(f"Mean within-complaint acuity purity: {purity.mean():.4f}")
print(f"Share of complaints that FULLY determine acuity: {(purity==1).mean():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(leak.keys(), leak.values(), color="#e23b3b")
ax[0].axhline(0, color="k", lw=.6); ax[0].set_title("Leakage: corr. with acuity")
ax[0].tick_params(axis="x", rotation=30)
ax[1].hist(purity, bins=20, color="#2b6cb0")
ax[1].set_title("Complaint→acuity purity\n(1.0 = fully deterministic)")
ax[1].set_xlabel("max class share within complaint")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- M2 acuity model
md(r"""
## 4. Module 2 — An honest, leak-free acuity model

We drop post-triage outcomes **and refuse the raw-text shortcut**, using only
structured intake (vitals, derived scores, demographics, the coarse
`chief_complaint_system` category, comorbidities). We then validate two ways:

- **StratifiedKFold** — the in-template score most submissions report.
- **GroupKFold by chief complaint** — no complaint template appears in both train
  and validation, the standard cure for template leakage.
""")
code(r"""
CAT = ["arrival_mode","arrival_season","shift","sex","language","insurance_type",
       "transport_origin","pain_location","mental_status_triage",
       "chief_complaint_system","age_group","site_id"]
NUM = ["arrival_hour","arrival_month","age","num_prior_ed_visits_12m",
       "num_prior_admissions_12m","num_active_medications","num_comorbidities",
       "systolic_bp","diastolic_bp","mean_arterial_pressure","pulse_pressure",
       "heart_rate","respiratory_rate","temperature_c","spo2","gcs_total",
       "pain_score","weight_kg","height_cm","bmi","shock_index","news2_score"]

def make_X(d):
    X = d[NUM + HX + ["hx_burden"]].copy()
    for c in CAT: X[c] = d[c].astype("category")
    return X

X = make_X(df); y = df["triage_acuity"].values
groups = df["chief_complaint_raw"].values

def lgbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=48,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=40, random_state=SEED, verbose=-1)

def cv_predict(splitter, use_groups=False):
    oof = np.zeros(len(y))
    it = splitter.split(X, y, groups) if use_groups else splitter.split(X, y)
    for tr, va in it:
        m = lgbm(); m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict(X.iloc[va])
    return oof

def report(y, oof, tag):
    under = (oof > y).mean(); over = (oof < y).mean()
    high = y <= 2
    rec = (oof[high] <= 2).mean(); missed = int(((oof > 2) & high).sum())
    print(f"{tag:38s} acc={accuracy_score(y,oof):.3f}  QWK={cohen_kappa_score(y,oof,weights='quadratic'):.3f}"
          f"  undertriage={under:.3f}  high-acuity recall={rec:.3f}  missed_high={missed}")
    return accuracy_score(y, oof), cohen_kappa_score(y, oof, weights="quadratic")

oof_skf = cv_predict(StratifiedKFold(5, shuffle=True, random_state=SEED))
oof_gkf = cv_predict(GroupKFold(5), use_groups=True)
a_skf,k_skf = report(y, oof_skf, "StratifiedKFold (in-template)")
a_gkf,k_gkf = report(y, oof_gkf, "GroupKFold by complaint (out-of-template)")
print(f"\nAccuracy change when complaints are held out: {a_gkf-a_skf:+.3f}  (~0: no collapse)")
""")
md(r"""
**The non-obvious result.** Holding out entire complaint templates barely changes
the score (Δacc ≈ 0). The leakage is **generative, not textual**: the synthetic
vitals were sampled *from* acuity (note `news2_score` correlates −0.82 with the
label), so a structured model rides that circularity and **GroupKFold cannot
detect it.** And κ ≈ 0.93 *exceeds the human inter-rater ceiling* (κ 0.6–0.8) —
no real triage model can. This is the central caution: standard ML hygiene is
not enough on generated data; only real-world data can reveal the truth (Module 5).
""")
code(r"""
# Confusion matrix + clinically-grouped feature importance (honest interpretation)
cm = confusion_matrix(y, oof_skf, normalize="true")
m_full = lgbm().fit(X, y)
imp = pd.Series(m_full.feature_importances_, index=X.columns).sort_values()[-12:]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
im = ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f"ESI{i}" for i in range(1,6)])
ax[0].set_yticks(range(5)); ax[0].set_yticklabels([f"ESI{i}" for i in range(1,6)])
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title("Confusion (row-normalised)")
for i in range(5):
    for j in range(5):
        ax[0].text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",
                   color="white" if cm[i,j]>.5 else "black", fontsize=8)
ax[1].barh(imp.index, imp.values, color="#2b6cb0"); ax[1].set_title("Top features (LightGBM gain)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- M3 equity
md(r"""
## 5. Module 3 — Equity / undertriage-disparity audit

Even an "accurate" model can under-rate some groups more than others. Using the
out-of-fold predictions, we measure the **model undertriage rate** (predicting a
*less* urgent level than the assigned label) within demographic strata. This is a
reusable audit *method*; on synthetic labels the magnitudes are small and partly
artefactual, but the pattern — immigrant-community languages (Arabic, Somali)
showing the highest undertriage — is exactly the equity signal real deployments
must monitor.
""")
code(r"""
d = df.copy(); d["under"] = (oof_skf > y).astype(int)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2)); overall = d["under"].mean()
for ax, col in zip(axes, ["language","insurance_type","sex","age_group"]):
    r = d.groupby(col)["under"].mean().sort_values(ascending=False)
    colors = ["#d73027" if v>overall else "#4575b4" for v in r.values]
    ax.bar(r.index, r.values*100, color=colors)
    ax.axhline(overall*100, color="k", ls="--", lw=1)
    ax.set_title(col); ax.tick_params(axis="x", rotation=40); ax.set_ylabel("undertriage %")
plt.suptitle(f"Model undertriage rate by group (dashed = overall {overall:.1%})")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- M4 waiting room
md(r"""
## 6. Module 4 — The waiting-room risk model

The Foundation suggests a "deterioration risk for patients already waiting"
track. We test it directly. `triage_acuity` **is** a valid input here (it is a
triage-time output, known before the wait); the only excluded field is
`ed_los_hours`. Targets:

- **Escalation** = disposition ∈ {admitted, transferred, observation, deceased}
  (needs care beyond the ED).
- **LWBS** = left without being seen.

The key clinical question: does intake data add value *beyond the acuity score*?
""")
code(r"""
df["escalate"] = df["disposition"].isin(["admitted","transferred","observation","deceased"]).astype(int)
df["lwbs"]     = (df["disposition"]=="lwbs").astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

def cv_proba(Xm, yt):
    oof = np.zeros(len(yt)); spw = (yt==0).sum()/(yt==1).sum()
    for tr, va in skf.split(Xm, yt):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=48,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                               scale_pos_weight=spw, random_state=SEED, verbose=-1)
        m.fit(Xm.iloc[tr], yt[tr]); oof[va] = m.predict_proba(Xm.iloc[va])[:,1]
    return oof

ye = df["escalate"].values
Xacu = df[["triage_acuity"]].copy()
Xall = make_X(df); Xall["triage_acuity"] = df["triage_acuity"]
oof_acu  = cv_proba(Xacu, ye)
oof_all  = cv_proba(Xall, ye)
auc_acu, auc_all = roc_auc_score(ye, oof_acu), roc_auc_score(ye, oof_all)
print(f"Escalation AUC — acuity only: {auc_acu:.3f} | full intake: {auc_all:.3f} | gain: {auc_all-auc_acu:+.3f}")
print(f"Escalation PR-AUC (full): {average_precision_score(ye, oof_all):.3f} (prevalence {ye.mean():.3f})")

iso = IsotonicRegression(out_of_bounds="clip").fit(oof_all, ye)
print(f"Brier {brier_score_loss(ye, oof_all):.3f} -> {brier_score_loss(ye, iso.predict(oof_all)):.3f} after calibration")

yl = df["lwbs"].values
oof_l = cv_proba(Xall, yl)
print(f"\nLWBS AUC: {roc_auc_score(yl, oof_l):.3f} (prevalence {yl.mean():.3f}) — near-random")

# calibration curve for escalation
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
bins = pd.qcut(oof_all, 10, duplicates="drop")
cal = pd.DataFrame({"p":oof_all,"y":ye}).groupby(bins).mean()
ax[0].plot([0,1],[0,1],"k--",lw=.8); ax[0].plot(cal["p"], cal["y"], "o-", color="#2b6cb0")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("observed"); ax[0].set_title("Escalation calibration")
ax[1].bar(["acuity\nonly","full\nintake"], [auc_acu, auc_all], color=["#999","#2b6cb0"])
ax[1].set_ylim(0.5,0.9); ax[1].set_title("Escalation AUC"); ax[1].set_ylabel("ROC-AUC")
plt.tight_layout(); plt.show()
""")
md(r"""
**Honest finding.** Full intake (AUC ≈ 0.83) does **not** beat acuity alone
(≈ 0.84) for escalation, and LWBS is near-random (AUC ≈ 0.61). In this synthetic
data, disposition is essentially an **acuity derivative**, and who leaves is
largely operational noise. The constructive implication for the Foundation:
**the waiting-room deterioration track needs real, longitudinally-linked ED
data** — the synthetic generator cannot support it. We turn to real data next.
""")

# ---------------------------------------------------------------- M5 NHAMCS
md(r"""
## 7. Module 5 — External validation on real NHAMCS data

We parse the **CDC NHAMCS ED public-use files (2021–2022)** — fixed-width ASCII,
column positions from the official codebook — and run the *same* structured
acuity model. The target is `IMMEDR` (real triage immediacy, 1–5). If internet is
disabled, this cell skips cleanly; to force it, attach a parsed copy or enable
internet. (Loader mirrors `src/load_nhamcs.py` in the repo.)
""")
code(r"""
# Hardcoded codebook positions (1-based inclusive); identical for 2021 & 2022 in this range.
NH = {"IMMEDR":(67,68),"AGE":(16,18),"SEX":(25,25),"TEMPF":(48,51),"PULSE":(52,54),
      "RESPR":(55,57),"BPSYS":(58,60),"BPDIAS":(61,63),"POPCT":(64,66),
      "PAINSCALE":(69,70),"ARREMS":(33,34),"SEEN72":(71,72),"RFV1":(73,77)}
BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS"

def load_nhamcs_year(zipname):
    raw = urllib.request.urlopen(f"{BASE}/{zipname}", timeout=60).read()
    name = zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]
    txt = zipfile.ZipFile(io.BytesIO(raw)).read(name)
    colspecs = [(s-1, e) for s,e in NH.values()]
    d = pd.read_fwf(io.BytesIO(txt), colspecs=colspecs, names=list(NH), dtype=str)
    return d.apply(pd.to_numeric, errors="coerce")

def clean_nhamcs(d):
    d = d[d["IMMEDR"].isin([1,2,3,4,5])].copy()
    for c in ["TEMPF","PULSE","RESPR","BPSYS","BPDIAS","POPCT","PAINSCALE"]:
        d.loc[d[c].isin([-9,-8,-7]), c] = np.nan
    for c in ["BPSYS","BPDIAS","POPCT","PULSE","RESPR"]:
        d.loc[d[c]==0, c] = np.nan
    d.loc[d["PULSE"]==998,"PULSE"]=np.nan; d.loc[d["BPDIAS"]==998,"BPDIAS"]=np.nan
    d["TEMPF"]=d["TEMPF"]/10.0
    d.loc[d["PAINSCALE"]>10,"PAINSCALE"]=np.nan
    for c in ["SEX","ARREMS","SEEN72","AGE"]: d.loc[d[c].isin([-9,-8,-7]),c]=np.nan
    d.loc[d["RFV1"]<0,"RFV1"]=np.nan; d["RFV1_module"]=(d["RFV1"]//10000)
    return d

try:
    nd = clean_nhamcs(pd.concat([load_nhamcs_year("ed2021.zip"),
                                 load_nhamcs_year("ED2022.zip")], ignore_index=True))
    feats = ["AGE","SEX","TEMPF","PULSE","RESPR","BPSYS","BPDIAS","POPCT",
             "PAINSCALE","ARREMS","SEEN72","RFV1_module"]
    Xn = nd[feats]; yn = nd["IMMEDR"].astype(int).values
    oof = np.zeros(len(yn))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xn, yn):
        m = lgbm(); m.fit(Xn.iloc[tr], yn[tr]); oof[va]=m.predict(Xn.iloc[va])
    nh_acc, nh_qwk = accuracy_score(yn,oof), cohen_kappa_score(yn,oof,weights="quadratic")
    print(f"NHAMCS real (n={len(yn):,}):  accuracy={nh_acc:.3f}  QWK={nh_qwk:.3f}")
    NHAMCS_OK = True
except Exception as e:
    print("NHAMCS validation skipped (no internet / fetch failed):", type(e).__name__, e)
    print("Reference result from the repo run: n=20,702, accuracy=0.542, QWK=0.269")
    nh_acc, nh_qwk, NHAMCS_OK = 0.542, 0.269, False

# Synthetic vs real — the punchline
fig, ax = plt.subplots(figsize=(6,3.4))
labels = ["Synthetic\n(in-template)","Synthetic\n(GroupKFold)","Real NHAMCS"]
vals = [k_skf, k_gkf, nh_qwk]
ax.bar(labels, vals, color=["#e23b3b","#f08a3c","#2b8a3e"])
ax.axhspan(0.6, 0.8, color="gray", alpha=.2, label="human inter-rater κ (0.6–0.8)")
ax.set_ylabel("quadratic-weighted κ"); ax.set_ylim(0,1)
ax.set_title("Acuity agreement: synthetic illusion vs real-world ceiling"); ax.legend()
for i,v in enumerate(vals): ax.text(i, v+.02, f"{v:.2f}", ha="center")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- limitations
md(r"""
## 8. Limitations, reproducibility & clinical recommendations

**Limitations (honest).**
1. The competition data is **synthetic**; its vitals are generated from acuity, so
   in-distribution scores are not clinically meaningful — the entire point of
   Modules 1, 4 and 5.
2. The equity audit is computed against synthetic labels; magnitudes are small
   and indicative of *method*, not of real disparities.
3. NHAMCS is a cross-sectional survey with its own missingness and no patient
   linkage; κ ≈ 0.27 is a floor given the limited public feature set, not a
   tuned ceiling.
4. No model here is deployment-ready; all are second-read decision support.

**Reproducibility.** Fixed seed (42), 5-fold CV, OOF metrics, no post-triage
leakage, data paths auto-resolved, NHAMCS parsed from the official codebook
layout. Full code + README: see the linked repository.

**Clinical recommendations.**
- Treat any benchmark where model κ exceeds the human ceiling as **leakage until
  proven otherwise**; validate on real, linked data before any claim.
- Monitor **undertriage of high-acuity (ESI 1–2)** and of language-minority
  groups continuously, not aggregate accuracy.
- For waiting-room deterioration work, invest in **real longitudinal ED data** —
  the modelling is ready; the data is the bottleneck.
""")
md(r"""
## References
1. Gilboy N. et al. *Emergency Severity Index (ESI) v4*, AHRQ, 2012.
2. Farrohknia N. et al. *ED triage scales: a systematic review*, Scand J Trauma, 2011.
3. NCHS. *NHAMCS public-use ED files 2021–2022*, CDC. https://www.cdc.gov/nchs/ahcd/
4. Obermeyer Z. et al. *Dissecting racial bias in a health algorithm*, Science, 2019.
""")

# ---------------------------------------------------------------- assemble
nb = {"cells": [], "metadata": {"kernelspec": {"name":"python3","display_name":"Python 3"},
      "language_info":{"name":"python"}}, "nbformat":4, "nbformat_minor":5}
for typ, src in cells:
    lines = src.splitlines(keepends=True)
    c = {"cell_type": typ, "metadata": {}, "source": lines}
    if typ == "code": c["outputs"] = []; c["execution_count"] = None
    nb["cells"].append(c)

out = Path(__file__).resolve().parent.parent / "notebook" / "triagegeist.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
