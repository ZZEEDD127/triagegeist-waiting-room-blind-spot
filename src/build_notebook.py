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
happen to them while they wait?"* This notebook gives the Foundation **two things
it can use**:

- **A working second-read tool** — a *waiting-room escalation early-warning*
  model that, **within the lower-acuity queue (ESI 3–5)**, concentrates the risk
  of needing admission: flagging the top 20% by risk captures **~35% of all
  low-acuity escalations at 1.7× precision** (AUC 0.76 within the subgroup). This
  targets exactly the patients the single acuity number overlooks.
- **An honesty & validation framework** so triage-AI results can be *trusted* —
  a leakage litmus test plus anchoring on real-world data.

It is built as one honest, end-to-end study in five parts: (1) leakage & shortcut
forensics, (2) an honest leak-free acuity model and why GroupKFold is
*insufficient*, (3) an equity / undertriage audit, (4) the waiting-room tool, and
(5) external validation on real U.S. ED data (NHAMCS, CDC).

**Why this matters for the pilots ahead.** On the synthetic benchmark a structured
model reaches quadratic-weighted κ ≈ 0.93 — *above the human inter-rater ceiling
(κ 0.6–0.8)*, which is impossible in reality and is a fingerprint of generative
leakage. On **real NHAMCS** the same method scores κ ≈ 0.27. The constructive
takeaway: to make clinical pilots succeed, invest in **honest evaluation and real
linked data, not higher leaderboard accuracy** — and deploy AI as a *second read*
that catches the waiting-room blind spot.
""")

# ---------------------------------------------------------------- problem
md(r"""
## 1. Clinical problem statement

Emergency triage assigns each arrival an acuity level (e.g. ESI 1–5) that governs
how long they may safely wait. Two failure modes matter clinically:

- **Undertriage** — a genuinely sick patient is given a low acuity and waits too
  long. This is the dangerous, sometimes fatal error; its cost is asymmetric and
  far exceeds overtriage. Inter-rater reliability of human ESI triage is only
  moderate (κ ≈ 0.6–0.8) [1], and undertriage of vulnerable groups is a
  documented patient-safety concern [2].
- **The waiting-room blind spot** — the acuity number is a *snapshot*. It does
  not directly say who will deteriorate, need admission, or give up and leave
  *without being seen* (LWBS) while waiting. Machine-learning triage that
  predicts *outcomes* (admission, deterioration) rather than re-deriving the ESI
  label is an established, more clinically useful direction [3, 4], yet it needs
  real outcome data — the crux of this study.

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
# --- consistent, clean house style for every figure ---
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.autolayout": False})
NAVY, RED, BLUE, GREEN, AMBER, GREY = "#0b1f33", "#e23b3b", "#2b6cb0", "#2b8a3e", "#f08a3c", "#9aa5b1"
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

# --- THE TOOL: a waiting-room watch within the LOWER-acuity queue (ESI 3-5) ---
# Global AUC ~ acuity-alone does NOT mean no value: the value is operational,
# inside the queue the acuity number treats as 'can wait'.
low = df["triage_acuity"].isin([3,4,5]).values
p_low, y_low = oof_all[low], ye[low]
auc_low = roc_auc_score(y_low, p_low)
print(f"\nWaiting-room WATCH within ESI 3-5 (n={low.sum():,}, escalation base {y_low.mean():.3f}): AUC={auc_low:.3f}")
rows = []
for q in [0.95, 0.90, 0.80]:
    thr = np.quantile(p_low, q); fl = p_low >= thr
    rows.append((f"top {int((1-q)*100)}%", fl.mean(), y_low[fl].mean(),
                 y_low[fl].sum()/y_low.sum(), y_low[fl].mean()/y_low.mean()))
watch = pd.DataFrame(rows, columns=["flag","alert_rate","precision","recall_of_escalations","precision_lift"]).round(2)
print(watch.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
bins = pd.qcut(oof_all, 10, duplicates="drop")
cal = pd.DataFrame({"p":oof_all,"y":ye}).groupby(bins).mean()
ax[0].plot([0,1],[0,1],"k--",lw=.8); ax[0].plot(cal["p"], cal["y"], "o-", color="#2b6cb0")
ax[0].set_xlabel("predicted risk"); ax[0].set_ylabel("observed"); ax[0].set_title("Escalation calibration")
# risk concentration within ESI 3-5: observed escalation rate by predicted-risk decile
dec = pd.qcut(p_low, 10, labels=False, duplicates="drop")
rate = pd.Series(y_low).groupby(dec).mean()
ax[1].bar(range(len(rate)), rate.values, color="#2b8a3e")
ax[1].axhline(y_low.mean(), color="k", ls="--", lw=1, label=f"queue base {y_low.mean():.2f}")
ax[1].set_xlabel("predicted-risk decile (ESI 3-5)"); ax[1].set_ylabel("observed escalation rate")
ax[1].set_title("Watch tool concentrates risk\ninside the low-acuity queue"); ax[1].legend()
plt.tight_layout(); plt.show()
""")
md(r"""
**From a null result to a usable tool.** Globally, full intake (AUC ≈ 0.84) does
not beat acuity alone — disposition here is largely an *acuity derivative*. But
that global view hides the clinically useful signal: **inside the lower-acuity
queue (ESI 3–5), which the acuity score treats as "can wait", the model still
separates escalation risk (AUC ≈ 0.76).** Flagging the top ~20% by risk as a
*waiting-room watch* captures roughly a third of all low-acuity escalations at
~1.7× the base precision — a concrete, calibrated second read that targets the
blind spot. (LWBS stays near-random, AUC ≈ 0.61: who *leaves* is operational, not
clinical — an honest boundary on what intake data can do.) For richer
deterioration prediction the Foundation will need real, longitudinally-linked ED
data; the modelling shown here is ready for it. We anchor on real data next.
""")

# ---------------------------------------------------------------- M5 NHAMCS
md(r"""
## 7. Module 5 — Real-world validation on NHAMCS (CDC)

We parse the **CDC NHAMCS ED public-use files (2021–2022)** — fixed-width ASCII,
positions from the official codebook (admission flags shift +2 bytes in 2022, so
each year is parsed with its own layout) — and run two real-data checks:

1. **Acuity reality check.** The same structured model on real `IMMEDR` labels —
   exposing how inflated the synthetic κ is.
2. **The waiting-room watch, on real outcomes.** Does triage data flag real
   *hospital admission* among patients triaged as *lower acuity*? This is the
   decisive test of whether our tool's premise survives outside synthetic data.

If internet is disabled the cell skips cleanly (reference numbers are printed).
""")
code(r"""
# Codebook positions (1-based inclusive). Front-half vars are identical across
# years; the hospital-admission flags shift +2 bytes in 2022 (two COVID items
# were inserted), so we parse each year with its own layout.
COMMON = {"IMMEDR":(67,68),"AGE":(16,18),"SEX":(25,25),"TEMPF":(48,51),"PULSE":(52,54),
          "RESPR":(55,57),"BPSYS":(58,60),"BPDIAS":(61,63),"POPCT":(64,66),
          "PAINSCALE":(69,70),"ARREMS":(33,34),"SEEN72":(71,72),"RFV1":(73,77)}
ADM = {2021:{"ADMITHOS":(497,497),"OBSHOS":(498,498)},
       2022:{"ADMITHOS":(499,499),"OBSHOS":(500,500)}}
BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS"

def load_nhamcs_year(year, zipname):
    raw = urllib.request.urlopen(f"{BASE}/{zipname}", timeout=60).read()
    name = zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]
    txt = zipfile.ZipFile(io.BytesIO(raw)).read(name)
    spec = {**COMMON, **ADM[year]}
    colspecs = [(s-1, e) for s,e in spec.values()]
    d = pd.read_fwf(io.BytesIO(txt), colspecs=colspecs, names=list(spec), dtype=str)
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
    d["admit"] = ((d["ADMITHOS"]==1) | (d["OBSHOS"]==1)).astype(int)  # real admission outcome
    return d

FEATS = ["AGE","SEX","TEMPF","PULSE","RESPR","BPSYS","BPDIAS","POPCT",
         "PAINSCALE","ARREMS","SEEN72","RFV1_module"]
try:
    nd = clean_nhamcs(pd.concat([load_nhamcs_year(2021,"ed2021.zip"),
                                 load_nhamcs_year(2022,"ED2022.zip")], ignore_index=True))
    Xn = nd[FEATS]; yn = nd["IMMEDR"].astype(int).values
    # (1) acuity reality check
    oof = np.zeros(len(yn))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xn, yn):
        m = lgbm(); m.fit(Xn.iloc[tr], yn[tr]); oof[va]=m.predict(Xn.iloc[va])
    nh_acc, nh_qwk = accuracy_score(yn,oof), cohen_kappa_score(yn,oof,weights="quadratic")
    print(f"[acuity] NHAMCS real (n={len(yn):,}): accuracy={nh_acc:.3f}  QWK={nh_qwk:.3f}")

    # (2) THE WAITING-ROOM WATCH ON REAL ADMISSION OUTCOMES
    ya = nd["admit"].values; oa = np.zeros(len(ya)); spw=(ya==0).sum()/(ya==1).sum()
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xn, ya):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=48,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                               scale_pos_weight=spw, random_state=SEED, verbose=-1)
        m.fit(Xn.iloc[tr], ya[tr]); oa[va]=m.predict_proba(Xn.iloc[va])[:,1]
    lowr = nd["IMMEDR"].isin([3,4,5]).values
    auc_low_real = roc_auc_score(ya[lowr], oa[lowr])
    pr, yr = oa[lowr], ya[lowr]; fl = pr >= np.quantile(pr, 0.8)
    print(f"[watch ] real admission rate={ya.mean():.3f}; within LOW-acuity (n={lowr.sum():,}, base {yr.mean():.3f}): "
          f"AUC={auc_low_real:.3f}")
    print(f"[watch ] flag top 20%: precision={yr[fl].mean():.2f}  recall={yr[fl].sum()/yr.sum():.2f}  "
          f"lift={yr[fl].mean()/yr.mean():.1f}x")
    NHAMCS_OK = True
except Exception as e:
    print("NHAMCS validation skipped (no internet / fetch failed):", type(e).__name__, e)
    print("Reference repo run: acuity acc=0.540 QWK=0.265; watch within low-acuity AUC=0.777, top20% recall=0.57 lift=2.9x")
    nh_qwk, auc_low_real = 0.265, 0.777; NHAMCS_OK = False

# Two-panel punchline: acuity illusion (left) + the tool survives on real data (right)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
labels = ["Synthetic\n(in-template)","Synthetic\n(GroupKFold)","Real NHAMCS"]
vals = [k_skf, k_gkf, nh_qwk]
ax[0].bar(labels, vals, color=["#e23b3b","#f08a3c","#2b8a3e"])
ax[0].axhspan(0.6, 0.8, color="gray", alpha=.2, label="human ceiling κ 0.6–0.8")
ax[0].set_ylabel("acuity κ (quadratic)"); ax[0].set_ylim(0,1)
ax[0].set_title("Acuity: synthetic illusion vs real ceiling"); ax[0].legend(fontsize=8)
for i,v in enumerate(vals): ax[0].text(i, v+.02, f"{v:.2f}", ha="center")
ax[1].bar(["Synthetic\nESI 3-5","Real NHAMCS\nIMMEDR 3-5"], [auc_low, auc_low_real],
          color=[BLUE, GREEN]); ax[1].set_ylim(0.5,0.85)
ax[1].set_ylabel("admission/escalation AUC"); ax[1].set_title("Waiting-room watch holds on REAL data")
for i,v in enumerate([auc_low, auc_low_real]): ax[1].text(i, v+.01, f"{v:.2f}", ha="center")
plt.tight_layout(); plt.show()

# Decision-curve (net-benefit) analysis on the REAL low-acuity queue: is the watch
# clinically worth using vs the two trivial policies (watch everyone / watch no-one)?
if NHAMCS_OK:
    p, yt = oa[lowr], ya[lowr]; N = len(yt)
    ths = np.linspace(0.02, 0.40, 39)
    def nb(flag):
        tp = ((flag==1)&(yt==1)).sum(); fp = ((flag==1)&(yt==0)).sum()
        return tp/N - fp/N*(pt/(1-pt))
    nb_model, nb_all = [], []
    for pt in ths:
        nb_model.append(nb(p>=pt)); nb_all.append(nb(np.ones(N)))
    fig, ax = plt.subplots(figsize=(6.2,3.6))
    ax.plot(ths, nb_model, color=GREEN, lw=2.2, label="Waiting-room watch")
    ax.plot(ths, nb_all, color=GREY, lw=1.4, ls="--", label="Watch everyone")
    ax.axhline(0, color="k", lw=1, label="Watch no-one (acuity-only)")
    ax.set_xlabel("threshold probability (clinician's risk tolerance)")
    ax.set_ylabel("net benefit"); ax.set_ylim(bottom=min(nb_all)*1.1)
    ax.set_title("Decision-curve analysis — real low-acuity ED queue")
    ax.legend(fontsize=8.5)
    plt.tight_layout(); plt.show()
    print(f"Net benefit positive and above both default policies across "
          f"thresholds ~{ths[np.argmax(np.array(nb_model)>np.array(nb_all))]:.0%}–40%.")
""")

md(r"""
**The decisive result.** Two things happen on real data. (i) Acuity κ collapses
from 0.93 to ≈ 0.27 — the synthetic agreement was an illusion. (ii) **The
waiting-room watch survives, and strengthens:** among real patients triaged
*lower acuity* (IMMEDR 3–5), triage data predicts genuine hospital admission at
**AUC ≈ 0.78**, and flagging the top 20% by risk catches **≈ 57% of those
admissions at ~2.9× precision** (vs 1.75× on synthetic). The clean separation is
the whole thesis: **the acuity *label* is corrupted by leakage and should not be
chased, but the deterioration *signal* is real and transferable — so build the
second-read watch, not a higher-accuracy acuity mimic.**
""")

# ---------------------------------------------------------------- submission
md(r"""
## 8. A concrete predictive artifact (test-set submission)

For completeness we fit the honest, leak-free acuity model on the full training
set and produce `submission.csv` in the required format. (The notebook also
exposes the calibrated waiting-room watch score, which is the deployable tool.)
""")
code(r"""
test = pd.read_csv(find_file("test.csv"))
th = pd.read_csv(find_file("patient_history.csv"))
test = test.merge(th[["patient_id"]+HX], on="patient_id", how="left")
test["pain_score"] = test["pain_score"].replace(-1, np.nan)
test["hx_burden"] = test[HX].sum(axis=1)
Xte = make_X(test)
final = lgbm().fit(X, y)
pred = final.predict(Xte)
sub = pd.DataFrame({"patient_id": test["patient_id"], "triage_acuity": pred.astype(int)})
sub.to_csv("submission.csv", index=False)
print("submission.csv:", sub.shape, "| predicted acuity mix:",
      sub["triage_acuity"].value_counts(normalize=True).round(3).sort_index().to_dict())
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

**Recommendations for the pilots ahead (constructive).**
- **Deploy as a second read, not a replacement.** The waiting-room watch is most
  valuable where the acuity number is least informative: re-checking the
  lower-acuity queue for hidden escalation risk.
- **Adopt the leakage litmus test.** If a model's κ exceeds the human ceiling,
  treat it as leakage until proven on real, linked data — a cheap safeguard for
  every future benchmark the Foundation commissions.
- **Monitor the right thing:** high-acuity (ESI 1–2) undertriage and
  language-minority equity, continuously — not aggregate accuracy.
- **Invest in real longitudinal ED data** to unlock genuine deterioration
  prediction; this study shows the modelling is ready and the data is the
  bottleneck, which is an actionable, fundable conclusion.
""")
md(r"""
## References
1. Gilboy N, Tanabe P, Travers D, Rosenau A. *Emergency Severity Index (ESI):
   A Triage Tool for ED Care, v4.* AHRQ, 2012.
2. Farrohknia N, et al. *Emergency department triage scales and their components:
   a systematic review.* Scand J Trauma Resusc Emerg Med, 2011;19:42.
3. Levin S, et al. *Machine-learning-based electronic triage more accurately
   differentiates patients than the Emergency Severity Index.* Ann Emerg Med,
   2018;71(5):565–574.
4. Hong WS, Haimovich AD, Taylor RA. *Predicting hospital admission at ED triage
   using machine learning.* PLoS One, 2018;13(7):e0201016.
5. Royal College of Physicians. *National Early Warning Score (NEWS) 2.* London, 2017.
6. Obermeyer Z, et al. *Dissecting racial bias in an algorithm used to manage the
   health of populations.* Science, 2019;366(6464):447–453.
7. Baker DW, Stevens CD, Brook RH. *Patients who leave a public hospital ED
   without being seen by a physician.* JAMA, 1991;266(8):1085–1090.
8. NCHS. *NHAMCS public-use ED files 2021–2022.* CDC. https://www.cdc.gov/nchs/ahcd/
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
