"""
Build the leakage-safe feature matrix for the WAITING-ROOM RISK task and run a
quick LightGBM baseline to confirm learnable signal for two targets:

  - LWBS  : patient left without being seen (disposition == 'lwbs')
  - ESCALATE : needs care beyond the ED
              (disposition in {admitted, transferred, observation, deceased})

Prediction time = the moment triage is completed (patient enters the waiting
room). Therefore `triage_acuity` IS a valid input (it is a triage-time output,
known before the wait). The only excluded post-hoc field is `ed_los_hours`
(length of stay, realised concurrently with the disposition outcome).
`disposition` is the target, not a feature.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "comp_data"

ESCALATE_SET = {"admitted", "transferred", "observation", "deceased"}

# Categorical intake fields (known at triage).
CAT = ["arrival_mode", "arrival_season", "shift", "sex", "language",
       "insurance_type", "transport_origin", "pain_location",
       "mental_status_triage", "chief_complaint_system", "age_group", "site_id"]

# Numeric intake fields (vitals + derived scores + utilization + context).
NUM = ["arrival_hour", "arrival_month", "age",
       "num_prior_ed_visits_12m", "num_prior_admissions_12m",
       "num_active_medications", "num_comorbidities",
       "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
       "pulse_pressure", "heart_rate", "respiratory_rate", "temperature_c",
       "spo2", "gcs_total", "pain_score", "weight_kg", "height_cm", "bmi",
       "shock_index", "news2_score", "triage_acuity"]

LEAK = ["ed_los_hours", "disposition"]  # never used as features


def load(split="train"):
    df = pd.read_csv(DATA / f"{split}.csv")
    hist = pd.read_csv(DATA / "patient_history.csv")
    hx = [c for c in hist.columns if c.startswith("hx_")]
    df = df.merge(hist[["patient_id"] + hx], on="patient_id", how="left")
    df["hx_burden"] = df[hx].sum(axis=1)
    # pain_score uses -1 as a missing sentinel.
    df["pain_score"] = df["pain_score"].replace(-1, np.nan)
    return df, hx


def make_targets(df):
    df = df.copy()
    df["lwbs"] = (df["disposition"] == "lwbs").astype(int)
    df["escalate"] = df["disposition"].isin(ESCALATE_SET).astype(int)
    return df


def feature_frame(df, hx):
    cols_num = NUM + hx + ["hx_burden"]
    X = df[cols_num].copy()
    for c in CAT:
        X[c] = df[c].astype("category")
    return X


if __name__ == "__main__":
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    df, hx = load("train")
    df = make_targets(df)
    X = feature_frame(df, hx)
    print(f"features: {X.shape[1]}  rows: {len(X)}")

    for target in ["lwbs", "escalate"]:
        y = df[target].values
        print(f"\n=== {target}  (prevalence {y.mean():.3f}) ===")
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        spw = (y == 0).sum() / (y == 1).sum()
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.03, num_leaves=48,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                scale_pos_weight=spw, random_state=42, verbose=-1)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"  ROC-AUC  {roc_auc_score(y, oof):.4f}")
        print(f"  PR-AUC   {average_precision_score(y, oof):.4f}  (baseline {y.mean():.4f})")
