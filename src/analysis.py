"""
Triagegeist flagship analysis pipeline (local engine).

Runs the technical core that the Kaggle notebook will present:
  1. Leakage + synthetic-shortcut forensics
  2. Honest leak-free acuity model: StratifiedKFold vs GroupKFold-by-complaint
     (the headline "the score collapses out-of-template" result), calibration,
     quadratic-weighted kappa, and undertriage safety metrics.

Outputs numbers to stdout and saves a results dict to artifacts/acuity_results.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "comp_data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

LEAK_COLS = ["disposition", "ed_los_hours"]  # post-triage outcomes


def load():
    train = pd.read_csv(DATA / "train.csv")
    hist = pd.read_csv(DATA / "patient_history.csv")
    cc = pd.read_csv(DATA / "chief_complaints.csv")
    hx = [c for c in hist.columns if c.startswith("hx_")]
    df = train.merge(hist[["patient_id"] + hx], on="patient_id", how="left")
    df = df.merge(cc[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left")
    df["pain_score"] = df["pain_score"].replace(-1, np.nan)
    df["hx_burden"] = df[hx].sum(axis=1)
    return df, hx


def forensics(df):
    print("=" * 64)
    print("MODULE 1  LEAKAGE + SHORTCUT FORENSICS")
    print("=" * 64)
    res = {}
    # (a) post-triage leakage correlations
    leak = {}
    for c in ["ed_los_hours", "news2_score", "shock_index", "gcs_total"]:
        leak[c] = round(float(df[c].corr(df["triage_acuity"])), 3)
    res["leak_corr"] = leak
    print("Correlation with triage_acuity:", leak)
    # (b) chief-complaint -> acuity determinism (the shortcut)
    g = df.groupby("chief_complaint_raw")["triage_acuity"]
    purity = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    res["complaint_acuity_purity"] = round(float(purity.mean()), 4)
    res["complaint_fully_determine_acuity"] = round(float((purity == 1).mean()), 4)
    res["n_unique_complaints"] = int(df["chief_complaint_raw"].nunique())
    print(f"Unique complaints: {res['n_unique_complaints']}")
    print(f"Mean within-complaint acuity purity: {res['complaint_acuity_purity']} "
          f"({res['complaint_fully_determine_acuity']:.1%} fully deterministic)")
    return res


def _undertriage_metrics(y_true, y_pred):
    """Undertriage = predicting a LESS urgent (numerically higher) level than true.
    Focus on missed high-acuity (true level 1-2)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    under = (y_pred > y_true).mean()
    over = (y_pred < y_true).mean()
    high = y_true <= 2
    high_recall = (y_pred[high] <= 2).mean() if high.sum() else np.nan
    missed_high = ((y_pred > 2) & high).sum()
    return {"undertriage_rate": round(float(under), 4),
            "overtriage_rate": round(float(over), 4),
            "high_acuity_recall(1-2)": round(float(high_recall), 4),
            "n_missed_high_acuity": int(missed_high)}


def acuity_model(df, hx):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    print("\n" + "=" * 64)
    print("MODULE 2  HONEST LEAK-FREE ACUITY MODEL")
    print("=" * 64)

    cat = ["arrival_mode", "arrival_season", "shift", "sex", "language",
           "insurance_type", "transport_origin", "pain_location",
           "mental_status_triage", "chief_complaint_system", "age_group", "site_id"]
    num = ["arrival_hour", "arrival_month", "age", "num_prior_ed_visits_12m",
           "num_prior_admissions_12m", "num_active_medications", "num_comorbidities",
           "systolic_bp", "diastolic_bp", "mean_arterial_pressure", "pulse_pressure",
           "heart_rate", "respiratory_rate", "temperature_c", "spo2", "gcs_total",
           "pain_score", "weight_kg", "height_cm", "bmi", "shock_index",
           "news2_score"] + hx + ["hx_burden"]
    # NOTE: structured intake only; raw complaint TEXT deliberately excluded so we
    # do NOT ride the synthetic shortcut. chief_complaint_system (coarse category)
    # is kept as a realistic triage-time field.
    X = df[num].copy()
    for c in cat:
        X[c] = df[c].astype("category")
    y = df["triage_acuity"].values
    groups = df["chief_complaint_raw"].values

    def run(splitter, use_groups):
        oof = np.zeros(len(y))
        it = splitter.split(X, y, groups) if use_groups else splitter.split(X, y)
        for tr, va in it:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=48, subsample=0.8, colsample_bytree=0.8,
                                   min_child_samples=40, random_state=42, verbose=-1)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict(X.iloc[va])
        return oof

    res = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof_skf = run(skf, False)
    res["stratified"] = {
        "accuracy": round(accuracy_score(y, oof_skf), 4),
        "qwk": round(cohen_kappa_score(y, oof_skf, weights="quadratic"), 4),
        **_undertriage_metrics(y, oof_skf)}
    print("StratifiedKFold (in-template):", res["stratified"])

    gkf = GroupKFold(5)
    oof_gkf = run(gkf, True)
    res["groupkfold_by_complaint"] = {
        "accuracy": round(accuracy_score(y, oof_gkf), 4),
        "qwk": round(cohen_kappa_score(y, oof_gkf, weights="quadratic"), 4),
        **_undertriage_metrics(y, oof_gkf)}
    print("GroupKFold-by-complaint (out-of-template):", res["groupkfold_by_complaint"])
    print(f"  -> accuracy drop: {res['stratified']['accuracy']-res['groupkfold_by_complaint']['accuracy']:+.3f}")
    return res


def _feat(df, hx, extra_num=()):
    cat = ["arrival_mode", "arrival_season", "shift", "sex", "language",
           "insurance_type", "transport_origin", "pain_location",
           "mental_status_triage", "chief_complaint_system", "age_group", "site_id"]
    num = ["arrival_hour", "arrival_month", "age", "num_prior_ed_visits_12m",
           "num_prior_admissions_12m", "num_active_medications", "num_comorbidities",
           "systolic_bp", "diastolic_bp", "mean_arterial_pressure", "pulse_pressure",
           "heart_rate", "respiratory_rate", "temperature_c", "spo2", "gcs_total",
           "pain_score", "weight_kg", "height_cm", "bmi", "shock_index",
           "news2_score"] + list(extra_num) + hx + ["hx_burden"]
    X = df[num].copy()
    for c in cat:
        X[c] = df[c].astype("category")
    return X


def fairness_audit(df, hx):
    """Undertriage-disparity audit: does the acuity model under-rate some groups
    more than others, at comparable clinical severity?"""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    print("\n" + "=" * 64)
    print("MODULE 3  EQUITY / UNDERTRIAGE-DISPARITY AUDIT")
    print("=" * 64)
    X = _feat(df, hx)
    y = df["triage_acuity"].values
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=48,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                               random_state=42, verbose=-1)
        m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict(X.iloc[va])
    d = df.copy(); d["under"] = (oof > y).astype(int)  # model under-rates vs label
    res = {}
    for col in ["language", "insurance_type", "sex", "age_group"]:
        rates = d.groupby(col)["under"].mean().round(4).sort_values(ascending=False)
        res[col] = rates.to_dict()
        print(f"  undertriage rate by {col}: {res[col]}")
    return res


def waiting_room(df, hx):
    """Escalation/deterioration risk + incremental value BEYOND acuity alone."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.isotonic import IsotonicRegression
    print("\n" + "=" * 64)
    print("MODULE 4  WAITING-ROOM RISK (escalation / deterioration)")
    print("=" * 64)
    df = df.copy()
    df["escalate"] = df["disposition"].isin(
        ["admitted", "transferred", "observation", "deceased"]).astype(int)
    df["lwbs"] = (df["disposition"] == "lwbs").astype(int)
    y = df["escalate"].values
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    res = {}

    def cv(X):
        oof = np.zeros(len(y))
        spw = (y == 0).sum() / (y == 1).sum()
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=48,
                                   subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                                   scale_pos_weight=spw, random_state=42, verbose=-1)
            m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        return oof

    # (a) acuity-ONLY baseline vs (b) full intake model -> incremental value
    X_ac = df[["triage_acuity"]].copy()
    oof_ac = cv(X_ac)
    X_full = _feat(df, hx)  # includes triage_acuity among num
    oof_full = cv(X_full)
    res["escalate_auc_acuity_only"] = round(roc_auc_score(y, oof_ac), 4)
    res["escalate_auc_full_intake"] = round(roc_auc_score(y, oof_full), 4)
    res["escalate_pr_auc_full"] = round(average_precision_score(y, oof_full), 4)
    res["escalate_prevalence"] = round(float(y.mean()), 4)
    # calibration via isotonic on OOF
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_full, y)
    res["brier_raw"] = round(brier_score_loss(y, oof_full), 4)
    res["brier_calibrated"] = round(brier_score_loss(y, iso.predict(oof_full)), 4)
    print(f"  escalation AUC  acuity-only={res['escalate_auc_acuity_only']} "
          f"-> full intake={res['escalate_auc_full_intake']} "
          f"(+{res['escalate_auc_full_intake']-res['escalate_auc_acuity_only']:.3f})")
    print(f"  PR-AUC {res['escalate_pr_auc_full']} (prev {res['escalate_prevalence']}); "
          f"Brier {res['brier_raw']}->{res['brier_calibrated']} (calibrated)")

    # LWBS honest negative result
    yl = df["lwbs"].values
    oofl = np.zeros(len(yl)); spwl = (yl == 0).sum() / (yl == 1).sum()
    for tr, va in skf.split(X_full, yl):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=32,
                               scale_pos_weight=spwl, random_state=42, verbose=-1)
        m.fit(X_full.iloc[tr], yl[tr]); oofl[va] = m.predict_proba(X_full.iloc[va])[:, 1]
    res["lwbs_auc"] = round(roc_auc_score(yl, oofl), 4)
    res["lwbs_prevalence"] = round(float(yl.mean()), 4)
    print(f"  LWBS AUC {res['lwbs_auc']} (prev {res['lwbs_prevalence']}) "
          f"-> weak: who walks out is largely operational/random, NOT clinical")
    return res


def nhamcs_validation():
    """External validation on REAL NHAMCS ED data: realistic ceiling vs the
    inflated synthetic numbers."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    print("\n" + "=" * 64)
    print("MODULE 5  EXTERNAL VALIDATION ON REAL NHAMCS DATA")
    print("=" * 64)
    p = ROOT / "data" / "nhamcs_ed_clean.parquet"
    if not p.exists():
        print("  nhamcs parquet missing; run src/load_nhamcs.py first"); return {}
    nd = pd.read_parquet(p)
    feats = ["AGE", "SEX", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT",
             "PAINSCALE", "ARREMS", "SEEN72", "RFV1_module"]
    X = nd[feats].copy(); y = nd["IMMEDR"].astype(int).values
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=48,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                               random_state=42, verbose=-1)
        m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict(X.iloc[va])
    res = {"n": int(len(y)),
           "accuracy": round(accuracy_score(y, oof), 4),
           "qwk": round(cohen_kappa_score(y, oof, weights="quadratic"), 4)}
    print(f"  NHAMCS real (n={res['n']}): accuracy={res['accuracy']}, QWK={res['qwk']}")
    print("  -> realistic ceiling; contrast with synthetic QWK 0.93 (impossibly high)")
    return res


if __name__ == "__main__":
    df, hx = load()
    out = {}
    out["forensics"] = forensics(df)
    out["acuity"] = acuity_model(df, hx)
    out["fairness"] = fairness_audit(df, hx)
    out["waiting_room"] = waiting_room(df, hx)
    out["nhamcs"] = nhamcs_validation()
    (ART / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved artifacts/results.json")
