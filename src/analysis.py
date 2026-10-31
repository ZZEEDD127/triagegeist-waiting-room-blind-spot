"""
Triagegeist analysis pipeline. Single source of truth for every number in the
write-up: run it and it writes artifacts/results.json (plus OOF predictions in
artifacts/oof.parquet for the figures).

  M1  How the synthetic data was generated (leakage and shortcut forensics)
  M2  Acuity model: leak-free, ablated, and compared with real data on a
      harmonised feature set
  M3  Equity audit, adjusted for the true acuity mix
  M4  Waiting-room escalation watch on the synthetic queue
  M5  External validation on NHAMCS: hospital-grouped CV, survey weights,
      incremental value over the real triage level, decision curve, equity

Usage:
  python src/load_nhamcs.py      # once, builds data/nhamcs_ed_clean.parquet
  python src/analysis.py         # ~10 minutes on a laptop CPU
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
SEED = 42
N_BOOT = 1000

# ---------------------------------------------------------------- features
CAT = ["arrival_mode", "arrival_season", "shift", "sex", "language", "insurance_type",
       "transport_origin", "pain_location", "mental_status_triage",
       "chief_complaint_system", "age_group", "site_id"]
NUM = ["arrival_hour", "arrival_month", "age", "num_prior_ed_visits_12m",
       "num_prior_admissions_12m", "num_active_medications", "num_comorbidities",
       "systolic_bp", "diastolic_bp", "mean_arterial_pressure", "pulse_pressure",
       "heart_rate", "respiratory_rate", "temperature_c", "spo2", "gcs_total",
       "pain_score", "weight_kg", "height_cm", "bmi", "shock_index", "news2_score"]
VITALS = ["systolic_bp", "diastolic_bp", "mean_arterial_pressure", "pulse_pressure",
          "heart_rate", "respiratory_rate", "temperature_c", "spo2", "gcs_total",
          "shock_index", "news2_score", "mental_status_triage"]
ESCALATE = ["admitted", "transferred", "observation", "deceased"]

# Same ten variables, same units, in both datasets.
HARMONISED = ["age", "male", "temp_c", "hr", "rr", "sbp", "dbp", "spo2", "pain", "ambulance"]
NH_INTAKE = ["AGE", "SEX", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT",
             "PAINSCALE", "ARREMS", "SEEN72", "RFV1_module"]


def find(name):
    for root in [ROOT / "comp_data", Path("/kaggle/input")]:
        if root.exists():
            hits = sorted(root.rglob(name), key=lambda p: len(str(p)))
            if hits:
                return hits[0]
    raise FileNotFoundError(name)


def load_synthetic():
    tr = pd.read_csv(find("train.csv"))
    hist = pd.read_csv(find("patient_history.csv"))
    cc = pd.read_csv(find("chief_complaints.csv"))
    hx = [c for c in hist.columns if c.startswith("hx_")]
    df = (tr.merge(hist[["patient_id"] + hx], on="patient_id", how="left")
            .merge(cc[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left"))
    df["pain_score"] = df["pain_score"].replace(-1, np.nan)
    df["hx_burden"] = df[hx].sum(axis=1)
    df["escalate"] = df["disposition"].isin(ESCALATE).astype(int)
    df["lwbs"] = (df["disposition"] == "lwbs").astype(int)
    df["complaint_base"] = df["chief_complaint_raw"].str.split(r",|—|–| with ", regex=True).str[0].str.strip()
    return df, hx


def make_X(df, hx, cols=None, with_acuity=False):
    num = [c for c in (cols or NUM + hx + ["hx_burden"]) if c not in CAT]
    cat = [c for c in (cols or CAT) if c in CAT]
    X = df[num].copy()
    for c in cat:
        X[c] = df[c].astype("category")
    if with_acuity:
        X["triage_acuity"] = df["triage_acuity"]
    return X


def harmonise_synthetic(df):
    return pd.DataFrame({
        "age": df["age"], "male": (df["sex"] == "M").astype(float),
        "temp_c": df["temperature_c"], "hr": df["heart_rate"], "rr": df["respiratory_rate"],
        "sbp": df["systolic_bp"], "dbp": df["diastolic_bp"], "spo2": df["spo2"],
        "pain": df["pain_score"], "ambulance": (df["arrival_mode"] == "ambulance").astype(float)})


def harmonise_nhamcs(nd):
    return pd.DataFrame({
        "age": nd["AGE"], "male": (nd["SEX"] == 2).astype(float).where(nd["SEX"].notna()),
        "temp_c": nd["TEMPC"], "hr": nd["PULSE"], "rr": nd["RESPR"], "sbp": nd["BPSYS"],
        "dbp": nd["BPDIAS"], "spo2": nd["POPCT"], "pain": nd["PAINSCALE"], "ambulance": nd["ambulance"]})


# ---------------------------------------------------------------- models
def clf(**kw):
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=48, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, min_child_samples=40, random_state=SEED, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def oof_multiclass(X, y, splitter, groups=None):
    oof = np.zeros(len(y), dtype=int)
    for tr, va in splitter.split(X, y, groups):
        oof[va] = clf().fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
    return oof


def oof_binary(X, y, splitter, groups=None):
    """Out-of-fold probabilities. No class re-weighting, so the raw scores are
    already on a probability scale; ranking metrics are unaffected either way."""
    oof = np.zeros(len(y))
    for tr, va in splitter.split(X, y, groups):
        m = clf(n_estimators=400, learning_rate=0.03).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def acuity_metrics(y, pred, clusters=None, n_boot=N_BOOT):
    high = y <= 2
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "qwk": S.bootstrap(S.qwk, y, pred, clusters=clusters, n_boot=n_boot),
            "undertriage_rate": round(float((pred > y).mean()), 4),
            "overtriage_rate": round(float((pred < y).mean()), 4),
            "high_acuity_recall": round(float((pred[high] <= 2).mean()), 4),
            "n_missed_high_acuity": int(((pred > 2) & high).sum())}


def cramers_v(table):
    chi2, p, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), float(p)


def news2_from_vitals(df):
    def band(x, cuts, pts):
        return pd.Series(np.select([x <= c for c in cuts], pts[:-1], pts[-1]), index=x.index).where(x.notna()).fillna(0)
    return (band(df["respiratory_rate"], [8, 11, 20, 24], [3, 1, 0, 2, 3])
            + band(df["spo2"], [91, 93, 95], [3, 2, 1, 0])
            + band(df["systolic_bp"], [90, 100, 110, 219], [3, 2, 1, 0, 3])
            + band(df["heart_rate"], [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
            + band(df["temperature_c"], [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
            + 3 * (df["mental_status_triage"] != "alert").astype(int))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===================================================================== M1
def m1_forensics(df, nd):
    log("M1 forensics")
    r = {}
    r["corr_with_acuity"] = {c: round(float(df[c].corr(df["triage_acuity"])), 3)
                             for c in ["ed_los_hours", "news2_score", "shock_index", "gcs_total", "spo2", "heart_rate"]}
    # Real-data reference: same vitals vs real triage level
    t = nd[nd.has_triage_level == 1]
    r["corr_with_acuity_nhamcs"] = {c: round(float(t[c].corr(t["IMMEDR"])), 3)
                                    for c in ["PULSE", "RESPR", "POPCT", "BPSYS", "TEMPF"]}
    r["corr_with_acuity_synthetic_same_vitals"] = {
        k: round(float(df[v].corr(df["triage_acuity"])), 3)
        for k, v in {"PULSE": "heart_rate", "RESPR": "respiratory_rate", "POPCT": "spo2",
                     "BPSYS": "systolic_bp", "TEMPF": "temperature_c"}.items()}

    # Complaint text -> acuity purity, overall and for frequent complaints only
    g = df.groupby("chief_complaint_raw")["triage_acuity"]
    cnt = g.size()
    pur = g.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    r["complaints"] = {"n_unique": int(cnt.size), "median_count": float(cnt.median()),
                       "purity_all": round(float(pur.mean()), 4),
                       "share_deterministic_all": round(float((pur == 1).mean()), 4),
                       "n_with_20plus": int((cnt >= 20).sum()),
                       "purity_20plus": round(float(pur[cnt >= 20].mean()), 4),
                       "share_deterministic_20plus": round(float((pur[cnt >= 20] == 1).mean()), 4)}

    # NEWS2 recomputed from the vitals with the RCP 2017 bands
    calc = news2_from_vitals(df)
    r["news2_recomputed"] = {"exact_match": round(float((calc == df["news2_score"]).mean()), 3),
                             "corr_given_vs_recomputed": round(float(calc.corr(df["news2_score"])), 3),
                             "corr_recomputed_with_acuity": round(float(calc.corr(df["triage_acuity"])), 3)}

    # Rater effect. Real triage varies by nurse and by hospital; generated labels do not.
    nurse_tab = pd.crosstab(df["triage_nurse_id"], df["triage_acuity"])
    v_n, p_n = cramers_v(nurse_tab)
    hosp = t[t.groupby("HOSP_ID")["IMMEDR"].transform("size") >= 20]
    v_h, p_h = cramers_v(pd.crosstab(hosp["HOSP_ID"], hosp["IMMEDR"]))
    r["rater_effect"] = {"synthetic_nurses": int(nurse_tab.shape[0]),
                         "synthetic_nurse_cramers_v": round(v_n, 4), "synthetic_nurse_p": round(p_n, 4),
                         "nhamcs_hospitals": int(hosp["HOSP_ID"].nunique()),
                         "nhamcs_hospital_cramers_v": round(v_h, 4), "nhamcs_hospital_p": float(f"{p_h:.2e}")}

    # Text shortcut: a bag-of-words model with and without template hold-out
    text = df["chief_complaint_raw"].fillna("")
    y = df["triage_acuity"].values
    Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(text)
    def text_oof(splitter, groups=None):
        oof = np.zeros(len(y), dtype=int)
        for tr, va in splitter.split(Xt, y, groups):
            m = LogisticRegression(C=5, max_iter=2000).fit(Xt[tr], y[tr])
            oof[va] = m.predict(Xt[va])
        return oof
    r["text_model"] = {}
    for name, sp, grp in [("stratified", StratifiedKFold(5, shuffle=True, random_state=SEED), None),
                          ("group_exact_complaint", GroupKFold(5), df["chief_complaint_raw"].values),
                          ("group_complaint_base", GroupKFold(5), df["complaint_base"].values)]:
        o = text_oof(sp, grp)
        r["text_model"][name] = {"accuracy": round(float(accuracy_score(y, o)), 4), "qwk": round(float(S.qwk(y, o)), 4)}
    r["text_model"]["n_complaint_bases"] = int(df["complaint_base"].nunique())
    return r


# ===================================================================== M2
def m2_acuity(df, hx, nd):
    log("M2 acuity models")
    y = df["triage_acuity"].values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    r = {}
    X = make_X(df, hx)
    oof_skf = oof_multiclass(X, y, skf)
    r["structured_stratified"] = acuity_metrics(y, oof_skf, n_boot=300)
    oof_gkf = oof_multiclass(X, y, GroupKFold(5), df["complaint_base"].values)
    r["structured_group_by_complaint_base"] = acuity_metrics(y, oof_gkf, n_boot=300)
    oof_site = oof_multiclass(X, y, GroupKFold(5), df["site_id"].values)
    r["structured_group_by_site"] = acuity_metrics(y, oof_site, n_boot=300)

    # Ablations: what carries the label?
    vit_only = make_X(df, hx, cols=VITALS)
    r["ablation_vitals_only"] = acuity_metrics(y, oof_multiclass(vit_only, y, skf), n_boot=300)
    no_vit = make_X(df, hx, cols=[c for c in NUM + CAT + hx + ["hx_burden"] if c not in VITALS])
    r["ablation_without_vitals"] = acuity_metrics(y, oof_multiclass(no_vit, y, skf), n_boot=300)

    # Harmonised comparison: identical 10 features, identical model, both datasets
    Hs = harmonise_synthetic(df)
    oof_hs = oof_multiclass(Hs, y, skf)
    r["harmonised_synthetic"] = acuity_metrics(y, oof_hs, n_boot=300)
    t = nd[nd.has_triage_level == 1].reset_index(drop=True)
    Hn = harmonise_nhamcs(t); yn = t["IMMEDR"].astype(int).values
    oof_hn = oof_multiclass(Hn, yn, StratifiedGroupKFold(5, shuffle=True, random_state=SEED), t["HOSP_ID"].values)
    r["harmonised_nhamcs"] = acuity_metrics(yn, oof_hn, clusters=t["HOSP_ID"].values)
    # Original 12-feature NHAMCS set, hospital-grouped vs random split
    Xn = t[NH_INTAKE]
    r["nhamcs_intake_random_split"] = acuity_metrics(yn, oof_multiclass(Xn, yn, skf), clusters=t["HOSP_ID"].values)
    r["nhamcs_intake_hospital_split"] = acuity_metrics(
        yn, oof_multiclass(Xn, yn, StratifiedGroupKFold(5, shuffle=True, random_state=SEED), t["HOSP_ID"].values),
        clusters=t["HOSP_ID"].values)
    return r, oof_skf


# ===================================================================== M3
def oe_by_group(under, acuity, group):
    """Observed / expected undertriage per group, where 'expected' uses the
    overall rate at each true acuity level (indirect standardisation)."""
    d = pd.DataFrame({"u": under, "a": acuity, "g": group})
    rate_at = d.groupby("a")["u"].mean()
    d["e"] = d["a"].map(rate_at)
    s = d.groupby("g").agg(n=("u", "size"), obs=("u", "sum"), exp=("e", "sum"))
    return s["obs"] / s["exp"], s


def m3_equity(df, oof_skf):
    log("M3 equity audit")
    y = df["triage_acuity"].values
    under = (oof_skf > y).astype(int)
    rng = np.random.default_rng(SEED)
    rate_at = pd.Series(under).groupby(y).mean()
    exp_row = pd.Series(y).map(rate_at).to_numpy()
    out, pvals = {}, []

    def heterogeneity(gg):
        oe, s = oe_by_group(under, y, gg)
        return float(np.average((oe - 1) ** 2, weights=s["n"]))

    for col in ["language", "insurance_type", "sex", "age_group"]:
        g = df[col].astype(str).values
        oe, s = oe_by_group(under, y, g)
        crude = pd.Series(under).groupby(g).mean()
        rows = {}
        for grp in oe.index:
            m = g == grp
            ci = S.bootstrap(lambda u, e: float(u.sum() / e.sum()), under[m], exp_row[m], n_boot=500)
            rows[grp] = {"n": int(s.loc[grp, "n"]), "crude_rate": round(float(crude[grp]), 4),
                         "oe_ratio": ci}
        # Permutation test of heterogeneity: shuffle group labels within each acuity level
        stat = heterogeneity
        obs = stat(g)
        null = []
        for _ in range(300):
            gp = g.copy()
            for a in np.unique(y):
                idx = np.where(y == a)[0]
                gp[idx] = rng.permutation(gp[idx])
            null.append(stat(gp))
        p = (1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null))
        pvals.append(p)
        out[col] = {"groups": rows, "perm_p": round(float(p), 4)}
    for col, p_adj in zip(out, S.holm(pvals)):
        out[col]["holm_p"] = round(float(p_adj), 4)
    # Why crude rates mislead: undertriage is impossible at ESI 5
    out["undertriage_rate_by_true_acuity"] = {int(k): round(float(v), 4)
                                              for k, v in pd.Series(under).groupby(y).mean().items()}
    return out


# ===================================================================== M4
def m4_waiting_room(df, hx):
    log("M4 waiting-room watch (synthetic)")
    ye = df["escalate"].values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X_ac = df[["triage_acuity"]]
    X_full = make_X(df, hx, with_acuity=True)
    p_ac = oof_binary(X_ac, ye, skf)
    p_full = oof_binary(X_full, ye, skf)
    p_site = oof_binary(X_full, ye, GroupKFold(5), df["site_id"].values)
    low = df["triage_acuity"].isin([3, 4, 5]).values
    r = {"prevalence_all": round(float(ye.mean()), 4), "prevalence_low": round(float(ye[low].mean()), 4),
         "n_low": int(low.sum())}
    r["auc_all"] = {"acuity_only": S.bootstrap(S.auc, ye, p_ac),
                    "full": S.bootstrap(S.auc, ye, p_full),
                    "delta_full_minus_acuity": S.bootstrap(S.delta_auc, ye, p_full, p_ac)}
    yl, pl, pal = ye[low], p_full[low], p_ac[low]
    r["auc_low"] = {"acuity_only": S.bootstrap(S.auc, yl, pal),
                    "full": S.bootstrap(S.auc, yl, pl),
                    "delta_full_minus_acuity": S.bootstrap(S.delta_auc, yl, pl, pal),
                    "full_site_held_out": S.bootstrap(S.auc, yl, p_site[low])}
    r["operating_points_low"] = {}
    for frac in [0.05, 0.10, 0.20]:
        op = S.operating_point(yl, pl, frac)
        op["recall_ci"] = S.bootstrap(lambda a, b: S.recall_at(a, b, frac), yl, pl, n_boot=500)
        op["lift_ci"] = S.bootstrap(lambda a, b: S.lift_at(a, b, frac), yl, pl, n_boot=500)
        r["operating_points_low"][f"top_{int(frac * 100)}pct"] = op
    cal = S.crossfit_isotonic(p_full, ye)
    r["calibration"] = {"raw": S.calibration_stats(ye, p_full), "crossfit_isotonic": S.calibration_stats(ye, cal)}

    yw = df["lwbs"].values
    p_l = oof_binary(X_full, yw, skf)
    r["lwbs"] = {"prevalence": round(float(yw.mean()), 4), "auc": S.bootstrap(S.auc, yw, p_l, n_boot=500)}

    # Demo cases, chosen by rule and labelled as such
    thr20 = np.quantile(cal[low], 0.80)
    flagged_tp = np.where(low & (cal >= thr20) & (ye == 1))[0]
    flagged = np.where(low & (cal >= thr20))[0]
    typical = flagged_tp[np.argsort(cal[flagged_tp])[len(flagged_tp) // 2]]
    best = flagged_tp[np.argmax(cal[flagged_tp])]
    fp_typ = np.setdiff1d(flagged, flagged_tp)
    fp_typical = fp_typ[np.argsort(cal[fp_typ])[len(fp_typ) // 2]]
    m = clf(n_estimators=400, learning_rate=0.03).fit(X_full, ye)
    def explain(i):
        c = np.asarray(m.booster_.predict(X_full.iloc[[i]], pred_contrib=True)).ravel()[:-1]
        top = sorted(zip(X_full.columns, c), key=lambda t: -abs(t[1]))[:4]
        return {"patient_id": df["patient_id"].iloc[i], "acuity": int(df["triage_acuity"].iloc[i]),
                "risk": round(float(cal[i]), 3), "escalated": int(ye[i]),
                "disposition": df["disposition"].iloc[i],
                "drivers": [(n, round(float(v), 3)) for n, v in top]}
    r["demo"] = {"threshold_top20": round(float(thr20), 3),
                 "precision_of_flags": round(float(ye[flagged].mean()), 3),
                 "typical_true_positive": explain(typical),
                 "typical_false_positive": explain(fp_typical),
                 "highest_risk_true_positive": explain(best)}
    oof = pd.DataFrame({"p_acuity": p_ac, "p_full": p_full, "p_cal": cal, "y": ye,
                        "acuity": df["triage_acuity"].values})
    return r, oof


# ===================================================================== M5
def m5_nhamcs(nd):
    log("M5 NHAMCS external validation")
    t = nd[(nd.has_triage_level == 1) & (nd.disp_known == 1)].reset_index(drop=True)
    hosp = t["HOSP_ID"].values; w = t["PATWT"].values
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X_imm = t[["IMMEDR"]]
    X_int = t[NH_INTAKE]
    X_both = t[NH_INTAKE + ["IMMEDR"]]
    low = t["IMMEDR"].isin([3, 4, 5]).values
    r = {"n": int(len(t)), "n_hospitals": int(t["HOSP_ID"].nunique()), "n_low": int(low.sum()),
         "share_visits_without_triage_level": round(float(1 - nd["has_triage_level"].mean()), 3),
         "admit_rate_with_vs_without_level": [round(float(nd.loc[nd.has_triage_level == 1, "admit"].mean()), 3),
                                              round(float(nd.loc[nd.has_triage_level == 0, "admit"].mean()), 3)]}
    oofs = {}
    for target in ["admit", "escalate"]:
        y = t[target].values
        p_imm = oof_binary(X_imm, y, sgkf, hosp)
        p_int = oof_binary(X_int, y, sgkf, hosp)
        p_both = oof_binary(X_both, y, sgkf, hosp)
        p_both_rand = oof_binary(X_both, y, skf)
        yl = y[low]; wl = w[low]; hl = hosp[low]
        res = {"prevalence": round(float(y.mean()), 4), "prevalence_low": round(float(yl.mean()), 4),
               "prevalence_low_weighted": round(float(np.average(yl, weights=wl)), 4)}
        res["auc_low"] = {
            "immedr_only": S.bootstrap(S.auc, yl, p_imm[low], clusters=hl),
            "intake_only": S.bootstrap(S.auc, yl, p_int[low], clusters=hl),
            "intake_plus_immedr": S.bootstrap(S.auc, yl, p_both[low], clusters=hl),
            "delta_vs_immedr_only": S.bootstrap(S.delta_auc, yl, p_both[low], p_imm[low], clusters=hl),
            "intake_plus_immedr_weighted": S.bootstrap(S.auc, yl, p_both[low], wl, clusters=hl),
            "intake_plus_immedr_random_split": S.bootstrap(S.auc, yl, p_both_rand[low], clusters=hl)}
        res["auc_all"] = {
            "immedr_only": S.bootstrap(S.auc, y, p_imm, clusters=hosp),
            "intake_plus_immedr": S.bootstrap(S.auc, y, p_both, clusters=hosp),
            "delta_vs_immedr_only": S.bootstrap(S.delta_auc, y, p_both, p_imm, clusters=hosp)}
        pl = p_both[low]
        ops = {}
        for frac in [0.10, 0.20]:
            op = S.operating_point(yl, pl, frac)
            op["weighted"] = S.operating_point(yl, pl, frac, wl)
            op["recall_ci"] = S.bootstrap(lambda a, b: S.recall_at(a, b, frac), yl, pl, clusters=hl)
            op["lift_ci"] = S.bootstrap(lambda a, b: S.lift_at(a, b, frac), yl, pl, clusters=hl)
            ops[f"top_{int(frac * 100)}pct"] = op
        res["operating_points_low"] = ops
        cal = S.crossfit_isotonic(p_both, y)
        res["calibration"] = {"raw": S.calibration_stats(y, p_both), "crossfit_isotonic": S.calibration_stats(y, cal)}
        if target == "admit":
            ths = np.round(np.arange(0.02, 0.41, 0.01), 2)
            res["decision_curve_low"] = S.decision_curve(yl, cal[low], ths, clusters=hl)
            dc = res["decision_curve_low"]
            pos = [th for th, lo in zip(dc["thresholds"], dc["adv_lo"]) if lo > 0]
            res["decision_curve_low"]["thresholds_with_ci_above_zero"] = [min(pos), max(pos)] if pos else None
            oofs = {"y": y, "p_imm": p_imm, "p_both": p_both, "p_cal": cal, "low": low, "hosp": hosp, "w": w,
                    "race": t["RACERETH_lbl"].values, "payer": t["PAYTYPER_lbl"].values}
        r[target] = res

    # Leaving early: does the LWBS null result replicate on real data?
    yle = t["left_early"].values
    p_le = oof_binary(X_both, yle, sgkf, hosp)
    r["left_early"] = {"prevalence": round(float(yle.mean()), 4), "auc": S.bootstrap(S.auc, yle, p_le, clusters=hosp)}

    # Equity of the watch on real data: who gets caught, and is risk calibrated per group?
    eq = {}
    y, pc, lw = oofs["y"], oofs["p_cal"], oofs["low"]
    flag = np.zeros(len(y), dtype=bool)
    flag[lw] = S.flag_top(pc[lw], 0.20)
    for col, vals in [("race_ethnicity", oofs["race"]), ("payer", oofs["payer"])]:
        rows = {}
        for g in pd.Series(vals[lw]).dropna().unique():
            m = lw & (vals == g)
            if m.sum() < 200 or y[m].sum() < 20:
                continue
            rows[str(g)] = {"n": int(m.sum()), "admissions": int(y[m].sum()),
                            "recall_of_watch": S.bootstrap(lambda a, f: a[f.astype(bool)].sum() / a.sum(),
                                                           y[m], flag[m], clusters=oofs["hosp"][m], n_boot=500),
                            "observed_rate": round(float(y[m].mean()), 3),
                            "mean_predicted": round(float(pc[m].mean()), 3)}
        eq[col] = rows
    r["equity_low_acuity_watch"] = eq
    return r, oofs


# ===================================================================== main
def main():
    ART.mkdir(exist_ok=True)
    nd = pd.read_parquet(ROOT / "data" / "nhamcs_ed_clean.parquet")
    df, hx = load_synthetic()
    out = {"meta": {"seed": SEED, "n_boot": N_BOOT, "n_train": int(len(df)),
                    "lightgbm": lgb.__version__}}
    out["m1_forensics"] = m1_forensics(df, nd)
    out["m2_acuity"], oof_skf = m2_acuity(df, hx, nd)
    out["m3_equity"] = m3_equity(df, oof_skf)
    out["m4_waiting_room"], oof4 = m4_waiting_room(df, hx)
    out["m5_nhamcs"], oof5 = m5_nhamcs(nd)
    (ART / "results.json").write_text(json.dumps(out, indent=2, default=str))
    oof4.to_parquet(ART / "oof_synthetic.parquet", index=False)
    pd.DataFrame({k: v for k, v in oof5.items()}).to_parquet(ART / "oof_nhamcs.parquet", index=False)
    log("saved artifacts/results.json")


if __name__ == "__main__":
    main()
