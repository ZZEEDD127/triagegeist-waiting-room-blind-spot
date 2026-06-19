"""
Load NHAMCS Emergency Department public-use micro-data into a tidy,
modeling-ready DataFrame.

The NHAMCS ED public-use file is a fixed-width ASCII file with no column
delimiters. Column positions are taken from the official codebook
(Section II of doc<YY>-ed-508.pdf), parsed into data/layout<YEAR>.json.

We deliberately keep ONLY features that are knowable at the moment of triage
(demographics, arrival mode, initial vital signs, chief-complaint / reason-for-
visit codes). Post-triage fields (diagnoses, tests ordered, disposition, length
of stay) are excluded to avoid target leakage, because the triage acuity score
(IMMEDR) is assigned at intake.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---- Triage-time predictors (no leakage) -----------------------------------
VITALS = ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE"]
DEMO = ["AGE", "SEX", "RESIDNCE", "ARREMS", "SEEN72"]
RFV = ["RFV1", "RFV2", "RFV3"]
# Sensitive attributes used ONLY for the fairness audit (not for the clinical
# model, to avoid building demographic bias directly into predictions).
SENSITIVE = ["RACERETH", "PAYTYPER"]
# Survey design + target
DESIGN = ["PATWT", "CPSUM", "CSTRATM"]
TARGET = "IMMEDR"

KEEP = VITALS + DEMO + RFV + SENSITIVE + DESIGN + [TARGET]

# Special "missing / not-applicable" codes used throughout NHAMCS.
SPECIAL_MISSING = {-9, -8, -7}

# Implausible physiologic floors used as additional sanity filters. NHAMCS
# stores 0 as a placeholder for several vitals; treat 0 as missing for those
# where 0 is not a real measurement.
ZERO_IS_MISSING = ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"]

LABELS = {
    "SEX": {1: "Female", 2: "Male"},
    "RACERETH": {1: "NH White", 2: "NH Black", 3: "Hispanic", 4: "NH Other"},
    "ARREMS": {1: "Ambulance", 2: "Not ambulance"},
    "SEEN72": {1: "Yes", 2: "No"},
    "RESIDNCE": {1: "Private residence", 2: "Nursing home",
                 3: "Homeless", 4: "Other"},
    "PAYTYPER": {1: "Private", 2: "Medicare", 3: "Medicaid/CHIP",
                 4: "Workers comp", 5: "Self-pay", 6: "No charge/charity",
                 7: "Other"},
}

ACUITY_LABELS = {1: "1 Immediate", 2: "2 Emergent", 3: "3 Urgent",
                 4: "4 Semi-urgent", 5: "5 Nonurgent"}


def _read_fixed_width(raw_path: Path, layout: dict, cols: list[str]) -> pd.DataFrame:
    colspecs = [(layout[c]["start"] - 1, layout[c]["end"]) for c in cols]
    df = pd.read_fwf(raw_path, colspecs=colspecs, names=cols, dtype=str)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_year(year: int) -> pd.DataFrame:
    layout = json.loads((DATA_DIR / f"layout{year}.json").read_text())
    raw = DATA_DIR / f"ed{year}"
    cols = [c for c in KEEP if c in layout]
    missing = [c for c in KEEP if c not in layout]
    if missing:
        print(f"[{year}] not in layout, skipped: {missing}")
    df = _read_fixed_width(raw, layout, cols)
    df["YEAR"] = year
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Target: keep only the 5 triage acuity levels (drop -9/-8/0/7).
    df = df[df[TARGET].isin([1, 2, 3, 4, 5])].copy()

    # Vitals: special codes -> NaN, then 0 -> NaN where 0 isn't a real value.
    for c in VITALS:
        df.loc[df[c].isin(SPECIAL_MISSING), c] = np.nan
    for c in ZERO_IS_MISSING:
        df.loc[df[c] == 0, c] = np.nan
    # 998 = Doppler/Palpated placeholder (not a real numeric value).
    df.loc[df["PULSE"] == 998, "PULSE"] = np.nan
    df.loc[df["BPDIAS"] == 998, "BPDIAS"] = np.nan
    # TEMPF has an implied decimal between 3rd and 4th digit (0986 -> 98.6).
    df["TEMPF"] = df["TEMPF"] / 10.0
    # PAINSCALE 0-10 valid; codes >10 (e.g. 96 unable) -> NaN.
    df.loc[df["PAINSCALE"] > 10, "PAINSCALE"] = np.nan

    # Demographic / categorical special codes -> NaN.
    for c in ["SEX", "RESIDNCE", "ARREMS", "SEEN72", "RACERETH", "PAYTYPER"]:
        df.loc[df[c].isin(SPECIAL_MISSING), c] = np.nan

    # AGE: -9 blank -> NaN (AGE is in years, 0-100+).
    df.loc[df["AGE"].isin(SPECIAL_MISSING), "AGE"] = np.nan

    # Reason-for-visit module = first digit of the 5-digit RFV code
    # (NHAMCS RFV classification: 1=symptom, 2=disease, 3=diagnostic,
    #  4=treatment, 5=injuries/adverse, 6=test results, 7=administrative).
    for c in RFV:
        df.loc[df[c] < 0, c] = np.nan
    df["RFV1_module"] = (df["RFV1"] // 10000).astype("Int64")

    # Derived physiologic danger flags (clinically motivated, transparent).
    df["tachycardic"] = (df["PULSE"] > 100).astype("Int64")
    df["hypoxic"] = (df["POPCT"] < 92).astype("Int64")
    df["hypotensive"] = (df["BPSYS"] < 90).astype("Int64")
    df["tachypneic"] = (df["RESPR"] > 22).astype("Int64")
    df["febrile"] = (df["TEMPF"] >= 100.4).astype("Int64")

    # Decoded labels for reporting / the fairness audit.
    for c, mp in LABELS.items():
        df[c + "_lbl"] = df[c].map(mp)
    df["acuity_lbl"] = df[TARGET].map(ACUITY_LABELS)

    # Binary high-acuity target for a focused under-triage analysis:
    # levels 1-2 = high acuity (must-not-miss), 3-5 = lower acuity.
    df["high_acuity"] = (df[TARGET] <= 2).astype(int)
    return df.reset_index(drop=True)


def build(years=(2021, 2022)) -> pd.DataFrame:
    frames = [load_year(y) for y in years]
    df = clean(pd.concat(frames, ignore_index=True))
    out = DATA_DIR / "nhamcs_ed_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out}  shape={df.shape}")
    return df


if __name__ == "__main__":
    d = build()
    print(d["acuity_lbl"].value_counts(dropna=False).sort_index())
