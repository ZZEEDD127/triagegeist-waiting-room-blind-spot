"""
Load the NHAMCS Emergency Department public-use files (2021-2022) into one
tidy, modelling-ready table.

The raw files are fixed-width ASCII with no delimiters. Most column positions
come from data/layout<YEAR>.json (parsed from the codebook) and are copied into
FRONT/BACK below; the loader asserts they still match when the JSON is present.
The visit
disposition check-boxes are missing from those layouts, so their positions are
written out below and were checked against the codebook (doc21-ed-508.pdf p.52:
items 224-239 at bytes 485-500; doc22-ed-508.pdf p.52: items 226-241 at bytes
487-502). Two COVID items added in 2022 push them 2 bytes to the right.

Predictors are limited to what is known at triage. Disposition, diagnoses and
length of stay are kept only as outcomes. Race/ethnicity and payer are kept
only for the equity audit and never enter a model. Survey design variables
(PATWT weight, hospital code, PSU and stratum) are kept for weighted and
hospital-clustered evaluation.

Run:  python src/load_nhamcs.py   ->  data/nhamcs_ed_clean.parquet
Raw files: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/
           (ed2021.zip, ED2022.zip; unzip into data/ as ed2021, ed2022)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VITALS = ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE"]
SENSITIVE = ["RACERETH", "PAYTYPER"]
TARGET = "IMMEDR"

# Visit disposition check-boxes (0 = no, 1 = yes), first byte of the block.
DISPOSITION = ["NODISP", "NOFU", "RETRNED", "RETREFFU", "LWBS", "LBTC", "LEFTAMA",
               "DOA", "DIEDED", "TRANNH", "TRANPSYC", "TRANOTH", "ADMITHOS",
               "OBSHOS", "OBSDIS", "OTHDISP"]
DISPOSITION_START = {2021: 485, 2022: 487}

SPECIAL_MISSING = [-9, -8, -7]
ZERO_IS_MISSING = ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"]

LABELS = {
    "SEX": {1: "Female", 2: "Male"},
    "RACERETH": {1: "NH White", 2: "NH Black", 3: "Hispanic", 4: "NH Other"},
    "PAYTYPER": {1: "Private", 2: "Medicare", 3: "Medicaid/CHIP", 4: "Workers comp",
                 5: "Self-pay", 6: "No charge/charity", 7: "Other"},
}


# Byte positions (1-based, inclusive) copied from data/layout<YEAR>.json so the
# loader also runs where the layout files are absent (e.g. a Kaggle notebook).
# Front-of-record fields are identical in 2021 and 2022.
FRONT = {"TEMPF": (48, 51), "PULSE": (52, 54), "RESPR": (55, 57), "BPSYS": (58, 60),
         "BPDIAS": (61, 63), "POPCT": (64, 66), "PAINSCALE": (69, 70), "AGE": (16, 18),
         "SEX": (25, 25), "RESIDNCE": (23, 24), "ARREMS": (33, 34), "SEEN72": (71, 72),
         "RFV1": (73, 77), "RACERETH": (32, 32), "PAYTYPER": (46, 47), "IMMEDR": (67, 68)}
BACK = {2021: {"HOSPCODE": (542, 544), "CSTRATM": (2343, 2350), "CPSUM": (2351, 2356), "PATWT": (2357, 2367)},
        2022: {"HOSPCODE": (544, 546), "CSTRATM": (2345, 2352), "CPSUM": (2353, 2358), "PATWT": (2359, 2369)}}
RAW = {2021: "ed2021.zip", 2022: "ED2022.zip"}
CDC = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS"


def _layout(year: int) -> dict:
    spec = {**FRONT, **BACK[year]}
    lay_file = DATA_DIR / f"layout{year}.json"
    if lay_file.exists():  # cross-check against the parsed codebook when available
        lay = json.loads(lay_file.read_text())
        bad = [c for c, (s, e) in spec.items() if (lay[c]["start"], lay[c]["end"]) != (s, e)]
        assert not bad, f"layout mismatch {year}: {bad}"
    start = DISPOSITION_START[year]
    for i, c in enumerate(DISPOSITION):
        spec[c] = (start + i, start + i)
    return spec


def _raw_bytes(year: int) -> bytes:
    """Raw fixed-width file: local data/ed<YEAR> if present, else download from CDC."""
    local = DATA_DIR / f"ed{year}"
    if local.exists():
        return local.read_bytes()
    import io, urllib.request, zipfile
    blob = urllib.request.urlopen(f"{CDC}/{RAW[year]}", timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    return z.read(z.namelist()[0])


def load_year(year: int) -> pd.DataFrame:
    import io
    spec = _layout(year)
    colspecs = [(s - 1, e) for s, e in spec.values()]
    df = pd.read_fwf(io.BytesIO(_raw_bytes(year)), colspecs=colspecs, names=list(spec), dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df["YEAR"] = year
    # Hospital codes are only unique within a survey year.
    df["HOSP_ID"] = df["YEAR"].astype(str) + "-" + df["HOSPCODE"].astype("Int64").astype(str)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_raw = len(df)
    df["immedr_raw"] = df[TARGET]
    df["has_triage_level"] = df[TARGET].isin([1, 2, 3, 4, 5]).astype(int)

    for c in VITALS + ["SEX", "RESIDNCE", "ARREMS", "SEEN72", "AGE"] + SENSITIVE:
        df.loc[df[c].isin(SPECIAL_MISSING), c] = np.nan
    for c in ZERO_IS_MISSING:
        df.loc[df[c] == 0, c] = np.nan
    df.loc[df["PULSE"] == 998, "PULSE"] = np.nan      # Doppler / palpated
    df.loc[df["BPDIAS"] == 998, "BPDIAS"] = np.nan    # "P" or "Palp"
    df["TEMPF"] = df["TEMPF"] / 10.0                  # implied decimal: 0986 -> 98.6
    df.loc[df["PAINSCALE"] > 10, "PAINSCALE"] = np.nan
    df.loc[df["RFV1"] < 0, "RFV1"] = np.nan
    df["RFV1_module"] = (df["RFV1"] // 10000).astype("Int64")
    df["TEMPC"] = (df["TEMPF"] - 32) * 5 / 9
    df["ambulance"] = (df["ARREMS"] == 1).astype(float).where(df["ARREMS"].notna())

    # Outcomes. A visit with every disposition box blank/0 has unknown outcome.
    disp = df[DISPOSITION].fillna(0)
    df["disp_known"] = (disp.drop(columns="NODISP").sum(axis=1) > 0).astype(int)
    df["admit"] = ((disp["ADMITHOS"] == 1) | (disp["OBSHOS"] == 1)).astype(int)
    df["escalate"] = ((df["admit"] == 1) | (disp["TRANPSYC"] == 1) | (disp["TRANOTH"] == 1)
                      | (disp["DIEDED"] == 1)).astype(int)
    df["left_early"] = ((disp["LWBS"] == 1) | (disp["LBTC"] == 1)).astype(int)

    for c, mp in LABELS.items():
        df[c + "_lbl"] = df[c].map(mp)
    df.attrs["n_raw"] = n_raw
    return df.reset_index(drop=True)


def build(years=(2021, 2022), save=True) -> pd.DataFrame:
    df = clean(pd.concat([load_year(y) for y in years], ignore_index=True))
    if not save:
        return df
    out = DATA_DIR / "nhamcs_ed_clean.parquet"
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {out}  shape={df.shape}")
    return df


if __name__ == "__main__":
    d = build()
    print("raw IMMEDR codes:", d["immedr_raw"].value_counts().sort_index().to_dict())
    print("admission rate (disposition known):", round(d.loc[d.disp_known == 1, "admit"].mean(), 3))
    print("hospitals:", d["HOSP_ID"].nunique())
