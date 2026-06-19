# Triagegeist — The Waiting-Room Blind Spot

Submission for the **Triagegeist: AI in Emergency Triage** competition
(Laitinen-Fredriksson Foundation).

A triage acuity score answers *"how sick is this patient?"* — but not
*"what will happen to them while they wait?"* This project does two things:

1. **An honest audit of acuity-prediction AI** on the Foundation's synthetic ED
   benchmark — exposing the leakage and generative shortcuts that make published
   accuracy/κ scores impossibly high, and showing why the usual fixes
   (GroupKFold) are *insufficient* here.
2. **A waiting-room deterioration-risk model** that predicts which triaged
   patients will need care *beyond the ED* (admission / escalation), delivering
   value **on top of** the acuity score — plus an honest negative result on
   predicting *left-without-being-seen* (LWBS).

Findings are cross-checked against **real U.S. ED data (NHAMCS, CDC)** to expose
the gap between synthetic and clinical reality.

## Reproduce

```bash
pip install -r requirements.txt

# 1. Competition data: download from the Kaggle competition page into comp_data/
#    (train.csv, test.csv, chief_complaints.csv, patient_history.csv)

# 2. Real-world data: NHAMCS ED public-use files (auto-downloaded from CDC FTP)
python src/load_nhamcs.py          # -> data/nhamcs_ed_clean.parquet

# 3. Run the full analysis (forensics + acuity audit + equity + waiting-room + NHAMCS)
python src/analysis.py             # -> artifacts/results.json
```

The Kaggle notebook (`notebook/triagegeist.ipynb`) runs the same pipeline
end-to-end on Kaggle with the competition data mounted at `/kaggle/input/`.

## Data sources

- **Triagegeist synthetic ED dataset** — provided via the Kaggle competition
  (synthetic, simulated Finnish multi-site ED network). Used under competition terms.
- **NHAMCS** (National Hospital Ambulatory Medical Care Survey), CDC/NCHS,
  public-use ED files 2021–2022. Public domain; fixed-width ASCII parsed using
  the official codebook layout. https://www.cdc.gov/nchs/ahcd/

## Layout

```
src/load_nhamcs.py    parse NHAMCS fixed-width ED files -> tidy parquet
src/analysis.py       5-module flagship pipeline (forensics -> NHAMCS validation)
src/build_waitingroom.py  standalone waiting-room baseline
src/make_cover.py     560x280 cover image
notebook/             public Kaggle notebook
```

> No credentials are committed. Kaggle tokens are git-ignored.
