# Triagegeist — The Waiting-Room Blind Spot
### What a synthetic triage benchmark can and cannot tell us, and a second-read watch validated on real emergency-department data

**Track:** Triagegeist: AI in Emergency Triage · **Version 2** (revised analysis; see "Changes from version 1" at the end)

All numbers below are produced by `src/analysis.py` and stored in `artifacts/results.json`. Brackets are 95% bootstrap intervals: patient-level on the synthetic data, hospital-cluster on NHAMCS.

---

## Clinical problem

Triage assigns each arriving patient an acuity level (ESI 1–5) that sets how long they may safely wait. The dangerous error is **undertriage**, where a sick patient is labelled "can wait". The acuity level is a snapshot. It does not say which of the patients in the lower-acuity queue will turn out to need a hospital bed.

This project asks two questions:

1. Can the competition's synthetic benchmark be used to judge triage AI at all?
2. Among patients triaged as lower acuity, does information collected at triage identify who will be admitted, **beyond what the triage level already says**?

## Data

- **Triagegeist synthetic ED data** (80,000 training visits, 5 sites, 50 triage nurses): vitals, demographics, chief-complaint text, 24 comorbidity flags, and the post-triage fields `disposition` and `ed_los_hours`.
- **NHAMCS ED public-use files 2021–2022** (CDC/NCHS): 32,232 real visits from a national probability sample. Fixed-width ASCII, parsed with byte positions taken from the official codebook. The disposition block sits at bytes 485–500 in 2021 and 487–502 in 2022, because two COVID items were inserted in 2022. 64.2% of visits carry a triage level (`IMMEDR` 1–5). Visits without one have the same admission rate (12.9% vs 12.9%), so restricting to triaged visits does not select on outcome. The analysis set is 20,616 visits from 298 hospitals with both a triage level and a recorded disposition.

Only information available at the end of triage is used as a predictor. Race/ethnicity and payer are used only to audit fairness and never enter a model.

## Methods

- **Models:** LightGBM gradient-boosted trees with fixed hyper-parameters (no tuning, so there is no tuning leakage), seed 42.
- **Validation:** 5-fold out-of-fold predictions. Synthetic data: stratified folds, plus grouped folds (by complaint phrase, by site) as stress tests. NHAMCS: **folds grouped by hospital**, so no hospital appears in both training and test data.
- **Uncertainty:** 1,000 bootstrap replicates. NHAMCS replicates resample whole hospitals, because visits within a hospital are correlated.
- **Incremental value:** each outcome model is compared with a model that sees **only the triage level**, using a paired bootstrap of the AUC difference.
- **Survey design:** NHAMCS results are also reported with the `PATWT` visit weights, which give nationally representative estimates.
- **Calibration:** isotonic regression, **cross-fitted** (the calibration map for each fold is learned on the other folds). Reported as Brier score, expected calibration error (ECE) and calibration slope.
- **Clinical utility:** decision-curve analysis (net benefit, Vickers & Elkin 2006) with hospital-cluster bootstrap bands.
- **Equity:** undertriage is compared as observed/expected ratios standardised to the true acuity mix. Heterogeneity is tested by permuting group labels within each acuity level, with Holm correction across attributes.

## Results

### 1. The synthetic labels were generated, not assigned by people

| Signal | Synthetic | Real (NHAMCS) |
|---|---|---|
| \|correlation\| of five vitals with triage level | 0.46–0.65 | ≤ 0.08 |
| Rater effect on triage level (Cramér's V) | 0.022 across 50 nurses (p = 0.99) | 0.35 across 239 hospitals (p < 10⁻¹⁰) |

- **No rater effect.** The 50 synthetic nurses, each with about 1,600 patients, produce statistically indistinguishable acuity distributions. Real triage varies widely between raters and sites. A label with no rater variation cannot have come from human triage.
- **The vitals encode the label.** NEWS2 recomputed from the vitals with the official RCP bands matches the provided score (r = 0.95, exact in 71% of rows). The NEWS2 field is consistent with the vitals; the vitals themselves were sampled from the acuity level.
- **The complaint text is a lookup table.** 99.7% of the 4,949 distinct complaints map to a single acuity level (99.9% among the 1,842 complaints seen at least 20 times). A text-only model reaches κ = 0.9997. Holding out exact complaint strings does not help, because templates share their modifiers (κ = 0.9997). Holding out the 874 base phrases does (κ = 0.93).
- `disposition` and `ed_los_hours` are recorded after triage and are absent from the test set. They are never used as acuity predictors.

### 2. Acuity prediction: the synthetic score does not transfer

| Model | κ (quadratic) |
|---|---|
| Synthetic, all structured intake fields (no text, no post-triage fields) | 0.931 [0.930, 0.932] |
| … holding out complaint base phrases | 0.919 [0.918, 0.920] |
| … holding out each site | 0.931 [0.930, 0.932] |
| … vitals only | 0.895 [0.893, 0.897] |
| … everything except vitals | 0.694 [0.691, 0.698] |
| **Synthetic, 10 vitals shared with NHAMCS** | **0.901 [0.899, 0.903]** |
| **Real NHAMCS, same 10 vitals, same model, hospital-grouped folds** | **0.240 [0.214, 0.263]** |

With identical features and an identical model, agreement falls from 0.90 to 0.24. On the synthetic data the label lives in the vitals: the vitals alone reach 0.895, and removing them drops κ to 0.69. Real triage depends on things the vitals do not capture, and agreement between human raters is itself only moderate (published κ ≈ 0.6–0.8). **Acuity accuracy on this benchmark measures recovery of the generator, not clinical skill.** On the real data, a random split gives κ 0.268 and a hospital-grouped split 0.257, so the gap is small but in the expected direction.

### 3. Equity audit: no disparity after adjusting for acuity mix

Crude undertriage rates differ across groups (Arabic speakers 8.3% vs 7.2–7.9% for the others). Crude rates mislead, though, because undertriage is impossible at ESI 5 and runs at about 10% at ESI 3–4, so groups with different acuity mixes differ for that reason alone. After standardising for true acuity, **no attribute shows significant heterogeneity**: language p = 0.73, insurance 0.73, sex 0.36, age group 0.73 (permutation tests, Holm-adjusted; LightGBM 4.6.0. With LightGBM 4.7 a few hundred predictions change and every p stays ≥ 0.39). Arabic speakers' O/E of 1.14 [1.03, 1.25] does not survive the multiple-group test. The audit method transfers to real data; on this synthetic data it finds no disparity, and the version 1 claim that it did is withdrawn.

### 4. The waiting-room watch cannot be tested on the synthetic data

Escalation means admission, transfer, observation or death.

| Synthetic, escalation | Triage level only | All intake fields | Difference |
|---|---|---|---|
| All patients | 0.837 | 0.838 | +0.001 [−0.001, 0.002] |
| ESI 3–5 queue (n = 63,339) | 0.762 | 0.763 | +0.001 [−0.001, 0.004] |

The synthetic disposition is a function of the acuity level and nothing else. The apparent within-queue AUC of 0.76 comes entirely from ranking ESI 3 above ESI 4 and 5 (the risk deciles form three steps, one per level). Version 1 presented this as the watch working; it was not. Leaving without being seen is also unpredictable (AUC 0.63 [0.62, 0.63]). **The synthetic data cannot show whether intake data adds anything to the triage level.** Only real outcomes can.

### 5. On real data, the watch adds substantial information beyond the triage level

NHAMCS, patients triaged as lower acuity (`IMMEDR` 3–5, n = 17,017, admission rate 8.5%; 8.1% survey-weighted). All folds are grouped by hospital.

| Admission AUC within the lower-acuity queue | |
|---|---|
| Triage level only | 0.641 [0.609, 0.670] |
| Intake data only | 0.775 [0.760, 0.790] |
| **Intake data + triage level** | **0.798 [0.782, 0.813]** |
| Gain over triage level only | **+0.156 [0.131, 0.182]** |
| Same model, survey-weighted | 0.817 [0.798, 0.834] |
| Same model, random (not hospital-grouped) folds | 0.808 [0.796, 0.821] |

| Operating point (lower-acuity queue) | Share of admissions caught | Lift over base rate | Precision |
|---|---|---|---|
| Flag top 10% | 36.9% [34.5, 39.9] | 3.7× [3.4, 4.0] | 31.5% |
| **Flag top 20%** | **57.9% [54.9, 61.0]** | **2.9× [2.7, 3.1]** | 24.7% |
| Flag top 20%, survey-weighted | 61.2% | 3.1× | 25.1% |

- **Calibration:** raw slope 0.80 (overconfident); after cross-fitted isotonic calibration the slope is 0.96, ECE 0.005 and Brier 0.091.
- **Clinical utility:** net benefit is above both default policies, "review everyone" and "review no-one", with the 95% hospital-cluster band above zero for every threshold from 2% to 35%.
- **Robustness:** a broader outcome (admission, transfer, or death in the ED) gives the same picture: AUC 0.791, a gain of +0.146 [0.124, 0.168] over the triage level alone.
- **Leaving early:** leaving without being seen or before treatment ends remains near-unpredictable on real data (AUC 0.58 [0.56, 0.61]), which replicates the synthetic null result. Who leaves depends on operations, not on the patient's condition.

**Equity on real data.** With one threshold for everyone, the share of admissions the watch catches follows baseline risk: Medicare 79%, Medicaid 43%, private 43%, self-pay 26% [13, 41]. By race/ethnicity it is NH White 62% [58, 67], Hispanic 55% [47, 64] and NH Black 52% [46, 59]. The lower capture rate is not caused by underestimating these groups' risk: predicted risk is slightly above observed for NH Black (8.0% vs 6.4%) and Hispanic (7.1% vs 6.4%) patients. Their baseline admission risk is lower, so a single cut-off flags a smaller share of their admissions. Before deployment, the threshold policy needs a deliberate equity decision, and capture rate by group should be monitored.

## Interpretation

1. **Do not rank triage AI on this benchmark by acuity accuracy.** The labels carry no rater variation and are encoded in the vitals; a model that reaches κ 0.90 here reaches 0.24 on real triage with the same inputs.
2. **A checklist for any synthetic clinical benchmark:** look for a rater effect (none means generated labels), compare feature–label strength with real data, run a model on features shared with real data, and test incremental value against the existing clinical score.
3. **The second-read watch is supported by real data, not by the synthetic data.** Among real patients told they can wait, information already collected at triage ranks admission risk far better than the triage level (AUC 0.80 vs 0.64). Flagging one patient in five for re-assessment would catch about 58% of later admissions.

## Limitations

1. NHAMCS is cross-sectional. Admission is a proxy for deterioration, and the timing of deterioration during the wait is not observed.
2. The real-data model uses 12 public-use variables. Richer local data (full complaint text, history, repeat vitals) would likely do better; the numbers here are a floor.
3. Hospital-grouped validation checks transfer to unseen US hospitals, not to Nordic EDs. A local silent-deployment pilot (predictions logged, not shown to staff) is the necessary next step.
4. Survey weights are used for weighted point estimates. The bootstrap resamples hospitals rather than reproducing the full stratified design, so intervals are approximate.
5. Decision support only. Not a medical device.

## Changes from version 1

| Version 1 claim | Version 2 finding |
|---|---|
| Within ESI 3–5 the synthetic watch separates risk (AUC 0.76) | That AUC is entirely the acuity level itself (gain +0.001). The synthetic data cannot evaluate the watch. |
| GroupKFold by complaint shows "generative" leakage | Replaced with direct evidence: no rater effect (V 0.02 vs 0.35 real), vitals–label correlations 0.46–0.65 vs ≤ 0.08 real, harmonised κ 0.90 vs 0.24 |
| NEWS2 is a leaking field | NEWS2 is consistent with the vitals (r 0.95); the vitals themselves are generated from acuity |
| Arabic/Somali speakers show the highest undertriage | Not significant after acuity standardisation and multiple-testing correction; Somali is below average |
| Real-data watch AUC 0.78, no comparison with triage level | 0.80 with hospital-grouped folds; +0.156 over the triage level alone; survey-weighted 0.82 |
| Calibration improved Brier (fit and scored on the same predictions) | Cross-fitted calibration; slope 0.80 → 0.96 on real data |
| Showcase patient at 81% risk | Removed; the cherry-picked case was the single highest-risk true positive |
| No uncertainty estimates | 95% intervals on every headline number, hospital-clustered on real data |

## Reproducibility

The notebook was re-executed from scratch in a clean folder (CDC files downloaded, no cached data). Every NHAMCS number matched the script run exactly.

```
pip install -r requirements.txt     # pins lightgbm==4.6.0, the version behind results.json
kaggle competitions download -c triagegeist -p comp_data && unzip comp_data/triagegeist.zip -d comp_data
python src/load_nhamcs.py     # downloads the CDC files if data/ed2021, data/ed2022 are absent
python src/analysis.py        # about 35 minutes on a laptop CPU -> artifacts/results.json
python src/figures.py         # -> artifacts/fig_*.png
```

## References

Gilboy N et al. *Emergency Severity Index v4.* AHRQ, 2012 · Mistry B et al. Accuracy and reliability of ED triage using the ESI: an international multicenter assessment. *Ann Emerg Med* 2018;71:581–587 · Levin S et al. Machine-learning-based electronic triage. *Ann Emerg Med* 2018;71:565–574 · Hong WS et al. Predicting hospital admission at ED triage using machine learning. *PLoS One* 2018;13:e0201016 · Vickers AJ, Elkin EB. Decision curve analysis. *Med Decis Making* 2006;26:565–574 · Royal College of Physicians. *NEWS2*, 2017 · Obermeyer Z et al. Dissecting racial bias in an algorithm. *Science* 2019;366:447–453 · NCHS. *NHAMCS 2021–2022 public-use ED files.* CDC.
