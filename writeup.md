# Triagegeist — The Waiting-Room Blind Spot
### Auditing acuity-prediction AI, and predicting who deteriorates before they are seen

**Track:** Triagegeist: AI in Emergency Triage

---

## Clinical problem statement

Emergency triage assigns every arriving patient an acuity level (e.g. ESI 1–5)
that governs how long they may safely wait. The dangerous failure mode is
**undertriage**: a genuinely sick patient is given a low acuity and waits too
long. Its cost is asymmetric — far greater than overtriage — and human triage
agreement is only moderate (inter-rater κ ≈ 0.6–0.8), with documented
undertriage of vulnerable groups.

Most Triagegeist submissions answer one question: *can a model reproduce the
acuity label?* We argue this is the wrong question on a synthetic benchmark, and
ask two better ones:

1. **Can we trust the score?** What has the benchmark actually encoded, and what
   does an *honest* acuity model look like once the shortcuts are removed?
2. **The waiting-room blind spot.** The acuity number is a snapshot. Can intake
   data flag who will *deteriorate, need admission, or leave without being seen*
   while waiting — adding value *beyond* the acuity score?

Every claim is then checked against **real U.S. emergency-department data
(NHAMCS, CDC)**. The result is a single, honest, end-to-end study rather than a
leaderboard chase.

## Methodology

**Data.** The provided **synthetic** Triagegeist dataset (80k train / 20k test;
a simulated Finnish multi-site ED network) with intake vitals, demographics,
chief-complaint text and 24 comorbidity flags. For external validation we parse
the **NHAMCS ED public-use files 2021–2022** (CDC/NCHS, public domain) — a
fixed-width ASCII format whose column positions we derive directly from the
official codebook. Target `IMMEDR` is real triage immediacy (1–5).

**Five modules.**
1. *Forensics* — quantify post-triage leakage (`disposition`, `ed_los_hours`) and
   the chief-complaint→acuity shortcut.
2. *Honest acuity model* — LightGBM on **structured intake only** (no raw text, no
   post-triage fields), validated with both StratifiedKFold and **GroupKFold by
   chief complaint**.
3. *Equity audit* — out-of-fold model-undertriage rate by language, insurance,
   sex and age.
4. *Waiting-room risk* — LightGBM for **escalation** (admission/transfer/
   observation/death) and **LWBS**, explicitly comparing *acuity alone* vs *full
   intake* to test incremental value; probability calibration via isotonic
   regression.
5. *External validation* — the same structured model on real NHAMCS data.

All results are out-of-fold, seed-fixed (42), 5-fold. Metrics: accuracy,
quadratic-weighted κ (acuity is ordinal), undertriage/high-acuity recall, ROC-
and PR-AUC, Brier score.

## Results

**Module 1 — The benchmark is over-determined.** The free-text chief complaint
**fully determines** acuity for **99.7%** of 4,949 unique complaints (mean within-
complaint label purity 0.999): any text model memorises templates. Separately,
generated vitals leak the label — `news2_score` correlates **−0.82** with acuity,
`ed_los_hours` −0.76. These are not learnable clinical relationships; they are
the generator's fingerprints.

**Module 2 — Honest model, and why GroupKFold is not enough.** Using structured
intake only, LightGBM reaches accuracy **0.857**, **κ = 0.931**. Holding out
entire complaint templates (GroupKFold) barely changes it (accuracy 0.858,
κ 0.931; Δ ≈ 0). This is the non-obvious finding: the leakage is **generative,
not textual** — the vitals were sampled *from* acuity — so the standard cure
(GroupKFold) cannot detect it. Decisively, **κ = 0.93 exceeds the human inter-
rater ceiling (κ 0.6–0.8)**; no real triage model can agree with human labels
better than humans agree with each other. The score is an artefact. (Undertriage
rate 7.3%, high-acuity ESI 1–2 recall 0.98, 294 missed high-acuity cases — these
*relative* error patterns remain useful.)

**Module 3 — Equity audit.** The method surfaces language-linked undertriage:
**Arabic (8.3%)** and Somali rank highest, Estonian/Russian lowest (~7.3%); the
"Other" sex category and military insurance also rank high. Magnitudes are small
and, on synthetic labels, partly artefactual — but immigrant-community languages
showing the highest undertriage is precisely the equity signal real deployments
must monitor, and the audit code transfers directly to real data.

**Module 4 — The waiting-room blind spot is real, but the synthetic data can't
support it.** Escalation is predictable (PR-AUC 0.79 at 43% prevalence), but a
model using **all** intake features (AUC **0.826**) does **not** beat one using
**acuity alone** (AUC **0.837**). In this synthetic world disposition is
essentially an *acuity derivative*, so there is no "beyond-acuity" signal to
find. LWBS is **near-random** (AUC **0.61**): who walks out is operational noise,
not clinical state. Calibration tightens the escalation Brier score (0.167 →
0.165). The constructive message for the Foundation: the waiting-room
deterioration track is worth pursuing, but **needs real, longitudinally-linked ED
data** — the modelling is ready; the data is the bottleneck.

**Module 5 — Reality check.** Running the *identical* structured model on
**20,702 real NHAMCS visits** yields accuracy **0.54**, **κ = 0.27** — far below
the synthetic κ 0.93 and squarely beneath the human ceiling, exactly as real
triage should behave. The side-by-side κ chart (synthetic in-template / synthetic
GroupKFold / real NHAMCS) is the paper's thesis in one figure: **synthetic
illusion versus real-world ceiling.**

## Insight and impact

The field's instinct — push acuity accuracy higher — is counterproductive on a
generated benchmark, where a higher score means *more* leakage, not better
clinical reasoning. The transferable contributions are:

- **A leakage litmus test:** if model κ exceeds the human inter-rater ceiling,
  treat it as leakage until proven otherwise on real data.
- **A demonstration that GroupKFold is insufficient** against *generative*
  leakage — a subtlety most submissions miss.
- **A reusable, real-data-ready equity audit** focused on language minorities,
  fitting the Foundation's Nordic deployment context.
- **A scoped negative result** that redirects the waiting-room track toward the
  real data it requires.

This is directly actionable for a foundation planning clinical pilots: it says
what to *stop* trusting, what to *measure*, and what data to *acquire*.

## Limitations

1. The competition data is synthetic with vitals generated from acuity; in-
   distribution scores are not clinically meaningful — which is the point of
   Modules 1, 4 and 5, not a flaw in the analysis.
2. The equity audit uses synthetic labels; magnitudes are indicative of *method*,
   not real disparities.
3. NHAMCS is a cross-sectional survey with its own missingness and no patient
   linkage; κ = 0.27 reflects a deliberately small public feature set and is a
   floor, not a tuned ceiling.
4. No model here is deployment-ready; all are positioned as second-read decision
   support, never autonomous triage.

## Reproducibility

Fixed seed (42), 5-fold OOF metrics, no post-triage leakage, data paths auto-
resolved (`/kaggle/input` or local). NHAMCS is parsed from the official codebook
layout (each year parsed with its own positions, since back-half columns shift
between 2021 and 2022). The public notebook runs end-to-end; the linked
repository contains the full pipeline, the NHAMCS loader, and setup instructions.
No credentials are committed.

**Notebook:** attached public Kaggle notebook · **Code:** linked public repository.
