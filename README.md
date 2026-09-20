# Gender-Stratified Thalassemia Minor Screening from CBC Data

Code, trained models, and interactive prototype for the paper:

> **Gender Stratified Thalassemia Minor Screening from CBC Data Using Population
> Evaluated Diagnostic Indices and Explainable Machine Learning**
> Abid Hussain, Usman Akram, Ehsan Usaf, Joddat Fatima
> BIOMISA Lab, School of Interdisciplinary Engineering and Sciences (SINES),
> National University of Sciences and Technology (NUST), Islamabad, Pakistan

---

## Overview

Thalassemia minor is frequently misclassified as iron deficiency anemia in
Pakistan because both conditions produce microcytic, hypochromic CBC profiles.
Classical discrimination indices used to separate them were derived from European
and Middle-Eastern cohorts, and their applicability to Pakistani patients had not
been systematically tested.

This work evaluates those indices on a Pakistani cohort, establishes that gender
stratification is statistically warranted rather than assumed, and trains separate
explainable models for male and female patients using only routine CBC parameters.

## Key findings

**Classical indices do not transfer.** Nine of ten diagnostic indices showed no
discriminative power on this cohort. The widely cited Mentzer index reached an AUC
of just 0.197. Only the Telmissani index exceeded the 0.60 retention threshold
(AUC 0.631), with a Youden-derived Pakistani cutoff of 7.266 against the published
European cutoff of 3.3.

![Diagnostic index discriminability](figures/index_auc_comparison.png)

**Gender stratification is statistically justified.** Mann–Whitney U testing found
significant distributional differences between male and female patients in six of
seven CBC features (p < 0.05), confirming that the two groups occupy distinct
statistical spaces.

**Raw CBC parameters are the optimal feature set.** An ablation study showed that
adding diagnostic indices provides no consistent additive value. Telmissani is a
ratio of MCHC, RDW, and MCV, all already present as independent features, so
gradient boosting recovers the relationship implicitly. The value of the index
evaluation lies in demonstrating non-transferability of European cutoffs, not in
finding new predictive features.

![Ablation study](figures/extended_ablation.png)

## Method

1. Cleaning against biologically plausible ranges
2. Computation of ten diagnostic indices
3. Youden's J cutoff derivation and AUC screening per index (retain AUC ≥ 0.60)
4. Mann–Whitney U testing to validate gender stratification
5. Gender-wise split, RobustScaler normalisation, then SMOTE on training data only
6. Eight-model comparison per gender with stratified 5-fold cross-validation
7. Optuna Bayesian tuning of the three top gradient boosting models
8. SHAP explainability on the held-out test set

Final model selection used cross-validation F1 during tuning rather than test-set
performance, to avoid overfitting the evaluation set.

![Model comparison](figures/model_comparison.png)

## Results

LightGBM achieved the highest cross-validation F1 for both genders.

| Gender | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity |
|--------|----------|-----|-----|-----|-------------|-------------|
| Male   | 92.2% | 81.4% | 0.934 | 0.764 | 82.8% | 94.6% |
| Female | 91.9% | 70.7% | 0.915 | 0.661 | 67.3% | 96.1% |

![Confusion matrices and ROC curves](figures/gender_cm_roc.png)

Sensitivity is substantially lower in the female model. Roughly one in three
female carriers is missed, so a negative result in female patients carries more
uncertainty than in male patients.

## Explainability

SHAP analysis reveals a gender-differential pattern in model-level feature
importance. RBC dominates male predictions (mean |SHAP| 1.598) while MCH dominates
female predictions (1.919). RBC importance is roughly 1.7 times higher for males,
and MCH importance roughly 3.7 times higher for females.

![SHAP gender comparison](figures/shap_gender_comparison.png)

These are model-level patterns. Whether they reflect underlying biological
differences in thalassemia minor presentation requires further clinical
investigation.

## Repository contents

```
streamlit_app.py              Interactive screening prototype
model_male.pkl                Male LightGBM model, scaler, features, cutoffs
model_female.pkl              Female LightGBM model, scaler, features, cutoffs
requirements.txt              Pinned dependencies
sample_batch.csv              Example batch input
custom.css                    App styling
.streamlit/config.toml        App theme
notebooks/
  thalassemia_pipeline.ipynb  Full pipeline: cleaning to SHAP and ablation
figures/                      Figures reproduced from the paper
SETUP.md                      Local setup and deployment instructions
```

Each model bundle is a dictionary containing the fitted `LGBMClassifier`, its
`RobustScaler`, the feature list, and the Pakistan-adjusted index cutoffs. The
scaler was fitted on unnamed arrays, so feature order is positional and must not
be changed:

`['RBC', 'HB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'Telmissani']`

where Telmissani = (MCHC × RDW) / MCV.

## Dataset

The dataset comprised 3,415 CBC records collected from multiple hospitals across
Pakistan, verified by a certified hematologist. **It is not included in this
repository**, as it contains patient clinical records shared for internship
research without documented consent for redistribution.

Detailed metadata including collection period, participating institutions, age
range, inclusion criteria, and the clinical reference standard used to confirm
diagnosis were not available for this study, and are noted as limitations.

## Running the prototype

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

See `SETUP.md` for full setup and deployment instructions.

The app accepts single-patient entry or batch upload from CSV or Excel. Batch
files require the seven CBC columns plus a `Gender` column; common column aliases
are recognised automatically, and rows with missing, non-numeric, or
implausible values are skipped and reported.

## Limitations

- Single-country dataset with no external validation set
- Trained to separate normal from thalassemia minor, not iron deficiency anemia
  from thalassemia trait, which is the harder and more clinically relevant
  discrimination
- Confidence intervals and PR-AUC were not computed
- Age-specific stratification was not applied
- The prototype has not undergone clinical evaluation

> **This is a research prototype.** It is not a diagnostic device and must not be
> used for patient care. A positive screening result requires confirmation by
> HbA2 quantification via HPLC or hemoglobin electrophoresis.

## Citation

Citation details will be added following the review outcome.

## Acknowledgement

The authors acknowledge the support of BIOMISA Lab, SINES, NUST, Islamabad. This
study used retrospectively collected anonymised CBC records; no patient-
identifiable information was accessed, and ethical oversight was provided by the
participating institutions.
