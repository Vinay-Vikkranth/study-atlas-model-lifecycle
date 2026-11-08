# Synthetic lifecycle model card

## Purpose and relationship to Study Atlas

This is an engineering extension to the existing Study Atlas research application: grid/city record inspection, explicit input validation, versioned model serving, source fingerprints and scenario exploration. It adds a new, separately versioned experiment rather than replacing or correcting the historical study in place. The original Gun Violence study covers 35 metropolitan areas and 11,448 cells. This public experiment covers **30 fictional city groups and 2,160 generated rows**. None of those generated rows is a research observation.

All six features are normalized synthetic proxies. The binary label is sampled from a documented logistic data generator. The dataset has arbitrary diagram coordinates and no real people, cities, addresses or incident outcomes. Missing values are injected independently of labels. Historical random-cell, city-holdout and regional-holdout results remain different experiments and are not reported as results of this project.

## Training, selection and calibration

1. An immutable generator seed creates 30 geography groups. The manifest binds the CSV bytes, feature schema and group assignments.
2. Eighteen groups train all five candidate families: logistic regression, random forest, gradient boosting, histogram gradient boosting and ExtraTrees. Each grouped CV fold clones a pipeline containing its own median imputer, scaler and classifier. Preprocessing statistics and fit-city lists are recorded per candidate/fold.
3. The family with the largest mean training-only grouped CV AUC is fitted on the training partition. No calibration, validation or test labels choose that family.
4. A sigmoid calibrator is fitted on four separate calibration cities while the base pipeline remains frozen.
5. Four separate validation cities select an F1 threshold and evaluate fixed release gates: AUC at least 0.60, F1 at least 0.40 and Brier score at most 0.30. These are demonstration gates, not established public-safety acceptance criteria.
6. Only after an administrator selects and promotes a release can four final-test cities be scored. The report is sealed by the complete held-out partition fingerprint, so changing training rows cannot unlock another candidate's test score. Repeating the same evaluation returns the original report. It is never used by the promotion gate.

Data fingerprinting reads and validates bytes across all partitions; this is provenance validation, not fitting or evaluating on the test set. Final-test labels and features do not enter fitting, calibration, threshold selection or promotion. A regression test flips all final-test labels and poisons their features and confirms unchanged selected family, validation metrics and fitted preprocessing.

## Serving and release behavior

Requests contain one to 256 rows. Unknown features, booleans, numeric strings, infinities and values outside the normalized schema are rejected; explicit missing values are handled by the fitted imputer. A response includes probability, threshold, model ID, dataset SHA256 and artifact SHA256. The active pointer is resolved once per request so its provenance matches the loaded artifact during a concurrent promotion.

Release changes use SQLite write transactions and an expected-active compare-and-swap. Artifacts are loaded only from generated registry paths after hash validation; there is no model upload or arbitrary pickle endpoint. Locally generated joblib artifacts still require a trusted writable volume. Anyone who controls both the registry database and artifact files is outside this integrity threat model.

## Monitoring and limitations

Input drift uses fixed training-derived bins, PSI with additive smoothing and a separate missingness threshold. It can detect distribution changes but cannot establish label drift, model-quality deterioration, fairness or policy usefulness. The bundled reference and shifted batches are controlled simulations.

No probability is a real-world risk estimate. No model is prospectively validated, causal, suitable for law enforcement, an individual decision system or evidence of production customer adoption. Sampling variation, synthetic-generator assumptions and small geography counts limit metrics. No subgroup fairness claim is made. Cloud or container recipes alone do not prove a live deployment.

The exact generated metrics and model fingerprints are in [experiment.json](evidence/experiment.json). Reproduce them rather than copying a historical paper score into this model card.
