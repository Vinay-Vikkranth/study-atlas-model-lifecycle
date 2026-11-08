# Research lineage and what changed

The source reviewed for this extension was the existing Study Atlas FastAPI application, model service and five-model registry builder. Their fingerprints are recorded in [lineage.json](lineage.json). The public source context is [gun-violence-geospatial-ml](https://github.com/Vinay-Vikkranth/gun-violence-geospatial-ml). No source directory was modified.

| Existing Study Atlas capability | Lifecycle extension |
|---|---|
| Grid identifier and city-level inspection | New generated SYN cell identifiers and fictional city groups through /api/hexes |
| Validated /api/predict scenario scoring | New strict normalized schema and bounded batched predictions with release fingerprints |
| Five saved historical research classifiers | Five fully exercised new synthetic candidate families; original weights remain private |
| Model/data fingerprints | Dataset, configuration, training-code and serialized-model SHA256 plus guarded registration |
| Local scenario persistence | Transactional release pointer, release events, sealed final test and monitoring reports |
| Historical research evaluation | Separate fold-contained grouped CV, disjoint calibration/validation/test cities and immutable final-test reporting |

This is not a fork that republishes restricted fitted models or simply wraps the prior paper scores in a dashboard. It is a new model-lifecycle experiment, connected to the original application through its schema, provenance practices, serving boundaries and inspection workflow. Reimplementing these concepts makes the public demo reproducible without research inputs whose redistribution is not approved.

The original collaboration's authorship and licensing notice remain unchanged. New synthetic metrics must not replace the historical LightGBM, XGBoost, random forest, gradient boosting or logistic-regression results in a paper or resume. The synthetic experiment is not evidence of prospective or causal validation. No cloud deployment or production users are implied.
