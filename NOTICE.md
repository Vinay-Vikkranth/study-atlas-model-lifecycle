# Attribution, scope and reference material

Study Atlas Model Lifecycle is a new synthetic-only engineering extension prepared for Vinay Vikkranth Subramanian with AI assistance. It builds on the actual Study Atlas application's concepts: grid/city inspection, validated scenario inputs, multiple classifiers, model/data provenance and FastAPI serving.

Historical research is collaborative ASU Quark Lab work with Dr. Asish Satpathy and Avantika Tiwari. The original public research repository is [gun-violence-geospatial-ml](https://github.com/Vinay-Vikkranth/gun-violence-geospatial-ml). The independently verified coauthored AMCIS 2026 paper is *A Harmonized Multi-City Architecture for Structural Spatial Risk Modeling*. This new lifecycle experiment does not reproduce its scientific results or establish sole authorship of that work.

No original research source file, raw dataset, trained research weight, private scenario database or unpublished manuscript is redistributed here. The historical repository's no-blanket-license notice remains applicable to its material. The new lifecycle code and synthetic generator are independently written for this extension; source fingerprints and conceptual lineage are recorded separately.

References used as design guidance, not copied implementations:

- [BentoML model-serving documentation](https://docs.bentoml.org/en/latest/): packaging, service boundaries and serving design patterns. BentoML is Apache-2.0 licensed. This project uses FastAPI and **does not claim to exercise BentoML adaptive batching**.
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): disjoint fitting and calibration guidance. scikit-learn retains its BSD-3-Clause license.
- [scikit-learn decision threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html): use data separate from base-estimator fitting when selecting a threshold.
- [FastAPI security documentation](https://fastapi.tiangolo.com/tutorial/security/): bearer authentication concepts. FastAPI retains its MIT license.

Dependency licenses remain with their respective authors. Repository visibility is not a license grant for the separate original research.
