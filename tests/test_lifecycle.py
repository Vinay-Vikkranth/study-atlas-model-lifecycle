from __future__ import annotations
import concurrent.futures
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from atlas_lifecycle.api import create_app
from atlas_lifecycle.data import FEATURES, digest, generate_dataset, load_dataset
from atlas_lifecycle.monitoring import drift
from atlas_lifecycle.registry import Registry
from atlas_lifecycle.training import TrainConfig, train

TEMP_ROOT = Path(__file__).resolve().parents[1]/".test-tmp"
TEMP_ROOT.mkdir(exist_ok=True)

class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="fx-", dir=TEMP_ROOT)
        cls.fixture_root = Path(cls.fixture.name)
        generate_dataset(cls.fixture_root/"synthetic.csv", rows_per_city=20)
        registry = Registry(cls.fixture_root)
        cls.first = registry.register(cls.fixture_root/"synthetic.csv", TrainConfig(seed=17))
        cls.second = registry.register(cls.fixture_root/"synthetic.csv", TrainConfig(seed=29, profile="compact"))
        registry.promote(cls.first["id"], None)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t-", dir=TEMP_ROOT)
        self.root = Path(self.temporary.name)/"registry"
        shutil.copytree(self.fixture_root, self.root)
        self.registry = Registry(self.root)
        self.dataset = self.root/"synthetic.csv"
        self.frame = load_dataset(self.dataset)

    def tearDown(self):
        self.temporary.cleanup()

    def test_geographies_do_not_cross_partitions(self):
        self.assertEqual(1, self.frame.groupby("city").split.nunique().max())
        self.assertEqual(30, self.frame.city.nunique())

    def test_all_five_candidate_families_are_exercised(self):
        self.assertEqual(5, len(self.first["grouped_cv"]))
        self.assertTrue(all(len(candidate["fold_auc"]) == 3 for candidate in self.first["grouped_cv"]))

    def test_fold_preprocessing_matches_only_fit_geography(self):
        for fold_index, audit in enumerate(self.first["folds"]):
            self.assertFalse(set(audit["fit_cities"]) & set(audit["validation_cities"]))
            fit = self.frame.loc[self.frame.city.isin(audit["fit_cities"]), FEATURES]
            medians = fit.median().to_numpy()
            mean = fit.fillna(fit.median()).mean().to_numpy()
            for candidate in self.first["grouped_cv"]:
                evidence = candidate["preprocessing"][fold_index]
                np.testing.assert_allclose(evidence["imputer_medians"], medians)
                np.testing.assert_allclose(evidence["scaler_mean"], mean)

    def test_final_test_poison_cannot_change_training_or_selection(self):
        poisoned = self.frame.copy()
        mask = poisoned.split.eq("test")
        poisoned.loc[mask, FEATURES] = 1.0
        poisoned.loc[mask, "outcome"] = 1-poisoned.loc[mask, "outcome"]
        model, report = train(poisoned, TrainConfig(seed=17))
        self.assertEqual(self.first["selected_family"], report["selected_family"])
        self.assertEqual(self.first["validation"], report["validation"])
        self.assertEqual(self.first["fitted_preprocessing"], report["fitted_preprocessing"])
        self.assertFalse(report["test_accessed"])

    def test_validation_poison_cannot_change_base_model_fitting(self):
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.split.eq("validation"), FEATURES] = .95
        _, report = train(poisoned, TrainConfig(seed=17))
        self.assertEqual(self.first["selected_family"], report["selected_family"])
        self.assertEqual(self.first["fitted_preprocessing"], report["fitted_preprocessing"])

    def test_calibration_and_threshold_use_distinct_geography(self):
        sets = [set(self.first[key]) for key in ("fit_geography", "calibration_geography", "validation_geography")]
        self.assertTrue(all(not sets[a] & sets[b] for a,b in ((0,1),(0,2),(1,2))))
        self.assertEqual("validation", self.first["gates"]["selection_partition"])
        self.assertFalse(self.first["gates"]["test_metrics_used"])

    def test_dataset_tampering_rejected(self):
        self.dataset.write_text(self.dataset.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_dataset(self.dataset)

    def test_cross_partition_city_rejected(self):
        self.frame.loc[self.frame.index[-1], "city"] = self.frame.iloc[0].city
        self.frame.to_csv(self.dataset, index=False)
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_dataset(self.dataset)

    def _refresh_manifest(self):
        path = self.dataset.with_suffix(".manifest.json")
        manifest = json.loads(path.read_text())
        manifest["sha256"] = digest(self.dataset)
        path.write_text(json.dumps(manifest))

    def test_model_tampering_rejected_before_deserialization(self):
        path = self.root/self.first["artifact"]
        path.write_bytes(path.read_bytes()+b"corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.registry.load_model(self.first["id"])

    def test_manifest_cannot_escape_model_directory(self):
        manifest = dict(self.first, artifact="../../arbitrary.joblib")
        with self.registry.db() as db:
            db.execute("UPDATE models SET manifest=? WHERE id=?", (json.dumps(manifest), self.first["id"]))
        with self.assertRaisesRegex(ValueError, "path"):
            self.registry.load_model(self.first["id"])

    def test_failed_gate_cannot_promote(self):
        manifest = json.loads(json.dumps(self.second))
        manifest["config"]["minimum_auc"] = 1.01
        with self.registry.db() as db:
            db.execute("UPDATE models SET manifest=? WHERE id=?", (json.dumps(manifest), manifest["id"]))
        with self.assertRaisesRegex(ValueError, "gates failed"):
            self.registry.promote(self.second["id"], self.first["id"])
        self.assertEqual(self.first["id"], self.registry.active_id())

    def test_restart_preserves_active_release_and_predictions(self):
        model, _ = self.registry.load_model(self.first["id"])
        restarted = Registry(self.root)
        recovered, _ = restarted.load_model(restarted.active_id())
        np.testing.assert_equal(model.predict_proba(self.frame[FEATURES].head(10)), recovered.predict_proba(self.frame[FEATURES].head(10)))

    def test_stale_promotion_rejected(self):
        self.registry.promote(self.second["id"], self.first["id"])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.registry.promote(self.first["id"], self.first["id"])

    def test_concurrent_promotions_compare_and_swap(self):
        def attempt(_):
            try:
                Registry(self.root).promote(self.second["id"], self.first["id"])
                return "success"
            except ValueError:
                return "stale"
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            result = list(pool.map(attempt, range(2)))
        self.assertCountEqual(["success", "stale"], result)

    def test_rollback_restores_exact_model(self):
        old, _ = self.registry.load_model(self.first["id"])
        expected = old.predict_proba(self.frame[FEATURES].head(8))
        self.registry.promote(self.second["id"], self.first["id"])
        self.registry.rollback(self.second["id"])
        restored, _ = self.registry.load_model(self.registry.active_id())
        np.testing.assert_equal(expected, restored.predict_proba(self.frame[FEATURES].head(8)))

    def test_rollback_without_history_rejected(self):
        with self.assertRaisesRegex(ValueError, "no prior"):
            self.registry.rollback(self.first["id"])

    def test_self_promotion_cannot_hide_release_history(self):
        self.registry.promote(self.second["id"], self.first["id"])
        with self.assertRaisesRegex(ValueError, "already active"):
            self.registry.promote(self.second["id"], self.second["id"])
        self.assertEqual(self.first["id"], self.registry.rollback(self.second["id"])["active"])

    def test_rollback_rechecks_target_gates(self):
        self.registry.promote(self.second["id"], self.first["id"])
        manifest = json.loads(json.dumps(self.first))
        manifest["config"]["minimum_auc"] = 1.01
        with self.registry.db() as db:
            db.execute("UPDATE models SET manifest=? WHERE id=?", (json.dumps(manifest), manifest["id"]))
        with self.assertRaisesRegex(ValueError, "gates failed"):
            self.registry.rollback(self.second["id"])
        self.assertEqual(self.second["id"], self.registry.active_id())

    def test_dataset_changed_during_training_cannot_register(self):
        def changed(frame, config):
            result = train(frame, config)
            self.dataset.write_text(self.dataset.read_text()+"\n")
            return result
        with patch("atlas_lifecycle.registry.train", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed during training"):
                self.registry.register(self.dataset, TrainConfig(seed=123))

    def test_final_test_idempotent_and_immutable(self):
        report = self.registry.final_test(self.first["id"], self.dataset)
        self.assertEqual(report, self.registry.final_test(self.first["id"], self.dataset))
        self.assertTrue(report["immutable"])
        self.registry.promote(self.second["id"], self.first["id"])
        with self.assertRaisesRegex(ValueError, "sealed"):
            self.registry.final_test(self.second["id"], self.dataset)

    def test_no_final_test_of_unreleased_candidate(self):
        with self.assertRaisesRegex(ValueError, "active"):
            self.registry.final_test(self.second["id"], self.dataset)

    def test_final_test_seal_survives_train_only_dataset_change(self):
        self.registry.final_test(self.first["id"], self.dataset)
        self.frame.loc[self.frame.split.eq("train"), "service_access"] = .3
        self.frame.to_csv(self.dataset, index=False, float_format="%.10f")
        self._refresh_manifest()
        new = self.registry.register(self.dataset, TrainConfig(seed=23))
        self.registry.promote(new["id"], self.first["id"])
        with self.assertRaisesRegex(ValueError, "sealed"):
            self.registry.final_test(new["id"], self.dataset)

    def test_drift_reference_stable_shift_alerts(self):
        frame = self.frame.loc[self.frame.split.eq("train"), FEATURES].copy()
        self.assertFalse(drift(frame, self.first["reference"])["alert"])
        frame["service_access"] = .01
        self.assertTrue(drift(frame, self.first["reference"])["alert"])

    def test_drift_small_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "30"):
            drift(self.frame[FEATURES].head(5), self.first["reference"])

    def test_readonly_deployment_blocks_administration(self):
        with TestClient(create_app(self.root, admin_token="")) as client:
            self.assertEqual(403, client.post("/api/promote", json={"model_id":self.second["id"], "expected_active":self.first["id"]}).status_code)
            self.assertEqual(403, client.post("/api/train", json={}).status_code)

    def test_admin_authentication_is_required(self):
        with TestClient(create_app(self.root, admin_token="local-unit-test-only")) as client:
            self.assertEqual(401, client.post("/api/rollback", json={"expected_active":self.first["id"]}).status_code)
            response = client.post("/api/promote", headers={"Authorization":"Bearer local-unit-test-only"}, json={"model_id":self.second["id"], "expected_active":self.first["id"]})
            self.assertEqual(200, response.status_code)

    def test_prediction_provenance_and_schema(self):
        with TestClient(create_app(self.root)) as client:
            response = client.post("/api/predict", json={"records":[{f:.5 for f in FEATURES}]})
            self.assertEqual(200, response.status_code)
            self.assertEqual(self.first["id"], response.json()["model_id"])
            self.assertEqual(self.first["artifact_sha256"], response.json()["artifact_sha256"])
            self.assertEqual(422, client.post("/api/predict", json={"records":[{"unexpected":.5}]}).status_code)
            self.assertEqual(422, client.post("/api/predict", json={"records":[{"income_index":2.0}]}).status_code)
            self.assertEqual(422, client.post("/api/predict", json={"records":[]}).status_code)

    def test_prediction_rejects_strings_and_boolean_values(self):
        with TestClient(create_app(self.root)) as client:
            for bad in (True, "0.5"):
                self.assertEqual(422, client.post("/api/predict", json={"records":[{"income_index":bad}]}).status_code)

    def test_grid_api_is_synthetic_and_bounded(self):
        with TestClient(create_app(self.root)) as client:
            response = client.get("/api/hexes?limit=7")
            self.assertEqual(7, len(response.json()["items"]))
            self.assertTrue(all(row["grid_id"].startswith("SYN-") for row in response.json()["items"]))
            self.assertEqual(422, client.get("/api/hexes?limit=10000").status_code)

    def test_health_metrics_and_request_correlation(self):
        with TestClient(create_app(self.root)) as client:
            health = client.get("/api/health")
            self.assertTrue(health.json()["ok"])
            self.assertIn("x-request-id", health.headers)
            self.assertIn("atlas_requests_total", client.get("/metrics").text)

    def test_readiness_fails_for_corrupt_active_artifact(self):
        path = self.root/self.first["artifact"]
        path.write_bytes(b"invalid")
        with TestClient(create_app(self.root)) as client:
            self.assertEqual(503, client.get("/api/ready").status_code)
            self.assertEqual(503, client.get("/api/health").status_code)

    def test_readiness_fails_without_a_promoted_model(self):
        with self.registry.db() as db:
            db.execute("DELETE FROM active")
        with TestClient(create_app(self.root)) as client:
            self.assertEqual(503, client.get("/api/ready").status_code)

if __name__ == "__main__":
    unittest.main()
