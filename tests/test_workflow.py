import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from fraud_detection.data import FEATURES, chronological_split, generate_transactions, prior_hour_counts
from fraud_detection.experiment import build_models, evaluate, select_threshold


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = generate_transactions(6000, seed=42)
        cls.parts = chronological_split(cls.frame)

    def test_fixed_seed_recreates_data_and_changes_with_another_seed(self):
        assert_frame_equal(self.frame, generate_transactions(6000, seed=42))
        self.assertFalse(self.frame.equals(generate_transactions(6000, seed=43)))

    def test_history_never_reads_future_transactions(self):
        np.testing.assert_array_equal(prior_hour_counts([1, 2, 1, 1, 1], [0, 1, 60, 3600, 3661]), [0, 0, 1, 2, 1])
        np.testing.assert_array_equal(prior_hour_counts([1, 1, 1], [0, 0, 1]), [0, 0, 2])

    def test_splits_are_disjoint_and_ordered(self):
        a, b, c = [self.parts[name] for name in ["train", "validation", "test"]]
        self.assertLess(a.timestamp.max(), b.timestamp.min())
        self.assertLess(b.timestamp.max(), c.timestamp.min())
        ids = pd.concat([a.transaction_id, b.transaction_id, c.transaction_id])
        self.assertEqual(len(ids), ids.nunique())
        self.assertEqual(len(ids), len(self.frame))
        self.assertFalse(set(["is_fraud", "account_id", "timestamp", "transaction_id"]) & set(FEATURES))

    def test_threshold_respects_ties_and_budget(self):
        scores = np.array([.9, .8, .8, .1, .1])
        self.assertEqual(select_threshold(scores, .4), .9)
        self.assertEqual(select_threshold(scores, .6), .8)
        threshold = select_threshold(np.ones(5), .2)
        self.assertEqual(int((np.ones(5) >= threshold).sum()), 0)
        with self.assertRaises(ValueError):
            select_threshold(scores, 0)

    def test_confusion_counts_and_zero_alert_behavior(self):
        result = evaluate(np.array([0, 0, 1, 1]), np.array([.1, .7, .8, .2]), .5)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 1, 1))
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        result = evaluate(np.array([0, 1]), np.array([.1, .2]), 1.)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["precision"], 0)

    def test_models_handle_unknown_categories_and_restorable_scores(self):
        import joblib
        import tempfile
        from pathlib import Path
        train = self.parts["train"]
        unseen = self.parts["test"].iloc[:12][FEATURES].copy()
        unseen.loc[:, "merchant_category"] = "new_merchant_type"
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as directory:
            for name, model in build_models(42).items():
                with self.subTest(model=name):
                    model.fit(train[FEATURES], train.is_fraud)
                    scores = model.predict_proba(unseen)[:, 1]
                    self.assertTrue(np.isfinite(scores).all())
                    self.assertTrue(((scores >= 0) & (scores <= 1)).all())
                    path = Path(directory) / "model.joblib"
                    joblib.dump(model, path)
                    np.testing.assert_array_equal(scores, joblib.load(path).predict_proba(unseen)[:, 1])


if __name__ == "__main__":
    unittest.main()
