"""Fit on training data; choose model and threshold on validation only."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .data import CATEGORICAL, FEATURES, NUMERIC, chronological_split, generate_transactions


def select_threshold(scores, budget):
    """Maximize alerts up to the budget; never split a tied score group."""
    if not 0 < budget < 1:
        raise ValueError("Review budget must lie strictly between 0 and 1.")
    scores = np.asarray(scores)
    values, counts = np.unique(scores, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    allowed = np.flatnonzero(np.cumsum(counts) <= int(len(scores) * budget))
    return float(values[allowed[-1]]) if len(allowed) else float(np.nextafter(values[0], np.inf))


def evaluate(y, scores, threshold):
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "alert_rate": float(predictions.mean()), "threshold": float(threshold),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_intervals(y, scores, threshold, seed, repetitions=200):
    rng = np.random.default_rng(seed)
    collected = {key: [] for key in ["average_precision", "precision", "recall"]}
    y = np.asarray(y)
    for _ in range(repetitions):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size != 2:
            continue
        result = evaluate(y[index], scores[index], threshold)
        for key in collected:
            collected[key].append(result[key])
    return {key: [float(x) for x in np.quantile(values, [0.025, 0.975])]
            for key, values in collected.items()}


def build_models(seed):
    def preprocessing():
        return ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
    return {
        "Logistic regression": Pipeline([("features", preprocessing()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed))]),
        "Boosted trees": Pipeline([("features", preprocessing()),
            ("classifier", HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15,
             learning_rate=0.07, l2_regularization=5, min_samples_leaf=60,
             early_stopping=False, random_state=seed))]),
    }


def run_experiment(root, n=40000, seed=42, budget=0.02):
    root = Path(root)
    for directory in ["data", "artifacts", "reports/figures"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    frame = generate_transactions(n, seed)
    csv = frame.to_csv(index=False, float_format="%.12g")
    (root / "data/transactions.csv").write_text(csv)
    fingerprint = hashlib.sha256(csv.encode()).hexdigest()
    # Fit exactly the persisted representation used by future reruns.
    frame = pd.read_csv(root / "data/transactions.csv", parse_dates=["timestamp"])
    parts = chronological_split(frame)
    train, validation, test = [parts[name] for name in ["train", "validation", "test"]]
    models, results, scores = build_models(seed), {}, {}
    with threadpool_limits(limits=1):
        for name, model in models.items():
            model.fit(train[FEATURES], train.is_fraud)
            validation_scores = model.predict_proba(validation[FEATURES])[:, 1]
            threshold = select_threshold(validation_scores, budget)
            results[name] = {"validation": evaluate(validation.is_fraud, validation_scores, threshold)}
        selected = max(results, key=lambda name: results[name]["validation"]["average_precision"])
        # All decisions are now frozen. Evaluate both fixed models for comparison.
        for name, model in models.items():
            scores[name] = model.predict_proba(test[FEATURES])[:, 1]
            results[name]["test"] = evaluate(test.is_fraud, scores[name], results[name]["validation"]["threshold"])
        importance = permutation_importance(models[selected], validation[FEATURES], validation.is_fraud,
             scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=1)
    threshold = results[selected]["validation"]["threshold"]
    intervals = bootstrap_intervals(test.is_fraud, scores[selected], threshold, seed)
    splits = {name: {"rows": len(part), "fraud_count": int(part.is_fraud.sum()),
                    "fraud_rate": float(part.is_fraud.mean()),
                    "start": part.timestamp.min().isoformat(), "end": part.timestamp.max().isoformat()}
              for name, part in parts.items()}
    summary = {"config": {"rows": n, "seed": seed, "review_budget": budget},
               "data_sha256": fingerprint, "selected_model": selected,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
               "splits": splits, "models": results, "test_bootstrap_95_percent": intervals,
               "baseline": {"no_alerts_accuracy": float(1 - test.is_fraud.mean()),
                            "no_alerts_recall": 0.0, "random_ranking_expected_ap": float(test.is_fraud.mean())},
               "importance": [{"feature": name, "mean_ap_drop": float(mean), "std_ap_drop": float(std)}
                              for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)]}
    (root / "reports/metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    joblib.dump({"model": models[selected], "threshold": threshold, "features": FEATURES,
                 "data_sha256": fingerprint, "config": summary["config"]}, root / "artifacts/model.joblib")
    assignments = pd.concat([part[["transaction_id"]].assign(split=name) for name, part in parts.items()])
    assignments.to_csv(root / "artifacts/split_assignments.csv", index=False)
    pd.DataFrame({"transaction_id": test.transaction_id, "is_fraud": test.is_fraud,
                  "score": scores[selected], "alert": scores[selected] >= threshold}).to_csv(
                      root / "artifacts/test_predictions.csv", index=False)
    from .report import write_report
    write_report(root, frame, parts, scores, summary)
    return summary
