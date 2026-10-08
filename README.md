# Fraud detection with transaction data

A complete educational experiment: generate simulated transactions, train two AI models, select an alert threshold, evaluate later transactions, and create a clean report with graphs. Every data point is simulated; the results demonstrate the workflow rather than real-world banking performance.

Start with the [full HTML report](reports/report.html), or read the [Markdown report](reports/report.md) directly in GitHub. The HTML file is self-contained, works offline, and has a print layout for exporting a PDF from your browser.

## Reproduce the study

Use **Python 3.12** (validated with 3.12.14). From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m fraud_detection run
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m fraud_detection verify
```

On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`. The cloud workspace already has the virtual environment and generated outputs.

`run` uses **40,000 transactions, seed 42, and a 2% validation review budget**. It overwrites the generated dataset, model, predictions, metrics, figures, and reports under its output directory. `verify` repeats the saved configuration in a temporary directory and checks exact equality of data fingerprints, splits, predictions, metrics, bootstrap intervals, importance, reports, and figures. It exits with an error if any comparison differs. Model files are tested for prediction-preserving serialization in the test suite.

`requirements.txt` records the main libraries; `requirements-lock.txt` pins all installed runtime dependencies and is the file to use for reproduction. No API keys, paid services, GPU, or external dataset download are needed. Installation uses PyPI; the experiment itself runs offline. Exact equality is verified in the current environment. Different Python, operating system, numerical libraries, or processor implementations may introduce small variations; the dataset hash and recorded versions help diagnose them.

## Study design

1. **Generate** transactions spanning 120 days and 1,200 possible accounts. The simulator creates overlapping legitimate and fraud patterns, transaction bursts, hidden noise, and mild time drift. Fraud labels are sampled from an explicit risk function.
2. **Split chronologically**: first 60% for training, next 20% for validation, final 20% for testing. Accounts may recur across periods; this tests later transactions for existing customers, not unseen-customer generalization.
3. **Train** logistic regression and histogram gradient-boosted trees. Numeric scaling and categorical one-hot encoding are fitted only on training data. Model hyperparameters are fixed in source. There is no oversampling, class weighting, or search over test results.
4. **Select** the model with the higher validation average precision (AP). For each model, choose the lowest score threshold that flags at most the validation review budget, including complete groups of tied scores. The choice is frozen before scoring test data. The fitted model remains trained on the training period; it is not refit after threshold selection.
5. **Evaluate** both fixed models on the later test period. Report AP, ROC AUC, precision, recall, F1, alert volume, confusion counts, and transaction-bootstrap intervals. Compute permutation importance on validation data only. Do not reselect the model because of test results.

AP summarizes precision across recall levels and is especially useful for rare fraud. It is not accuracy or trapezoidal area under the PR curve. ROC AUC measures ranking and should be read alongside precision–recall. A review threshold is an operational choice: flagging more transactions usually catches more fraud but creates more false alerts. A budget satisfied on validation data need not hold in a future period.

## Default results

The validation-selected boosted tree model, evaluated on 8,000 later transactions:

| Measure | Result |
|---|---:|
| Test fraud cases | 139 |
| Detected fraud | 48 |
| Missed fraud | 91 |
| False alerts | 221 |
| Precision | 17.84% |
| Recall | 34.53% |
| Average precision | 0.158 |
| ROC AUC | 0.834 |
| Transactions flagged | 3.36% |

The always-legitimate baseline achieves 98.26% accuracy while finding zero fraud. The random-ranking AP reference is the test fraud prevalence, approximately 0.017. The model improves ranking, but most fraud remains missed at this operating point and most alerts are false. Its frozen threshold exceeds the 2% validation review budget on test data. These outcomes are useful teaching results, not a production recommendation.

## Files and outputs

```text
fraud_detection/
  data.py                  simulator and chronological splitting
  experiment.py            preprocessing, training, selection and evaluation
  report.py                figures and HTML/Markdown reporting
  __main__.py              run and verify commands
tests/test_workflow.py      causal history, split, threshold and model checks
requirements-lock.txt      exact dependency versions
data/transactions.csv      generated transactions (ignored in Git)
artifacts/
  model.joblib             selected fitted model, threshold, feature names
  split_assignments.csv    transaction IDs and their split
  test_predictions.csv     held-out labels, scores and alert decisions
reports/
  report.html              self-contained presentation with embedded graphs
  report.md                GitHub-readable summary
  metrics.json             exact metrics, configuration and dataset SHA-256
  figures/                 six exportable PNG charts
```

Data and binary models are recreated by `run`; reports and figures can be committed to share the results. The six tests cover seeded data generation, causal history calculation, disjoint time splits and feature exclusions, score ties under the review budget, confusion counts, unseen categories, and model save/load behavior. The `verify` command is the end-to-end reproducibility check. Never load untrusted `joblib` model files.

## Data dictionary

| Column | Meaning | Used by model? |
|---|---|---|
| `transaction_id` | Unique generated transaction identifier | No |
| `timestamp` | Decision time on the simulated timeline | Split/history only |
| `account_id` | Generated account identifier; customers can repeat | History only |
| `amount` | Positive amount in arbitrary synthetic currency units | Yes |
| `amount_to_historical_mean` | Amount divided by an idealized pre-period account mean | Yes |
| `hour` | Hour of day, 0–23 | Yes |
| `account_age_days` | Account tenure at the transaction time | Yes |
| `transactions_last_hour` | Same-account count in the prior hour, excluding simultaneous and future transactions | Yes |
| `distance_from_home_km` | Simulated distance from the account's home location | Yes |
| `new_device` | 1 if a newly observed device is used | Yes |
| `foreign_transaction` | 1 if a simulated foreign transaction | Yes |
| `merchant_category` | Groceries, travel, electronics, services, or cash | Yes |
| `channel` | Online, in store, or ATM | Yes |
| `is_fraud` | Sampled target label: 1 fraud, 0 legitimate | Target only |

The historical mean is generated before the observation window, not estimated using future transactions. The simulator is fully visible in `data.py`; it does not insert fraud labels, hidden risk probabilities, or post-investigation facts into the feature table. Synthetic label availability is immediate, unlike real-world chargebacks or investigations.

## Explore without replacing the reference report

```bash
.venv/bin/python -m fraud_detection run --seed 7 --rows 60000 --review-budget 0.04 --output /tmp/fraud-experiment
```

Use a different output directory for each experiment. The HTML report also shows the validation precision–recall tradeoff as review volume changes. More observations, a different seed, or a different budget change the study; record those choices and use `verify --output /tmp/fraud-experiment` to reproduce them. If you redesign features or tune hyperparameters after seeing test performance, create a new holdout rather than treating the old test results as unbiased.

## Conclusions and next steps

AI fraud detection learns patterns from labeled historical transactions and ranks new ones. It does not establish that an individual transaction is fraudulent. This project shows how class imbalance, threshold choice, time drift, and review capacity affect the usefulness of those rankings.

The simulation simplifies adversarial behavior, delayed labels, correlation, missing data, and the costs of investigations. Scores are not calibrated real-world probabilities. Bootstrap intervals use 200 transaction-level resamples with a fixed model and threshold; they exclude retraining uncertainty and account/time dependence. Permutation importance measures predictive associations and can be affected by correlated features; it is not causal evidence.

For a real-data extension, document provenance and label definitions, construct features using only information available at decision time, allow labels to mature, use rolling time validation and grouped uncertainty estimates, evaluate new-customer behavior separately, calibrate scores, and choose review thresholds using explicit costs. Those are future extensions; the present project is a complete simulated-data study.
