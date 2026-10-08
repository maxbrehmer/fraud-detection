"""Create exportable figures and a self-contained HTML research report."""
import base64
import html
import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / ".cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = ["#2056a5", "#c35c2b"]
STYLE = """
:root{--ink:#162b41;--muted:#526678;--blue:#2056a5;--line:#dbe3eb}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f6f9;color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
header{background:var(--ink);color:white;padding:64px max(24px,calc((100vw - 1080px)/2));}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:#afc9ea}
h1{font-size:clamp(32px,5vw,52px);line-height:1.1;margin:18px 0}header p{max-width:720px;color:#dce5f0}
.tag{display:inline-block;border:1px solid #58718d;padding:5px 12px;border-radius:20px;font-size:12px;margin:4px 8px 0 0}
nav{background:white;border-bottom:1px solid var(--line);padding:14px 24px;text-align:center}nav a{display:inline-block;color:var(--blue);margin:0 12px;text-decoration:none;font-size:14px}
main{max-width:1130px;padding:32px 24px 64px;margin:auto}section{background:white;border:1px solid var(--line);border-radius:12px;padding:30px;margin-bottom:24px}
h2{font-size:26px;line-height:1.25;margin-top:0}h3{font-size:19px;margin-top:28px}.muted,figcaption{color:var(--muted)}
.callout{border-left:4px solid var(--blue);background:#edf3fb;padding:15px 20px;margin:20px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{background:#f4f7fb;border:1px solid var(--line);border-radius:8px;padding:20px}.card strong{font-size:30px;display:block;color:var(--blue)}.card span{font-size:13px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line)}th{background:#f4f7fb}.table-wrap{overflow-x:auto}
figure{margin:28px 0}figure img{display:block;width:100%;max-width:920px;margin:auto}figcaption{font-size:14px;margin:10px 10px 0}
.workflow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.workflow div{border-top:3px solid var(--blue);background:#f4f7fb;padding:15px;font-size:14px}.workflow b{display:block;margin-bottom:8px}
code{background:#edf1f5;padding:2px 5px;border-radius:4px;font-size:13px}pre{background:#162b41;color:#f3f6f9;padding:20px;overflow:auto;border-radius:8px}pre code{background:none;color:inherit}
.hash{overflow-wrap:anywhere;font-size:12px}footer{color:var(--muted);font-size:13px;text-align:center;padding:20px}
@media(max-width:700px){.cards,.workflow{grid-template-columns:repeat(2,1fr)}section{padding:20px}th,td{padding:8px}}
@media print{body{background:white;font-size:11pt}header{padding:24px;color:#162b41;background:white}header p,.eyebrow{color:#526678}nav{display:none}main{padding:0}section{border:none;padding:12px 0;break-inside:auto}figure,.cards,table{break-inside:avoid}h2,h3{break-after:avoid}pre{white-space:pre-wrap}.tag{border-color:#aaa}}
"""


def pct(value):
    return f"{value:.1%}"


def write_report(root, frame, parts, scores, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.18, "axes.axisbelow": True, "font.size": 11})
    figures = root / "reports/figures"
    selected = summary["selected_model"]
    result = summary["models"][selected]["test"]
    threshold = result["threshold"]
    test = parts["test"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rates = [summary["splits"][name]["fraud_rate"] * 100 for name in parts]
    axes[0].bar(list(parts), rates, color=["#2056a5", "#5982b8", "#94b0d2"])
    axes[0].set(ylabel="Fraud prevalence (%)", title="Class imbalance across time periods")
    for i, value in enumerate(rates):
        axes[0].text(i, value, f"{value:.2f}%", ha="center", va="bottom")
    axes[0].set_ylim(0, max(rates) * 1.22)
    for label, group, color in [("Legitimate", 0, COLORS[0]), ("Fraud", 1, COLORS[1])]:
        axes[1].hist(frame.loc[frame.is_fraud == group, "amount"], bins=np.geomspace(.5, frame.amount.max()+1, 45),
                     density=True, alpha=.55, color=color, label=label)
    axes[1].set(xscale="log", yscale="log", xlabel="Transaction amount (synthetic currency units)",
                ylabel="Density (log scale)", title="Amount distributions overlap")
    axes[1].legend()
    save(fig, "dataset")

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, values), color in zip(scores.items(), COLORS):
        precision, recall, _ = precision_recall_curve(test.is_fraud, values)
        ap = summary["models"][name]["test"]["average_precision"]
        ax.step(recall, precision, where="post", color=color, label=f"{name} · AP {ap:.3f}")
        metrics = summary["models"][name]["test"]
        ax.scatter(metrics["recall"], metrics["precision"], color=color, s=55, zorder=3)
    ax.axhline(test.is_fraud.mean(), color="#718096", linestyle="--", label="Random-ranking reference: prevalence")
    ax.set(xlabel="Recall: share of fraud detected", ylabel="Precision: fraud among alerts",
           title="Precision–recall on the untouched test period", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    save(fig, "precision_recall")

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, values), color in zip(scores.items(), COLORS):
        fpr, tpr, _ = roc_curve(test.is_fraud, values)
        ax.plot(fpr, tpr, color=color, label=f"{name} · AUC {summary['models'][name]['test']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#718096", label="Random-ranking reference")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC on the untouched test period", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, "roc")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    ax.grid(False)
    ax.imshow(np.log1p(matrix), cmap="Blues")
    for i in range(2):
        for j in range(2):
            label = [["Correctly not flagged", "False alert"], ["Missed fraud", "Detected fraud"]][i][j]
            ax.text(j, i, f"{matrix[i,j]:,}\n{label}", ha="center", va="center",
                    color="white" if np.log1p(matrix[i,j]) > np.log1p(matrix.max())*.65 else "#162b41")
    ax.set(xticks=[0, 1], xticklabels=["No alert", "Alert"], yticks=[0, 1],
           yticklabels=["Legitimate", "Fraud"], xlabel="Model decision", ylabel="Actual label",
           title=f"{selected}: test decisions at the frozen threshold")
    save(fig, "confusion_matrix")

    # Validation scores are reconstructed from the retained fitted model, without refitting.
    import joblib
    from .data import FEATURES
    from threadpoolctl import threadpool_limits
    bundle = joblib.load(root / "artifacts/model.joblib")
    with threadpool_limits(limits=1):
        val_scores = bundle["model"].predict_proba(parts["validation"][FEATURES])[:, 1]
    val_y = parts["validation"].is_fraud.to_numpy()
    precision, recall, thresholds = precision_recall_curve(val_y, val_scores)
    # One entry per unique threshold; the PR endpoint has no associated threshold.
    sorted_scores = np.sort(val_scores)
    alert_rates = (len(sorted_scores) - np.searchsorted(sorted_scores, thresholds, side="left")) / len(sorted_scores)
    fig, ax = plt.subplots(figsize=(9, 5))
    within = alert_rates <= .15
    ax.plot(alert_rates[within]*100, precision[:-1][within]*100, color=COLORS[0], label="Precision")
    ax.plot(alert_rates[within]*100, recall[:-1][within]*100, color=COLORS[1], label="Recall")
    ax.axvline(summary["config"]["review_budget"]*100, color="#718096", linestyle="--", label="Chosen review budget")
    ax.set(xlabel="Validation transactions flagged (%)", ylabel="Metric (%)", ylim=(0, 100),
           title=f"{selected}: more reviews trade precision for recall")
    ax.legend()
    save(fig, "review_tradeoff")

    importance = sorted(summary["importance"], key=lambda item: item["mean_ap_drop"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([item["feature"].replace("_", " ") for item in importance],
            [item["mean_ap_drop"] for item in importance],
            xerr=[item["std_ap_drop"] for item in importance], color=COLORS[0], alpha=.85)
    ax.axvline(0, color="#718096", linewidth=.8)
    ax.set(xlabel="Decrease in validation average precision when shuffled",
           title="Which recorded features help the selected model?")
    save(fig, "feature_importance")

    def figure(name, caption):
        encoded = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    split_rows = "".join(f"<tr><td>{name.title()}</td><td>{s['rows']:,}</td><td>{s['fraud_count']:,}</td><td>{pct(s['fraud_rate'])}</td><td>{s['start'][:10]} → {s['end'][:10]}</td></tr>"
                         for name, s in summary["splits"].items())
    comparison_rows = "".join(
        f"<tr><td>{name}{' ★' if name == selected else ''}</td><td>{m['validation']['average_precision']:.3f}</td>"
        f"<td>{m['test']['average_precision']:.3f}</td><td>{m['test']['roc_auc']:.3f}</td>"
        f"<td>{pct(m['test']['precision'])}</td><td>{pct(m['test']['recall'])}</td><td>{pct(m['test']['alert_rate'])}</td></tr>"
        for name, m in summary["models"].items())
    ci_rows = "".join(f"<tr><td>{name.replace('_', ' ').title()}</td><td>{result[name]:.3f}</td><td>{bounds[0]:.3f}–{bounds[1]:.3f}</td></tr>"
                      for name, bounds in summary["test_bootstrap_95_percent"].items())
    dataset_caption = "Fraud is rare in every period. Amounts overlap, so a large transaction alone cannot establish fraud. Amount histograms are normalized separately within each class."
    pr_caption = "Dots show thresholds fixed using validation data. Average precision (AP) summarizes precision across recall levels; it is not accuracy and is not trapezoidal PR area."
    roc_caption = "ROC measures ranking across thresholds. With rare fraud, a low false-positive rate can still produce many false alerts; read it alongside precision–recall."
    confusion_caption = "Cell labels are exact transaction counts; background colors use a log scale to keep small cells visible. Alerts mean review candidates, not proven fraud."
    tradeoff_caption = "This curve uses validation data only. The threshold fills the budget as closely as score ties allow; the test alert rate may exceed or fall below that budget."
    importance_caption = "Validation permutation importance: five shuffles per feature; error bars show standard deviation across shuffles. Associations reflect the simulation and model, not causal explanations."
    summary_text = (f"The validation-selected {selected.lower()} flags {result['tp'] + result['fp']:,} of {len(test):,} test transactions. "
                    f"It detects {result['tp']:,} of {int(test.is_fraud.sum()):,} fraud cases ({pct(result['recall'])} recall), "
                    f"and {pct(result['precision'])} of its alerts are fraud. There are {result['fp']:,} false alerts and {result['fn']:,} missed fraud cases.")
    budget_note = (f"The threshold was selected under a {pct(summary['config']['review_budget'])} validation review budget, "
                   f"but flagged {pct(result['alert_rate'])} of test transactions. The budget is a validation constraint, "
                   "not a guarantee on future workload; a real review team would need volume monitoring and an overflow policy.")
    conclusion = (f"The model concentrates probable fraud into a small review queue: test average precision is {result['average_precision']:.3f}, "
                  f"compared with a random-ranking reference of {test.is_fraud.mean():.3f}. "
                  f"At the chosen workload, most fraud is {'still missed' if result['recall'] < .5 else 'detected'}, "
                  "showing why threshold selection is a business decision. This result demonstrates how the workflow works on this simulator; "
                  "it does not establish effectiveness on real transactions.")
    body = f"""
<header><div class="eyebrow">An educational machine learning experiment</div><h1>Finding fraud<br>in transaction data</h1>
<p>A reproducible study of rare-event detection: how models learn patterns, how alerts are selected, and what validation tells us.</p>
<span class="tag">Simulated data</span><span class="tag">{len(frame):,} transactions</span><span class="tag">Seed {summary['config']['seed']}</span><span class="tag">Chronological holdout</span></header>
<nav><a href="#overview">Results</a><a href="#data">Data</a><a href="#workflow">Workflow</a><a href="#evaluation">Validation</a><a href="#conclusions">Conclusions</a><a href="#reproduce">Reproduce</a></nav>
<main><section id="overview"><h2>What did the experiment find?</h2><p>{summary_text}</p>
<div class="cards"><div class="card"><strong>{result['average_precision']:.3f}</strong><span>Test average precision</span></div>
<div class="card"><strong>{pct(result['precision'])}</strong><span>Fraud among alerts · precision</span></div>
<div class="card"><strong>{pct(result['recall'])}</strong><span>Fraud detected · recall</span></div>
<div class="card"><strong>{pct(result['alert_rate'])}</strong><span>Test transactions sent to review</span></div></div>
<div class="callout"><b>Scope of the evidence.</b> Every transaction and fraud label is simulated. These results teach the modeling process and describe this specific generator. They are not a claim about real bank, card, or payment performance.</div>
<p>{budget_note}</p><p>An always-legitimate baseline reaches <b>{pct(summary['baseline']['no_alerts_accuracy'])} accuracy</b> while detecting <b>zero fraud</b>. That is why accuracy is not our main metric.</p></section>
<section id="data"><h2>01 · Transaction data</h2><p>The dataset spans 120 simulated days and 1,200 possible customer accounts. It includes ordinary transactions and bursts of activity. Fraud labels are sampled from a risk function with overlapping legitimate and fraudulent patterns, hidden noise, and mild time drift.</p>
<p>The model sees amount, amount relative to a simulated pre-period historical mean, hour, account age, prior-hour transaction count, distance from home, new-device status, foreign-transaction status, merchant category, and channel. Transaction IDs, account IDs, timestamps, and labels are excluded from the predictors.</p>
<p>The historical mean is an idealized account profile generated before this study window. Prior-hour counts use earlier transactions only. In real data, each feature must be available at the decision time; investigations, chargebacks, and future transactions would leak information.</p>
<div class="table-wrap"><table><thead><tr><th>Period</th><th>Transactions</th><th>Fraud</th><th>Prevalence</th><th>Dates (simulated)</th></tr></thead><tbody>{split_rows}</tbody></table></div>
{figure('dataset', dataset_caption)}</section>
<section id="workflow"><h2>02 · From data to a frozen decision rule</h2>
<div class="workflow"><div><b>1. Generate</b>Fixed seed, explicit risk function, saved CSV and SHA-256 fingerprint.</div><div><b>2. Train · first 60%</b>Fit preprocessing and two classifiers using earlier transactions.</div><div><b>3. Validate · next 20%</b>Select model by AP; select each threshold within the review budget.</div><div><b>4. Test · final 20%</b>Freeze decisions, score later transactions, and measure results.</div></div>
<h3>Models and preprocessing</h3><p>Logistic regression is a transparent linear baseline in transformed feature space. Histogram gradient-boosted trees can learn nonlinear relationships and interactions. Both use standardized numeric features and one-hot categorical features fitted on training data only; unknown categories are ignored. Hyperparameters are fixed in the source, with no test-driven tuning, resampling, or class weighting.</p>
<h3>A review budget rather than an arbitrary score of 0.5</h3><p>We allow at most {pct(summary['config']['review_budget'])} of validation transactions to be flagged. For each model, we choose the lowest threshold that satisfies that budget, including all score ties or none. The selected model is <b>{selected}</b>, based on validation AP. Its frozen threshold is <b>{threshold:.6f}</b>.</p>
<p>Scores rank suspicion. They have not been calibrated to real fraud probabilities. A flagged transaction is a candidate for human review. The same threshold is applied unchanged to test data, where alert volume can drift.</p>
{figure('review_tradeoff', tradeoff_caption)}</section>
<section id="evaluation"><h2>03 · Evaluation on later transactions</h2>
<p>The star identifies the validation-selected model. Both models are evaluated on test data after all decisions are frozen; a better test result does not trigger model reselection.</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Validation AP</th><th>Test AP</th><th>Test ROC AUC</th><th>Test precision</th><th>Test recall</th><th>Test alerts</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<p><b>Precision = detected fraud / all alerts.</b> Low precision creates review work and potential customer friction. <b>Recall = detected fraud / all fraud.</b> Low recall leaves more fraud undetected. AP summarizes the ranking; ROC AUC provides a complementary view.</p>
{figure('precision_recall', pr_caption)}{figure('roc', roc_caption)}{figure('confusion_matrix', confusion_caption)}
<h3>How uncertain are the test metrics?</h3><table><thead><tr><th>Selected model metric</th><th>Estimate</th><th>95% bootstrap interval</th></tr></thead><tbody>{ci_rows}</tbody></table>
<p class="muted">200 transaction-level bootstrap samples, fixed seed, fixed trained model and threshold. These percentile intervals describe sample variability, not uncertainty from retraining, account dependence, temporal correlation, or model selection. Real transaction clusters would require a grouped or time-block bootstrap.</p>
{figure('feature_importance', importance_caption)}</section>
<section id="conclusions"><h2>04 · Conclusions and limits</h2><p>{conclusion}</p><p>{budget_note}</p>
<ul><li><b>Learning:</b> model scores are useful only when paired with an explicit operating threshold and an honest holdout evaluation.</li>
<li><b>Review capacity:</b> a {pct(summary['config']['review_budget'])} validation budget is illustrative. Costs of missed fraud, false alerts, and review effort have not been measured, so the threshold is not economically optimized.</li>
<li><b>Generalization:</b> accounts can appear in multiple chronological splits, matching repeat-customer scoring. This does not test unseen-account performance.</li>
<li><b>Simulation:</b> a chosen risk formula simplifies adversarial behavior, delayed labels, missing data, and real customer patterns. Feature importance partly rediscovers that formula.</li>
<li><b>Next experiment:</b> use documented real data, features available at transaction time, mature fraud labels, rolling time validation, calibration, and explicit review costs. Keep a new holdout for later improvements.</li></ul></section>
<section id="reproduce"><h2>05 · Reproduce and inspect</h2><p>Use Python 3.12 and the committed dependency lock. Run from the repository root:</p>
<pre><code>python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m fraud_detection run --seed {summary['config']['seed']} --rows {summary['config']['rows']} --review-budget {summary['config']['review_budget']}
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m fraud_detection verify</code></pre>
<p>The verification command repeats the experiment in a temporary directory and checks the dataset hash, predictions, model metrics, bootstrap intervals, and feature importance. Single-thread model execution reduces numerical variation. Exact equality is checked within the current environment; different platforms may produce small differences.</p>
<p><b>Artifacts:</b> <code>data/transactions.csv</code>, <code>artifacts/model.joblib</code>, <code>artifacts/test_predictions.csv</code>, <code>artifacts/split_assignments.csv</code>, <code>reports/metrics.json</code>, and exportable PNG figures. This HTML embeds every figure and opens without a server; use your browser’s print function to export a PDF.</p>
<p class="hash"><b>Dataset SHA-256:</b> {summary['data_sha256']}</p><p class="muted">Runtime: Python {summary['versions']['python']} · NumPy {summary['versions']['numpy']} · pandas {summary['versions']['pandas']} · scikit-learn {summary['versions']['scikit-learn']}.</p></section></main>
<footer>Fraud detection study · Simulated transactions · Reproducible educational experiment</footer>
"""
    (root / "reports/report.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fraud detection · Experiment report</title><style>' + STYLE + '</style></head><body>' + body + '</body></html>')
    lines = ["# Fraud detection: experiment report", "", "**Simulated data · Educational experiment · No real-world performance claim**", "",
             "## Results", "", summary_text, "", f"Selected by validation average precision: **{selected}**.", "",
             "| Model | Validation AP | Test AP | Test ROC AUC | Test precision | Test recall | Test alerts |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in summary["models"].items():
        v, t = metrics["validation"], metrics["test"]
        lines.append(f"| {name} | {v['average_precision']:.3f} | {t['average_precision']:.3f} | {t['roc_auc']:.3f} | {pct(t['precision'])} | {pct(t['recall'])} | {pct(t['alert_rate'])} |")
    lines.extend(["", f"Always-legitimate baseline: {pct(summary['baseline']['no_alerts_accuracy'])} accuracy, zero recall. Random-ranking AP reference: {test.is_fraud.mean():.3f}.",
                  "", "## Workflow", "", "1. Generate transactions with the fixed seed and persist a dataset fingerprint.",
                  "2. Split in chronological order: 60% train, 20% validation, 20% test. IDs and timestamps are not predictors.",
                  "3. Fit numeric scaling, categorical encoding, logistic regression and boosted trees on training data only.",
                  f"4. Select the model using validation AP and thresholds under a {pct(summary['config']['review_budget'])} validation review budget.",
                  f"5. Freeze decisions and evaluate later transactions. Selected threshold: {threshold:.6f}.", "",
                  "The generator samples fraud from amount, device, foreign status, time, category, velocity and interactions plus hidden noise and mild time drift. Historical account means are idealized pre-period profiles. Prior-hour counts use only earlier transactions. Labels are never predictors.", "", "## Figures", ""])
    for name, caption in [("dataset", dataset_caption), ("precision_recall", pr_caption), ("roc", roc_caption),
                          ("confusion_matrix", confusion_caption), ("review_tradeoff", tradeoff_caption), ("feature_importance", importance_caption)]:
        lines.extend([f"![{name.replace('_', ' ').title()}](figures/{name}.png)", "", caption, ""])
    lines.extend(["## Test uncertainty", "", "| Metric | Estimate | 95% bootstrap interval |", "|---|---:|---:|"])
    for name, bounds in summary["test_bootstrap_95_percent"].items():
        lines.append(f"| {name.replace('_', ' ')} | {result[name]:.3f} | {bounds[0]:.3f}–{bounds[1]:.3f} |")
    lines.extend(["", "200 fixed-model transaction-level bootstrap samples. Intervals exclude retraining and model-selection uncertainty and do not account for repeated accounts or temporal dependence.",
                  "", "## Conclusions", "", conclusion, "", budget_note, "", "Scores are not calibrated real-world fraud probabilities. Review costs are unspecified; the review budget is illustrative. Accounts may repeat across periods, so unseen-account generalization is untested. Real deployment needs mature labels, causal features, rolling validation, calibration and cost analysis.",
                  "", "## Reproducibility", "", "See [README](../README.md) for exact commands and the self-contained [HTML report](report.html) for the full explanation.", "",
                  f"Configuration: `{summary['config']}`", "", f"Dataset SHA-256: `{summary['data_sha256']}`", ""])
    (root / "reports/report.md").write_text("\n".join(lines))
