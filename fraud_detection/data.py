"""Synthetic transactions. Labels are sampled, never copied into predictors."""
from collections import deque

import numpy as np
import pandas as pd

NUMERIC = ["amount", "amount_to_historical_mean", "hour", "account_age_days",
           "transactions_last_hour", "distance_from_home_km", "new_device", "foreign_transaction"]
CATEGORICAL = ["merchant_category", "channel"]
FEATURES = NUMERIC + CATEGORICAL


def prior_hour_counts(accounts, timestamps):
    """Count only strictly earlier transactions in the previous hour."""
    histories = {}
    counts = []
    for account, timestamp in zip(accounts, timestamps):
        history = histories.setdefault(account, deque())
        while history and history[0] < timestamp - 3600:
            history.popleft()
        counts.append(sum(t < timestamp for t in history))
        history.append(timestamp)
    return np.asarray(counts)


def generate_transactions(n=40000, seed=42):
    if n < 5000:
        raise ValueError("Use at least 5,000 transactions for meaningful chronological splits.")
    rng = np.random.default_rng(seed)
    accounts = rng.integers(0, 1200, n)
    seconds = rng.uniform(0, 120 * 86400, n)
    # Some accounts transact in bursts. These features exist before labels.
    burst = rng.random(n) < 0.15
    centers = rng.uniform(0, 120 * 86400 - 7200, 60)
    burst_accounts = rng.integers(0, 1200, 60)
    groups = rng.integers(0, 60, burst.sum())
    seconds[burst] = centers[groups] + rng.uniform(0, 7200, burst.sum())
    accounts[burst] = burst_accounts[groups]
    order = np.argsort(seconds, kind="stable")
    seconds, accounts = seconds[order], accounts[order]
    historical_means = rng.lognormal(3.6, 0.65, 1200)
    initial_age = rng.integers(5, 2000, 1200)
    ratio = rng.lognormal(-0.35, 1.05, n)
    amount = np.round(np.maximum(0.5, historical_means[accounts] * ratio), 2)
    ratio = amount / historical_means[accounts]
    hour = (seconds // 3600 % 24).astype(int)
    new_device = (rng.random(n) < 0.09).astype(int)
    foreign = (rng.random(n) < 0.12).astype(int)
    distance = rng.exponential(18, n) + foreign * rng.uniform(100, 4000, n)
    category = rng.choice(["groceries", "travel", "electronics", "services", "cash"], n,
                          p=[0.35, 0.12, 0.18, 0.25, 0.10])
    channel = rng.choice(["online", "in_store", "atm"], n, p=[0.55, 0.35, 0.10])
    velocity = prior_hour_counts(accounts, seconds)
    # Fraud has overlapping patterns with legitimate transactions. Unobserved
    # factors and a sampled outcome deliberately make prediction imperfect.
    log_odds = (-6.5 + 1.1 * np.clip(np.log(ratio), -2, 4)
                + 1.45 * new_device + 1.05 * foreign + 0.65 * (hour < 6)
                + 0.65 * np.isin(category, ["electronics", "cash"])
                + 0.14 * np.minimum(velocity, 12)
                + 1.25 * new_device * foreign + 0.35 * (channel == "online")
                + 0.35 * seconds / (120 * 86400) + rng.normal(0, 0.65, n))
    probability = 1 / (1 + np.exp(-log_odds))
    fraud = (rng.random(n) < probability).astype(int)
    return pd.DataFrame({
        "transaction_id": [f"TX{i:07d}" for i in range(n)],
        "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(seconds, unit="s"),
        "account_id": accounts,
        "amount": amount,
        "amount_to_historical_mean": ratio,
        "hour": hour,
        "account_age_days": initial_age[accounts] + seconds / 86400,
        "transactions_last_hour": velocity,
        "distance_from_home_km": np.round(distance, 2),
        "new_device": new_device,
        "foreign_transaction": foreign,
        "merchant_category": category,
        "channel": channel,
        "is_fraud": fraud,
    })


def chronological_split(frame):
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    a, b = int(len(ordered) * 0.6), int(len(ordered) * 0.8)
    parts = {"train": ordered.iloc[:a].copy(), "validation": ordered.iloc[a:b].copy(),
             "test": ordered.iloc[b:].copy()}
    if not (parts["train"].timestamp.max() < parts["validation"].timestamp.min()
            and parts["validation"].timestamp.max() < parts["test"].timestamp.min()):
        raise ValueError("Chronological boundaries must have distinct timestamps.")
    for name, part in parts.items():
        if part.is_fraud.nunique() != 2:
            raise ValueError(f"{name} requires both classes; increase dataset size.")
    return parts
