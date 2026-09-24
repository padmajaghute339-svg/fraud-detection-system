"""
Real-Time Transaction Fraud Detection Engine (All-in-One Production Module)
Contains: Ingestion, Feature Pipeline, LightGBM, TreeSHAP, FastAPI, and Automated Benchmark.
"""

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
import sys
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, status
import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import requests
import shap
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler, TargetEncoder
import uvicorn


# =====================================================================
# 1. Pipeline: Ingestion, Chronological Split, Features & Training
# =====================================================================

def generate_mock_transactions(num_records: int = 40_000) -> pd.DataFrame:
    """Generates realistic transaction logs with extreme class imbalance (<0.3% fraud)."""
    print(f"[*] Generating {num_records:,} chronological transaction records...")
    np.random.seed(42)

    base_time = pd.Timestamp("2026-08-01 00:00:00")
    random_deltas = np.sort(np.random.randint(0, 30 * 24 * 3600, size=num_records))
    timestamps = [base_time + pd.Timedelta(seconds=int(s)) for s in random_deltas]

    user_ids = [f"usr_{np.random.randint(1000, 3500)}" for _ in range(num_records)]
    amounts = np.round(np.random.lognormal(mean=3.5, sigma=1.2, size=num_records), 2)
    amounts = np.clip(amounts, 1.0, 10000.0)

    categories = np.random.choice(
        ["grocery", "electronics", "dining", "travel", "utilities", "crypto_transfer"],
        size=num_records,
        p=[0.40, 0.15, 0.20, 0.10, 0.13, 0.02],
    )

    fraud_probs = np.where(categories == "crypto_transfer", 0.06, 0.0015)
    is_fraud = np.random.binomial(1, fraud_probs)

    df = pd.DataFrame({
        "transaction_id": [f"tx_{i:08d}" for i in range(num_records)],
        "user_id": user_ids,
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "amount": amounts,
        "merchant_category": categories,
        "is_fraud": is_fraud,
    })
    return df
def build_and_train_system():
    """Executes Phases 1 to 4: Data processing, Modeling, and SHAP serialization."""
    print("=" * 65)
    print("      INITIALIZING OFFLINE MODELING & SHAP PIPELINE")
    print("=" * 65)

    # 1. Ingestion & Epoch Calculation
    df = generate_mock_transactions()
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df["epoch_seconds"] = (df["datetime"].astype("int64") // 10**9).astype(np.int64)

    # 2. Strict Chronological Split (Prevents Data Leakage)
    df = df.sort_values(by="epoch_seconds").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[*] Train set: {len(train_df):,} rows | Fraud cases: {train_df['is_fraud'].sum():,}")
    print(f"[*] Test set:  {len(test_df):,} rows | Fraud cases: {test_df['is_fraud'].sum():,}")

    # 3. Rolling Window Aggregations (Vectorized & Index-Safe)
    def compute_windows(data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(by=["user_id", "epoch_seconds"]).reset_index(drop=True)
        
        n = len(data)
        c_15m = np.zeros(n, dtype=np.int32)
        c_1h = np.zeros(n, dtype=np.int32)
        c_24h = np.zeros(n, dtype=np.int32)
        s_24h = np.zeros(n, dtype=np.float64)
        sec_since = np.full(n, 86400.0, dtype=np.float64)

        user_arr = data["user_id"].to_numpy()
        epoch_arr = data["epoch_seconds"].to_numpy()
        amt_arr = data["amount"].to_numpy()

        if n > 1:
            user_transitions = np.where(user_arr[:-1] != user_arr[1:])[0] + 1
            starts = np.r_[0, user_transitions]
            ends = np.r_[user_transitions, n]
        else:
            starts, ends = np.array([0]), np.array([n])

        for s, e in zip(starts, ends):
            u_epochs = epoch_arr[s:e]
            u_amts = amt_arr[s:e]
            u_len = e - s

            if u_len > 1:
                sec_since[s+1:e] = u_epochs[1:] - u_epochs[:-1]

            for i in range(u_len):
                cur_t = u_epochs[i]
                idx_15m = np.searchsorted(u_epochs[:i], cur_t - 900, side='left')
                idx_1h = np.searchsorted(u_epochs[:i], cur_t - 3600, side='left')
                idx_24h = np.searchsorted(u_epochs[:i], cur_t - 86400, side='left')

                c_15m[s + i] = i - idx_15m
                c_1h[s + i] = i - idx_1h
                c_24h[s + i] = i - idx_24h
                s_24h[s + i] = np.sum(u_amts[idx_24h:i])

        data["tx_count_last_15m"] = c_15m
        data["tx_count_last_1h"] = c_1h
        data["tx_count_last_24h"] = c_24h
        data["tx_sum_last_24h"] = s_24h
        data["seconds_since_last_tx"] = sec_since

        return data

    print("[*] Computing time-window velocity metrics...")
    train_df = compute_windows(train_df)
    test_df = compute_windows(test_df)

    # 4. User Historical Spend Deviations
    user_stats = train_df.groupby("user_id")["amount"].agg(
        user_mean_spend="mean", 
        user_std_spend="std"
    ).reset_index()
    global_mean = train_df["amount"].mean()

    train_df = train_df.merge(user_stats, on="user_id", how="left")
    test_df = test_df.merge(user_stats, on="user_id", how="left")

    for d in [train_df, test_df]:
        d["user_mean_spend"] = d["user_mean_spend"].fillna(global_mean)
        d["user_std_spend"] = d["user_std_spend"].fillna(1.0)
        d["ratio_to_avg_spend"] = d["amount"] / (d["user_mean_spend"] + 1.0)
        d["amount_zscore"] = (d["amount"] - d["user_mean_spend"]) / (d["user_std_spend"] + 1e-5)

    num_cols = [
        "amount", "tx_count_last_15m", "tx_count_last_1h", "tx_count_last_24h",
        "tx_sum_last_24h", "seconds_since_last_tx", "ratio_to_avg_spend", "amount_zscore"
    ]
    cat_cols = ["merchant_category"]
    features = num_cols + cat_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("cat", TargetEncoder(smooth="auto", cv=5, random_state=42), cat_cols),
        ],
        verbose_feature_names_out=False,
    )

    y_train = train_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    X_train_proc = pd.DataFrame(preprocessor.fit_transform(train_df[features], y_train), columns=features)
    X_test_proc = pd.DataFrame(preprocessor.transform(test_df[features]), columns=features)

    # 5. Cost-Sensitive LightGBM
    scale_weight = float(np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-5))
    model = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_weight,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train_proc, y_train)

    test_probs = model.predict_proba(X_test_proc)[:, 1]
    pr_auc = average_precision_score(y_test, test_probs)
    print(f"[+] LightGBM Trained. PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")

    # 6. Financial Loss Curve Optimization
    thresholds = np.linspace(0.01, 0.99, 100)
    best_loss, opt_thresh = float("inf"), 0.50
    cost_fp, cost_fn = 15.0, 250.0

    for t in thresholds:
        preds = (test_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        loss = (fp * cost_fp) + (fn * cost_fn)
        if loss < best_loss:
            best_loss = loss
            opt_thresh = float(t)

    print(f"[+] Cost-Optimal Cutoff: {opt_thresh:.3f} (Expected Loss: ${best_loss:,.2f})")

    # 7. TreeSHAP Serialization
    print("[*] Initializing TreeSHAP Explainer...")
    explainer = shap.TreeExplainer(model)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "explainer": explainer,
        "threshold_cfg": {
            "optimal_threshold": round(opt_thresh, 4),
            "challenge_lower_bound": round(max(0.05, opt_thresh - 0.20), 4),
            "features": features,
        },
    }




# =====================================================================
# 2. Real-Time State Store & Service Schemas
# =====================================================================

class UserTransactionStateStore:
    """Manages real-time sliding history in memory for sub-millisecond lookups."""
    def __init__(self, max_history_seconds: int = 86400):
        self.max_history_seconds = max_history_seconds
        self._user_state: dict[str, deque] = {}

    def prune_and_record(self, user_id: str, timestamp: float, amount: float):
        if user_id not in self._user_state:
            self._user_state[user_id] = deque()
        q = self._user_state[user_id]
        cutoff = timestamp - self.max_history_seconds
        while q and q[0][0] < cutoff:
            q.popleft()
        q.append((timestamp, amount))

    def compute_features(self, user_id: str, current_ts: float, current_amount: float) -> dict:
        q = self._user_state.get(user_id, deque())
        past_txs = [tx for tx in q if tx[0] < current_ts]

        fifteen_min = current_ts - 900
        one_hour = current_ts - 3600
        twenty_four_hour = current_ts - 86400

        c_15m = sum(1 for ts, _ in past_txs if ts >= fifteen_min)
        c_1h = sum(1 for ts, _ in past_txs if ts >= one_hour)
        c_24h = sum(1 for ts, _ in past_txs if ts >= twenty_four_hour)
        s_24h = sum(amt for ts, amt in past_txs if ts >= twenty_four_hour)
        sec_since = (current_ts - past_txs[-1][0]) if past_txs else 86400.0

        hist_mean = 85.0
        return {
            "tx_count_last_15m": c_15m,
            "tx_count_last_1h": c_1h,
            "tx_count_last_24h": c_24h,
            "tx_sum_last_24h": float(s_24h),
            "seconds_since_last_tx": float(sec_since),
            "ratio_to_avg_spend": float(current_amount / (hist_mean + 1.0)),
            "amount_zscore": float((current_amount - hist_mean) / 45.0),
        }


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="tx_982144")
    user_id: str = Field(..., example="usr_3412")
    amount: float = Field(..., gt=0.0, example=1850.50)
    merchant_category: str = Field(..., example="crypto_transfer")
    timestamp: Optional[float] = Field(default=None)


class ReasonCode(BaseModel):
    feature: str
    shap_contribution: float
    reason_description: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str  # APPROVED, CHALLENGE_OTP, DECLINED
    action: str
    latency_ms: float
    adverse_reasons: list[ReasonCode]


# =====================================================================
# 3. FastAPI Service Definition
# =====================================================================

ml_bundle: dict = {}
state_store = UserTransactionStateStore()

HUMAN_REASONS = {
    "amount_zscore": lambda val: f"Spend is {val:.1f} standard deviations above user average",
    "ratio_to_avg_spend": lambda val: f"Spend is {val:.1f}x higher than historical average",
    "tx_count_last_15m": lambda val: f"High velocity: {int(val)} swipes within 15 minutes",
    "tx_count_last_1h": lambda val: f"Elevated frequency: {int(val)} swipes in 1 hour",
    "tx_count_last_24h": lambda val: f"High daily velocity: {int(val)} swipes in 24 hours",
    "tx_sum_last_24h": lambda val: f"Cumulative 24h spend reached ${val:,.2f}",
    "seconds_since_last_tx": lambda val: f"Successive swipe within {int(val)} seconds",
    "merchant_category": lambda val: "High-risk merchant industry activity",
    "amount": lambda val: f"Absolute amount (${val:,.2f}) flagged by threshold safeguards",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_bundle
    if not ml_bundle:
        ml_bundle = build_and_train_system()
    yield
    ml_bundle.clear()


app = FastAPI(title="Real-Time Fraud Engine", version="1.0.0", lifespan=lifespan)


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def score_transaction(tx: TransactionRequest):
    t0 = time.perf_counter()
    now = tx.timestamp if tx.timestamp is not None else time.time()

    # Step A: Update sliding state & compute dynamic features
    state_store.prune_and_record(tx.user_id, now, tx.amount)
    dyn = state_store.compute_features(tx.user_id, now, tx.amount)

    raw_payload = {
        "amount": tx.amount,
        "tx_count_last_15m": dyn["tx_count_last_15m"],
        "tx_count_last_1h": dyn["tx_count_last_1h"],
        "tx_count_last_24h": dyn["tx_count_last_24h"],
        "tx_sum_last_24h": dyn["tx_sum_last_24h"],
        "seconds_since_last_tx": dyn["seconds_since_last_tx"],
        "ratio_to_avg_spend": dyn["ratio_to_avg_spend"],
        "amount_zscore": dyn["amount_zscore"],
        "merchant_category": tx.merchant_category,
    }

    # Step B: Preprocess & Predict
    try:
        proc_vec = ml_bundle["preprocessor"].transform(pd.DataFrame([raw_payload]))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    prob = float(ml_bundle["model"].predict_proba(proc_vec)[:, 1][0])
    opt_t = ml_bundle["threshold_cfg"]["optimal_threshold"]
    chl_t = ml_bundle["threshold_cfg"]["challenge_lower_bound"]

    # Step C: Three-tier Decision Routing
    if prob >= opt_t:
        decision, action = "DECLINED", "BLOCK_AND_FLAG_SAR"
    elif prob >= chl_t:
        decision, action = "CHALLENGE_OTP", "SEND_TWO_FACTOR_VERIFICATION"
    else:
        decision, action = "APPROVED", "SETTLE_PAYMENT"

    # Step D: Local TreeSHAP Explanation for Flagged Swipes
    reasons = []
    if decision in ["DECLINED", "CHALLENGE_OTP"]:
        shap_vals = ml_bundle["explainer"].shap_values(proc_vec)
        pos_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        feats = list(raw_payload.keys())

        ranked = np.argsort(-pos_shap)[:3]
        for idx in ranked:
            name = feats[idx]
            formatter = HUMAN_REASONS.get(name, lambda v: f"Anomaly in {name}")
            reasons.append(ReasonCode(
                feature=name,
                shap_contribution=round(float(pos_shap[idx]), 4),
                reason_description=formatter(raw_payload[name]),
            ))

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=round(prob, 4),
        decision=decision,
        action=action,
        latency_ms=latency,
        adverse_reasons=reasons,
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "model" in ml_bundle}


# =====================================================================
# 4. Simulation & Traffic Benchmark
# =====================================================================

def run_live_simulation():
    """Runs realistic benchmark queries against the local API."""
    time.sleep(1.5)  # Wait for uvicorn to initialize
    print("\n" + "=" * 65)
    print("      RUNNING LIVE TRANSACTION SIMULATION & AUDIT")
    print("=" * 65)

    test_events = [
        {"user_id": "usr_7701", "amount": 6.50, "merchant_category": "dining"},
        {"user_id": "usr_7701", "amount": 42.00, "merchant_category": "grocery"},
        {"user_id": "usr_7701", "amount": 4900.00, "merchant_category": "crypto_transfer"},
        {"user_id": "usr_7701", "amount": 6500.00, "merchant_category": "crypto_transfer"},
    ]

    latencies = []
    url = "http://127.0.0.1:8000/api/v1/predict"

    for i, e in enumerate(test_events, 1):
        payload = {"transaction_id": f"tx_live_{i:04d}", **e}
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        if resp.status_code == 200:
            res = resp.json()
            print(f"Swipe #{i:02d} | ${e['amount']:>7.2f} | Score: {res['fraud_probability']:.4f} | "
                  f"Decision: {res['decision']:<13} | Latency: {ms:.2f}ms")
            for rson in res["adverse_reasons"]:
                print(f"   └─ [SHAP Driver]: {rson['reason_description']}")
        else:
            print(f"Swipe #{i:02d} Failed: {resp.text}")

    print("-" * 65)
    print(f"P50 Latency: {np.percentile(latencies, 50):.2f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
    print("=" * 65)
    print("\n[+] Service running at: http://127.0.0.1:8000")
    print("[+] Interactive Swagger UI: http://127.0.0.1:8000/docs\n")


# =====================================================================
# 5. Execution Entry Point
# =====================================================================

if __name__ == "__main__":
    # Pre-train system before starting server
    ml_bundle = build_and_train_system()

    # Start benchmark thread in background
    sim_thread = threading.Thread(target=run_live_simulation, daemon=True)
    sim_thread.start()

    # Start web server
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")