"""
Train Bagging, Boosting, Stacking, and Hybrid (RF + MLP) models
on Sri Lanka telecom churn data. Saves pickles + metrics for the dashboard.

AI-assisted code (Claude / Cursor). Declare this in submissions.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sri_lanka_telecom_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churned"
ID_COL = "customer_id"
CAT_COLS = [
    "operator",
    "district",
    "plan_type",
    "device_type",
    "payment_method",
]
NUM_COLS = [
    "age",
    "tenure_months",
    "monthly_bill_lkr",
    "avg_topup_lkr",
    "data_usage_gb",
    "voice_minutes",
    "sms_count",
    "support_calls_3m",
    "network_drop_events",
    "contract_months",
    "has_family_bundle",
    "has_ott_addon",
    "days_since_last_topup",
    "received_competitor_offer",
    "is_rural_district",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )


def metrics_dict(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        from generate_sl_telecom_churn import main as gen_main

        gen_main()

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Bagging family ---
    bagging = make_pipeline(
        BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, random_state=42),
            n_estimators=80,
            max_samples=0.8,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
    )
    random_forest = make_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    )

    # --- Boosting family ---
    ada_boost = make_pipeline(
        AdaBoostClassifier(
            n_estimators=120,
            learning_rate=0.6,
            random_state=42,
        )
    )
    grad_boost = make_pipeline(
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        )
    )

    try:
        from xgboost import XGBClassifier

        xgb = make_pipeline(
            XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        )
        has_xgb = True
    except Exception:
        xgb = None
        has_xgb = False

    try:
        from lightgbm import LGBMClassifier

        lgbm = make_pipeline(
            LGBMClassifier(
                n_estimators=180,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        )
        has_lgbm = True
    except Exception:
        lgbm = None
        has_lgbm = False

    # --- Stacking ---
    stacking_estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)),
        ("ada", AdaBoostClassifier(n_estimators=60, random_state=42)),
    ]
    stacking = make_pipeline(
        StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            stack_method="predict_proba",
            n_jobs=-1,
        )
    )

    # --- Hybrid: RF + MLP (ANN) soft voting ---
    hybrid_rf_ann = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                VotingClassifier(
                    estimators=[
                        (
                            "rf",
                            RandomForestClassifier(
                                n_estimators=150,
                                max_depth=12,
                                class_weight="balanced_subsample",
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                        (
                            "ann",
                            MLPClassifier(
                                hidden_layer_sizes=(64, 32),
                                activation="relu",
                                max_iter=400,
                                early_stopping=True,
                                random_state=42,
                            ),
                        ),
                    ],
                    voting="soft",
                    weights=[1.2, 1.0],
                ),
            ),
        ]
    )

    models = {
        "Bagging (Trees)": bagging,
        "Random Forest (Bagging)": random_forest,
        "AdaBoost (Boosting)": ada_boost,
        "Gradient Boosting": grad_boost,
        "Stacking (RF+GB+Ada->LR)": stacking,
        "Hybrid RF+ANN Voting": hybrid_rf_ann,
    }
    if has_xgb:
        models["XGBoost (Boosting)"] = xgb
    if has_lgbm:
        models["LightGBM (Boosting)"] = lgbm

    results = {}
    feature_importance = {}

    print("Training models...")
    for name, pipe in models.items():
        print(f"  -> {name}")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = metrics_dict(y_test, pred, proba)
        # 5-fold CV on train for stability story
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        m["cv_roc_auc_mean"] = float(cv.mean())
        m["cv_roc_auc_std"] = float(cv.std())
        results[name] = m

        # Persist model
        safe = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "_")
            .replace("->", "_to_")
        )
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
        joblib.dump(pipe, MODEL_DIR / f"{safe}.joblib")

        # Feature importance where available (RF)
        if name == "Random Forest (Bagging)":
            pre = pipe.named_steps["preprocess"]
            clf = pipe.named_steps["model"]
            cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
            feat_names = cat_names + NUM_COLS
            importances = clf.feature_importances_
            order = np.argsort(importances)[::-1][:15]
            feature_importance = {
                feat_names[i]: float(importances[i]) for i in order
            }

    # Soft-voting ensemble of best bagging + boosting + stacking for dashboard demo
    soft_vote = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                VotingClassifier(
                    estimators=[
                        (
                            "bag",
                            RandomForestClassifier(
                                n_estimators=120, max_depth=10, random_state=42, n_jobs=-1
                            ),
                        ),
                        (
                            "boost",
                            GradientBoostingClassifier(
                                n_estimators=100, max_depth=3, random_state=42
                            ),
                        ),
                        (
                            "meta",
                            LogisticRegression(max_iter=800, class_weight="balanced"),
                        ),
                    ],
                    voting="soft",
                ),
            ),
        ]
    )
    soft_vote.fit(X_train, y_train)
    proba = soft_vote.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results["Soft Voting (Bag+Boost+LR)"] = metrics_dict(y_test, pred, proba)
    joblib.dump(soft_vote, MODEL_DIR / "soft_voting_bag_boost_lr.joblib")

    # Save artifacts for dashboard / notebook
    joblib.dump(
        {
            "feature_columns": CAT_COLS + NUM_COLS,
            "categorical": CAT_COLS,
            "numeric": NUM_COLS,
            "operators": sorted(df["operator"].unique().tolist()),
            "districts": sorted(df["district"].unique().tolist()),
            "plan_types": sorted(df["plan_type"].unique().tolist()),
            "device_types": sorted(df["device_type"].unique().tolist()),
            "payment_methods": sorted(df["payment_method"].unique().tolist()),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "churn_rate": float(y.mean()),
        },
        MODEL_DIR / "meta.joblib",
    )

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(MODEL_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    # Rank by ROC-AUC for interpretation narrative
    ranked = sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
    print("\n=== Model ranking by ROC-AUC (test) ===")
    for i, (name, m) in enumerate(ranked, 1):
        print(
            f"{i:2d}. {name:32s}  AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}"
        )

    print(f"\nSaved models -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
