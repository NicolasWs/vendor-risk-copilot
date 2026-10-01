"""
Vendor approval scoring pipeline — mirrors the Dataiku DSS flow:
Split recipe -> Prepare -> Train (Quick Prototypes) -> Score -> Evaluate.

Produces:
  - trained model
  - validation metrics (accuracy, AUC, precision/recall)
  - feature importances (for the explainability agent downstream)
  - scored_mandates.csv (predictions + probabilities, consumed by the agent)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/vendor_mandates.csv"
TARGET = "approved"
CAT_COLS = ["vendor", "asset_class", "region"]
NUM_COLS = [
    "cost_bps", "latency_ms", "fill_rate", "coverage_pct",
    "compliance_flags_12m", "data_quality_score", "relationship_years",
    "sla_breach_rate",
]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        random_state=1337, class_weight="balanced",
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


def feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_COLS))
    all_names = NUM_COLS + cat_names
    importances = pipeline.named_steps["clf"].feature_importances_
    return (
        pd.DataFrame({"feature": all_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1337, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("Validation metrics:", json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred))

    fi = feature_importances(pipe)
    fi.to_csv("data/feature_importances.csv", index=False)
    print("\nTop drivers of vendor approval:")
    print(fi.head(8).to_string(index=False))

    # Score the full dataset for downstream agent consumption
    df["pred_approved"] = pipe.predict(X)
    df["approval_probability"] = pipe.predict_proba(X)[:, 1]
    df.to_csv("data/scored_mandates.csv", index=False)

    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, "data/model.joblib")
    print(f"\nWrote data/scored_mandates.csv ({len(df)} rows), data/feature_importances.csv, data/model.joblib")


if __name__ == "__main__":
    main()
