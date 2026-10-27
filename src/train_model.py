#!/usr/bin/env python3
"""
Train the cancellation model on the synthetic dataset and save it for the app.

Reproduces the final configuration of the project: LightGBM with native
categorical features, `is_unbalance=True` and the hyperparameters found with
Optuna on the real data. Only the training data changes (synthetic), so the
metrics written to models/metrics.json describe the synthetic data, not the
results reported in the README.

Usage:
    python src/generate_synthetic_data.py   # if data/ is empty
    python src/train_model.py
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET = "CANCELADO"
FEATURES = [
    "FUENTE_NEGOCIO_SEGMENTO_CLIENTE", "PAIS_TOP_200", "LEAD_TIME", "LOYALTY_REWARDS", "COMPLEJO_RESERVA", "ADR",
    "HORA_TOMA_COS", "MES_LLEGADA_SIN", "TIPO_VIAJERO", "REV_PAX", "ES_GRUPO", "TIENE_FIDELIDAD", "MES_LLEGADA_COS",
    "HOTEL_HABITACION_TOP", "NOCHES", "ADULTOS", "HORA_TOMA_SIN",
]
CATEGORICAL = ["FUENTE_NEGOCIO_SEGMENTO_CLIENTE", "PAIS_TOP_200", "COMPLEJO_RESERVA", "TIPO_VIAJERO",
               "HOTEL_HABITACION_TOP"]

# Best hyperparameters found with Optuna on the real dataset
PARAMS = {
    "learning_rate": 0.0875, "num_leaves": 141, "max_depth": 12, "min_child_samples": 18,
    "subsample": 0.739, "colsample_bytree": 0.901, "reg_alpha": 2.4e-07, "reg_lambda": 2.8e-04,
    "objective": "binary", "is_unbalance": True, "n_estimators": 100, "random_state": 42, "verbose": -1,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "df_unificado_completo_v8.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc_roc": roc_auc_score(y_test, proba),
        "average_precision_score": average_precision_score(y_test, proba),
        "recall_score": recall_score(y_test, pred),
        "precision_score": precision_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
    }

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "lightgbm.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "trained_on": "synthetic data (src/generate_synthetic_data.py)",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "model_type": "LightGBM",
            "n_train": len(X_train), "n_test": len(X_test),
            "parameters": PARAMS,
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "features": FEATURES,
            "categorical_features": CATEGORICAL,
        }, f, indent=2)

    print("Saved models/lightgbm.joblib")
    for k, v in metrics.items():
        print(f"  {k:24s} {v:.4f}")


if __name__ == "__main__":
    main()
