"""Обучение baseline-модели SMS без ноутбука (задание 7).

Запуск из корня проекта:
    python -m ml.train_baseline

Повторяет шаги notebooks/03_baseline_sms.ipynb: Logistic Regression
(class_weight='balanced', max_iter=1000) в Pipeline с векторизатором задания 6,
обучение на train, оценка на validation, сохранение в ml/artifacts/.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from ml.features import (MANUAL_FEATURES, PROJECT_ROOT, RANDOM_STATE, SPLIT_PATH,
                         build_vectorizer, load_split, save_split, split_dataset)
from ml.predictor import ARTIFACTS_DIR, BASELINE_META_PATH, BASELINE_PATH

FEATURE_COLUMNS = ["text_raw", "text_clean"]
THRESHOLD = 0.5


def build_baseline() -> Pipeline:
    return Pipeline([
        ("features", build_vectorizer()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=RANDOM_STATE)),
    ])


def evaluate(y_true, proba, threshold: float = THRESHOLD) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fraud": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_fraud": round(recall_score(y_true, y_pred), 4),
        "f1_fraud": round(f1_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sms_clean.csv", keep_default_na=False)
    if not SPLIT_PATH.exists():
        save_split(split_dataset(df), df)
    parts = load_split(df)
    train, val = parts["train"], parts["val"]

    model = build_baseline()
    start = time.perf_counter()
    model.fit(train[FEATURE_COLUMNS], train["label"])
    train_seconds = time.perf_counter() - start

    proba = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    metrics = evaluate(val["label"].to_numpy(), proba)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_PATH)
    meta = {
        "model": "sms_baseline",
        "version": "0.1.0",
        "algorithm": "LogisticRegression(class_weight='balanced', max_iter=1000)",
        "features": "TF-IDF words (1,2) + TF-IDF char_wb (3,5) + manual: " + ", ".join(MANUAL_FEATURES),
        "n_features": int(len(model.named_steps["features"].get_feature_names_out())),
        "threshold": THRESHOLD,
        "trained_on": f"train ({len(train)} SMS), data/processed/sms_split.json",
        "metrics_split": f"validation ({len(val)} SMS)",
        "metrics": metrics,
        "train_seconds": round(train_seconds, 2),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    BASELINE_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Сохранено: {BASELINE_PATH.relative_to(PROJECT_ROOT)}, {BASELINE_META_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
