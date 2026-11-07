"""Обучение baseline-модели SMS без ноутбука (задание 7).

Запуск из корня проекта:
    python -m ml.train_baseline           # только английский набор UCI → sms_baseline.joblib
    python -m ml.train_baseline --multi   # UCI + синтетический ru/kk → sms_baseline_multi.joblib

Повторяет шаги notebooks/03_baseline_sms.ipynb: Logistic Regression
(class_weight='balanced', max_iter=1000) в Pipeline с векторизатором задания 6,
обучение на train, оценка на validation, сохранение в ml/artifacts/.
"""
from __future__ import annotations

import argparse
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
                         build_vectorizer, load_split, load_synthetic, save_split, split_dataset)
from ml.predictor import (ARTIFACTS_DIR, BASELINE_META_PATH, BASELINE_PATH, MULTI_META_PATH,
                          MULTI_PATH)

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


def load_training_data(multi: bool) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Возвращает обучающую выборку и словарь валидационных выборок по языкам."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sms_clean.csv", keep_default_na=False)
    if not SPLIT_PATH.exists():
        save_split(split_dataset(df), df)
    parts = load_split(df)
    train, vals = parts["train"].assign(lang="en"), {"en": parts["val"]}
    if multi:
        synth = load_synthetic()
        cols = FEATURE_COLUMNS + ["label", "lang"]
        train = pd.concat([train[cols], synth["train"][cols]], ignore_index=True)
        for lang in ("ru", "kk"):
            vals[lang] = synth["val"][synth["val"]["lang"] == lang]
    return train, vals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", action="store_true",
                        help="обучить на UCI + синтетическом ru/kk-наборе (sms_baseline_multi.joblib)")
    multi = parser.parse_args(argv).multi
    model_path, meta_path = (MULTI_PATH, MULTI_META_PATH) if multi else (BASELINE_PATH, BASELINE_META_PATH)

    train, vals = load_training_data(multi)
    val = vals["en"]

    model = build_baseline()
    start = time.perf_counter()
    model.fit(train[FEATURE_COLUMNS], train["label"])
    train_seconds = time.perf_counter() - start

    metrics_by_lang = {
        lang: evaluate(part["label"].to_numpy(), model.predict_proba(part[FEATURE_COLUMNS])[:, 1])
        for lang, part in vals.items()
    }
    metrics = metrics_by_lang["en"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        "model": "sms_baseline_multi" if multi else "sms_baseline",
        "version": "0.1.0",
        "algorithm": "LogisticRegression(class_weight='balanced', max_iter=1000)",
        "features": "TF-IDF words (1,2) + TF-IDF char_wb (3,5) + manual: " + ", ".join(MANUAL_FEATURES),
        "n_features": int(len(model.named_steps["features"].get_feature_names_out())),
        "threshold": THRESHOLD,
        "languages": sorted(vals),
        "trained_on": (f"train ({len(train)} SMS: " +
                       ", ".join(f"{k} {v}" for k, v in train["lang"].value_counts().sort_index().items()) +
                       "), data/processed/sms_split.json" +
                       (" + data/processed/synthetic_split.json" if multi else "")),
        "metrics_split": "validation: " + ", ".join(f"{k} {len(v)}" for k, v in vals.items()),
        "metrics": metrics,
        "metrics_by_lang": metrics_by_lang,
        "train_seconds": round(train_seconds, 2),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for lang, m in metrics_by_lang.items():
        print(f"[{lang}] validation: precision={m['precision_fraud']}, recall={m['recall_fraud']}, "
              f"f1={m['f1_fraud']}, roc_auc={m['roc_auc']}, cm={m['confusion_matrix']}")
    print(f"Сохранено: {model_path.relative_to(PROJECT_ROOT)}, {meta_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
