"""Общие функции метрик и подбора порога (задания 8, 11, 13)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Метрики для положительного класса (1 = мошенничество / фишинг) и macro-F1."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = 0 < y_true.sum() < len(y_true)
    return {
        "threshold": round(float(threshold), 4),
        "n": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f2": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if both else None,
        "pr_auc": round(average_precision_score(y_true, proba), 4) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_precision(y_true, proba, min_precision: float = 0.85) -> float:
    """Минимальный порог, при котором Precision ≥ min_precision (а значит, Recall максимален).

    Перебираются все различные значения вероятности. Если условие недостижимо,
    возвращается порог с наибольшей Precision.
    """
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true), np.asarray(proba))
    precision, recall = precision[:-1], recall[:-1]          # последняя точка без порога
    ok = np.where(precision >= min_precision)[0]
    if len(ok) == 0:
        return float(thresholds[int(np.argmax(precision))])
    best = ok[np.argmax(recall[ok])]                          # максимальный Recall среди допустимых
    # при равном Recall — минимальный порог
    same = ok[recall[ok] == recall[best]]
    return float(thresholds[same].min())
