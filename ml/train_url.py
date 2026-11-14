"""Обучение и сравнение моделей URL (задание 11).

Запуск из корня проекта (после python -m ml.data.prepare_urls), около 5 минут:
    python -m ml.train_url

1. Признаки: 17 числовых признаков домена (ml.url_features.MODEL_FEATURES) + символьные
   2–5-граммы канонического домена (TF-IDF). Версия только с числовыми признаками дала на
   validation ROC-AUC 0,81, n-граммы домена подняли его до ≈ 0,90 (см. docs/11_url_models.md).
2. Алгоритмы: Logistic Regression (StandardScaler + TF-IDF в Pipeline), Random Forest
   (TF-IDF ограничен 3000 n-граммами — RF на разреженной матрице обучается медленно).
   Подбор параметров — GridSearchCV на train, 3 фолда StratifiedGroupKFold по
   регистрируемому домену, метрика F2.
3. Пороги на validation:
   t1 (граница SAFE / SUSPICIOUS)     — максимальный порог, при котором Recall ≥ 0,85
                                        (минимальное требование раздела 3.4 к URL-модели);
   t2 (граница SUSPICIOUS / DANGEROUS) — минимальный порог, при котором Precision ≥ 0,90:
                                        категория DANGEROUS верна не менее чем в 90% случаев.
   Первая версия порогов (Recall ≥ 0,95 / Precision ≥ 0,95) относила к SUSPICIOUS 60% безопасных
   доменов validation, поэтому t1 был пересмотрен (см. docs/11_url_models.md).
4. Выбор модели — максимум F2 на validation (порог t1) при Precision ≥ 0,85; при равенстве
   (разница < 0,005) — более быстрая и интерпретируемая.
5. Однократная оценка на test, сохранение ml/artifacts/url_model.joblib + url_model_meta.json.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.prepare_urls import DATASET_PATH, load_url_features
from ml.metrics import compute_metrics
from ml.url_features import MODEL_FEATURES, MODEL_TEXT_COLUMN, model_input

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS / "url_model.joblib"
META_PATH = ARTIFACTS / "url_model_meta.json"
COMPARISON_PATH = ARTIFACTS / "url_comparison.json"
RANDOM_STATE = 42
VERSION = "1.0.0"
T1_RECALL = 0.85
T2_PRECISION = 0.90
MIN_PRECISION = 0.85
TIE_MARGIN = 0.005
PREFERENCE = ["logreg", "random_forest"]
TITLES = {"logreg": "Logistic Regression", "random_forest": "Random Forest"}
X_COLS = MODEL_FEATURES + [MODEL_TEXT_COLUMN]
CV_FOLDS = 3


def _features(scale: bool, max_ngrams: int | None = None) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", MODEL_FEATURES),
        ("host", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=max_ngrams), MODEL_TEXT_COLUMN),
    ])


def algorithms() -> dict:
    return {
        "logreg": (Pipeline([("features", _features(scale=True)),
                             ("clf", LogisticRegression(class_weight="balanced", max_iter=3000,
                                                        random_state=RANDOM_STATE))]),
                   {"clf__C": [0.3, 1, 3, 10]}),
        "random_forest": (Pipeline([("features", _features(scale=False, max_ngrams=3000)),
                                    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                                   n_jobs=-1, random_state=RANDOM_STATE))]),
                          {"clf__min_samples_leaf": [2, 5]}),
    }


def load_data() -> dict[str, pd.DataFrame]:
    feats = load_url_features()
    meta = pd.read_csv(DATASET_PATH)[["id", "url", "host", "registered_domain"]]
    df = feats.merge(meta, on="id", how="left")
    return {part: df[df["split"] == part].reset_index(drop=True) for part in ("train", "val", "test")}


def thresholds(y, proba) -> tuple[float, float]:
    """t1: максимальный порог с Recall ≥ T1_RECALL; t2: минимальный порог с Precision ≥ T2_PRECISION."""
    precision, recall, thr = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok1 = np.where(recall >= T1_RECALL)[0]
    t1 = float(thr[ok1].max()) if len(ok1) else float(thr.min())
    ok2 = np.where(precision >= T2_PRECISION)[0]
    t2 = float(thr[ok2].min()) if len(ok2) else float(thr.max())
    if t2 <= t1:                       # категории должны идти по возрастанию
        t2 = min(1.0, t1 + 0.05)
    return round(t1, 4), round(t2, 4)


def category(p: float, t1: float, t2: float) -> str:
    return "DANGEROUS" if p >= t2 else "SUSPICIOUS" if p >= t1 else "SAFE"


def predict_ms(model, frame: pd.DataFrame, n: int = 200) -> float:
    sample = frame[X_COLS].head(n)
    start = time.perf_counter()
    for i in range(len(sample)):
        model.predict_proba(sample.iloc[[i]])
    return (time.perf_counter() - start) / len(sample) * 1000


def run_comparison(data: dict, verbose: bool = True) -> tuple[list[dict], dict]:
    train, val = data["train"], data["val"]
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, models = [], {}
    for name, (pipe, grid) in algorithms().items():
        start = time.perf_counter()
        search = GridSearchCV(pipe, grid, scoring=make_scorer(fbeta_score, beta=2), cv=cv, n_jobs=1)
        search.fit(train[X_COLS], train["label"], groups=train["registered_domain"])
        fit_seconds = time.perf_counter() - start
        model = search.best_estimator_
        proba = model.predict_proba(val[X_COLS])[:, 1]
        t1, t2 = thresholds(val["label"], proba)
        res = {
            "name": name, "title": TITLES[name],
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f2": round(float(search.best_score_), 4),
            "t1": t1, "t2": t2,
            "val_at_0_5": compute_metrics(val["label"], proba, 0.5),
            "val_at_t1": compute_metrics(val["label"], proba, t1),
            "val_at_t2": compute_metrics(val["label"], proba, t2),
            "fit_seconds": round(fit_seconds, 1),
            "predict_ms": round(predict_ms(model, val), 2),
        }
        results.append(res)
        models[name] = model
        if verbose:
            m = res["val_at_t1"]
            print(f"{name:14s} cv_F2={res['cv_f2']:.4f} AUC={m['roc_auc']} t1={t1} t2={t2} "
                  f"R@t1={m['recall']} P@t1={m['precision']} F2@t1={m['f2']} {res['best_params']} ({res['fit_seconds']} c)")
    return results, models


def choose_best(results: list[dict]) -> dict:
    ok = [r for r in results if r["val_at_t1"]["precision"] >= MIN_PRECISION] or results
    top = max(r["val_at_t1"]["f2"] for r in ok)
    near = [r for r in ok if top - r["val_at_t1"]["f2"] < TIE_MARGIN]
    return min(near, key=lambda r: PREFERENCE.index(r["name"]))


def feature_importance(model) -> pd.Series:
    """Коэффициенты LR (по стандартизованным признакам) или feature_importances_ RF, по всем признакам."""
    names = [n.split("__", 1)[1] for n in model.named_steps["features"].get_feature_names_out()]
    names = [f"n-грамма «{n}»" if n not in MODEL_FEATURES else n for n in names]
    clf = model.named_steps["clf"]
    values = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    return pd.Series(values, index=names)


def evaluate_test(model, best: dict, test: pd.DataFrame) -> dict:
    proba = model.predict_proba(test[X_COLS])[:, 1]
    cats = pd.Series([category(p, best["t1"], best["t2"]) for p in proba])
    return {
        "at_t1": compute_metrics(test["label"], proba, best["t1"]),
        "at_t2": compute_metrics(test["label"], proba, best["t2"]),
        "at_0_5": compute_metrics(test["label"], proba, 0.5),
        "categories": {cat: {"safe": int(((cats == cat) & (test["label"] == 0)).sum()),
                             "phishing": int(((cats == cat) & (test["label"] == 1)).sum())}
                       for cat in ("SAFE", "SUSPICIOUS", "DANGEROUS")},
    }


def save(model, best: dict, test_metrics: dict, n_train: int) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "model": "url_model", "version": VERSION, "algorithm": best["title"], "params": best["best_params"],
        "features": MODEL_FEATURES + ["char 2-5-граммы домена (TF-IDF)"],
        "input": "ml.url_features.model_input(url) — 17 признаков канонического домена + сам домен",
        "thresholds": {"t1": best["t1"], "t2": best["t2"]},
        "threshold": best["t1"],
        "categories": {"SAFE": f"P < {best['t1']}", "SUSPICIOUS": f"{best['t1']} ≤ P < {best['t2']}",
                       "DANGEROUS": f"P ≥ {best['t2']}"},
        "threshold_rules": {"t1": f"максимальный порог на validation с Recall ≥ {T1_RECALL}",
                            "t2": f"минимальный порог на validation с Precision ≥ {T2_PRECISION}"},
        "trained_on": f"train ({n_train} URL, PhiUSIIL, разбиение по домену)",
        "metrics_validation": best["val_at_t1"], "metrics_test": test_metrics,
        "predict_ms_per_url": best["predict_ms"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def predict_url_proba(model, urls: list[str]) -> np.ndarray:
    frame = pd.DataFrame([model_input(u) for u in urls])[X_COLS]
    return model.predict_proba(frame)[:, 1]


def main() -> int:
    data = load_data()
    print({k: len(v) for k, v in data.items()})
    results, models = run_comparison(data)
    best = choose_best(results)
    model = models[best["name"]]
    test_metrics = evaluate_test(model, best, data["test"])
    meta = save(model, best, test_metrics, len(data["train"]))
    COMPARISON_PATH.write_text(json.dumps({"algorithms": results, "best": best["name"]}, ensure_ascii=False,
                                          indent=1), encoding="utf-8")
    print(f"Выбрана: {best['title']} {best['best_params']}, t1={best['t1']}, t2={best['t2']}")
    print("Test @t1:", json.dumps(test_metrics["at_t1"], ensure_ascii=False))
    print(f"Сохранено: ml/artifacts/url_model.joblib (версия {meta['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
