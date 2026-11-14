"""Сравнение алгоритмов классификации SMS и обучение итоговой модели (задание 8).

Запуск из корня проекта (несколько минут):
    python -m ml.train_sms

Что делает:
1. Данные: UCI SMS (английский, разбиение задания 6) + синтетический ru/kk-набор
   (разбиение по шаблонам). Ручной набор custom_ru_kk.csv — только для проверки.
2. Подбор гиперпараметров GridSearchCV на обучающей выборке, 5 фолдов, метрика F2.
   Фолды — StratifiedGroupKFold: сообщения одного синтетического шаблона не попадают
   одновременно в обучающий и проверочный фолд (иначе CV завышается).
3. Алгоритмы: Multinomial NB, Logistic Regression, Linear SVM (+ CalibratedClassifierCV),
   Random Forest. Варианты признаков: слова; слова без стоп-слов; слова + символы;
   слова + символы + ручные признаки (со стоп-словами и без).
4. Порог — на validation: минимальный, при котором Precision ≥ 0,90. Правило задания 8
   (Precision ≥ 0,85) выжимает Precision ровно до 0,85 и на test даёт меньше 0,90, что
   нарушает раздел 3.4; поэтому целевая Precision поднята до 0,90.
5. Итоговая модель выбирается по F2 на validation при Precision ≥ 0,85; при разнице F2
   меньше 0,005 предпочитается более быстрая и интерпретируемая модель. Она один раз
   оценивается на test, переобучается на train + val и сохраняется:
   ml/artifacts/sms_model.joblib + sms_model_meta.json (версия 1.0.0).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.features import (PROJECT_ROOT, RANDOM_STATE, SPLIT_PATH, build_vectorizer, load_split,
                         load_synthetic, make_input, save_split, split_dataset)
from ml.metrics import compute_metrics, threshold_for_precision

ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS / "sms_model.joblib"
META_PATH = ARTIFACTS / "sms_model_meta.json"
COMPARISON_PATH = ARTIFACTS / "sms_comparison.json"
X_COLS = ["text_raw", "text_clean"]
MIN_PRECISION = 0.85          # условие выбора модели (раздел 3.4: F2 при Precision ≥ 0,85)
THRESHOLD_PRECISION = 0.90    # цель при подборе порога: требование 3.4 к SMS-модели — Precision ≥ 0,90
TIE_MARGIN = 0.005            # модели с F2 в пределах 0,005 считаются равными
PREFERENCE = ["logreg", "linear_svm", "naive_bayes", "random_forest"]  # быстрее и понятнее — раньше
VERSION = "1.0.0"

FEATURE_VARIANTS = {
    "words": dict(use_word=True, use_char=False, use_manual=False, remove_stop_words=False),
    "words_nostop": dict(use_word=True, use_char=False, use_manual=False, remove_stop_words=True),
    "words_char": dict(use_word=True, use_char=True, use_manual=False, remove_stop_words=False),
    "words_char_manual": dict(use_word=True, use_char=True, use_manual=True, remove_stop_words=False),
    "words_char_manual_nostop": dict(use_word=True, use_char=True, use_manual=True, remove_stop_words=True),
}
VARIANT_TITLES = {
    "words": "Слова (1–2-граммы)",
    "words_nostop": "Слова без стоп-слов",
    "words_char": "Слова + символьные n-граммы",
    "words_char_manual": "Слова + символы + 8 ручных признаков",
    "words_char_manual_nostop": "Слова + символы + ручные, без стоп-слов",
}


def _algorithms() -> dict:
    return {
        "naive_bayes": (MultinomialNB(), {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}),
        "logreg": (LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
                   {"clf__C": [0.1, 1, 10, 100]}),
        "linear_svm": (CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
                                              cv=3, method="sigmoid"),
                       {"clf__estimator__C": [0.01, 0.1, 1, 10]}),
        "random_forest": (RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                                 random_state=RANDOM_STATE),
                          {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 60]}),
    }


ALGO_TITLES = {"naive_bayes": "Multinomial Naive Bayes", "logreg": "Logistic Regression",
               "linear_svm": "Linear SVM (калибровка)", "random_forest": "Random Forest"}


@dataclass
class SmsData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    custom: pd.DataFrame
    groups_train: np.ndarray = field(repr=False)


def load_data() -> SmsData:
    """UCI (en) + синтетика (ru/kk) с единым набором столбцов: text_raw, text_clean, label, lang, group."""
    uci = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sms_clean.csv", keep_default_na=False)
    if not SPLIT_PATH.exists():
        save_split(split_dataset(uci), uci)
    uci_parts = load_split(uci)
    synth = load_synthetic()
    cols = X_COLS + ["label", "lang", "group", "source"]
    parts = {}
    for name in ("train", "val", "test"):
        u = uci_parts[name].assign(lang="en", group=lambda d: "uci-" + d["id"].astype(str), source="uci")
        s = synth[name].assign(group=lambda d: d["template_id"], source="synthetic")
        parts[name] = pd.concat([u[cols], s[cols]], ignore_index=True)
    custom = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "custom_ru_kk.csv")
    custom = pd.concat([custom, make_input(custom["text"]).set_axis(custom.index)], axis=1)
    return SmsData(parts["train"], parts["val"], parts["test"], custom, parts["train"]["group"].to_numpy())


def make_pipeline(algo: str, variant: str = "words_char_manual") -> tuple[Pipeline, dict]:
    estimator, grid = _algorithms()[algo]
    return Pipeline([("features", build_vectorizer(**FEATURE_VARIANTS[variant])), ("clf", estimator)]), grid


def grid_search(algo: str, variant: str, data: SmsData, n_jobs: int = 1) -> GridSearchCV:
    pipe, grid = make_pipeline(algo, variant)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, grid, scoring=make_scorer(fbeta_score, beta=2), cv=cv,
                          n_jobs=n_jobs, refit=True)
    search.fit(data.train[X_COLS], data.train["label"], groups=data.groups_train)
    return search


def predict_ms(model, frame: pd.DataFrame, n: int = 200) -> float:
    """Среднее время предсказания одного SMS (мс), по одному сообщению за вызов."""
    sample = frame[X_COLS].head(n)
    start = time.perf_counter()
    for i in range(len(sample)):
        model.predict_proba(sample.iloc[[i]])
    return (time.perf_counter() - start) / len(sample) * 1000


def evaluate_model(name: str, search: GridSearchCV, data: SmsData, fit_seconds: float) -> dict:
    model = search.best_estimator_
    proba = model.predict_proba(data.val[X_COLS])[:, 1]
    thr = threshold_for_precision(data.val["label"], proba, THRESHOLD_PRECISION)
    by_lang = {lang: compute_metrics(data.val.loc[data.val["lang"] == lang, "label"],
                                     proba[(data.val["lang"] == lang).to_numpy()], 0.5)
               for lang in ("en", "ru", "kk")}
    return {
        "name": name,
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "cv_f2": round(float(search.best_score_), 4),
        "val_at_0_5": compute_metrics(data.val["label"], proba, 0.5),
        "threshold": round(thr, 4),
        "val_at_threshold": compute_metrics(data.val["label"], proba, thr),
        "val_by_lang": by_lang,
        "fit_seconds": round(fit_seconds, 1),
        "predict_ms": round(predict_ms(model, data.val), 2),
        "n_features": int(len(model.named_steps["features"].get_feature_names_out())),
    }


def run_comparison(data: SmsData, verbose: bool = True) -> tuple[list[dict], list[dict], dict]:
    """Возвращает (результаты алгоритмов, результаты вариантов признаков, обученные модели)."""
    algo_results, models = [], {}
    for algo in _algorithms():
        start = time.perf_counter()
        search = grid_search(algo, "words_char_manual", data)
        res = evaluate_model(algo, search, data, time.perf_counter() - start)
        res["title"] = ALGO_TITLES[algo]
        algo_results.append(res)
        models[algo] = search.best_estimator_
        if verbose:
            print(f"{algo:14s} cv_F2={res['cv_f2']:.4f} val_F2={res['val_at_threshold']['f2']:.4f} "
                  f"P={res['val_at_threshold']['precision']:.3f} R={res['val_at_threshold']['recall']:.3f} "
                  f"thr={res['threshold']:.3f} {res['best_params']} ({res['fit_seconds']} c)")
    variant_results = []
    for algo in ("naive_bayes", "logreg"):
        for variant in FEATURE_VARIANTS:
            start = time.perf_counter()
            search = grid_search(algo, variant, data)
            res = evaluate_model(algo, search, data, time.perf_counter() - start)
            res.update(variant=variant, variant_title=VARIANT_TITLES[variant], title=ALGO_TITLES[algo])
            variant_results.append(res)
            if verbose:
                print(f"{algo:12s} {variant:26s} val_F2={res['val_at_threshold']['f2']:.4f} "
                      f"R={res['val_at_threshold']['recall']:.3f} P={res['val_at_threshold']['precision']:.3f}")
    return algo_results, variant_results, models


def choose_best(algo_results: list[dict]) -> dict:
    """Максимум F2 на validation (при выбранном пороге) среди моделей с Precision ≥ 0,85.

    Модели, отстающие от лучшей меньше чем на TIE_MARGIN, считаются равными; из них
    выбирается первая по списку PREFERENCE (быстрее предсказание, понятнее веса).
    """
    ok = [r for r in algo_results if r["val_at_threshold"]["precision"] >= MIN_PRECISION]
    pool = ok or algo_results
    top = max(r["val_at_threshold"]["f2"] for r in pool)
    near = [r for r in pool if top - r["val_at_threshold"]["f2"] < TIE_MARGIN]
    return min(near, key=lambda r: PREFERENCE.index(r["name"]))


def evaluate_on_test(model, threshold: float, data: SmsData) -> dict:
    """Однократная оценка на test + ручной ru/kk-набор, по языкам."""
    proba = model.predict_proba(data.test[X_COLS])[:, 1]
    result = {"test_all": compute_metrics(data.test["label"], proba, threshold)}
    for lang in ("en", "ru", "kk"):
        mask = (data.test["lang"] == lang).to_numpy()
        result[f"test_{lang}"] = compute_metrics(data.test.loc[mask, "label"], proba[mask], threshold)
    proba_c = model.predict_proba(data.custom[X_COLS])[:, 1]
    for lang in ("ru", "kk"):
        mask = (data.custom["lang"] == lang).to_numpy()
        result[f"custom_{lang}"] = compute_metrics(data.custom.loc[mask, "label"], proba_c[mask], threshold)
    return result


def train_final(best: dict, data: SmsData):
    """Переобучение выбранной конфигурации на train + val."""
    pipe, _ = make_pipeline(best["name"], best.get("variant", "words_char_manual"))
    pipe.set_params(**{f"clf__{k}": v for k, v in best["best_params"].items()})
    full = pd.concat([data.train, data.val], ignore_index=True)
    start = time.perf_counter()
    pipe.fit(full[X_COLS], full["label"])
    return pipe, time.perf_counter() - start, len(full)


def save_model(model, best: dict, test_metrics: dict, fit_seconds: float, n_train: int,
               final_test_metrics: dict | None = None) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "model": "sms_model",
        "version": VERSION,
        "algorithm": best["title"],
        "params": best["best_params"],
        "features": VARIANT_TITLES[best.get("variant", "words_char_manual")],
        "n_features": int(len(model.named_steps["features"].get_feature_names_out())),
        "threshold": best["threshold"],
        "threshold_rule": f"минимальный порог на validation, при котором Precision ≥ {THRESHOLD_PRECISION}",
        "languages": ["en", "ru", "kk"],
        "trained_on": f"train + validation ({n_train} SMS: UCI en + синтетика ru/kk)",
        "metrics_validation": best["val_at_threshold"],
        "metrics_test": test_metrics,
        "metrics_test_after_refit": final_test_metrics,
        "train_seconds": round(fit_seconds, 2),
        "predict_ms_per_sms": best["predict_ms"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    data = load_data()
    print(f"train {len(data.train)}, val {len(data.val)}, test {len(data.test)}, custom {len(data.custom)}")
    algo_results, variant_results, models = run_comparison(data)
    best = choose_best(algo_results)
    print(f"\nВыбрана модель: {best['title']} {best['best_params']}, порог {best['threshold']}")

    test_metrics = evaluate_on_test(models[best["name"]], best["threshold"], data)
    print("Test (модель, обученная на train):", json.dumps(test_metrics["test_all"], ensure_ascii=False))

    final, fit_seconds, n_train = train_final(best, data)
    final_test = evaluate_on_test(final, best["threshold"], data)
    meta = save_model(final, best, test_metrics, fit_seconds, n_train, final_test)
    COMPARISON_PATH.write_text(json.dumps({"algorithms": algo_results, "variants": variant_results,
                                           "best": best["name"]}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    print(f"Сохранено: {MODEL_PATH.relative_to(PROJECT_ROOT)}, {META_PATH.name} (версия {meta['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
