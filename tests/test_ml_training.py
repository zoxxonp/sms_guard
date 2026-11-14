"""Модульные тесты обучающего кода (задания 8, 11, 13) на маленьких искусственных данных.

Полное обучение на реальных наборах выполняется командами python -m ml.train_sms / ml.train_url;
здесь проверяется, что функции сравнения, выбора, оценки, сохранения и калибровки работают корректно.
"""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from ml import calibrate_risk as cr
from ml import train_sms as ts
from ml import train_url as tu
from ml.features import make_input
from ml.metrics import compute_metrics, threshold_for_precision
from ml.url_features import model_input, registered_domain

from .conftest import BAD_URLS, GOOD_URLS, HAM, SPAM


# --- metrics ---------------------------------------------------------------------------------------
def test_compute_metrics_values():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5 and m["roc_auc"] == 0.75


def test_compute_metrics_single_class_has_no_auc():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.9], 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["confusion_matrix"]["fp"] == 1


def test_threshold_for_precision():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.3, 0.6, 0.5, 0.7, 0.9]
    t = threshold_for_precision(y, p, 0.9)
    assert t == pytest.approx(0.7)                     # при 0.5 Precision 0.75 < 0.9
    assert compute_metrics(y, p, t)["precision"] >= 0.9
    # недостижимое условие → порог с максимальной Precision (при 0,1 Precision 0,5, при 0,9 — 0)
    assert threshold_for_precision([0, 1], [0.9, 0.1], 0.99) == pytest.approx(0.1)


# --- SMS: сравнение, выбор, оценка, сохранение ----------------------------------------------------------
def _sms_frame(texts, labels, langs):
    frame = make_input(texts).reset_index(drop=True)
    frame["label"], frame["lang"] = labels, langs
    frame["group"] = [f"g{i}" for i in range(len(frame))]
    return frame


def _lang(text):
    if any(ch in text for ch in "әғқңөұүһі"):
        return "kk"
    return "ru" if any("а" <= ch.lower() <= "я" for ch in text) else "en"


@pytest.fixture(scope="module")
def sms_data():
    texts = (SPAM + HAM) * 4
    labels = ([1] * len(SPAM) + [0] * len(HAM)) * 4
    langs = [_lang(t) for t in texts]
    train = _sms_frame(texts, labels, langs)
    val = _sms_frame((SPAM + HAM) * 2, ([1] * len(SPAM) + [0] * len(HAM)) * 2, [_lang(t) for t in (SPAM + HAM) * 2])
    test = val.copy()
    custom = val[val["lang"].isin(["ru", "kk"])].reset_index(drop=True)
    return ts.SmsData(train, val, test, custom, train["group"].to_numpy())


@pytest.fixture
def small_grids(monkeypatch):
    original = ts._algorithms

    def algorithms():
        return {"naive_bayes": (MultinomialNB(), {"clf__alpha": [0.1]}),
                "logreg": (LogisticRegression(max_iter=500), {"clf__C": [1, 10]}),
                "linear_svm": (original()["linear_svm"][0], {"clf__estimator__C": [1]}),
                "random_forest": (RandomForestClassifier(n_estimators=10, random_state=0), {"clf__max_depth": [None]})}
    monkeypatch.setattr(ts, "_algorithms", algorithms)
    return original


def test_sms_full_training_cycle(sms_data, small_grids, tmp_path, monkeypatch):
    algo_results, variant_results, models = ts.run_comparison(sms_data, verbose=False)
    assert {r["name"] for r in algo_results} == {"naive_bayes", "logreg", "linear_svm", "random_forest"}
    assert len(variant_results) == 2 * len(ts.FEATURE_VARIANTS)
    for r in algo_results:
        assert 0 <= r["threshold"] <= 1 and set(r["val_by_lang"]) == {"en", "ru", "kk"}

    best = ts.choose_best(algo_results)
    assert best["name"] in models
    test_metrics = ts.evaluate_on_test(models[best["name"]], best["threshold"], sms_data)
    assert {"test_all", "test_en", "test_ru", "test_kk", "custom_ru", "custom_kk"} <= set(test_metrics)

    model, seconds, n = ts.train_final(best, sms_data)
    assert n == len(sms_data.train) + len(sms_data.val)
    monkeypatch.setattr(ts, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(ts, "MODEL_PATH", tmp_path / "sms_model.joblib")
    monkeypatch.setattr(ts, "META_PATH", tmp_path / "sms_model_meta.json")
    meta = ts.save_model(model, best, test_metrics, seconds, n)
    saved = json.loads((tmp_path / "sms_model_meta.json").read_text(encoding="utf-8"))
    assert saved["version"] == meta["version"] == ts.VERSION and saved["threshold"] == best["threshold"]
    assert (tmp_path / "sms_model.joblib").exists()


def test_choose_best_prefers_simple_model_on_tie():
    def res(name, f2, precision=0.9):
        return {"name": name, "val_at_threshold": {"f2": f2, "precision": precision}}
    assert ts.choose_best([res("random_forest", 0.950), res("logreg", 0.947)])["name"] == "logreg"
    assert ts.choose_best([res("random_forest", 0.97), res("logreg", 0.90)])["name"] == "random_forest"
    # модели с Precision < 0,85 не выбираются, если есть другие
    assert ts.choose_best([res("logreg", 0.99, 0.5), res("naive_bayes", 0.9)])["name"] == "naive_bayes"


# --- URL: пороги, сравнение, важность признаков, сохранение -----------------------------------------------
def test_url_thresholds_and_category():
    y = np.array([0] * 50 + [1] * 50)
    p = np.round(np.concatenate([np.linspace(0, 0.6, 50), np.linspace(0.3, 1, 50)]), 2)
    t1, t2 = tu.thresholds(y, p)
    assert 0 < t1 < t2 <= 1
    assert compute_metrics(y, p, t1)["recall"] >= tu.T1_RECALL
    assert compute_metrics(y, p, t2)["precision"] >= tu.T2_PRECISION
    assert tu.category(t1 - 0.01, t1, t2) == "SAFE" and tu.category(t1, t1, t2) == "SUSPICIOUS"
    assert tu.category(t2, t1, t2) == "DANGEROUS"


def _url_frame(urls, labels):
    frame = pd.DataFrame([model_input(u) for u in urls])
    frame["label"] = labels
    frame["registered_domain"] = [registered_domain(u) or u for u in urls]
    return frame


def test_url_training_cycle(tmp_path, monkeypatch):
    urls = BAD_URLS + GOOD_URLS
    labels = [1] * len(BAD_URLS) + [0] * len(GOOD_URLS)
    many = [u.replace("example", f"example{i}") for i in range(3) for u in urls]
    data = {"train": _url_frame(many, labels * 3), "val": _url_frame(urls, labels), "test": _url_frame(urls, labels)}

    original = tu.algorithms

    def algorithms():
        algos = original()
        lr, rf = algos["logreg"][0], algos["random_forest"][0]
        lr.set_params(features__host__min_df=1)
        rf.set_params(features__host__min_df=1, clf__n_estimators=10)
        return {"logreg": (lr, {"clf__C": [1]}), "random_forest": (rf, {"clf__min_samples_leaf": [1]})}
    monkeypatch.setattr(tu, "algorithms", algorithms)
    results, models = tu.run_comparison(data, verbose=False)
    best = tu.choose_best(results)
    model = models[best["name"]]
    importance = tu.feature_importance(model)
    assert "has_ip" in importance.index and any(i.startswith("n-грамма") for i in importance.index)
    metrics = tu.evaluate_test(model, best, data["test"])
    assert sum(v["safe"] + v["phishing"] for v in metrics["categories"].values()) == len(urls)

    monkeypatch.setattr(tu, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(tu, "MODEL_PATH", tmp_path / "url_model.joblib")
    monkeypatch.setattr(tu, "META_PATH", tmp_path / "url_model_meta.json")
    meta = tu.save(model, best, metrics, len(data["train"]))
    assert meta["thresholds"] == {"t1": best["t1"], "t2": best["t2"]}
    proba = tu.predict_url_proba(model, ["http://192.0.2.15/x", "https://www.example.com/about"])
    assert proba.shape == (2,) and proba[0] > proba[1]


# --- Калибровка риска ------------------------------------------------------------------------------------
@pytest.fixture
def components():
    rows = [  # label, p_sms, p_url_max, groups
        (1, 0.95, 0.9, {"URGENCY": ["срочно"], "BANK_CARD": ["карта"]}),
        (1, 0.70, None, {"CODE_REQUEST": ["назовите код"]}),
        (1, 0.30, None, {"URGENCY": ["срочно"], "MONEY_TRANSFER": ["переведи"]}),
        (0, 0.02, None, {}),
        (0, 0.10, 0.2, {}),
        (0, 0.40, None, {"BANK_CARD": ["карта"]}),
    ]
    frame = pd.DataFrame([{"label": l, "lang": "ru", "p_sms": p, "n_urls": int(u is not None), "p_url_max": u,
                           "groups": g, "has_ip": 0, "has_punycode": 0, "k": len(g)} for l, p, u, g in rows])
    return frame


def test_calibration_grid_and_choice(components):
    table = cr.run_grid(components, url_t2=0.66)
    assert len(table) == len(cr.WEIGHT_VARIANTS) * len(cr.LEVEL_VARIANTS)
    assert table[["recall_medium_plus", "safe_high_share"]].apply(lambda c: c.between(0, 1).all()).all()
    best = cr.choose(table)
    assert best["meets_goal"] and best["w_se"] >= 0.2
    assert (best["low_max"], best["medium_max"]) == (34, 69)          # границы ТЗ сохраняются


def test_calibration_evaluate(components):
    cfg = cr.load_config()
    scores = cr.scores_for(components, cfg, 0.66)
    metrics = cr.evaluate(components, scores, cfg)
    assert metrics["recall_medium_plus"] == 1.0 and metrics["safe_high_share"] == 0.0
    assert scores[1] >= 70                                               # CODE_REQUEST


def test_save_choice_keeps_comments(tmp_path):
    from ml.risk import CONFIG_PATH
    target = tmp_path / "risk_config.yaml"
    target.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    cr.save_choice(pd.Series({"w_sms": 0.7, "w_url": 0.2, "w_se": 0.1, "low_max": 29, "medium_max": 59}), target)
    text = target.read_text(encoding="utf-8")
    cfg = cr.load_config(target)
    assert cfg["weights"] == {"w_sms": 0.7, "w_url": 0.2, "w_se": 0.1}
    assert cfg["levels"] == {"low_max": 29, "medium_max": 59} and text.startswith("# Параметры модуля оценки риска")
