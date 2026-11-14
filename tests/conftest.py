"""Общие фикстуры: маленькие SMS- и URL-модели, обученные на игрушечных данных.

Настоящие модели (ml/artifacts/*.joblib) в git не хранятся, поэтому интеграционные тесты
сервиса и API работают на этих моделях и не зависят от обучения.
"""
import json

import joblib
import pandas as pd
import pytest

from ml.features import make_input
from ml.predictor import Predictor
from ml.train_baseline import build_baseline
from ml.url_features import MODEL_FEATURES, MODEL_TEXT_COLUMN, model_input

SPAM = ["WINNER! Claim your £900 prize now, call 09061701461", "URGENT! Your account is blocked, verify your card",
        "Free entry to win cash! Txt WIN to 87121", "Ваша карта заблокирована! Срочно подтвердите данные",
        "Вы выиграли приз! Оплатите доставку 990 тг", "Картаңыз бұғатталды. Жедел растаңыз",
        "Мама, срочно переведи 50 000 на этот номер", "Назовите код из SMS оператору"]
HAM = ["Ok see you at home later", "Are we meeting for lunch tomorrow?", "Thanks for the notes",
       "Привет! Встречаемся завтра в 18:00", "Купи хлеб по дороге домой", "Сәлем! Ертең кездесеміз бе?",
       "Ваш код подтверждения 4821. Никому не сообщайте этот код", "Your order has been shipped"]
BAD_URLS = ["http://secure-bank-verify.example/login", "http://kaspi-bonus.test/claim", "http://192.0.2.15/x",
            "bit.ly/prize-claim-test", "http://login-verify-account.example", "http://free-prize-win.example",
            "http://halyk-secure.test", "http://xn--exmple-cua.test/login", "http://a1-b2-c3.verify.example",
            "http://paypa1.example/signin"]
GOOD_URLS = ["https://www.example.com/about", "https://example.org", "https://docs.example.com",
             "https://news.example.org/today", "https://shop.example.com/cart", "https://example.net",
             "https://www.wikipedia.org", "https://library.example.org", "https://weather.example.com",
             "https://example.edu"]


@pytest.fixture(scope="session")
def toy_models(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    texts = SPAM * 4 + HAM * 4
    labels = [1] * len(SPAM) * 4 + [0] * len(HAM) * 4
    sms = build_baseline().fit(make_input(texts), labels)
    joblib.dump(sms, folder / "sms_model.joblib")
    (folder / "sms_model_meta.json").write_text(json.dumps(
        {"version": "test-1", "algorithm": "Logistic Regression", "threshold": 0.5, "languages": ["en", "ru", "kk"]}))

    from ml.train_url import algorithms
    url_pipe = algorithms()["logreg"][0]
    url_pipe.set_params(features__host__min_df=1)
    rows = pd.DataFrame([model_input(u) for u in BAD_URLS + GOOD_URLS])[MODEL_FEATURES + [MODEL_TEXT_COLUMN]]
    url_pipe.fit(rows, [1] * len(BAD_URLS) + [0] * len(GOOD_URLS))
    joblib.dump(url_pipe, folder / "url_model.joblib")
    (folder / "url_model_meta.json").write_text(json.dumps(
        {"version": "test-1", "algorithm": "Logistic Regression", "thresholds": {"t1": 0.3, "t2": 0.7}}))
    return folder


@pytest.fixture
def predictor(toy_models):
    return Predictor(sms_path=toy_models / "sms_model.joblib", sms_meta=toy_models / "sms_model_meta.json",
                     url_path=toy_models / "url_model.joblib", url_meta=toy_models / "url_model_meta.json")


@pytest.fixture
def use_toy_predictor(predictor, monkeypatch):
    """Подменяет синглтон Predictor игрушечными моделями (для тестов сервиса и API)."""
    monkeypatch.setattr(Predictor, "_instance", predictor)
    yield predictor
    Predictor.reset()
