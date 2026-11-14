"""Интеграционные тесты сервиса анализа (задание 12)."""
import logging
import socket

import pytest

from apps.analysis.services import AnalysisInputError, analyze_sms, analyze_url, extract_urls
from ml.predictor import ModelUnavailableError, Predictor


def test_sms_without_url(predictor):
    r = analyze_sms("Привет! Встречаемся завтра в 18:00 у входа в библиотеку.", predictor)
    assert r.type == "SMS" and r.language == "ru"
    assert r.urls == [] and r.p_url_max is None
    assert 0 <= r.p_sms <= 1 and 0 <= r.risk_score <= 100
    assert r.category in ("SAFE", "SUSPICIOUS", "FRAUDULENT")
    assert len(r.explanation) >= 1 and r.recommendation and r.disclaimer


def test_sms_with_one_url(predictor):
    r = analyze_sms("Ваша карта заблокирована! Срочно подтвердите данные по ссылке: "
                    "http://secure-bank-verify.example/login", predictor)
    assert [u.url for u in r.urls] == ["http://secure-bank-verify.example/login"]
    assert r.urls[0].domain == "secure-bank-verify.example"
    assert r.p_url_max == r.urls[0].p_url
    assert {"URGENCY", "ACCOUNT_BLOCK", "LINK_REQUEST"} <= set(r.features)


def test_sms_with_three_urls_and_limit(predictor):
    text = ("Check http://a.example, www.b.example.com and bit.ly/c-test, also http://d.example")
    r = analyze_sms(text, predictor)
    assert len(r.urls) == 3                                   # не более 3 ссылок
    assert r.p_url_max == max(u.p_url for u in r.urls)
    assert len({u.url for u in r.urls}) == 3


def test_url_only(predictor):
    r = analyze_url("http://192.0.2.15/account-update", predictor)
    assert r.type == "URL" and r.p_sms is None and r.language is None
    assert r.category in ("SAFE", "SUSPICIOUS", "DANGEROUS")
    assert r.risk_score >= 60                                 # IP-адрес → правило усиления
    assert "has_ip" in r.features
    assert any("IP-адрес" in reason for reason in r.explanation)


def test_trusted_official_domain_is_safe(predictor):
    r = analyze_url("https://kaspi.kz/shop", predictor)
    assert r.category == "SAFE" and r.urls[0].trusted
    assert r.explanation[0].startswith("Официальный домен kaspi.kz")
    fake = analyze_url("http://kaspl.kz/shop", predictor)    # похожий домен не доверенный
    assert not fake.urls[0].trusted


def test_code_request_without_url_is_high(predictor):
    r = analyze_sms("Сотрудник службы безопасности банка: назовите код из SMS для отмены операции.", predictor)
    assert "CODE_REQUEST" in r.features and r.risk_score >= 70 and r.category == "FRAUDULENT"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 1001, None])
def test_invalid_sms_input(predictor, bad):
    with pytest.raises(AnalysisInputError):
        analyze_sms(bad, predictor)


@pytest.mark.parametrize("bad", ["", "ftp://example.com", "http://", "not a url", "http://exa mple.com"])
def test_invalid_url_input(predictor, bad):
    with pytest.raises(AnalysisInputError):
        analyze_url(bad, predictor)


def test_control_characters_removed(predictor):
    r = analyze_sms("Hello\x00\x07 world", predictor)
    assert r.input_text == "Hello world"


def test_language_note_for_unsupported_language(predictor):
    r = analyze_sms("Hallo, wie geht es dir heute? Ich komme morgen nach Hause", predictor)
    assert r.language == "other" and "не представлен в обучающих данных" in r.language_note


def test_no_url_is_opened(predictor, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("сетевой запрос запрещён")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    analyze_sms("Go to http://secure-bank-verify.example/login now", predictor)


def test_extract_urls_variants():
    assert extract_urls("Перейдите: https://сайт.рф/путь.") == ["https://сайт.рф/путь"]
    assert extract_urls("mail me at a@b.com") == []
    assert extract_urls("claim at bit.ly/prize-claim-test!") == ["bit.ly/prize-claim-test"]


def test_predictor_singleton_loads_once(toy_models, monkeypatch, caplog):
    import ml.predictor as pred
    monkeypatch.setattr(pred, "default_model_paths",
                        lambda: (toy_models / "sms_model.joblib", toy_models / "sms_model_meta.json"))
    monkeypatch.setattr(pred, "URL_MODEL_PATH", toy_models / "url_model.joblib")
    Predictor.reset()
    monkeypatch.setattr(Predictor.__init__, "__defaults__",
                        (None, None, toy_models / "url_model.joblib", toy_models / "url_model_meta.json"))
    with caplog.at_level(logging.INFO, logger="sms_guard"):
        first, second = Predictor.instance(), Predictor.instance()
    assert first is second
    assert sum("Модели загружены" in rec.message for rec in caplog.records) == 1
    Predictor.reset()


def test_missing_model_gives_clear_error(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="sms_guard"):
        p = Predictor(sms_path=tmp_path / "nope.joblib", sms_meta=tmp_path / "nope.json",
                      url_path=tmp_path / "nope2.joblib", url_meta=tmp_path / "nope2.json")
    assert not p.sms_ready and not p.url_ready
    assert p.status()["sms"]["loaded"] is False
    assert any("не загружена" in rec.message for rec in caplog.records)
    with pytest.raises(ModelUnavailableError):
        analyze_sms("Hello", p)
