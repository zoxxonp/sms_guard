"""Тесты REST API (задание 14, п. 9): успешные сценарии, ошибки ввода, доступ, лимиты, безопасность."""
import logging

import pytest
from rest_framework.test import APIClient

from apps.analysis.models import Check, DetectedFeature, UrlResult
from ml.predictor import Predictor

from .conftest import PASSWORD

pytestmark = pytest.mark.django_db

FRAUD_SMS = "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login"


def error_of(response):
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}
    return body["error"]


# --- Аутентификация ---------------------------------------------------------------------------
def test_register_creates_user_and_logs_in(db):
    client = APIClient()
    r = client.post("/api/v1/auth/register/", {"username": "bob_01", "email": "bob@example.com",
                                                "password": "Secret2026x"}, format="json")
    assert r.status_code == 201
    assert r.json()["username"] == "bob_01" and "password" not in r.json()
    assert client.get("/api/v1/checks/").status_code == 200          # сессия создана


@pytest.mark.parametrize("payload, message_part", [
    ({"username": "ab", "email": "x@example.com", "password": "Secret2026x"}, "Логин"),
    ({"username": "кириллица", "email": "x@example.com", "password": "Secret2026x"}, "Логин"),
    ({"username": "good_name", "email": "not-an-email", "password": "Secret2026x"}, "e-mail"),
    ({"username": "good_name", "email": "x@example.com", "password": "short1"}, "8 символов"),
    ({"username": "good_name", "email": "x@example.com", "password": "onlyletters"}, "буквы и цифры"),
])
def test_register_validation(db, payload, message_part):
    r = APIClient().post("/api/v1/auth/register/", payload, format="json")
    assert r.status_code == 400
    err = error_of(r)
    assert err["code"] == "VALIDATION_ERROR" and message_part in err["message"]


def test_register_duplicate_username_and_email(user):
    client = APIClient()
    r = client.post("/api/v1/auth/register/", {"username": "ALICE", "email": "new@example.com",
                                                "password": "Secret2026x"}, format="json")
    assert r.status_code == 400 and "уже существует" in error_of(r)["message"]
    r = client.post("/api/v1/auth/register/", {"username": "carol", "email": "alice@example.com",
                                                "password": "Secret2026x"}, format="json")
    assert r.status_code == 400 and "уже зарегистрирован" in error_of(r)["message"]


def test_password_is_hashed(user):
    assert user.password.startswith("pbkdf2_sha256$") and PASSWORD not in user.password


def test_login_logout(user):
    client = APIClient()
    r = client.post("/api/v1/auth/login/", {"username": "alice", "password": PASSWORD}, format="json")
    assert r.status_code == 200 and r.json()["username"] == "alice"
    assert client.post("/api/v1/auth/logout/").status_code == 204
    assert client.get("/api/v1/checks/").status_code == 401              # сессия уничтожена


def test_login_wrong_password_401(user):
    r = APIClient().post("/api/v1/auth/login/", {"username": "alice", "password": "wrong"}, format="json")
    assert r.status_code == 401 and error_of(r)["code"] == "INVALID_CREDENTIALS"


def test_login_lockout_after_5_failures(user):
    client = APIClient()
    codes = [client.post("/api/v1/auth/login/", {"username": "alice", "password": f"bad{i}"}, format="json").status_code
             for i in range(5)]
    assert codes[:4] == [401] * 4 and codes[4] == 429
    # даже правильный пароль не принимается во время паузы
    r = client.post("/api/v1/auth/login/", {"username": "alice", "password": PASSWORD}, format="json")
    assert r.status_code == 429 and error_of(r)["code"] == "TOO_MANY_ATTEMPTS"
    assert int(r["Retry-After"]) <= 60


def test_unauthenticated_access_401(db):
    client = APIClient()
    for method, url in [("post", "/api/v1/check/sms/"), ("post", "/api/v1/check/url/"), ("get", "/api/v1/checks/"),
                        ("get", "/api/v1/checks/1/"), ("get", "/api/v1/stats/summary/")]:
        r = getattr(client, method)(url, {}, format="json")
        assert r.status_code == 401, url
        assert error_of(r)["code"] == "NOT_AUTHENTICATED"


def test_csrf_required_for_session_api(user, use_toy_predictor):
    client = APIClient(enforce_csrf_checks=True)
    client.force_login(user)
    r = client.post("/api/v1/check/sms/", {"text": "Привет"}, format="json")
    assert r.status_code == 403 and error_of(r)["code"] == "CSRF_FAILED"
    # логин без токена тоже отклоняется (защита от login CSRF)
    r = APIClient(enforce_csrf_checks=True).post("/api/v1/auth/login/", {"username": "alice", "password": PASSWORD},
                                                 format="json")
    assert r.status_code == 403


def test_csrf_with_token_passes(user, use_toy_predictor):
    client = APIClient(enforce_csrf_checks=True)
    client.get("/api/v1/auth/csrf/")
    token = client.cookies["csrftoken"].value
    r = client.post("/api/v1/auth/login/", {"username": "alice", "password": PASSWORD}, format="json",
                    HTTP_X_CSRFTOKEN=token)
    assert r.status_code == 200
    token = client.cookies["csrftoken"].value                       # токен ротируется при входе
    r = client.post("/api/v1/check/sms/", {"text": "Привет"}, format="json", HTTP_X_CSRFTOKEN=token)
    assert r.status_code == 200


# --- Проверки ---------------------------------------------------------------------------------
def test_check_sms_success_and_saved(api, user):
    r = api.post("/api/v1/check/sms/", {"text": FRAUD_SMS}, format="json")
    assert r.status_code == 200
    data = r.json()
    for key in ("id", "type", "language", "category", "risk_level", "risk_score", "p_sms", "urls", "features",
                "explanation", "recommendation", "disclaimer", "model_versions", "created_at"):
        assert key in data
    assert data["type"] == "SMS" and data["language"] == "ru" and len(data["urls"]) == 1
    assert data["urls"][0]["url"] == "http://secure-bank-verify.example/login"
    assert data["explanation"] and data["recommendation"]
    check = Check.objects.get(pk=data["id"])
    assert check.user == user and UrlResult.objects.filter(check_ref=check).count() == 1
    assert DetectedFeature.objects.filter(check_ref=check).exists()


def test_check_sms_without_url_has_null_p_url(api):
    data = api.post("/api/v1/check/sms/", {"text": "Привет! Встречаемся завтра в 18:00"}, format="json").json()
    assert data["urls"] == [] and data["p_url_max"] is None


def test_check_url_success(api):
    r = api.post("/api/v1/check/url/", {"url": "http://xn--exmple-cua.test/secure/login?verify=1"}, format="json")
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "URL" and data["category"] in ("SAFE", "SUSPICIOUS", "DANGEROUS")
    assert data["urls"][0]["features"]["has_punycode"] == 1
    assert data["risk_score"] >= 60                                  # правило IP/Punycode


@pytest.mark.parametrize("payload, message", [
    ({"text": ""}, "не может быть пустым"),
    ({"text": "   \n\t  "}, "не может быть пустым"),
    ({}, "не может быть пустым"),
    ({"text": "a" * 1001}, "длиннее 1000"),
    ({"text": 12345}, None),
])
def test_check_sms_invalid_input(api, payload, message):
    r = api.post("/api/v1/check/sms/", payload, format="json")
    if message is None:          # число приводится к строке "12345" — это корректный ввод
        assert r.status_code == 200
        return
    assert r.status_code == 400
    err = error_of(r)
    assert err["code"] == "VALIDATION_ERROR" and message in err["message"]


def test_check_sms_exactly_1000_chars_ok(api):
    assert api.post("/api/v1/check/sms/", {"text": "б" * 1000}, format="json").status_code == 200


def test_control_characters_removed(api):
    data = api.post("/api/v1/check/sms/", {"text": "Привет\x00\x07 мир"}, format="json").json()
    assert "\x00" not in Check.objects.get(pk=data["id"]).input_masked


@pytest.mark.parametrize("url, status", [("", 400), ("http://", 400), ("ftp://example.com/file", 400),
                                         ("http://" + "a" * 2050 + ".example", 400), ("example.com", 200)])
def test_check_url_validation(api, url, status):
    r = api.post("/api/v1/check/url/", {"url": url}, format="json")
    assert r.status_code == status
    if status == 400:
        assert error_of(r)["code"] == "VALIDATION_ERROR"


def test_invalid_json_body(api):
    r = api.post("/api/v1/check/sms/", data=b"\x00\xff{not json", content_type="application/json")
    assert r.status_code == 400 and error_of(r)["code"] == "VALIDATION_ERROR"


def test_double_submit_does_not_duplicate(api):
    first = api.post("/api/v1/check/sms/", {"text": FRAUD_SMS}, format="json").json()
    second = api.post("/api/v1/check/sms/", {"text": FRAUD_SMS}, format="json").json()
    assert second["id"] == first["id"] and second["duplicate"] is True
    assert Check.objects.count() == 1


def test_throttling_30_per_minute(api):
    codes = [api.post("/api/v1/check/sms/", {"text": f"Сообщение номер {i}"}, format="json").status_code
             for i in range(31)]
    assert codes[:30] == [200] * 30 and codes[30] == 429
    r = api.post("/api/v1/check/url/", {"url": "https://www.example.com"}, format="json")
    assert r.status_code == 429 and error_of(r)["code"] == "THROTTLED" and "30 проверок" in error_of(r)["message"]


def test_model_unavailable_503(user, tmp_path, monkeypatch):
    broken = Predictor(sms_path=tmp_path / "missing.joblib", sms_meta=tmp_path / "missing.json",
                       url_path=tmp_path / "missing2.joblib", url_meta=tmp_path / "missing2.json")
    monkeypatch.setattr(Predictor, "_instance", broken)
    client = APIClient()
    client.force_login(user)
    r = client.post("/api/v1/check/sms/", {"text": "Привет"}, format="json")
    assert r.status_code == 503
    err = error_of(r)
    assert err["code"] == "MODEL_UNAVAILABLE" and "модель не загружена" in err["message"]
    health = client.get("/api/v1/health/").json()
    assert health["status"] == "degraded" and health["models"]["sms"]["loaded"] is False
    Predictor.reset()


# --- История и доступ к чужим записям -----------------------------------------------------------
def _seed(api, n=3):
    texts = [FRAUD_SMS, "Привет! Встречаемся завтра в 18:00", "Your order has been shipped"][:n]
    return [api.post("/api/v1/check/sms/", {"text": t}, format="json").json()["id"] for t in texts]


def test_history_list_filters_and_pagination(api):
    ids = _seed(api)
    api.post("/api/v1/check/url/", {"url": "https://www.example.com/about"}, format="json")
    data = api.get("/api/v1/checks/").json()
    assert data["count"] == 4 and data["page"] == 1 and data["page_size"] == 20
    assert [r["id"] for r in data["results"]][1:] == ids[::-1]          # новые сверху
    assert api.get("/api/v1/checks/?type=URL").json()["count"] == 1
    sms_safe = api.get("/api/v1/checks/?type=SMS&category=SAFE").json()
    assert all(r["category"] == "SAFE" and r["type"] == "SMS" for r in sms_safe["results"])
    assert api.get("/api/v1/checks/?date_from=2000-01-01&date_to=2000-01-02").json()["count"] == 0
    assert api.get("/api/v1/checks/?page=99").status_code == 404
    assert api.get("/api/v1/checks/?type=XXX").status_code == 400
    assert api.get("/api/v1/checks/?date_from=31-31-2026").status_code == 400


def test_detail_and_delete_own(api):
    check_id = _seed(api, 1)[0]
    r = api.get(f"/api/v1/checks/{check_id}/")
    assert r.status_code == 200 and r.json()["id"] == check_id
    assert api.delete(f"/api/v1/checks/{check_id}/").status_code == 204
    assert api.get(f"/api/v1/checks/{check_id}/").status_code == 404
    assert not UrlResult.objects.exists() and not DetectedFeature.objects.exists()     # каскадное удаление


def test_foreign_record_returns_404(api, make_user, use_toy_predictor):
    check_id = _seed(api, 1)[0]
    other = APIClient()
    other.force_login(make_user("mallory"))
    assert other.get(f"/api/v1/checks/{check_id}/").status_code == 404
    assert other.delete(f"/api/v1/checks/{check_id}/").status_code == 404
    assert other.get("/api/v1/checks/").json()["count"] == 0
    assert Check.objects.filter(pk=check_id).exists()


def test_sql_injection_is_plain_text(api):
    _seed(api, 2)
    payload = "' OR 1=1; DROP TABLE checks; --"
    r = api.post("/api/v1/check/sms/", {"text": payload}, format="json")
    assert r.status_code == 200
    assert api.get("/api/v1/checks/?category=" + "SAFE' OR '1'='1").status_code == 400
    assert api.get("/api/v1/checks/").json()["count"] == 3                  # таблица цела


def test_xss_payload_returned_as_data(api, user):
    from apps.accounts.models import get_profile
    profile = get_profile(user)
    profile.store_sms_text = True
    profile.save()
    payload = "<script>alert(1)</script>"
    data = api.post("/api/v1/check/sms/", {"text": payload}, format="json").json()
    assert api.get(f"/api/v1/checks/{data['id']}/").json()["input"] == payload      # данные не изменены
    # страница результата экранирует ввод (json_script), тег не исполняется
    from django.test import Client
    page = Client()
    page.force_login(user)
    html = page.get(f"/checks/{data['id']}/").content.decode()
    assert "<script>alert(1)</script>" not in html and "\\u003Cscript\\u003E" in html


def test_stats_summary(api):
    _seed(api)
    data = api.get("/api/v1/stats/summary/?days=7").json()
    assert data["period_days"] == 7 and data["totals"]["total"] == 3 and data["totals"]["sms"] == 3
    assert sum(data["categories"].values()) == 3 and len(data["daily"]["dates"]) == 7
    assert sum(sum(data["daily"][k]) for k in ("SAFE", "SUSPICIOUS", "FRAUD")) == 3
    assert len(data["recent"]) == 3 and "models" not in data
    assert api.get("/api/v1/stats/summary/?days=5").json()["period_days"] == 30       # недопустимый период


def test_admin_stats_all_users_without_foreign_texts(api, make_user, use_toy_predictor):
    _seed(api, 2)
    admin = APIClient()
    admin.force_login(make_user("boss", is_staff=True))
    data = admin.get("/api/v1/stats/summary/").json()
    assert data["scope"] == "all" and data["totals"]["total"] == 2 and "models" in data
    assert all(r["fragment"] is None for r in data["recent"])                         # ФТ-1.9


def test_logs_do_not_contain_sms_text(api, caplog):
    secret = "Уникальная фраза 7781 для проверки логов"
    with caplog.at_level(logging.DEBUG):
        r = api.post("/api/v1/check/sms/", {"text": secret}, format="json")
    assert r.status_code == 200
    assert "Уникальная фраза" not in caplog.text
    assert f"check id={r.json()['id']}" in caplog.text and "category=" in caplog.text
