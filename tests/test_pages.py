"""Тесты страниц, хранения и маскирования (задания 15–17)."""
import pytest
from django.core.management import call_command
from django.test import Client

from apps.accounts.models import UserProfile, get_profile
from apps.analysis.masking import mask_text
from apps.analysis.models import Check, ModelVersion
from apps.analysis.services import analyze_sms
from apps.analysis.storage import save_result

from .conftest import PASSWORD

pytestmark = pytest.mark.django_db

PROTECTED = ["/", "/check/sms/", "/check/url/", "/history/", "/stats/", "/profile/", "/checks/1/"]


@pytest.fixture
def web(user, use_toy_predictor):
    client = Client()
    client.force_login(user)
    return client


def _save(user, text, predictor):
    return save_result(user, analyze_sms(text, predictor), predictor)


# --- Маскирование ---------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("Карта 4400 4301 2345 6789 заблокирована", "Карта *** заблокирована"),
    ("card 4400430123456789012", "card ***"),
    ("Позвоните +7 701 123 45 67", "Позвоните ***"),
    ("или 8(727)123-45-67", "или ***"),
    ("Переведи 50 000 на этот номер", "Переведи *** на этот номер"),
    ("Оплатите 350 T сегодня", "Оплатите *** сегодня"),
    ("You won a $1000 gift card", "You won a *** gift card"),
    ("Код 4821, встреча в 18:00", "Код 4821, встреча в 18:00"),
    ("http://192.0.2.15/account-update", "http://192.0.2.15/account-update"),
])
def test_mask_text(text, expected):
    assert mask_text(text) == expected


def test_profile_created_with_consent_off(user):
    assert UserProfile.objects.get(user=user).store_sms_text is False


def test_text_not_stored_without_consent(user, predictor):
    check = _save(user, "Срочно переведи 50 000 на карту 4400 4301 2345 6789, звони +7 701 123 45 67", predictor)
    assert check.input_text is None
    assert "4400" not in check.input_masked and "701" not in check.input_masked and "***" in check.input_masked


def test_text_stored_with_consent(user, predictor):
    profile = get_profile(user)
    profile.store_sms_text = True
    profile.save()
    check = _save(user, "Переведи 50 000", predictor)
    assert check.input_text == "Переведи 50 000" and check.input_masked == "Переведи ***"


def test_model_versions_registered(user, predictor):
    _save(user, "Привет", predictor)
    active = ModelVersion.objects.filter(is_active=True)
    assert {m.model_type for m in active} == {"sms", "url"}


# --- Доступ к страницам ------------------------------------------------------------------------------
@pytest.mark.parametrize("url", PROTECTED)
def test_pages_require_login(db, url):
    r = Client().get(url)
    assert r.status_code == 302 and r.url.startswith("/login/?next=")


@pytest.mark.parametrize("url", ["/", "/check/sms/", "/check/url/", "/history/", "/stats/", "/profile/"])
def test_pages_open_for_user(web, url):
    r = web.get(url)
    assert r.status_code == 200
    html = r.content.decode()
    assert "Проверка SMS" in html and "Выход" in html


def test_login_page_and_errors(user):
    client = Client()
    assert client.get("/login/").status_code == 200
    r = client.post("/login/", {"username": "alice", "password": "wrong"})
    assert "Неверный логин или пароль" in r.content.decode()
    r = client.post("/login/", {"username": "alice", "password": PASSWORD})
    assert r.status_code == 302 and r.url == "/"


def test_login_page_lockout(user):
    client = Client()
    for _ in range(5):
        r = client.post("/login/", {"username": "alice", "password": "wrong"})
    assert r.status_code == 429 and "Повторите через" in r.content.decode()


def test_login_open_redirect_blocked(user):
    r = Client().post("/login/?next=https://evil.example/", {"username": "alice", "password": PASSWORD,
                                                              "next": "https://evil.example/"})
    assert r.status_code == 302 and r.url == "/"


def test_register_page(db):
    client = Client()
    r = client.post("/register/", {"username": "x", "email": "bad", "password1": "123", "password2": "456"})
    html = r.content.decode()
    assert r.status_code == 200 and "Логин: от 3 до 30 символов" in html and "Введите правильный адрес" in html
    r = client.post("/register/", {"username": "new_user", "email": "new@example.com",
                                   "password1": "Secret2026x", "password2": "Secret2026x"})
    assert r.status_code == 302 and client.get("/").status_code == 200


def test_logout_requires_post_and_destroys_session(web):
    assert web.get("/logout/").status_code == 405
    assert web.post("/logout/").status_code == 302
    assert web.get("/history/").status_code == 302


def test_session_and_csrf_cookie_flags(user):
    client = Client(enforce_csrf_checks=True)
    page = client.get("/login/")
    token = page.cookies["csrftoken"].value
    r = client.post("/login/", {"username": "alice", "password": PASSWORD, "csrfmiddlewaretoken": token})
    cookie = r.cookies["sessionid"]
    assert cookie["httponly"] and cookie["samesite"] == "Lax"
    # POST-форма без токена отклоняется
    assert Client(enforce_csrf_checks=True).post("/login/", {"username": "alice", "password": PASSWORD}).status_code == 403


# --- История, результат, удаление ----------------------------------------------------------------------
def test_history_filters_in_query_string(web, user, predictor):
    for text in ["Привет! Встречаемся завтра", "Ваша карта заблокирована! Срочно подтвердите данные",
                 "Your order has been shipped"]:
        _save(user, text, predictor)
    html = web.get("/history/").content.decode()
    assert "Показано 1–3 из 3" in html
    fraud = Check.objects.filter(user=user).exclude(category="SAFE").count()
    html = web.get("/history/?category=SAFE&type=SMS").content.decode()
    assert f"из {3 - fraud}" in html and 'value="SAFE" selected' in html


def test_history_pagination_20(web, user, predictor):
    for i in range(23):
        _save(user, f"Сообщение {i}", predictor)
    html = web.get("/history/").content.decode()
    assert "Показано 1–20 из 23" in html and "page=2" in html
    assert "Показано 21–23 из 23" in web.get("/history/?page=2").content.decode()


def test_history_escapes_xss(web, user, predictor):
    profile = get_profile(user)
    profile.store_sms_text = True
    profile.save()
    _save(user, "<script>alert(1)</script>", predictor)
    html = web.get("/history/").content.decode()
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_detail_delete_with_confirmation(web, user, predictor):
    check = _save(user, "Привет", predictor)
    assert web.get(f"/checks/{check.pk}/").status_code == 200
    confirm = web.get(f"/checks/{check.pk}/delete/")
    assert confirm.status_code == 200 and "Да, удалить" in confirm.content.decode()
    assert Check.objects.filter(pk=check.pk).exists()                     # GET ничего не удаляет
    assert web.post(f"/checks/{check.pk}/delete/").status_code == 302
    assert not Check.objects.filter(pk=check.pk).exists()


def test_foreign_detail_and_delete_404(web, make_user, predictor):
    other = make_user("mallory")
    check = _save(other, "Чужое сообщение", predictor)
    assert web.get(f"/checks/{check.pk}/").status_code == 404
    assert web.post(f"/checks/{check.pk}/delete/").status_code == 404
    assert Check.objects.filter(pk=check.pk).exists()


def test_profile_consent_and_delete_history(web, user, predictor):
    web.post("/profile/", {"action": "consent", "store_sms_text": "on"})
    assert get_profile(user).store_sms_text is True
    web.post("/profile/", {"action": "consent"})
    assert get_profile(user).store_sms_text is False
    _save(user, "Раз", predictor)
    _save(user, "Два", predictor)
    r = web.post("/profile/", {"action": "delete_history"})               # без подтверждения
    assert "Подтвердите удаление" in r.content.decode() and Check.objects.filter(user=user).count() == 2
    web.post("/profile/", {"action": "delete_history", "confirm": "on"})
    assert Check.objects.filter(user=user).count() == 0


def test_profile_password_change(web, user):
    r = web.post("/profile/", {"action": "password", "old_password": PASSWORD,
                               "new_password1": "NewSecret2027", "new_password2": "NewSecret2027"})
    assert r.status_code == 302
    user.refresh_from_db()
    assert user.check_password("NewSecret2027")
    assert web.get("/profile/").status_code == 200                        # сессия сохранена


def test_check_page_shows_model_unavailable(user, tmp_path, monkeypatch):
    from ml.predictor import Predictor
    monkeypatch.setattr(Predictor, "_instance", Predictor(sms_path=tmp_path / "no.joblib", sms_meta=tmp_path / "no.json",
                                                          url_path=tmp_path / "no2.joblib", url_meta=tmp_path / "no2.json"))
    client = Client()
    client.force_login(user)
    html = client.get("/check/sms/").content.decode()
    assert "Анализ временно недоступен" in html
    Predictor.reset()


# --- seed_demo и статистика -------------------------------------------------------------------------------
def test_seed_demo_creates_30_checks(use_toy_predictor):
    call_command("seed_demo", "--password", "DemoPass2026")
    from django.contrib.auth import get_user_model
    demo = get_user_model().objects.get(username="demo_user")
    assert demo.check_password("DemoPass2026")
    assert Check.objects.filter(user=demo).count() == 30
    assert Check.objects.filter(user=demo, check_type="URL").count() == 3
    call_command("seed_demo")                                              # повторный запуск не дублирует
    assert Check.objects.filter(user=demo).count() == 30


def test_stats_cards_match_db(use_toy_predictor, make_user):
    from apps.analysis.stats import summary
    call_command("seed_demo")
    from django.contrib.auth import get_user_model
    demo = get_user_model().objects.get(username="demo_user")
    data = summary(demo, 90)
    qs = Check.objects.filter(user=demo)
    assert data["totals"]["total"] == qs.count() == 30
    assert data["totals"]["safe"] == qs.filter(category="SAFE").count()
    assert data["totals"]["fraud"] == qs.filter(category__in=["FRAUDULENT", "DANGEROUS"]).count()
    assert data["totals"]["url"] == qs.filter(check_type="URL").count()
    assert sum(data["risk_levels"].values()) == 30
    assert len(data["top_features"]) <= 5


def test_empty_dashboard_state(db, make_user):
    from apps.analysis.stats import summary
    data = summary(make_user("empty"), 30)
    assert data["has_checks"] is False and data["totals"]["total"] == 0 and data["totals"]["avg_score"] is None
