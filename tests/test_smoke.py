"""Дымовые тесты каркаса (задание 2)."""
import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_home_page_opens(client):
    response = client.get(reverse("dashboard:home"))
    assert response.status_code == 200
    assert "SMS Guard" in response.content.decode("utf-8")


def test_health_endpoint(client, use_toy_predictor):
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models"]["sms"]["loaded"] and data["models"]["url"]["loaded"]
    assert "error" not in data["models"]["sms"]


def test_session_cookie_settings(settings):
    assert settings.SESSION_COOKIE_HTTPONLY is True
    assert settings.SESSION_COOKIE_SAMESITE == "Lax"
