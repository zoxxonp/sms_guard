"""Дымовые тесты каркаса (задание 2)."""
import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_home_page_opens(client):
    response = client.get(reverse("dashboard:home"))
    assert response.status_code == 200
    assert "SMS Guard" in response.content.decode("utf-8")


def test_health_endpoint(client):
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_session_cookie_settings(settings):
    assert settings.SESSION_COOKIE_HTTPONLY is True
    assert settings.SESSION_COOKIE_SAMESITE == "Lax"
