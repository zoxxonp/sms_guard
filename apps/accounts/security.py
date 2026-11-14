"""Регистрация и вход: правила ФТ-1.1–ФТ-1.3, общие для страниц и API."""
from __future__ import annotations

import re

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.cache import cache
from django.core.exceptions import ValidationError

MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")


def _key(username: str) -> str:
    return f"login-fail:{(username or '').strip().lower()}"


def lockout_remaining(username: str) -> int:
    """Секунд до снятия блокировки (0 — вход разрешён)."""
    data = cache.get(_key(username))
    if not data or data["count"] < MAX_FAILED_LOGINS:
        return 0
    import time
    return max(0, int(data["locked_until"] - time.time()) + 1)


def register_failure(username: str) -> None:
    """5 неудачных попыток подряд → пауза 60 с (ФТ-1.3)."""
    import time
    data = cache.get(_key(username)) or {"count": 0, "locked_until": 0}
    data["count"] += 1
    if data["count"] >= MAX_FAILED_LOGINS:
        data["locked_until"] = time.time() + LOCKOUT_SECONDS
        cache.set(_key(username), data, LOCKOUT_SECONDS)
    else:
        cache.set(_key(username), data, 15 * 60)


def reset_failures(username: str) -> None:
    cache.delete(_key(username))


def lockout_message(seconds: int) -> str:
    return f"Слишком много неудачных попыток входа. Повторите через {seconds} с."


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise ValidationError("Логин: от 3 до 30 символов, только латинские буквы, цифры и «_».")
    if get_user_model().objects.filter(username__iexact=username).exists():
        raise ValidationError("Пользователь с таким логином уже существует.")
    return username


def validate_email_unique(email: str) -> str:
    email = (email or "").strip().lower()
    if get_user_model().objects.filter(email__iexact=email).exists():
        raise ValidationError("Этот e-mail уже зарегистрирован.")
    return email


def validate_new_password(password: str, user=None) -> str:
    """Не менее 8 символов, буквы и цифры (ФТ-1.1) + стандартные валидаторы Django."""
    errors = []
    if len(password or "") < 8:
        errors.append("Пароль должен содержать не менее 8 символов.")
    if not re.search(r"[A-Za-zА-Яа-яЁё]", password or "") or not re.search(r"\d", password or ""):
        errors.append("Пароль должен содержать буквы и цифры.")
    if errors:
        raise ValidationError(errors)
    validate_password(password, user)
    return password
