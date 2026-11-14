"""Единый формат ошибок REST API (раздел 4.4 ТЗ)."""
from __future__ import annotations

import logging

from django.core.exceptions import PermissionDenied
from django.http import Http404
from rest_framework import exceptions, status
from rest_framework.response import Response
from rest_framework.views import exception_handler

from ml.predictor import ModelUnavailableError

logger = logging.getLogger("sms_guard.api")


class ApiError(exceptions.APIException):
    def __init__(self, message: str, code: str = "VALIDATION_ERROR", status_code: int = 400):
        self.status_code = status_code
        super().__init__(detail=message, code=code)
        self.error_code = code


def _first_message(detail) -> str:
    if isinstance(detail, dict):
        for value in detail.values():
            return _first_message(value)
    if isinstance(detail, (list, tuple)) and detail:
        return _first_message(detail[0])
    return str(detail)


_CODES = {
    exceptions.ValidationError: ("VALIDATION_ERROR", None),
    exceptions.ParseError: ("VALIDATION_ERROR", "Некорректный формат запроса (ожидается JSON)."),
    exceptions.UnsupportedMediaType: ("VALIDATION_ERROR", "Неподдерживаемый тип данных: используйте JSON."),
    exceptions.NotAuthenticated: ("NOT_AUTHENTICATED", "Требуется вход в систему."),
    exceptions.AuthenticationFailed: ("NOT_AUTHENTICATED", "Ошибка аутентификации."),
    exceptions.PermissionDenied: ("PERMISSION_DENIED", "Недостаточно прав для этого действия."),
    exceptions.NotFound: ("NOT_FOUND", "Запись не найдена."),
    exceptions.MethodNotAllowed: ("METHOD_NOT_ALLOWED", "Метод не поддерживается."),
    exceptions.Throttled: ("THROTTLED", None),
}


def error_response(code: str, message: str, http_status: int, headers=None) -> Response:
    return Response({"error": {"code": code, "message": message}}, status=http_status, headers=headers)


def api_exception_handler(exc, context):
    """Все ошибки API: {"error": {"code": ..., "message": ...}}; 503, если модель не загружена."""
    if isinstance(exc, ModelUnavailableError):
        logger.error("Анализ недоступен: %s", exc)
        return error_response("MODEL_UNAVAILABLE",
                              "Анализ временно недоступен: модель не загружена. Обратитесь к администратору.",
                              status.HTTP_503_SERVICE_UNAVAILABLE)
    if isinstance(exc, Http404):
        exc = exceptions.NotFound()
    elif isinstance(exc, PermissionDenied):
        exc = exceptions.PermissionDenied()

    response = exception_handler(exc, context)
    if response is None:
        logger.exception("Необработанная ошибка API", exc_info=exc)
        return error_response("INTERNAL_ERROR", "Внутренняя ошибка сервера. Попробуйте позже.", 500)

    if isinstance(exc, ApiError):
        code, message = exc.error_code, str(exc.detail)
    else:
        code, message = "ERROR", None
        for cls, (c, m) in _CODES.items():
            if isinstance(exc, cls):
                code, message = c, m
                break
        if isinstance(exc, exceptions.Throttled):
            wait = int(exc.wait or 60)
            message = f"Превышен лимит: не более 30 проверок в минуту. Повторите через {wait} с."
        elif isinstance(exc, exceptions.PermissionDenied) and "CSRF" in str(exc.detail):
            code, message = "CSRF_FAILED", "Запрос отклонён: отсутствует или неверен CSRF-токен."
        if message is None:
            message = _first_message(exc.detail)
    headers = {k: v for k, v in response.items() if k in ("Retry-After", "WWW-Authenticate")}
    return error_response(code, message, response.status_code, headers=headers)
