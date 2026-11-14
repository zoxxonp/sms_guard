"""Аутентификация API: сессия + CSRF; без входа ответ 401 (раздел 4.4 ТЗ)."""
from rest_framework.authentication import SessionAuthentication


class SessionAuth401(SessionAuthentication):
    """Отдельный модуль без импорта rest_framework.views — иначе циклический импорт при загрузке настроек DRF."""

    def authenticate_header(self, request):
        return 'Session realm="api"'
