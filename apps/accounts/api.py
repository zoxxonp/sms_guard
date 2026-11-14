"""API регистрации, входа и выхода (раздел 4.4 ТЗ, ФТ-1.1–ФТ-1.8)."""
import logging

from django.contrib.auth import login, logout
from django.core.exceptions import ValidationError as DjangoValidationError
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from rest_framework import serializers, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.analysis.apiutils import ApiError, error_response

from . import security
from .forms import LoginForm

logger = logging.getLogger("sms_guard.auth")


def _django_errors(exc: DjangoValidationError) -> list[str]:
    return list(exc.messages)


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=30, error_messages={"blank": "Введите логин.",
                                                                    "required": "Введите логин."})
    email = serializers.EmailField(error_messages={"invalid": "Введите корректный e-mail.",
                                                   "required": "Введите e-mail.", "blank": "Введите e-mail."})
    password = serializers.CharField(write_only=True, trim_whitespace=False,
                                     error_messages={"required": "Введите пароль.", "blank": "Введите пароль."})

    def validate_username(self, value):
        try:
            return security.validate_username(value)
        except DjangoValidationError as exc:
            raise serializers.ValidationError(_django_errors(exc))

    def validate_email(self, value):
        try:
            return security.validate_email_unique(value)
        except DjangoValidationError as exc:
            raise serializers.ValidationError(_django_errors(exc))

    def validate(self, attrs):
        from django.contrib.auth import get_user_model
        try:
            security.validate_new_password(attrs["password"],
                                           get_user_model()(username=attrs["username"], email=attrs["email"]))
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"password": _django_errors(exc)})
        return attrs


def _user_data(user) -> dict:
    return {"id": user.pk, "username": user.username, "email": user.email, "is_staff": user.is_staff}


@method_decorator(csrf_protect, name="dispatch")
class RegisterView(APIView):
    """POST /api/v1/auth/register/ → 201. После регистрации пользователь сразу входит (сессия)."""

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        from django.contrib.auth import get_user_model
        user = get_user_model().objects.create_user(**serializer.validated_data)
        login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        logger.info("Регистрация пользователя id=%s", user.pk)
        return Response(_user_data(user), status=status.HTTP_201_CREATED)


@method_decorator(csrf_protect, name="dispatch")
class LoginView(APIView):
    """POST /api/v1/auth/login/ → 200 / 401; после 5 неудачных попыток подряд — 429 на 60 с."""

    permission_classes = [AllowAny]

    def post(self, request):
        username = str(request.data.get("username", "")).strip()
        password = str(request.data.get("password", ""))
        if not username or not password:
            raise ApiError("Введите логин и пароль.")
        remaining = security.lockout_remaining(username)
        if remaining:
            return error_response("TOO_MANY_ATTEMPTS", security.lockout_message(remaining), 429,
                                  headers={"Retry-After": str(remaining)})
        form = LoginForm(request, data={"username": username, "password": password})
        if not form.is_valid():
            remaining = security.lockout_remaining(username)
            logger.warning("Неудачный вход (логин скрыт), блокировка=%s", bool(remaining))
            if remaining:
                return error_response("TOO_MANY_ATTEMPTS", security.lockout_message(remaining), 429,
                                      headers={"Retry-After": str(remaining)})
            return error_response("INVALID_CREDENTIALS", "Неверный логин или пароль.", 401)
        login(request, form.user)
        logger.info("Вход пользователя id=%s", form.user.pk)
        return Response(_user_data(form.user))


class LogoutView(APIView):
    """POST /api/v1/auth/logout/ → 204, сессия уничтожается (ФТ-1.5)."""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


@method_decorator(ensure_csrf_cookie, name="dispatch")
class CsrfView(APIView):
    """GET /api/v1/auth/csrf/ — выдаёт CSRF-cookie для клиентов API (curl, тесты)."""

    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"detail": "CSRF cookie set"})
