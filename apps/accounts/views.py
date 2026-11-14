"""Страницы входа, регистрации, выхода и профиля."""
import logging

from django.contrib import messages
from django.contrib.auth import login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_http_methods, require_POST

from apps.analysis.models import Check

from .forms import ConsentForm, DeleteHistoryForm, GuardPasswordChangeForm, LoginForm, RegisterForm
from .models import get_profile

logger = logging.getLogger("sms_guard.auth")


def _safe_next(request) -> str:
    target = request.POST.get("next") or request.GET.get("next") or "/"
    if url_has_allowed_host_and_scheme(target, allowed_hosts={request.get_host()},
                                       require_https=request.is_secure()):
        return target
    return "/"


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")
    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.user)
        logger.info("Вход пользователя id=%s", form.user.pk)
        return redirect(_safe_next(request))
    status = 429 if any(e.code == "locked" for e in form.non_field_errors().as_data()) else 200
    return render(request, "accounts/login.html", {"form": form, "next": _safe_next(request)}, status=status)


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        logger.info("Регистрация пользователя id=%s", user.pk)
        messages.success(request, "Аккаунт создан. Добро пожаловать в SMS Guard!")
        return redirect("dashboard:home")
    return render(request, "accounts/register.html", {"form": form})


@require_POST
def logout_view(request):
    logout(request)
    messages.info(request, "Вы вышли из системы.")
    return redirect("accounts:login")


@login_required
@require_http_methods(["GET", "POST"])
def profile_view(request):
    profile = get_profile(request.user)
    action = request.POST.get("action") if request.method == "POST" else None
    consent_form = ConsentForm(request.POST if action == "consent" else None,
                               initial={"store_sms_text": profile.store_sms_text})
    password_form = GuardPasswordChangeForm(request.user, request.POST if action == "password" else None)
    delete_form = DeleteHistoryForm(request.POST if action == "delete_history" else None)

    if action == "consent" and consent_form.is_valid():
        profile.store_sms_text = consent_form.cleaned_data["store_sms_text"]
        profile.save(update_fields=["store_sms_text"])
        messages.success(request, "Согласие на хранение текстов " +
                         ("включено." if profile.store_sms_text else "выключено. Новые проверки хранятся в "
                                                                    "маскированном виде."))
        return redirect("accounts:profile")
    if action == "password" and password_form.is_valid():
        user = password_form.save()
        update_session_auth_hash(request, user)
        messages.success(request, "Пароль изменён.")
        return redirect("accounts:profile")
    if action == "delete_history" and delete_form.is_valid():
        deleted = Check.objects.filter(user=request.user).count()
        Check.objects.filter(user=request.user).delete()
        logger.info("Пользователь id=%s удалил историю: %s записей", request.user.pk, deleted)
        messages.success(request, f"История удалена: {deleted} записей.")
        return redirect("accounts:profile")

    return render(request, "accounts/profile.html", {
        "profile": profile, "consent_form": consent_form, "password_form": password_form,
        "delete_form": delete_form, "checks_count": Check.objects.filter(user=request.user).count(),
        "active": "profile",
    })


def csrf_failure(request, reason=""):
    """CSRF-ошибка: JSON для API (единый формат), понятная страница для форм."""
    if request.path.startswith("/api/"):
        return JsonResponse({"error": {"code": "CSRF_FAILED",
                                       "message": "Запрос отклонён: отсутствует или неверен CSRF-токен."}},
                            status=403)
    return render(request, "403_csrf.html", status=403)
