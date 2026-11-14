"""Страницы проверки, результата и истории + эндпоинт health."""
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import get_predictor

from .api import PAGE_SIZE, ApiError, filter_checks
from .models import Category, Check, CheckType, RiskLevel
from .storage import check_to_dict

SMS_EXAMPLES = [
    {"label": "Безопасное", "icon": "bi-shield-check",
     "text": "Привет! Встречаемся завтра в 18:00 у входа в библиотеку."},
    {"label": "Подозрительное", "icon": "bi-exclamation-triangle",
     "text": "Ваша посылка ожидает оплаты таможенного сбора 350 T: http://post-delivery-pay.example"},
    {"label": "Мошенническое", "icon": "bi-x-octagon",
     "text": "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: "
             "http://secure-bank-verify.example/login"},
]

URL_EXAMPLES = ["https://www.example.com/about", "https://login-account.example.com/update",
                "http://xn--exmple-cua.test/secure/login?verify=1"]


class HealthView(APIView):
    """GET /api/v1/health/ — статус приложения и загруженных моделей. Доступен без авторизации."""

    permission_classes = [AllowAny]
    authentication_classes: list = []

    def get(self, request):
        # Текст ошибки (пути к файлам) наружу не отдаётся — он есть в logs/app.log
        status = {kind: {k: v for k, v in info.items() if k != "error"}
                  for kind, info in get_predictor().status().items()}
        ok = status["sms"]["loaded"] and status["url"]["loaded"]
        return Response({"status": "ok" if ok else "degraded", "models": status})


def _models_status():
    p = get_predictor()
    return {"sms": p.sms_ready, "url": p.url_ready}


@login_required
def check_sms_page(request):
    return render(request, "analysis/check_sms.html", {"examples": SMS_EXAMPLES, "models": _models_status(),
                                                       "active": "check_sms", "max_length": 1000})


@login_required
def check_url_page(request):
    return render(request, "analysis/check_url.html", {"examples": URL_EXAMPLES, "models": _models_status(),
                                                       "active": "check_url"})


@login_required
def check_detail_page(request, pk):
    check = get_object_or_404(Check.objects.prefetch_related("url_results", "detected_features"),
                              pk=pk, user=request.user)
    return render(request, "analysis/check_detail.html", {"check": check, "data": check_to_dict(check),
                                                          "active": "history"})


@login_required
@require_http_methods(["GET", "POST"])
def check_delete_page(request, pk):
    check = get_object_or_404(Check, pk=pk, user=request.user)
    if request.method == "POST":
        check.delete()
        messages.success(request, f"Проверка №{pk} удалена.")
        return redirect("analysis:history")
    return render(request, "analysis/check_delete.html", {"check": check, "active": "history"})


@login_required
def history_page(request):
    params = request.GET
    error = None
    try:
        qs, filters = filter_checks(request.user, params)
    except ApiError as exc:
        error = str(exc.detail)
        qs, filters = Check.objects.filter(user=request.user).order_by("-created_at", "-id"), {}
    paginator = Paginator(qs, PAGE_SIZE)
    page = paginator.get_page(params.get("page"))
    query = params.copy()
    query.pop("page", None)
    return render(request, "analysis/history.html", {
        "page": page, "paginator": paginator,
        "page_range": paginator.get_elided_page_range(page.number, on_each_side=1, on_ends=1),
        "filters": filters, "raw": params, "query": query.urlencode(), "error": error,
        "types": CheckType.choices, "categories": Category.choices, "levels": RiskLevel.choices,
        "active": "history",
    })
