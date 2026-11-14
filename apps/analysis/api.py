"""REST API проверок, истории и статистики (раздел 4.4 ТЗ, задание 14)."""
from __future__ import annotations

import logging
import time
from datetime import datetime

from django.core.paginator import EmptyPage, Paginator
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import get_predictor

from . import stats
from .apiutils import ApiError
from .models import Category, Check, CheckType, RiskLevel
from .services import MAX_SMS_LENGTH, AnalysisInputError, analyze_sms, analyze_url, clean_text
from .storage import check_summary, check_to_dict, find_recent_duplicate, save_result

logger = logging.getLogger("sms_guard.api")

MAX_URL_LENGTH = 2048
PAGE_SIZE = 20


class CleanCharField(serializers.CharField):
    """Управляющие символы (в т. ч. \x00) удаляются до проверок — бинарный мусор не даёт ошибку 500."""

    def run_validation(self, data=serializers.empty):
        if isinstance(data, str):
            data = clean_text(data)
        return super().run_validation(data)


class SmsInputSerializer(serializers.Serializer):
    text = CleanCharField(allow_blank=True, trim_whitespace=False, max_length=MAX_SMS_LENGTH * 4,
                                 error_messages={"required": "Текст SMS не может быть пустым",
                                                 "null": "Текст SMS не может быть пустым",
                                                 "invalid": "Текст SMS должен быть строкой",
                                                 "max_length": f"Текст SMS длиннее {MAX_SMS_LENGTH} символов"})

    def validate_text(self, value):
        value = clean_text(value)           # управляющие символы удаляются
        if not value:
            raise serializers.ValidationError("Текст SMS не может быть пустым")
        if len(value) > MAX_SMS_LENGTH:
            raise serializers.ValidationError(
                f"Текст SMS длиннее {MAX_SMS_LENGTH} символов (сейчас {len(value)})")
        return value


class UrlInputSerializer(serializers.Serializer):
    url = CleanCharField(allow_blank=True, trim_whitespace=True, max_length=MAX_URL_LENGTH * 2,
                                error_messages={"required": "Введите ссылку", "null": "Введите ссылку",
                                                "invalid": "Ссылка должна быть строкой",
                                                "max_length": f"Ссылка длиннее {MAX_URL_LENGTH} символов"})

    def validate_url(self, value):
        value = clean_text(value)
        if not value:
            raise serializers.ValidationError("Введите ссылку")
        if len(value) > MAX_URL_LENGTH:
            raise serializers.ValidationError(f"Ссылка длиннее {MAX_URL_LENGTH} символов")
        return value


class _CheckView(APIView):
    """Общий код check/sms и check/url: анализ → сохранение → JSON. Лимит 30 проверок в минуту."""

    throttle_scope = "checks"
    serializer_class: type[serializers.Serializer]
    field: str
    check_type: str

    def analyze(self, value, predictor):
        raise NotImplementedError

    def post(self, request):
        started = time.perf_counter()
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        value = serializer.validated_data[self.field]

        duplicate = find_recent_duplicate(request.user, self.check_type, value)
        if duplicate is not None:                   # двойной клик / повторная отправка (НФТ-3)
            logger.info("check id=%s type=%s повтор запроса — возвращена существующая запись",
                        duplicate.pk, self.check_type)
            return Response({**check_to_dict(duplicate), "duplicate": True})

        predictor = get_predictor()
        try:
            result = self.analyze(value, predictor)
        except AnalysisInputError as exc:
            raise ApiError(str(exc))
        check = save_result(request.user, result, predictor)
        logger.info("check id=%s type=%s category=%s risk=%s analysis_ms=%.1f total_ms=%.1f",
                    check.pk, check.check_type, check.category, check.risk_score, result.processing_ms,
                    (time.perf_counter() - started) * 1000)
        return Response(check_to_dict(check), status=status.HTTP_200_OK)


class CheckSmsView(_CheckView):
    """POST /api/v1/check/sms/ {"text": "..."}"""

    serializer_class = SmsInputSerializer
    field = "text"
    check_type = "SMS"

    def analyze(self, value, predictor):
        return analyze_sms(value, predictor)


class CheckUrlView(_CheckView):
    """POST /api/v1/check/url/ {"url": "..."}"""

    serializer_class = UrlInputSerializer
    field = "url"
    check_type = "URL"

    def analyze(self, value, predictor):
        return analyze_url(value, predictor)


def _parse_date(value: str | None, name: str):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ApiError(f"Параметр {name}: дата в формате ГГГГ-ММ-ДД")


def filter_checks(user, params) -> tuple:
    """Фильтры истории — общие для API и страницы /history/. Выборка всегда ограничена user."""
    qs = Check.objects.filter(user=user)
    date_from = _parse_date(params.get("date_from"), "date_from")
    date_to = _parse_date(params.get("date_to"), "date_to")
    if date_from:
        qs = qs.filter(created_at__date__gte=date_from)
    if date_to:
        qs = qs.filter(created_at__date__lte=date_to)
    check_type = (params.get("type") or "").upper()
    if check_type:
        if check_type not in CheckType.values:
            raise ApiError("Параметр type: SMS или URL")
        qs = qs.filter(check_type=check_type)
    category = (params.get("category") or "").upper()
    if category:
        if category not in Category.values:
            raise ApiError("Параметр category: SAFE, SUSPICIOUS, FRAUDULENT или DANGEROUS")
        qs = qs.filter(category=category)
    level = (params.get("risk_level") or "").upper()
    if level:
        if level not in RiskLevel.values:
            raise ApiError("Параметр risk_level: LOW, MEDIUM или HIGH")
        qs = qs.filter(risk_level=level)
    return qs.order_by("-created_at", "-id"), {"date_from": date_from, "date_to": date_to, "type": check_type,
                                                "category": category, "risk_level": level}


class CheckListView(APIView):
    """GET /api/v1/checks/?date_from=&date_to=&type=&category=&risk_level=&page="""

    def get(self, request):
        qs, _ = filter_checks(request.user, request.query_params)
        paginator = Paginator(qs, PAGE_SIZE)
        try:
            page_number = int(request.query_params.get("page", 1))
            page = paginator.page(page_number)
        except (ValueError, EmptyPage):
            raise ApiError("Страница не найдена", code="NOT_FOUND", status_code=404)
        return Response({"count": paginator.count, "page": page.number, "pages": paginator.num_pages,
                         "page_size": PAGE_SIZE, "results": [check_summary(c) for c in page.object_list]})


class CheckDetailView(APIView):
    """GET / DELETE /api/v1/checks/{id}/ — только своя запись, иначе 404."""

    def get_object(self, request, pk):
        return get_object_or_404(Check.objects.prefetch_related("url_results", "detected_features"),
                                 pk=pk, user=request.user)

    def get(self, request, pk):
        return Response(check_to_dict(self.get_object(request, pk)))

    def delete(self, request, pk):
        check = self.get_object(request, pk)
        check_id = check.pk
        check.delete()
        logger.info("check id=%s удалена пользователем", check_id)
        return Response(status=status.HTTP_204_NO_CONTENT)


class StatsSummaryView(APIView):
    """GET /api/v1/stats/summary/?days=7|30|90 — администратору по всем пользователям."""

    def get(self, request):
        days = stats.parse_period(request.query_params.get("days"))
        all_users = request.user.is_staff and request.query_params.get("scope", "all") != "user"
        data = stats.summary(request.user, days, all_users=all_users)
        data["generated_at"] = timezone.localtime().isoformat()
        return Response(data)
