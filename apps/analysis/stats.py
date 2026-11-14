"""Сводная статистика для Dashboard (задание 17). Вся агрегация выполняется в БД."""
from __future__ import annotations

from datetime import timedelta

from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone

from .models import FRAUD_CATEGORIES, Category, Check, DetectedFeature, ModelVersion, RiskLevel
from .storage import check_summary

PERIODS = (7, 30, 90)
GROUPS = ("SAFE", "SUSPICIOUS", "FRAUD")          # опасные ссылки считаются вместе с мошенническими SMS
GROUP_NAMES = {"SAFE": "Безопасные", "SUSPICIOUS": "Подозрительные", "FRAUD": "Мошеннические / опасные"}


def parse_period(value, default: int = 30) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value in PERIODS else default


def model_card(m: ModelVersion) -> dict:
    """Основные метрики на test: для SMS — все языки, для URL — при пороге t1."""
    main = m.metrics_json.get("test_all") or m.metrics_json.get("at_t1") or m.metrics_json
    return {"type": m.model_type, "version": m.version, "algorithm": m.algorithm, "threshold": m.threshold,
            "trained_at": m.trained_at,
            "metrics": {k: main.get(k) for k in ("precision", "recall", "f1", "roc_auc") if k in main}}


def summary(user, days: int = 30, all_users: bool = False) -> dict:
    today = timezone.localdate()
    start_date = today - timedelta(days=days - 1)
    start = timezone.make_aware(timezone.datetime.combine(start_date, timezone.datetime.min.time()))
    base = Check.objects.all() if all_users else Check.objects.filter(user=user)
    qs = base.filter(created_at__gte=start)

    fraud_q = Q(category__in=FRAUD_CATEGORIES)
    totals = qs.aggregate(
        total=Count("id"),
        sms=Count("id", filter=Q(check_type="SMS")),
        url=Count("id", filter=Q(check_type="URL")),
        safe=Count("id", filter=Q(category=Category.SAFE)),
        suspicious=Count("id", filter=Q(category=Category.SUSPICIOUS)),
        fraud=Count("id", filter=fraud_q),
        avg_score=Avg("risk_score"),
    )
    totals["avg_score"] = round(totals["avg_score"], 1) if totals["avg_score"] is not None else None

    per_day = (qs.annotate(day=TruncDate("created_at"))
               .values("day")
               .annotate(SAFE=Count("id", filter=Q(category=Category.SAFE)),
                         SUSPICIOUS=Count("id", filter=Q(category=Category.SUSPICIOUS)),
                         FRAUD=Count("id", filter=fraud_q))
               .order_by("day"))
    by_day = {row["day"]: row for row in per_day}
    dates = [start_date + timedelta(days=i) for i in range(days)]
    daily = {"dates": [d.isoformat() for d in dates],
             **{g: [by_day.get(d, {}).get(g, 0) for d in dates] for g in GROUPS}}

    levels = dict(qs.values_list("risk_level").annotate(n=Count("id")).order_by())
    top_features = list(DetectedFeature.objects.filter(check_ref__in=qs.filter(check_type="SMS"))
                        .values("code", "title").annotate(count=Count("id")).order_by("-count", "code")[:5])

    recent_qs = base.order_by("-created_at", "-id").select_related("user")[:10]
    recent = []
    for c in recent_qs:
        own = c.user_id == user.pk
        row = check_summary(c, include_text=own)       # администратор не видит тексты чужих SMS (ФТ-1.9)
        row["own"] = own
        if all_users:
            row["username"] = c.user.username
        recent.append(row)

    data = {
        "period_days": days,
        "scope": "all" if all_users else "user",
        "totals": totals,
        "categories": {g: totals[g.lower()] for g in GROUPS},
        "category_names": GROUP_NAMES,
        "risk_levels": {lvl: levels.get(lvl, 0) for lvl in RiskLevel.values},
        "daily": daily,
        "top_features": top_features,
        "recent": recent,
        "has_checks": base.exists(),
    }
    if all_users:
        data["models"] = [model_card(m) for m in ModelVersion.objects.filter(is_active=True)]
        data["users_with_checks"] = base.values("user").distinct().count()
    return data
