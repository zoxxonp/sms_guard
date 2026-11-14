"""Сохранение результата проверки в БД (задание 16, п. 3–4) и восстановление ответа API из БД."""
from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.accounts.models import get_profile
from ml.risk import CATEGORY_NAMES_RU, DISCLAIMER, URL_CATEGORY_NAMES_RU, load_config

from .masking import mask_text
from .models import Check, DetectedFeature, ModelVersion, UrlResult

logger = logging.getLogger("sms_guard.storage")

# Повторная отправка той же формы в течение этого времени не создаёт новую запись (НФТ-3)
DUPLICATE_WINDOW = timedelta(seconds=10)

URL_FLAG_TITLES = {
    "has_ip": "IP-адрес вместо домена",
    "has_punycode": "Punycode (xn--) в домене",
    "mixed_scripts": "Смешение алфавитов в домене",
    "brand_similarity": "Домен похож на известный бренд",
    "brand_in_url": "Название бренда в чужом домене",
    "is_shortener": "Сервис сокращения ссылок",
    "suspicious_tld": "Подозрительная доменная зона",
}

LEVEL_NAMES_RU = {"LOW": "низкий", "MEDIUM": "средний", "HIGH": "высокий"}


def input_hash(user_id: int, check_type: str, text: str) -> str:
    """HMAC (а не простой хеш), чтобы по значению нельзя было подобрать текст SMS перебором."""
    msg = f"{user_id}:{check_type}:{text}".encode("utf-8")
    return hmac.new(settings.SECRET_KEY.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def find_recent_duplicate(user, check_type: str, text: str) -> Check | None:
    since = timezone.now() - DUPLICATE_WINDOW
    return (Check.objects.filter(user=user, check_type=check_type, input_hash=input_hash(user.pk, check_type, text),
                                 created_at__gte=since).order_by("-created_at").first())


def sync_model_versions(predictor) -> None:
    """Регистрирует активные версии моделей из метаданных (таблица model_versions)."""
    for kind in ("sms", "url"):
        meta = getattr(predictor, f"{kind}_meta", {}) or {}
        if not meta.get("version"):
            continue
        obj, _ = ModelVersion.objects.get_or_create(
            model_type=kind, version=meta["version"], trained_at=str(meta.get("trained_at", "")),
            defaults={"algorithm": meta.get("algorithm", ""), "metrics_json": meta.get("metrics_test", {}),
                      "threshold": meta.get("threshold")})
        if not obj.is_active:
            ModelVersion.objects.filter(model_type=kind, is_active=True).update(is_active=False)
            obj.is_active = True
            obj.save(update_fields=["is_active"])


def save_result(user, result, predictor=None, created_at=None) -> Check:
    """Сохраняет AnalysisResult одной транзакцией: проверка, ссылки, признаки."""
    store_text = get_profile(user).store_sms_text
    weights = load_config()["weights"]
    with transaction.atomic():
        check = Check.objects.create(
            user=user, check_type=result.type,
            input_text=result.input_text if store_text else None,
            input_masked=mask_text(result.input_text),
            input_hash=input_hash(user.pk, result.type, result.input_text),
            language=result.language, category=result.category, risk_level=result.risk_level,
            risk_score=result.risk_score, p_sms=result.p_sms, p_url_max=result.p_url_max,
            explanation=result.explanation, recommendation=result.recommendation,
            language_note=result.language_note,
            sms_model_version=result.model_versions.get("sms"), url_model_version=result.model_versions.get("url"),
            processing_ms=result.processing_ms,
        )
        if created_at is not None:                      # для seed_demo: даты в прошлом
            Check.objects.filter(pk=check.pk).update(created_at=created_at)
            check.created_at = created_at
        UrlResult.objects.bulk_create([
            UrlResult(check_ref=check, url=u.url if store_text else mask_text(u.url), domain=u.domain,
                      p_url=u.p_url, category=u.category,
                      features_json={"reasons": u.reasons, "suspicious_words": u.suspicious_words,
                                     "trusted": u.trusted, "features": u.features})
            for u in result.urls
        ])
        if result.type == "SMS":
            per_group = weights["w_se"] / 3
            features = [DetectedFeature(check_ref=check, code=f["code"], title=f["title"], weight=round(per_group, 4))
                        for f in result.features_detail]
        else:
            features = [DetectedFeature(check_ref=check, code=code.upper(), title=URL_FLAG_TITLES.get(code, code),
                                        weight=0) for code in result.features]
        DetectedFeature.objects.bulk_create(features)
        if predictor is not None:
            sync_model_versions(predictor)
    return check


def check_to_dict(check: Check, include_text: bool = True) -> dict:
    """Ответ API (формат раздела 4.4) по сохранённой записи."""
    urls = list(check.url_results.all())
    features = list(check.detected_features.all())
    is_url = check.check_type == "URL"
    names = URL_CATEGORY_NAMES_RU if is_url else CATEGORY_NAMES_RU
    return {
        "id": check.pk,
        "type": check.check_type,
        "input": check.display_text if include_text else None,
        "text_stored": check.input_text is not None,
        "language": check.language,
        "language_note": check.language_note,
        "category": check.category,
        "category_ru": names.get(check.category, check.category),
        "risk_level": check.risk_level,
        "risk_level_ru": LEVEL_NAMES_RU.get(check.risk_level, check.risk_level),
        "risk_score": check.risk_score,
        "p_sms": check.p_sms,
        "p_url_max": check.p_url_max,
        "urls": [{"url": u.url, "domain": u.domain, "p_url": u.p_url, "category": u.category,
                  "category_ru": URL_CATEGORY_NAMES_RU.get(u.category, u.category),
                  "reasons": u.features_json.get("reasons", []),
                  "suspicious_words": u.features_json.get("suspicious_words", []),
                  "trusted": u.features_json.get("trusted", False),
                  "features": u.features_json.get("features", {}) if is_url else None} for u in urls],
        "features": [f.code for f in features],
        "features_detail": [{"code": f.code, "title": f.title} for f in features],
        "explanation": check.explanation,
        "recommendation": check.recommendation,
        "disclaimer": DISCLAIMER,
        "model_versions": {"sms": check.sms_model_version, "url": check.url_model_version},
        "processing_ms": check.processing_ms,
        "created_at": timezone.localtime(check.created_at).isoformat(),
    }


def check_summary(check: Check, include_text: bool = True) -> dict:
    """Строка истории: без причин и ссылок."""
    names = URL_CATEGORY_NAMES_RU if check.check_type == "URL" else CATEGORY_NAMES_RU
    return {"id": check.pk, "type": check.check_type, "fragment": check.fragment if include_text else None,
            "category": check.category, "category_ru": names.get(check.category, check.category),
            "risk_level": check.risk_level, "risk_level_ru": LEVEL_NAMES_RU.get(check.risk_level),
            "risk_score": check.risk_score, "created_at": timezone.localtime(check.created_at).isoformat()}
