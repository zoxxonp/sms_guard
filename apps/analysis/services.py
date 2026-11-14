"""Сервис совместного анализа SMS и URL (задание 12).

    analyze_sms(text) -> AnalysisResult   — текст SMS: SMS-модель + словарь правил + каждая найденная ссылка
    analyze_url(url)  -> AnalysisResult   — отдельная ссылка без текста

Ни одна ссылка не открывается: URL анализируются только как строки (ml.url_features).
Модели берутся из синглтона ml.predictor.Predictor (загружаются один раз).
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime

from django.utils import timezone

from ml.language import LANGUAGE_NAMES, detect_language
from ml.predictor import ModelUnavailableError, Predictor, get_predictor  # noqa: F401 (реэкспорт)
from ml.risk import (CATEGORY_NAMES_RU, DISCLAIMER, LEVEL_NAMES_RU, URL_CATEGORY_NAMES_RU, compute_risk,
                     load_config, url_probability_to_score)
from ml.rules import GROUP_TITLES, match_rules
from ml.url_features import (InvalidURLError, explain_url, extract_url_features, registered_domain,
                             suspicious_words_in, validate_url)

MAX_SMS_LENGTH = 1000
MAX_URLS = 3

FEATURE_TITLES = {**GROUP_TITLES, "LINK_REQUEST": "Требование перейти по ссылке"}

# --- Извлечение URL из текста -------------------------------------------------------------
_TLDS = (
    "com|net|org|info|biz|ru|kz|su|ua|by|uz|kg|рф|қаз|бел|укр|uk|co|io|me|ly|gl|cc|tk|ml|ga|cf|gq|pw|xyz|top|club|"
    "online|site|app|link|click|live|dev|page|ai|gg|cn|eu|edu|gov|mobi|asia|tech|space|website|win|bid|loan|work|icu|"
    "buzz|monster|cfd|sbs|zip|mov|shop|store|pro|tv|ws|to|de|fr|in|us|lt|lv|ee|pl|cz|tr|ir|br|jp|kr|es|it|nl|be|ch|"
    "at|se|no|fi|dk|au|ca|nz|za|mx|ar|vn|th|id|my|ph|sg|hk|tw|il|ae|sa|eg|ng|ke|example|test"
)
_URL_RE = re.compile(
    r"(?:https?://[^\s<>\"'«»]+"
    r"|www\.[^\s<>\"'«»]+"
    r"|(?<![@\w.\-])(?:[a-zа-яё0-9](?:[a-zа-яё0-9\-]{0,62})\.)+(?:" + _TLDS + r")(?![\w\-])(?::\d{1,5})?(?:/[^\s<>\"'«»]*)?)",
    re.IGNORECASE,
)
_TRAILING = ".,;:!?)]}»\"'…"

_LINK_REQUEST_RE = re.compile(
    r"перейд|по ссылке|по ссылк|нажмите|откройте|подтверд|оплат\w* по|введите|"
    r"сілтеме|өтіңіз|басыңыз|растаңыз|"
    r"\bclick\b|\bvisit\b|\bfollow\b|\btap\b|\bgo to\b|\blog ?in\b|\bverify\b|\bclaim\b|\bconfirm\b",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class AnalysisInputError(ValueError):
    """Некорректный ввод; сообщение на русском языке показывается пользователю."""


def clean_text(text) -> str:
    """Удаляет управляющие символы (кроме перевода строки и табуляции) и пробелы по краям."""
    if not isinstance(text, str):
        raise AnalysisInputError("Текст SMS должен быть строкой")
    text = unicodedata.normalize("NFC", text)
    return _CONTROL_RE.sub("", text).strip()


def extract_urls(text: str, limit: int = MAX_URLS) -> list[str]:
    """Ссылки из текста: со схемой, www., без схемы (bit.ly/..., домены с кириллицей); не более limit."""
    found: list[str] = []
    for match in _URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(_TRAILING)
        if url and url.lower() not in (u.lower() for u in found) and validate_url(url).ok:
            found.append(url)
        if len(found) >= limit:
            break
    return found


def detect_social_engineering(text: str, urls: list[str]) -> dict[str, list[str]]:
    """Группы словаря правил + LINK_REQUEST (есть ссылка и призыв перейти по ней)."""
    groups = match_rules(text)
    if urls:
        without_urls = text
        for u in urls:
            without_urls = without_urls.replace(u, " ")
        m = _LINK_REQUEST_RE.search(without_urls)
        if m:
            groups["LINK_REQUEST"] = [m.group(0)]
    return groups


# --- Результат ----------------------------------------------------------------------------
@dataclass
class UrlResult:
    url: str
    domain: str
    p_url: float                 # вероятность URL-модели
    category: str
    category_ru: str
    reasons: list[str]
    suspicious_words: list[str]
    features: dict
    trusted: bool = False        # официальный домен из списка risk_config.yaml

    @property
    def p_effective(self) -> float:
        """Вероятность для расчёта риска: для доверенного домена — 0."""
        return 0.0 if self.trusted else self.p_url


def is_trusted(url: str) -> bool:
    cfg = load_config().get("trusted_domains", {})
    if not cfg.get("enabled"):
        return False
    feats_ok = validate_url(url)
    if not feats_ok.ok:
        return False
    return registered_domain(url) in set(cfg.get("domains", []))


@dataclass
class AnalysisResult:
    type: str                                   # SMS / URL
    input_text: str
    language: str | None
    language_name: str | None
    language_note: str | None
    category: str
    category_ru: str
    risk_level: str
    risk_level_ru: str
    risk_score: int
    p_sms: float | None
    p_url_max: float | None
    urls: list[UrlResult]
    features: list[str]
    features_detail: list[dict]
    explanation: list[str]
    recommendation: str
    disclaimer: str
    model_versions: dict
    base_score: float
    applied_overrides: list[str]
    processing_ms: float
    created_at: datetime = field(default_factory=timezone.now)
    id: int | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        for u in data["urls"]:
            u.pop("features", None)
        return {"id": self.id, **{k: v for k, v in data.items() if k != "id"}}


def _url_result(url: str, predictor: Predictor) -> UrlResult:
    features = extract_url_features(url)
    p_url = predictor.predict_url_proba(url)
    trusted = is_trusted(url)
    category = "SAFE" if trusted else predictor.url_category(p_url)
    reasons = explain_url(features, url)
    if trusted:
        reasons = [f"Официальный домен {registered_domain(url)} из списка доверенных"] + reasons
    return UrlResult(url=url, domain=validate_url(url).host, p_url=round(p_url, 4), category=category,
                     category_ru=URL_CATEGORY_NAMES_RU[category], reasons=reasons,
                     suspicious_words=suspicious_words_in(url), features=features, trusted=trusted)


def _language_note(lang: str, predictor: Predictor) -> str | None:
    trained = set(predictor.sms_meta.get("languages", ["en"]))
    if lang not in trained:
        return ("Язык сообщения не представлен в обучающих данных — оценка текста менее надёжна. "
                "Работают словарь признаков и анализ ссылок.")
    if lang in ("ru", "kk"):
        return ("Для русского и казахского модель обучена на искусственных сообщениях; ложные срабатывания "
                "на этих языках встречаются чаще, чем на английском.")
    return None


def analyze_sms(text: str, predictor: Predictor | None = None) -> AnalysisResult:
    start = time.perf_counter()
    text = clean_text(text)
    if not text:
        raise AnalysisInputError("Текст SMS не может быть пустым")
    if len(text) > MAX_SMS_LENGTH:
        raise AnalysisInputError(f"Текст SMS длиннее {MAX_SMS_LENGTH} символов")
    predictor = predictor or get_predictor()

    lang = detect_language(text)
    urls = extract_urls(text)
    p_sms = predictor.predict_sms_proba(text)
    groups = detect_social_engineering(text, urls)

    url_results = [_url_result(u, predictor) for u in urls] if urls else []
    p_url_max = max((u.p_effective for u in url_results), default=None)
    riskiest = max(url_results, key=lambda u: u.p_effective) if url_results else None
    flags = {"has_ip": int(any(u.features["has_ip"] for u in url_results)),
             "has_punycode": int(any(u.features["has_punycode"] for u in url_results))}
    url_reasons = [f"{riskiest.domain}: {r[0].lower() + r[1:]}" for r in riskiest.reasons] if riskiest else []

    risk = compute_risk(p_sms, p_url_max, groups, flags, url_t2=predictor.url_thresholds[1],
                        se_titles=FEATURE_TITLES, url_reasons=url_reasons, kind="SMS")
    return AnalysisResult(
        type="SMS", input_text=text, language=lang, language_name=LANGUAGE_NAMES[lang],
        language_note=_language_note(lang, predictor), category=risk.category,
        category_ru=CATEGORY_NAMES_RU[risk.category], risk_level=risk.level,
        risk_level_ru=LEVEL_NAMES_RU[risk.level], risk_score=risk.score, p_sms=round(p_sms, 4),
        p_url_max=p_url_max, urls=url_results, features=list(groups),
        features_detail=[{"code": c, "title": FEATURE_TITLES.get(c, c), "phrases": ph} for c, ph in groups.items()],
        explanation=risk.reasons, recommendation=risk.recommendation, disclaimer=DISCLAIMER,
        model_versions=predictor.versions(), base_score=risk.base_score, applied_overrides=risk.applied_overrides,
        processing_ms=round((time.perf_counter() - start) * 1000, 1),
    )


def analyze_url(url: str, predictor: Predictor | None = None) -> AnalysisResult:
    start = time.perf_counter()
    if not isinstance(url, str):
        raise AnalysisInputError("Ссылка должна быть строкой")
    url = _CONTROL_RE.sub("", url).strip()
    check = validate_url(url)
    if not check.ok:
        raise AnalysisInputError(check.error)
    predictor = predictor or get_predictor()
    try:
        result = _url_result(url, predictor)
    except InvalidURLError as exc:
        raise AnalysisInputError(str(exc)) from exc
    t1, t2 = predictor.url_thresholds
    score = url_probability_to_score(result.p_effective, t1, t2)
    risk = compute_risk(None, result.p_effective, {}, result.features, url_only_score=score,
                        url_reasons=result.reasons, kind="URL")
    category = {"LOW": "SAFE", "MEDIUM": "SUSPICIOUS", "HIGH": "DANGEROUS"}[risk.level]
    return AnalysisResult(
        type="URL", input_text=url, language=None, language_name=None, language_note=None,
        category=category, category_ru=URL_CATEGORY_NAMES_RU[category], risk_level=risk.level,
        risk_level_ru=LEVEL_NAMES_RU[risk.level], risk_score=risk.score, p_sms=None, p_url_max=result.p_url,
        urls=[result], features=[k for k in ("has_ip", "has_punycode", "mixed_scripts", "brand_similarity",
                                            "brand_in_url", "is_shortener", "suspicious_tld") if result.features[k]],
        features_detail=[], explanation=risk.reasons, recommendation=risk.recommendation, disclaimer=DISCLAIMER,
        model_versions=predictor.versions(), base_score=risk.base_score, applied_overrides=risk.applied_overrides,
        processing_ms=round((time.perf_counter() - start) * 1000, 1),
    )
