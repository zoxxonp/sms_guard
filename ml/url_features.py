"""Извлечение признаков URL без обращения к сети (задание 9).

URL анализируется ТОЛЬКО как строка: модуль не открывает ссылку, не делает
HTTP-запросов, не разрешает DNS и не использует сокеты. tldextract работает со
встроенным в пакет снимком Public Suffix List (``suffix_list_urls=()``), поэтому
тоже не обращается к сети.

Основные функции:
    validate_url(url)          -> ValidationResult (ok, normalized, error)
    extract_url_features(url)  -> dict из FEATURE_NAMES (фиксированный порядок ключей)
    explain_url(features)      -> list[str] — причины на русском языке

Одна и та же функция используется для подготовки набора данных (задание 10)
и в веб-приложении (задание 12).
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import parse_qsl, urlsplit

import tldextract

__all__ = [
    "FEATURE_NAMES", "MODEL_FEATURES", "MODEL_TEXT_COLUMN", "canonical_url", "model_features", "model_input", "InvalidURLError", "ValidationResult", "validate_url", "extract_url_features",
    "explain_url", "registered_domain", "SHORTENERS", "BRAND_DOMAINS", "SUSPICIOUS_TLDS",
]

MAX_URL_LENGTH = 2048

# Порядок ключей фиксирован: от него зависит порядок столбцов матрицы признаков.
FEATURE_NAMES = [
    "url_length",            # 1  длина URL
    "host_length",           # 2  длина домена (host)
    "path_length",           # 3  длина пути
    "n_dots",                # 4  число точек в URL
    "n_subdomains",          # 5  число поддоменов (по tldextract), без «www»
    "n_hyphens_host",        # 6  число дефисов в домене
    "n_digits",              # 7  число цифр в URL
    "digit_ratio_host",      # 8  доля цифр в домене
    "n_special",             # 9  число спецсимволов @ - % = & _ ?
    "has_ip",                # 10 IP-адрес вместо домена (IPv4/IPv6, в т. ч. десятичная запись)
    "is_https",              # 11 схема HTTPS
    "nonstandard_port",      # 12 порт не 80/443
    "has_at",                # 13 символ @ в URL
    "double_slash_path",     # 14 двойной // в пути
    "is_shortener",          # 15 URL-сокращатель
    "n_suspicious_words",    # 16 число подозрительных слов
    "has_punycode",          # 17 Punycode (xn--) в домене
    "mixed_scripts",         # 18 смешение алфавитов в домене
    "brand_similarity",      # 19 домен похож на бренд (Левенштейн < 2), но не совпадает с ним
    "host_entropy",          # 20 энтропия Шеннона символов домена
    "suspicious_tld",        # 21 подозрительная доменная зона
    "n_query_params",        # 22 число параметров запроса
    "brand_in_url",          # 23 название бренда в поддомене/пути при чужом регистрируемом домене
    "path_depth",            # 24 число сегментов пути
]

# Признаки, которые получает ML-модель (задание 10). Признаки пути, схемы и символа @ исключены:
# в наборе PhiUSIIL ВСЕ легитимные URL имеют вид https://www.домен без пути, поэтому модель выучила бы
# правило «есть путь / нет https → фишинг». Модель обучается на канонической форме http://домен
# (см. canonical_url); исключённые признаки по-прежнему считаются и показываются в объяснении.
MODEL_FEATURES = [
    "host_length", "n_dots", "n_subdomains", "n_hyphens_host", "n_digits", "digit_ratio_host",
    "n_special", "has_ip", "nonstandard_port", "is_shortener", "n_suspicious_words", "has_punycode",
    "mixed_scripts", "brand_similarity", "host_entropy", "suspicious_tld", "brand_in_url",
]

# Не менее 15 сервисов сокращения ссылок
SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly", "ow.ly", "clck.ru", "buff.ly",
    "rebrand.ly", "tiny.cc", "shorturl.at", "rb.gy", "bitly.com", "v.gd", "s.id", "lnkd.in",
    "t.ly", "short.io", "u.to", "vk.cc", "qps.ru", "trib.al", "soo.gd", "bl.ink",
})

# Подозрительные слова в URL (ФТ-3.7)
SUSPICIOUS_WORDS = (
    "login", "signin", "logon", "verify", "verification", "secure", "security", "account", "update",
    "bank", "confirm", "password", "passwd", "bonus", "free", "prize", "gift", "claim", "winner",
    "unlock", "suspend", "wallet", "billing", "payment", "invoice", "support", "webscr", "auth",
    "recover", "restore", "kaspi", "halyk", "egov", "kazpost",
)
_SUSPICIOUS_RE = re.compile("|".join(sorted(SUSPICIOUS_WORDS, key=len, reverse=True)))

# Не менее 20 официальных доменов банков и популярных сервисов (для сходства с брендом)
BRAND_DOMAINS = (
    "kaspi.kz", "halykbank.kz", "jusan.kz", "bcc.kz", "forte.kz", "homecredit.kz", "freedombank.kz",
    "berekebank.kz", "egov.kz", "kazpost.kz", "olx.kz", "google.com", "gmail.com", "youtube.com",
    "facebook.com", "instagram.com", "whatsapp.com", "telegram.org", "apple.com", "icloud.com",
    "microsoft.com", "outlook.com", "paypal.com", "amazon.com", "netflix.com", "ebay.com",
    "sberbank.ru", "tinkoff.ru", "yandex.ru", "mail.ru", "vk.com", "wildberries.ru", "ozon.ru",
)
_BRAND_NAMES = {d.split(".")[0]: d for d in BRAND_DOMAINS}
# Слова-бренды для признака brand_in_url (≥ 4 символов, чтобы не ловить случайные совпадения)
# Общие слова (mail, outlook, forte) исключены: mail.google.com не должен считаться подделкой mail.ru
_GENERIC_BRAND_WORDS = {"mail", "outlook", "forte"}
_BRAND_TOKENS = sorted({n for n in _BRAND_NAMES if len(n) >= 4 and n not in _GENERIC_BRAND_WORDS}
                       | {"halyk", "sber", "kaspi"}, key=len, reverse=True)
_BRAND_TOKEN_RE = re.compile("|".join(_BRAND_TOKENS))

# Подозрительные доменные зоны: зоны из списков Spamhaus «The World's Most Abused TLDs»
# (публикации 2016–2020: .top, .click, .work, .download, .review, .loan, .men, .pw, .gq, .cf, .ga,
# .tk, .ml) и отчётов о злоупотреблениях доменами (.xyz, .live, .monster, .cfd, .buzz, .icu, .cyou,
# .rest, .sbs), а также .zip и .mov, которые легко спутать с именами файлов.
SUSPICIOUS_TLDS = frozenset({
    "top", "click", "work", "download", "review", "loan", "men", "pw", "gq", "cf", "ga", "tk", "ml",
    "xyz", "live", "monster", "cfd", "buzz", "icu", "cyou", "rest", "sbs", "zip", "mov", "country",
    "stream", "gdn", "racing", "win", "bid", "date", "faith", "party", "science", "cricket",
})

_SPECIAL_CHARS = "@-%=&_?"
_ALLOWED_SCHEMES = {"http", "https"}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://")
_HOST_CHARS_RE = re.compile(r"^[\w.\-]+$", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[а-яёәғқңөұүһі]", re.IGNORECASE)
_LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)

# Встроенный снимок Public Suffix List, без сетевого обновления
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=False)


class InvalidURLError(ValueError):
    """Некорректный URL; сообщение на русском языке предназначено для пользователя."""


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    normalized: str | None = None   # URL со схемой (http:// добавляется только для разбора)
    host: str | None = None
    error: str | None = None


def validate_url(url) -> ValidationResult:
    """Проверяет формат URL (ФТ-3.1) и возвращает понятную ошибку вместо исключения."""
    if not isinstance(url, str):
        return ValidationResult(False, error="URL должен быть строкой")
    url = url.strip()
    if not url:
        return ValidationResult(False, error="Введите ссылку")
    if len(url) > MAX_URL_LENGTH:
        return ValidationResult(False, error=f"Ссылка длиннее {MAX_URL_LENGTH} символов")
    if _CONTROL_RE.search(url) or " " in url:
        return ValidationResult(False, error="Ссылка содержит пробелы или управляющие символы")
    match = _SCHEME_RE.match(url)
    if match:
        if match.group(1).lower() not in _ALLOWED_SCHEMES:
            return ValidationResult(False, error="Допустимы только схемы http и https")
        normalized = url
    else:
        if url.startswith("//"):
            url = url[2:]
        normalized = "http://" + url
    try:
        parts = urlsplit(normalized)
        _ = parts.port                      # некорректный порт вызывает ValueError
        host = (parts.hostname or "").strip(".")
    except ValueError:
        return ValidationResult(False, error="Некорректный формат ссылки (порт или адрес)")
    if not host:
        return ValidationResult(False, error="В ссылке нет доменного имени")
    if not _is_ip(host):
        if "." not in host:
            return ValidationResult(False, error="Доменное имя должно содержать точку, например example.com")
        if not _HOST_CHARS_RE.match(host) or ".." in host:
            return ValidationResult(False, error="Доменное имя содержит недопустимые символы")
    return ValidationResult(True, normalized=normalized, host=host.lower())


def _is_ip(host: str) -> bool:
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host.isdigit() and 0 < int(host) < 2 ** 32:      # десятичная запись IPv4: http://3232235777/
        return True
    return bool(re.fullmatch(r"0x[0-9a-f]{1,8}", host, re.IGNORECASE))  # шестнадцатеричная


@dataclass(frozen=True)
class _Parts:
    subdomain: str
    domain: str
    suffix: str


@lru_cache(maxsize=65536)
def _extract(host: str) -> _Parts:
    """Разбор домена; для зон вне Public Suffix List (.test, .example, .local)
    последняя метка считается зоной, предпоследняя — доменом."""
    ext = _EXTRACT(host)
    if ext.suffix:
        return _Parts(ext.subdomain, ext.domain, ext.suffix)
    labels = host.split(".")
    if len(labels) >= 2:
        return _Parts(".".join(labels[:-2]), labels[-2], labels[-1])
    return _Parts("", host, "")


def registered_domain(url_or_host: str) -> str:
    """Регистрируемый домен (domain + suffix), например login.kaspi.kz → kaspi.kz.

    Используется для разбиения набора URL по группам (задание 10).
    """
    res = validate_url(url_or_host)
    host = res.host if res.ok else str(url_or_host).lower()
    if _is_ip(host):
        return host
    ext = _extract(host)
    return ".".join(p for p in (ext.domain, ext.suffix) if p) or host


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _levenshtein(a: str, b: str, limit: int = 3) -> int:
    """Расстояние Левенштейна с ранним выходом, если оно заведомо больше limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _decode_idna(host: str) -> str:
    labels = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                label = label[4:].encode("ascii").decode("punycode")
            except (UnicodeError, ValueError):
                pass
        labels.append(label)
    return ".".join(labels)


def extract_url_features(url: str) -> dict[str, float]:
    """Возвращает словарь признаков URL с ключами FEATURE_NAMES в фиксированном порядке.

    Некорректный URL вызывает InvalidURLError (подкласс ValueError) с понятным сообщением.
    """
    res = validate_url(url)
    if not res.ok:
        raise InvalidURLError(res.error)
    full = res.normalized
    parts = urlsplit(full)
    host = res.host
    path = parts.path or ""
    is_ip = _is_ip(host)
    ext = _extract(host) if not is_ip else None
    reg_domain = ".".join(p for p in (ext.domain, ext.suffix) if p) if ext else host
    sub_labels = [s for s in (ext.subdomain.split(".") if ext and ext.subdomain else []) if s and s != "www"]
    decoded = _decode_idna(host)
    lower = full.lower()

    port = parts.port
    host_digits = sum(ch.isdigit() for ch in host)

    brand_similarity = 0
    if ext and ext.domain:
        name = _decode_idna(ext.domain)
        for brand, brand_domain in _BRAND_NAMES.items():
            if reg_domain != brand_domain and _levenshtein(name, brand, 2) < 2:
                brand_similarity = 1
                break

    brand_in_url = 0
    if not is_ip:
        rest = ".".join(sub_labels) + " " + (ext.domain if ext else "") + " " + path.lower()
        for m in _BRAND_TOKEN_RE.finditer(rest):
            official = [d for n, d in _BRAND_NAMES.items() if n.startswith(m.group(0)) or m.group(0).startswith(n)]
            if reg_domain not in official:
                brand_in_url = 1
                break

    features = {
        "url_length": len(full),
        "host_length": len(host),
        "path_length": len(path),
        "n_dots": full.count("."),
        "n_subdomains": len(sub_labels),
        "n_hyphens_host": host.count("-"),
        "n_digits": sum(ch.isdigit() for ch in full),
        "digit_ratio_host": round(host_digits / len(host), 4) if host else 0.0,
        "n_special": sum(full.count(ch) for ch in _SPECIAL_CHARS),
        "has_ip": int(is_ip),
        "is_https": int(parts.scheme.lower() == "https"),
        "nonstandard_port": int(port is not None and port not in (80, 443)),
        "has_at": int("@" in full),
        "double_slash_path": int("//" in path),
        "is_shortener": int(reg_domain in SHORTENERS or host in SHORTENERS),
        "n_suspicious_words": len(_suspicious_matches(lower, reg_domain)),
        "has_punycode": int("xn--" in host),
        "mixed_scripts": int(any(_CYRILLIC_RE.search(label) and _LATIN_RE.search(label)
                                 for label in decoded.split("."))),
        "brand_similarity": brand_similarity,
        "host_entropy": round(_entropy(host), 4),
        "suspicious_tld": int(bool(ext) and ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "n_query_params": len(parse_qsl(parts.query, keep_blank_values=True)),
        "brand_in_url": brand_in_url,
        "path_depth": len([s for s in path.split("/") if s]),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def _suspicious_matches(lower_url: str, reg_domain: str) -> list[str]:
    """Подозрительные слова; название бренда на его официальном домене не считается
    (kaspi на kaspi.kz — норма, kaspi на kaspi-login.example — нет)."""
    found = _SUSPICIOUS_RE.findall(lower_url)
    if reg_domain in BRAND_DOMAINS:
        own = reg_domain.split(".")[0]
        found = [w for w in found if w not in own]
    return found


def canonical_url(url: str) -> str:
    """Каноническая форма для модели: http://домен[:порт] — без схемы, «www.», пути и запроса.

    Убирает артефакты разметки набора PhiUSIIL (все легитимные URL — https://www.домен).
    """
    res = validate_url(url)
    if not res.ok:
        raise InvalidURLError(res.error)
    host = res.host[4:] if res.host.startswith("www.") else res.host
    port = urlsplit(res.normalized).port
    if ":" in host and not host.startswith("["):      # IPv6
        host = f"[{host}]"
    return f"http://{host}" + (f":{port}" if port else "")


def model_features(url: str) -> dict[str, float]:
    """Признаки для URL-модели: extract_url_features по канонической форме, только MODEL_FEATURES."""
    full = extract_url_features(canonical_url(url))
    return {name: full[name] for name in MODEL_FEATURES}


MODEL_TEXT_COLUMN = "host"   # канонический домен — для символьных n-грамм (TF-IDF) в URL-модели


def model_input(url: str) -> dict:
    """Строка входа URL-модели: 17 числовых признаков домена + сам домен (для символьных n-грамм)."""
    row = model_features(url)
    row[MODEL_TEXT_COLUMN] = canonical_url(url).replace("http://", "", 1)
    return row


def suspicious_words_in(url: str) -> list[str]:
    """Список подозрительных слов, найденных в URL (для показа пользователю)."""
    res = validate_url(url)
    if not res.ok:
        return []
    return sorted(set(_suspicious_matches(res.normalized.lower(), registered_domain(res.host))))


# --- Объяснения ---------------------------------------------------------------------
# (признак, условие, текст причины, вес для сортировки)
_REASONS = [
    ("has_ip", lambda v: v == 1, "Используется IP-адрес вместо доменного имени", 10),
    ("has_punycode", lambda v: v == 1, "Домен записан в Punycode (xn--) — так маскируют похожие на бренды адреса", 9),
    ("mixed_scripts", lambda v: v == 1, "В домене смешаны кириллица и латиница", 9),
    ("brand_similarity", lambda v: v == 1, "Домен похож на название известного банка или сервиса, но не совпадает с ним", 9),
    ("brand_in_url", lambda v: v == 1, "Название бренда стоит в поддомене или пути чужого домена", 8),
    ("has_at", lambda v: v == 1, "В адресе есть символ @ — часть до него браузер игнорирует", 8),
    ("is_shortener", lambda v: v == 1, "Используется сервис сокращения ссылок — настоящий адрес скрыт", 7),
    ("nonstandard_port", lambda v: v == 1, "Указан нестандартный порт", 6),
    ("suspicious_tld", lambda v: v == 1, "Доменная зона часто используется для мошеннических сайтов", 6),
    ("n_suspicious_words", lambda v: v >= 1, "Подозрительные слова в адресе", 5),
    ("double_slash_path", lambda v: v == 1, "Двойной слеш // в пути — признак перенаправления", 5),
    ("n_subdomains", lambda v: v >= 3, "Много поддоменов", 4),
    ("n_hyphens_host", lambda v: v >= 2, "Много дефисов в домене", 4),
    ("is_https", lambda v: v == 0, "Соединение без шифрования (HTTP, а не HTTPS)", 3),
    ("url_length", lambda v: v > 100, "Очень длинный адрес", 2),
    ("host_entropy", lambda v: v > 4.5, "Домен похож на случайный набор символов", 2),
]


def explain_url(features: dict, url: str | None = None, limit: int = 6) -> list[str]:
    """Человекочитаемые причины подозрительности URL, отсортированные по важности."""
    reasons = []
    for key, cond, text, weight in _REASONS:
        value = features.get(key)
        if value is not None and cond(value):
            if key == "n_suspicious_words" and url:
                words = suspicious_words_in(url)
                text = f"{text}: {', '.join(words)}" if words else text
            elif key == "n_subdomains":
                text = f"{text} ({int(value)})"
            reasons.append((weight, text))
    reasons.sort(key=lambda r: -r[0])
    return [text for _, text in reasons[:limit]]
