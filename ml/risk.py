"""Модуль оценки риска: Risk Score 0–100, уровень, категория, причины, рекомендация (задание 13).

Формула (раздел 2.5 и задание 13 ТЗ):

    RiskScore = 100 · (w_sms·P_sms + w_url·P_url,max + w_se·S_se) / (w_sms + w_url·[URL есть] + w_se)
    S_se = min(k / 3, 1)

Если URL нет, слагаемое w_url исключается из числителя и знаменателя.
После формулы применяются правила усиления (override), затем определяется уровень:
низкий 0–34 → SAFE, средний 35–69 → SUSPICIOUS, высокий 70–100 → FRAUDULENT.

Все веса, пороги, правила и тексты рекомендаций читаются из risk_config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "risk_config.yaml"

LEVELS = ("LOW", "MEDIUM", "HIGH")
LEVEL_TO_CATEGORY = {"LOW": "SAFE", "MEDIUM": "SUSPICIOUS", "HIGH": "FRAUDULENT"}
LEVEL_NAMES_RU = {"LOW": "низкий", "MEDIUM": "средний", "HIGH": "высокий"}
CATEGORY_NAMES_RU = {"SAFE": "Безопасное", "SUSPICIOUS": "Подозрительное", "FRAUDULENT": "Мошенническое"}
URL_CATEGORY_NAMES_RU = {"SAFE": "Безопасная", "SUSPICIOUS": "Подозрительная", "DANGEROUS": "Опасная"}
DISCLAIMER = "Оценка носит вероятностный характер и не гарантирует безопасность"

_cache: dict = {"mtime": None, "path": None, "config": None}


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Читает risk_config.yaml; перечитывает файл, если он изменился (без перезапуска)."""
    path = Path(path)
    mtime = path.stat().st_mtime
    if _cache["config"] is None or _cache["mtime"] != mtime or _cache["path"] != path:
        with path.open(encoding="utf-8") as fh:
            _cache.update(config=yaml.safe_load(fh), mtime=mtime, path=path)
    return _cache["config"]


@dataclass
class RiskResult:
    score: int
    level: str                      # LOW / MEDIUM / HIGH
    category: str                   # SAFE / SUSPICIOUS / FRAUDULENT
    reasons: list[str]
    recommendation: str
    base_score: float               # значение формулы до правил усиления
    applied_overrides: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @property
    def level_ru(self) -> str:
        return LEVEL_NAMES_RU[self.level]


def level_for(score: float, config: dict | None = None) -> str:
    cfg = config or load_config()
    if score <= cfg["levels"]["low_max"]:
        return "LOW"
    if score <= cfg["levels"]["medium_max"]:
        return "MEDIUM"
    return "HIGH"


def base_score(p_sms: float | None, p_url_max: float | None, k_groups: int, config: dict | None = None) -> float:
    """Значение формулы 0–100 без правил усиления."""
    cfg = config or load_config()
    w = cfg["weights"]
    s_se = min(k_groups / cfg.get("se_saturation", 3), 1.0)
    num = w["w_se"] * s_se
    den = w["w_se"]
    if p_sms is not None:
        num += w["w_sms"] * p_sms
        den += w["w_sms"]
    if p_url_max is not None:
        num += w["w_url"] * p_url_max
        den += w["w_url"]
    return 100.0 * num / den if den else 0.0


def url_probability_to_score(p_url: float, t1: float, t2: float, config: dict | None = None) -> float:
    """Кусочно-линейное преобразование P_url → 0–100, при котором t1 → 35, t2 → 70."""
    cfg = config or load_config()
    low = cfg["levels"]["low_max"] + 1           # 35
    high = cfg["levels"]["medium_max"] + 1       # 70
    if p_url < t1:
        return low * p_url / t1 if t1 > 0 else 0.0
    if p_url < t2:
        return low + (high - low) * (p_url - t1) / (t2 - t1)
    return high + (100 - high) * (p_url - t2) / (1 - t2) if t2 < 1 else 100.0


def compute_risk(p_sms: float | None, p_url_max: float | None, se_groups: dict | list | None,
                 url_flags: dict | None = None, *, url_t2: float = 0.8, se_titles: dict | None = None,
                 url_reasons: list[str] | None = None, url_only_score: float | None = None,
                 kind: str = "SMS", config: dict | None = None) -> RiskResult:
    """Рассчитывает риск.

    p_sms        — вероятность мошенничества по тексту (None для проверки только URL);
    p_url_max    — максимальная вероятность опасности среди ссылок (None, если ссылок нет);
    se_groups    — сработавшие группы словаря: {код: [найденные фразы]} или список кодов;
    url_flags    — флаги самой опасной ссылки: {"has_ip": 1, "has_punycode": 0, ...};
    url_t2       — порог DANGEROUS URL-модели (для правила «P_sms ≥ 0,8 и P_url ≥ t2»);
    url_reasons  — причины от explain_url для самой опасной ссылки;
    url_only_score — для проверки отдельной ссылки: P_url, переведённая в шкалу 0–100.
    """
    cfg = config or load_config()
    groups = dict(se_groups) if isinstance(se_groups, dict) else {g: [] for g in (se_groups or [])}
    flags = url_flags or {}
    ov = cfg["overrides"]

    score = url_only_score if url_only_score is not None else base_score(p_sms, p_url_max, len(groups), cfg)
    base = score
    applied: list[str] = []
    if "CODE_REQUEST" in groups and score < ov["code_request_min"]:
        score = ov["code_request_min"]
        applied.append("CODE_REQUEST")
    if (p_sms is not None and p_url_max is not None and p_sms >= ov["sms_high"] and p_url_max >= url_t2
            and score < ov["sms_and_url_high_min"]):
        score = ov["sms_and_url_high_min"]
        applied.append("SMS_AND_URL_HIGH")
    if (flags.get("has_ip") or flags.get("has_punycode")) and score < ov["ip_or_punycode_min"]:
        score = ov["ip_or_punycode_min"]
        applied.append("IP_OR_PUNYCODE")

    score_int = int(round(min(max(score, 0.0), 100.0)))
    level = level_for(score_int, cfg)
    reasons = _reasons(p_sms, p_url_max, groups, url_reasons or [], se_titles or {}, cfg, kind, applied)
    rec_key = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}[level]
    recommendation = (cfg["url_recommendations"] if kind == "URL" else cfg["recommendations"])[rec_key].strip()
    return RiskResult(score_int, level, LEVEL_TO_CATEGORY[level], reasons, recommendation,
                      round(base, 2), applied)


_OVERRIDE_REASONS = {
    "SMS_AND_URL_HIGH": "И текст, и ссылка одновременно похожи на мошеннические",
    "IP_OR_PUNYCODE": "Ссылка использует IP-адрес или Punycode — так маскируют адреса мошеннических сайтов",
}


def _reasons(p_sms, p_url_max, groups, url_reasons, se_titles, cfg, kind, applied=()) -> list[str]:
    """Причины, отсортированные по вкладу; не более max_reasons; всегда хотя бы одна."""
    w = cfg["weights"]
    threshold = cfg.get("reason_min_probability", 0.5)
    weighted: list[tuple[float, str]] = [(0.9, _OVERRIDE_REASONS[a]) for a in applied if a in _OVERRIDE_REASONS]
    if p_sms is not None and p_sms > threshold:
        weighted.append((w["w_sms"] * p_sms,
                         f"Текст похож на мошеннические сообщения (вероятность по SMS-модели {p_sms:.2f})"))
    if p_url_max is not None and p_url_max > threshold:
        label = "Ссылка" if kind == "URL" else "Ссылка в сообщении"
        weighted.append((w["w_url"] * p_url_max,
                         f"{label} похожа на опасную (вероятность по URL-модели {p_url_max:.2f})"))
    per_group = w["w_se"] / cfg.get("se_saturation", 3)
    for code, phrases in groups.items():
        title = se_titles.get(code, code)
        text = f"{title}" + (f": «{phrases[0]}»" if phrases else "")
        bonus = 1.0 if code == "CODE_REQUEST" else 0.0       # просьба назвать код — самый опасный признак
        weighted.append((per_group + bonus, text))
    for i, reason in enumerate(url_reasons):
        weighted.append((w["w_url"] * 0.5 - i * 0.01, reason))
    weighted.sort(key=lambda r: -r[0])
    seen, out = set(), []
    for _, text in weighted:
        if text not in seen:
            seen.add(text)
            out.append(text)
    out = out[: cfg.get("max_reasons", 6)]
    if not out:
        out = ["Явных признаков мошенничества не обнаружено" if kind == "SMS"
               else "Явных признаков опасной ссылки не обнаружено"]
    return out
