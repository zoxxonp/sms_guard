"""Определение языка SMS (ФТ-2.2): ru, kk, en, other.

Правило (раздел 3.2 ТЗ, допустимая альтернатива langdetect):
* есть буквы, характерные только для казахского (ә, ғ, қ, ң, ө, ұ, ү, һ, і) → kk;
* иначе кириллицы больше, чем латиницы → ru;
* латиница → en, если langdetect уверенно не относит текст к другому языку
  (короткие английские SMS со сленгом langdetect часто путает, поэтому по умолчанию en);
* букв нет или другой алфавит → other.

langdetect не поддерживает казахский и ненадёжен на коротких русских текстах
(«Ваша карта заблокирована» он определяет как болгарский), поэтому для кириллицы
используется правило.
"""
from __future__ import annotations

import re

_KAZAKH = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+|\S+\.(?:com|ru|kz|org|net|example|test)\S*", re.IGNORECASE)

SUPPORTED = {"en", "ru", "kk"}
LANGUAGE_NAMES = {"en": "английский", "ru": "русский", "kk": "казахский", "other": "другой"}


def detect_language(text: str) -> str:
    if not isinstance(text, str):
        return "other"
    body = _URL.sub(" ", text)                       # ссылки латиницей не должны влиять на язык
    if _KAZAKH.search(body):
        return "kk"
    cyr, lat = len(_CYRILLIC.findall(body)), len(_LATIN.findall(body))
    if cyr == 0 and lat == 0:
        return "other"
    if cyr >= lat:
        return "ru"
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        guesses = detect_langs(body)
        top = guesses[0]
        if top.lang != "en" and top.prob >= 0.95 and len(body.split()) >= 4:
            return "other"
    except Exception:                                 # langdetect не установлен или текст без признаков
        pass
    return "en"
