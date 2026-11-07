"""Очистка и нормализация SMS (задание 4).

Одна и та же функция ``normalize_sms`` используется в ноутбуках, при обучении
и в веб-приложении, поэтому текст на входе модели всегда обработан одинаково.

Порядок шагов:
    1. Unicode-нормализация NFKC (полноширинные символы, лигатуры → обычные)
       и раскодирование HTML-сущностей (&lt; &gt; &amp;). В UCI-наборе числа и ссылки
       анонимизированы метками &lt;#&gt;, &lt;DECIMAL&gt;, &lt;TIME&gt;, &lt;URL&gt; —
       они приводятся к токенам <num> и <url>.
    2. Ошибочные C1-символы cp1252 (\\x92 → ') исправляются, прочие управляющие
       и невидимые символы → пробел.
    3. Нижний регистр.
    4. Замена сущностей токенами: e-mail → <email>, URL → <url>,
       сумма с валютой → <money>, телефон → <phone>, прочие числа → <num>.
    5. «!» и «?» выделяются в отдельные токены — это признаки давления.
    6. Повторяющиеся пробелы схлопываются.
    7. (опция) удаление стоп-слов.

Функция чистая: не меняет аргументы и глобальное состояние, не пишет в лог
и для одинакового входа всегда возвращает одинаковый результат.
"""
from __future__ import annotations

import html
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

__all__ = [
    "normalize_sms",
    "remove_stopwords",
    "TOKEN_PATTERN",
    "SPECIAL_TOKENS",
    "STOP_WORDS",
]

SPECIAL_TOKENS = ("<url>", "<email>", "<phone>", "<money>", "<num>")

# Шаблон токенизации для TfidfVectorizer: спецтокены, «!», «?» и слова от 2 букв.
# \w в Python учитывает Unicode, поэтому кириллица и казахские буквы сохраняются.
TOKEN_PATTERN = r"<(?:url|email|phone|money|num)>|[!?]|(?u)\b\w\w+\b"

# --- Регулярные выражения сущностей ---------------------------------------------

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zа-я]{2,}(?![\w])")

_TLDS = (
    "com|net|org|info|biz|ru|kz|рф|қаз|uk|co|io|me|ly|gl|cc|tk|xyz|top|club|online|site|"
    "app|link|click|example|test|su|ua|by|uz|kg|de|fr|in|us|tv|ws|to|pro|shop|store|live"
)
_URL_RE = re.compile(
    r"(?:"
    r"(?:https?|ftp)://[^\s<>\"']+"                 # со схемой
    r"|www\d?\.[^\s<>\"']+"                          # www.
    r"|(?<![\w@.-])(?:[a-zа-я0-9][a-zа-я0-9-]*\.)+(?:" + _TLDS + r")\b(?:[/?#][^\s<>\"']*)?"  # без схемы
    r")",
    re.IGNORECASE,
)

_NUMBER = r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?"
_CURRENCY_SYMBOLS = r"[£$€₸₽]"
_CURRENCY_WORDS = (
    r"тг|тенге|теңге|₸|руб(?:лей|ля|ль)?|р\.|₽|"
    r"usd|eur|gbp|kzt|rub|pounds?|pence|dollars?|euros?|p"
)
_MONEY_RE = re.compile(
    r"(?:"
    + _CURRENCY_SYMBOLS + r"\s?" + _NUMBER                      # £900, $ 50
    + r"|" + _NUMBER + r"\s?" + _CURRENCY_SYMBOLS               # 900£
    + r"|" + _NUMBER + r"\s?(?:" + _CURRENCY_WORDS + r")(?![\w])"  # 990 тг, 150p
    + r")",
    re.IGNORECASE,
)

# Телефон: необязательный «+», затем цифры с пробелами, дефисами и скобками;
# всего от 7 до 15 цифр (короткие номера вроде 87121 станут <num>).
# Номер может быть «приклеен» к слову (08452810075over18's) — это частый приём спамеров.
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\d<])\+?\(?\d[\d\s\-()]{5,}\d(?![\d>])")

_NUM_RE = re.compile(r"(?<![\w<])\d+(?:[.,:/]\d+)*(?![\w>])")

# Метки анонимизации из исходного корпуса (после html.unescape и нижнего регистра)
_PLACEHOLDER_RE = re.compile(r"<(#|decimal|time|url|email|ukp)>")
_PLACEHOLDER_MAP = {"#": " <num> ", "decimal": " <num> ", "time": " <num> ",
                    "url": " <url> ", "email": " <email> ", "ukp": " <money> "}

_EXCLAIM_QUESTION_RE = re.compile(r"\s*([!?])\s*")
_SPACES_RE = re.compile(r"\s+")


# Символы C1 (U+0080–U+009F), которые на деле являются знаками препинания cp1252,
# сохранёнными в неверной кодировке (в UCI-наборе встречается \x92 вместо апострофа).
_CP1252_C1 = str.maketrans({
    "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
    "\x96": "-", "\x97": "-", "\x85": "...",
})


def _strip_control(text: str) -> str:
    """Исправляет C1-символы cp1252, остальные управляющие (Cc) и форматирующие (Cf)
    символы заменяет пробелом."""
    text = text.translate(_CP1252_C1)
    return "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in text
    )


def _replace_phone(match: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in match.group(0))
    return " <phone> " if 7 <= digits <= 15 else match.group(0)


def normalize_sms(text: str | None, remove_stop_words: bool = False) -> str:
    """Возвращает нормализованный текст SMS.

    >>> normalize_sms("СРОЧНО!!! Переведите 5 000 тг на +7 000 123 45 67")
    'срочно ! ! ! переведите <money> на <phone>'

    Пустой ввод (None, "", только пробелы) даёт пустую строку.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    result = unicodedata.normalize("NFKC", text)
    result = html.unescape(result)
    result = _strip_control(result)
    result = result.lower()
    result = _PLACEHOLDER_RE.sub(lambda m: _PLACEHOLDER_MAP[m.group(1)], result)

    result = _EMAIL_RE.sub(" <email> ", result)
    result = _URL_RE.sub(" <url> ", result)
    result = _MONEY_RE.sub(" <money> ", result)
    result = _PHONE_CANDIDATE_RE.sub(_replace_phone, result)
    result = _NUM_RE.sub(" <num> ", result)

    result = _EXCLAIM_QUESTION_RE.sub(r" \1 ", result)
    result = _SPACES_RE.sub(" ", result).strip()

    if remove_stop_words:
        result = remove_stopwords(result)
    return result


# --- Стоп-слова ---------------------------------------------------------------------
# Английский — список scikit-learn; русский и казахский — короткие списки служебных слов.
# NLTK не используется, чтобы модуль не требовал загрузки корпусов из сети.
_RU_STOP_WORDS = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее её мне было вот от меня еще ещё нет о из ему теперь когда даже ну ли если уже или ни быть был
него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней
для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того
потому этого какой совсем ним здесь этом один почти мой тем чтобы нее неё сейчас были куда зачем
всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего
них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя
такой им более всегда конечно всю между это
""".split())

_KK_STOP_WORDS = frozenset("""
және мен да де та те бен пен бір бұл ол сол осы мына сен сіз біз олар ал бірақ үшін туралы
дейін кейін немесе не ма ме ба бе па пе ғой қой ғана қана тек әр еш барлық өз оның менің
сенің сіздің біздің олардың маған саған сізге бізге оған бар жоқ емес екен еді болды болып
""".split())

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | _RU_STOP_WORDS | _KK_STOP_WORDS


def remove_stopwords(text: str) -> str:
    """Удаляет стоп-слова из уже нормализованного текста (спецтокены сохраняются)."""
    return " ".join(tok for tok in text.split() if tok not in STOP_WORDS)
