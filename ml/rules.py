"""Словарь признаков социальной инженерии (ФТ-2.5, задание 12; используется с задания 6).

Шесть обязательных групп: URGENCY, MONEY_TRANSFER, BANK_CARD, ACCOUNT_BLOCK, PRIZE,
CODE_REQUEST. В каждой группе не менее 10 выражений на русском, казахском и английском.

Выражения — регулярные выражения по основам слов, они применяются к тексту ПОСЛЕ
``normalize_sms`` (нижний регистр, числа заменены на <num>). Поэтому вместо «24 часа»
ищется «<num> час».

Группа CODE_REQUEST в модуле риска поднимает Risk Score минимум до 70 (раздел 2.5 ТЗ),
поэтому её выражения требуют именно ПРОСЬБЫ назвать код. Легитимное «никому не
сообщайте код» в эту группу не попадает (отрицание «не» исключается).
"""
from __future__ import annotations

import re
from functools import lru_cache

from ml.preprocessing import normalize_sms

__all__ = ["RULES", "GROUP_TITLES", "match_rules", "count_rule_groups"]

_N = r"(?:<num>|\d+)"            # число после нормализации
_NOT = r"(?<!не )(?<!не\s)"      # не «не сообщайте»
_NOT_EN = r"(?<!not )(?<!never )(?<!t )"  # не «do not / never / don't send»

GROUP_TITLES = {
    "URGENCY": "Требование срочного действия",
    "MONEY_TRANSFER": "Просьба перевести или оплатить деньги",
    "BANK_CARD": "Упоминание банка, карты или счёта",
    "ACCOUNT_BLOCK": "Угроза блокировки или требование подтвердить данные",
    "PRIZE": "Обещание выигрыша, приза или бонуса",
    "CODE_REQUEST": "Просьба сообщить код из SMS или пароль",
}

RULES: dict[str, dict[str, list[str]]] = {
    "URGENCY": {
        "ru": [
            r"срочн", r"немедленн", rf"в течение {_N} (?:час|мин|дн|сут)", r"до конца дня",
            r"прямо сейчас", r"последн(?:ий|яя|ее) (?:шанс|попытк|предупрежден)", r"только сегодня",
            r"не откладыва", r"истека", r"как можно скорее", r"осталось \S+ мест", r"сегодня до", r"мест ограничено|количество мест",
        ],
        "kk": [
            r"шұғыл", r"жедел", r"дереу", r"тез арада", rf"{_N} сағат", r"бүгін(?:нен)?",
            r"соңғы мүмкіндік", r"кешіктірмей", r"мерзімі (?:аяқтал|бітеді)", r"асығыңыз",
            r"уақыт шектеулі", r"орын қалды", r"саны шектеулі",
        ],
        "en": [
            r"urgent", r"immediately", r"\basap\b", r"act now", rf"within {_N} ?(?:hrs?|hours?|days?)",
            r"today only", r"last chance", r"final (?:notice|attempt|reminder)", r"expires?\b",
            r"hurry", r"limited time", r"don'?t miss", r"before midnight", r"\b2nd attempt",
        ],
    },
    "MONEY_TRANSFER": {
        "ru": [
            r"перевед", r"перевод(?:а|ом)? на", r"скин(?:ь|те) ", r"отправ(?:ь|ьте) деньги",
            r"оплат(?:ите|ить|у)", r"предоплат", r"комисси", r"депозит", r"внесите", r"займ",
            r"деньги нужн|нужн\S* деньги", r"верну\b", r"передай(?:те)? <money>|передай(?:те)? деньги",
        ],
        "kk": [
            r"аудар", r"ақша жібер", r"төле(?:ңіз|у)", r"төлем", r"жібер(?:іңіз)? <money>",
            r"депозит", r"комиссия", r"салып", r"салым", r"қарыз", r"ақша керек", r"қайтарамын",
        ],
        "en": [
            r"transfer", r"send (?:me |us )?money", r"pay(?:ment)? (?:now|today)", r"\bpayment",
            r"deposit", r"\bfee\b", r"\bwire\b", r"\bcash\b", r"refund", r"\bloan\b",
            r"top ?up", r"\bowe\b", r"pay (?:the |a )?(?:fee|charge|delivery)",
        ],
    },
    "BANK_CARD": {
        "ru": [
            r"банк", r"карт(?:а|у|ы|ой|е|ах)\b", r"\bcvv|\bcvc", r"срок действия", r"номер карты",
            r"\bсч[её]т", r"интернет-банк|онлайн-банк", r"личн\S+ кабинет", r"kaspi|каспи",
            r"halyk|халык", r"данные карты",
        ],
        "kk": [
            r"банк", r"карта(?:ңыз|ның|ны|сы|ға)?\b", r"\bшот", r"\bcvv", r"карта нөмірі",
            r"kaspi|каспи", r"halyk|халық банк", r"жеке кабинет", r"карта деректер", r"банк қосымша",
            r"аккаунт",
        ],
        "en": [
            r"\bbank", r"\bcard\b", r"\baccount", r"\bcvv", r"credit card", r"debit",
            r"\bvisa\b", r"mastercard", r"online banking", r"sort code", r"statement",
        ],
    },
    "ACCOUNT_BLOCK": {
        "ru": [
            r"заблокир", r"блокировк", r"разблокир", r"приостановл", r"ограничен\S* доступ|доступ\S* (?:будет )?ограничен",
            r"заморож", r"аннулир", r"деактивир", r"будет удал", r"подтвердите (?:личность|данные|владельца|вход)",
            r"верификац", r"подозрительн", r"взлом",
        ],
        "kk": [
            r"бұғатта", r"бұғаттан шығар", r"тоқтатыл", r"шектел", r"жабылады", r"растаңыз",
            r"верификация", r"күдікті", r"жойылады", r"тексеру қажет", r"қайта тексер",
        ],
        "en": [
            r"\bblocked", r"suspend", r"\blocked\b", r"deactivat", r"verify your", r"confirm your",
            r"unusual activity", r"suspicious", r"unauthori[sz]ed", r"will be (?:closed|terminated)",
            r"restricted", r"security alert",
        ],
    },
    "PRIZE": {
        "ru": [
            r"выигр", r"приз", r"победител", r"розыгрыш", r"бонус", r"подар(?:ок|оч)", r"сертификат",
            r"лотере", r"поздравля", r"наград", r"начислен", r"бесплатн",
        ],
        "kk": [
            r"ұтып ал", r"ұтыс", r"жеңімпаз", r"сыйлық", r"бонус", r"құттықтаймыз", r"лотерея",
            r"тегін", r"сыйақы", r"акция", r"жүлде",
        ],
        "en": [
            r"\bwin\b|\bwins\b", r"\bwon\b", r"winner", r"prize", r"award", r"bonus", r"\bgift",
            r"congratulation", r"lottery", r"\bfree\b", r"\bclaim", r"reward", r"jackpot",
            r"you have been selected|you've been selected",
        ],
    },
    "CODE_REQUEST": {
        "ru": [
            rf"{_NOT}назов(?:и|ите) код", rf"{_NOT}сообщ(?:и|ите) (?:его|нам|код|оператору)",
            rf"{_NOT}продикту(?:й|йте) код", rf"{_NOT}переш(?:ли|лите) (?:его|код)",
            rf"{_NOT}отправ(?:ь|ьте) код", rf"{_NOT}скаж(?:и|ите) код", r"код из (?:sms|смс|сообщения)",
            r"(?:назов|сообщ|продикту)\S* (?:пароль|пин)", rf"{_NOT}сообщите его",
            r"логин и пароль", r"укаж\S* (?:пароль|cvv|пин)",
        ],
        "kk": [
            r"кодты айт(?:ыңыз|шы)?\b", r"кодты хабарла", r"кодты .{0,25}жібер", r"кодты операторға",
            r"(?:sms|смс)-тегі кодты (?:айт|хабарла|жібер|оператор)", r"құпиясөз\S* айт", r"растау кодын айт",
            r"пин-?код\S* айт", r"келетін кодты", r"кодты бізге", r"оны (?:біздің )?қызметкерге",
        ],
        "en": [
            rf"{_NOT_EN}(?:tell|give) (?:me|us) (?:the |your )?(?:code|pin|password)",
            rf"{_NOT_EN}send (?:me |us )?(?:the |your )?(?:code|pin|password)",
            rf"{_NOT_EN}share (?:the |your )?(?:code|pin|password)", r"read (?:out )?the code",
            r"reply with (?:the |your )?(?:code|pin)", rf"{_NOT_EN}forward (?:the |this )?code",
            rf"{_NOT_EN}provide (?:the |your )?(?:code|pin|password)", r"code (?:you )?received",
            r"your (?:pin|password) (?:to|for)", r"confirm (?:the |your )?(?:otp|code)",
        ],
    },
}


@lru_cache(maxsize=1)
def _compiled() -> dict[str, list[tuple[str, re.Pattern]]]:
    return {
        group: [(lang, re.compile(rf"(?<![\w]){pat}" if pat[0].isalpha() else pat))
                for lang, patterns in langs.items() for pat in patterns]
        for group, langs in RULES.items()
    }


def match_rules(text: str, normalized: bool = False) -> dict[str, list[str]]:
    """Возвращает сработавшие группы: {группа: [найденные фрагменты]}.

    ``normalized=True`` — текст уже обработан ``normalize_sms`` (экономит повторную очистку).
    """
    clean = text if normalized else normalize_sms(text)
    found: dict[str, list[str]] = {}
    for group, patterns in _compiled().items():
        hits = []
        for _lang, rx in patterns:
            m = rx.search(clean)
            if m and m.group(0).strip() not in hits:
                hits.append(m.group(0).strip())
        if hits:
            found[group] = hits
    return found


def count_rule_groups(text: str, normalized: bool = False) -> int:
    """Число сработавших групп словаря (0–6) — ручной признак для модели."""
    return len(match_rules(text, normalized=normalized))
