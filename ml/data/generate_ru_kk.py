"""Генерация синтетического набора SMS на русском и казахском языках.

Запуск из корня проекта:
    python -m ml.data.generate_ru_kk

Результат: data/raw/synthetic_ru_kk.csv — 1000 SMS на русском и 1000 на казахском,
в каждом языке 50% мошеннических. Столбцы:
    id, lang, text, label (0 safe / 1 fraud), label_name, category, template_id, author

Как устроено. Сообщения собираются из шаблонов со слотами: {bank}, {amount}, {url},
{phone}, {name} и т. д. Значения слотов выбираются случайно (seed = 42, результат
воспроизводим). На каждый язык 41 шаблон мошеннических и 44 безопасных SMS.

Ограничения (указаны в паспорте docs/03b_synthetic_dataset.md):
* данные искусственные — это не реальные SMS, распределение схем условное;
* сообщения одного шаблона похожи друг на друга, поэтому выборки делятся
  ПО ШАБЛОНАМ (template_id), а не по отдельным сообщениям: в тест попадают
  шаблоны, которых модель не видела при обучении;
* 150 SMS из data/raw/custom_ru_kk.csv в генерацию не входят и остаются
  независимым тестом.

Персональные данные: номера телефонов вида +7 000 … (код 000 не существует),
карты только последними 4 цифрами, ссылки только на зарезервированные домены
.example / .test / example.com и несуществующие пути сокращателей.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

from ml.preprocessing import normalize_sms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_ru_kk.csv"
CUSTOM_PATH = PROJECT_ROOT / "data" / "raw" / "custom_ru_kk.csv"

SEED = 42
PER_CLASS = 500          # на язык и класс → 1000 SMS на язык
AUTHOR = "сгенерировано ml/data/generate_ru_kk.py (Омаров М.Е.)"

# --------------------------------------------------------------------------------------
# Значения слотов
# --------------------------------------------------------------------------------------
BANKS = ["Kaspi", "Halyk", "Jusan", "Freedom", "Forte", "BCC", "Bereke", "Home Credit"]
SHOPS = ["Magnum", "Small", "Technodom", "Sulpak", "Wildberries", "Ozon", "Mechta", "Lamoda"]
DELIVERY = {"ru": ["Казпочта", "CDEK", "Почта", "Курьерская служба", "Доставка"],
            "kk": ["Қазпошта", "CDEK", "Пошта", "Курьерлік қызмет", "Жеткізу қызметі"]}
NAMES = ["Айгерим", "Данияр", "Асель", "Ерлан", "Мадина", "Нурлан", "Алия", "Тимур", "Дана",
         "Арман", "Жанна", "Бауыржан"]
PRODUCTS = {"ru": ["iPhone 17", "Samsung Galaxy", "ноутбук", "PlayStation 5", "автомобиль", "телевизор"],
            "kk": ["iPhone 17", "Samsung Galaxy", "ноутбук", "PlayStation 5", "көлік", "теледидар"]}
RELATIVE = {"ru": ["Мама", "Мам", "Папа", "Пап", "Сестрёнка", "Брат", "Тётя"],
            "kk": ["Апа", "Мама", "Әке", "Папа", "Аға", "Әпке", "Тәте"]}
PLACES = {"ru": ["у библиотеки", "у входа в универ", "в кафе", "возле метро", "у ТРЦ", "в столовой"],
          "kk": ["кітапхана жанында", "университет кіреберісінде", "кафеде", "метро жанында",
                 "сауда орталығында", "асханада"]}
SUBJECTS = {"ru": ["базам данных", "сетям", "криптографии", "Python", "матанализу", "английскому"],
            "kk": ["дерекқор", "желілер", "криптография", "Python", "математика", "ағылшын тілі"]}

URL_BRANDS = ["kaspi", "halyk", "jusan", "egov", "kazpost", "bank", "card", "pay", "bonus", "prize",
              "post", "tax", "gov", "invest", "crypto", "delivery", "parcel", "secure", "account"]
URL_WORDS = ["secure", "verify", "login", "confirm", "pay", "bonus", "claim", "update", "unlock",
             "check", "kz", "online", "help", "support"]
URL_PATHS = ["", "/login", "/verify", "/pay", "/claim", "/confirm", "/account", "/form",
             "/confirm?id={code}", "/p/{code}"]


def _fmt_amount(value: int, lang: str, rnd: random.Random) -> str:
    num = f"{value:,}".replace(",", " ") if rnd.random() < 0.7 else str(value)
    cur = rnd.choice(["тг", "тенге", "₸"] if lang == "ru" else ["тг", "теңге", "₸"])
    return f"{num} {cur}" if cur != "₸" or rnd.random() < 0.5 else f"{cur}{num}"


def _phone(rnd: random.Random) -> str:
    a, b, c = rnd.randint(100, 999), rnd.randint(10, 99), rnd.randint(10, 99)
    return rnd.choice([f"+7 000 {a} {b} {c}", f"8 000 {a} {b} {c}", f"+7000{a}{b}{c}",
                       f"8 (000) {a}-{b}-{c}"])


def _fraud_url(rnd: random.Random) -> str:
    if rnd.random() < 0.18:
        return rnd.choice(["bit.ly/", "tinyurl.com/", "clck.ru/"]) + \
            f"{rnd.choice(URL_BRANDS)}-{rnd.randint(10, 99)}-test"
    scheme = rnd.choice(["http://", "http://", "https://", ""])
    name = f"{rnd.choice(URL_BRANDS)}{rnd.choice(['-', '-', ''])}{rnd.choice(URL_WORDS)}"
    tld = rnd.choice(["example", "test"])
    path = rnd.choice(URL_PATHS).format(code=rnd.randint(1000, 99999))
    return f"{scheme}{name}.{tld}{path}"


def _safe_url(rnd: random.Random) -> str:
    return rnd.choice([
        f"https://www.example.com/receipt/{rnd.randint(10000, 99999)}",
        f"https://shop.example/order/{rnd.randint(10000, 99999)}",
        f"https://track.example/{rnd.randint(100000, 999999)}",
        "https://www.example.com/menu",
        f"https://example.com/check/{rnd.randint(1000, 9999)}",
    ])


def make_slots(lang: str, rnd: random.Random) -> dict:
    big = rnd.choice([15000, 20000, 25000, 30000, 50000, 75000, 100000, 120000, 150000, 200000, 350000])
    small = rnd.choice([290, 350, 490, 590, 690, 990, 1200, 1490, 1990, 2500])
    mid = rnd.choice([1500, 2000, 2500, 3450, 4990, 5000, 7500, 10000, 12000])
    return {
        "bank": rnd.choice(BANKS),
        "shop": rnd.choice(SHOPS),
        "delivery": rnd.choice(DELIVERY[lang]),
        "name": rnd.choice(NAMES),
        "product": rnd.choice(PRODUCTS[lang]),
        "relative": rnd.choice(RELATIVE[lang]),
        "place": rnd.choice(PLACES[lang]),
        "subject": rnd.choice(SUBJECTS[lang]),
        "amount": _fmt_amount(big, lang, rnd),
        "small": _fmt_amount(small, lang, rnd),
        "mid": _fmt_amount(mid, lang, rnd),
        "balance": _fmt_amount(rnd.randint(5, 900) * 100, lang, rnd),
        "url": _fraud_url(rnd),
        "safe_url": _safe_url(rnd),
        "phone": _phone(rnd),
        "last4": f"{rnd.randint(1000, 9999)}",
        "code": f"{rnd.randint(1000, 999999)}",
        "order": f"{rnd.randint(10000, 999999)}",
        "hours": rnd.choice([2, 3, 6, 12, 24, 48]),
        "days": rnd.choice([1, 2, 3, 5, 7]),
        "pct": rnd.choice([20, 30, 40, 50, 70]),
        "time": f"{rnd.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])}:{rnd.choice(['00', '15', '30', '45'])}",
        "date": f"{rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}",
        "room": rnd.choice([101, 204, 305, 312, 407, 512]),
        "gb": rnd.choice([3, 5, 10, 12, 20, 30]),
        "places_left": rnd.choice([3, 5, 7, 10]),
    }


# --------------------------------------------------------------------------------------
# Шаблоны: (категория, текст)
# --------------------------------------------------------------------------------------
RU_FRAUD = [
    # поддельный банк
    ("fake_bank", "{bank}: обнаружен вход с нового устройства. Если это не вы, подтвердите личность: {url}"),
    ("fake_bank", "Ваши данные в приложении {bank} устарели. Обновите их по ссылке, иначе доступ будет ограничен: {url}"),
    ("fake_bank", "Служба поддержки {bank}: требуется повторная верификация аккаунта. Перейдите: {url}"),
    ("fake_bank", "Зафиксирована попытка списания {amount} с вашего счёта. Отменить операцию: {url}"),
    ("fake_bank", "Срок действия вашей карты {bank} истекает. Для перевыпуска введите данные карты на сайте {url}"),
    ("fake_bank", "Ваш личный кабинет {bank} будет удалён через {hours} часов. Подтвердите вход: {url}"),
    # блокировка карты
    ("card_block", "Ваша карта *{last4} заблокирована из-за подозрительной операции! Для разблокировки срочно позвоните: {phone}"),
    ("card_block", "ВНИМАНИЕ! Карта *{last4} заблокирована. Для восстановления доступа перейдите по ссылке: {url}"),
    ("card_block", "Ваш счёт в {bank} временно заморожен. Подтвердите данные в течение {hours} часов: {url}"),
    ("card_block", "По вашей карте приостановлены операции. Немедленно позвоните в службу безопасности {bank}: {phone}"),
    ("card_block", "Карта *{last4} будет заблокирована сегодня в 23:59. Подтвердите владельца карты: {url}"),
    # приз
    ("prize", "Поздравляем! Ваш номер выиграл {product} в акции оператора. Оплатите доставку {small} до конца дня: {url}"),
    ("prize", "Вы стали победителем розыгрыша {amount}! Для получения приза перейдите по ссылке: {url}"),
    ("prize", "Вам начислен бонус {mid} от {shop}. Активируйте до полуночи: {url}"),
    ("prize", "Вы выиграли {product}! Чтобы забрать приз, оплатите страховку {small} на карту *{last4}."),
    ("prize", "Ваш номер выбран случайным образом. Вы получаете сертификат {shop} на {mid}. Подробнее: {url}"),
    # доставка
    ("delivery", "{delivery}: ваша посылка ожидает оплаты таможенного сбора {small}. Оплатите в течение {hours} часов: {url}"),
    ("delivery", "Курьер не смог доставить заказ №{order}: неверный адрес. Оплатите повторную доставку {small}: {url}"),
    ("delivery", "{delivery}: посылка задержана на складе. Для получения подтвердите данные карты: {url}"),
    ("delivery", "Заказ №{order} будет возвращён отправителю, если вы не оплатите хранение {small} до завтра: {url}"),
    ("delivery", "Последняя попытка доставки! Подтвердите получение посылки: {url}"),
    # родственник
    ("relative", "{relative}, это я, пишу с чужого номера. У меня проблемы, срочно переведи {amount} на этот номер, потом всё объясню. Никому не говори."),
    ("relative", "{relative}, телефон сломался, пишу с номера друга. Срочно нужно {amount}, верну завтра. Только никому пока не говори."),
    ("relative", "{relative}, я попал в неприятную ситуацию, нужны деньги прямо сейчас. Переведи {amount} на карту *{last4}, позвонить не могу."),
    ("relative", "{relative}, это {name}. Срочно скинь {amount} на этот номер, потом объясню, свой телефон потерял."),
    ("relative", "{relative}, срочно! Попал в аварию, нужны деньги на лечение. Переведи {amount}, не звони мне сейчас."),
    # запрос кода
    ("code_request", "Служба безопасности {bank}: назовите код из SMS оператору, чтобы отменить перевод на {amount}."),
    ("code_request", "Вам пришёл код подтверждения. Сообщите его нашему сотруднику для завершения проверки счёта."),
    ("code_request", "Оператор связи: для продления номера продиктуйте код, который придёт вам в SMS."),
    ("code_request", "Мы ошибочно отправили вам код. Перешлите его на номер {phone}, это срочно."),
    ("code_request", "Сотрудник {bank}: чтобы защитить ваши деньги, назовите код из сообщения. Это безопасно."),
    # госуслуги
    ("gov", "eGov: на ваше имя выписан штраф {mid}. При неоплате в течение {days} дней сумма удвоится. Оплатить со скидкой {pct}%: {url}"),
    ("gov", "Вам положена социальная выплата {amount}. Для получения заполните анкету: {url}"),
    ("gov", "Налоговая служба: у вас задолженность {mid}. Во избежание ограничения выезда оплатите сегодня: {url}"),
    ("gov", "Ваша ЭЦП будет аннулирована. Для продления введите данные по ссылке: {url}"),
    ("gov", "Уведомление о штрафе ПДД на {mid}. Фото нарушения и оплата: {url}"),
    # инвестиции
    ("invest", "Доход от {amount} в неделю без опыта! Платформа с ИИ-трейдингом, осталось {places_left} мест: {url}"),
    ("invest", "Гарантированная прибыль {pct}% в месяц. Вложите от {mid} и начните зарабатывать: {url}"),
    ("invest", "Секретная инвестиционная программа. Узнайте, как удвоить деньги за неделю: {url}"),
    ("invest", "Вам одобрен инвестиционный счёт. Внесите первый депозит {mid}, чтобы активировать: {url}"),
    ("invest", "Пассивный доход без риска! Пишите на номер {phone}, количество мест ограничено."),
]

RU_SAFE = [
    # личная переписка
    ("personal", "Привет! Встречаемся завтра в {time} {place}."),
    ("personal", "Я уже еду, буду минут через {days}0."),
    ("personal", "{name}, купи, пожалуйста, хлеб и молоко по дороге домой."),
    ("personal", "Пара перенесена на {time}, аудитория {room}."),
    ("personal", "Можешь скинуть конспект по {subject}?"),
    ("personal", "Я опоздаю минут на {days}0, начинайте без меня."),
    ("personal", "{relative}, я доехал, всё хорошо. Позвоню вечером."),
    ("personal", "Давай в кино в пятницу? Сеанс в {time}."),
    ("personal", "Защита курсовой {date}, не забудь распечатать титульный лист."),
    ("personal", "Я закинул тебе {mid} за обед, проверь."),
    ("personal", "Собрание группы завтра в {time} в аудитории {room}."),
    ("personal", "{name}, с днём рождения! Счастья, здоровья и успехов!"),
    # настоящие уведомления банка
    ("bank_legit", "{bank}: код подтверждения {code}. Никому не сообщайте этот код, даже сотрудникам банка."),
    ("bank_legit", "{bank}: покупка {mid}, {shop}. Доступно: {balance}. Карта *{last4}."),
    ("bank_legit", "{bank}: зачисление {amount}. Зарплата. Карта *{last4}."),
    ("bank_legit", "Перевод {mid} от {name} успешно зачислен на карту *{last4}."),
    ("bank_legit", "{bank}: код для входа в приложение {code}. Если вы не запрашивали код, просто проигнорируйте сообщение."),
    ("bank_legit", "{bank}: напоминаем о платеже по кредиту {mid} до {date}. Оплатить можно в приложении банка."),
    ("bank_legit", "{bank}: ваша новая карта готова. Получите её в отделении, адрес указан в приложении."),
    ("bank_legit", "Банк никогда не просит назвать код из SMS. Будьте бдительны! {bank}"),
    ("bank_legit", "{bank}: начислен кешбэк {mid}. Карта *{last4}."),
    ("bank_legit", "{bank}: оплата {shop} {mid}. Остаток: {balance}."),
    ("bank_legit", "{bank}: пополнение счёта на {mid}. Карта *{last4}."),
    ("bank_legit", "{bank}: оплата коммунальных услуг на {mid} прошла успешно."),
    # доставка
    ("delivery_legit", "{delivery}: заказ №{order} передан в доставку. Ожидаемая дата: {date}."),
    ("delivery_legit", "{delivery}: посылка прибыла в пункт выдачи. Срок хранения {days} дней. Код получения {code}."),
    ("delivery_legit", "Курьер приедет {date} с {time}. Номер заказа {order}. Отследить: {safe_url}"),
    ("delivery_legit", "{shop}: ваш заказ №{order} доставлен. Спасибо за покупку!"),
    ("delivery_legit", "{shop}: заказ №{order} собран и ожидает курьера."),
    ("delivery_legit", "Курьер {delivery}: буду у вас через 20 минут, телефон {phone}."),
    # сервисные уведомления
    ("service", "Ваша запись к врачу подтверждена: {date} в {time}, кабинет {room}."),
    ("service", "Баланс вашего номера: {balance}. Пополнить можно в приложении оператора."),
    ("service", "Ваш тариф продлён до {date}. Остаток интернета {gb} ГБ."),
    ("service", "Плановые работы на сети {date} с 02:00 до 04:00. Приносим извинения."),
    ("service", "Ваш код для входа на портал университета: {code}."),
    ("service", "Библиотека: срок возврата книги истекает {date}."),
    ("service", "Счёт за коммунальные услуги за месяц сформирован. Сумма {mid}."),
    ("service", "Вы записаны на экзамен {date} в {time}. Возьмите удостоверение личности."),
    # легитимная реклама
    ("promo_legit", "{shop}: скидка {pct}% на всю обувь в эти выходные. Ждём вас в магазинах!"),
    ("promo_legit", "{shop}: новая коллекция уже в продаже. Подробнее: {safe_url}"),
    ("promo_legit", "Кофейня: каждый пятый кофе в подарок по карте лояльности."),
    ("promo_legit", "{shop}: спасибо за покупку! Ваш электронный чек: {safe_url}"),
    ("promo_legit", "Оцените качество обслуживания в {shop} в мобильном приложении. Нам важно ваше мнение."),
    ("promo_legit", "Распродажа в книжном: до {pct}% на учебную литературу до конца месяца."),
]

KK_FRAUD = [
    ("fake_bank", "{bank}: жаңа құрылғыдан кіру анықталды. Бұл сіз болмасаңыз, мына жерде растаңыз: {url}"),
    ("fake_bank", "Құрметті клиент! {bank} қосымшасындағы деректеріңізді жаңартыңыз, әйтпесе қолжетімділік шектеледі: {url}"),
    ("fake_bank", "{bank} қолдау қызметі: аккаунтыңызды қайта тексеру қажет. Өтіңіз: {url}"),
    ("fake_bank", "Шотыңыздан {amount} шешу әрекеті тіркелді. Операцияны болдырмау: {url}"),
    ("fake_bank", "{bank} картаңыздың мерзімі аяқталады. Қайта шығару үшін карта деректерін енгізіңіз: {url}"),
    ("fake_bank", "{bank} жеке кабинетіңіз {hours} сағаттан кейін жойылады. Кіруді растаңыз: {url}"),
    ("card_block", "Сіздің *{last4} картаңыз күдікті операцияға байланысты бұғатталды! Бұғаттан шығару үшін дереу хабарласыңыз: {phone}"),
    ("card_block", "НАЗАР АУДАРЫҢЫЗ! *{last4} картасы бұғатталды. Қолжетімділікті қалпына келтіру үшін сілтемеге өтіңіз: {url}"),
    ("card_block", "{bank} шотыңыз уақытша тоқтатылды. {hours} сағат ішінде деректеріңізді растаңыз: {url}"),
    ("card_block", "Картаңыз бойынша операциялар тоқтатылды. {bank} қауіпсіздік қызметіне дереу қоңырау шалыңыз: {phone}"),
    ("card_block", "*{last4} картасы бүгін 23:59-да бұғатталады. Карта иесін растаңыз: {url}"),
    ("prize", "Құттықтаймыз! Сіздің нөміріңіз акцияда {product} ұтып алды. Жеткізу ақысын {small} бүгін төлеңіз: {url}"),
    ("prize", "Сіз {amount} ұтыс ойынының жеңімпазы атандыңыз! Сыйлықты алу үшін сілтемеге өтіңіз: {url}"),
    ("prize", "{shop} сізге {mid} бонус берді. Түн ортасына дейін белсендіріңіз: {url}"),
    ("prize", "Сіз {product} ұтып алдыңыз! Сыйлықты алу үшін *{last4} картасына {small} сақтандыру ақысын төлеңіз."),
    ("prize", "Сіздің нөміріңіз кездейсоқ таңдалды. Сізге {shop} сертификаты {mid} берілді. Толығырақ: {url}"),
    ("delivery", "{delivery}: сәлемдемеңіз кедендік алымды төлеуді күтуде, {small}. {hours} сағат ішінде төлеңіз: {url}"),
    ("delivery", "Курьер №{order} тапсырысты жеткізе алмады: мекенжай қате. Қайта жеткізу үшін {small} төлеңіз: {url}"),
    ("delivery", "{delivery}: сәлемдеме қоймада ұсталды. Алу үшін карта деректерін растаңыз: {url}"),
    ("delivery", "Ертеңге дейін {small} сақтау ақысын төлемесеңіз, №{order} тапсырыс жіберушіге қайтарылады: {url}"),
    ("delivery", "Жеткізудің соңғы әрекеті! Сәлемдемені алғаныңызды растаңыз: {url}"),
    ("relative", "{relative}, бұл менмін, басқа нөмірден жазып тұрмын. Шұғыл {amount} аудар, кейін бәрін түсіндіремін. Ешкімге айтпа."),
    ("relative", "{relative}, телефоным сынып қалды, досымның нөмірінен жазамын. Дереу {amount} керек, ертең қайтарамын."),
    ("relative", "{relative}, қиын жағдайға тап болдым, қазір ақша керек. *{last4} картасына {amount} аудар, қоңырау шала алмаймын."),
    ("relative", "{relative}, бұл {name}. Шұғыл осы нөмірге {amount} жібер, кейін түсіндіремін, телефонымды жоғалттым."),
    ("relative", "{relative}, шұғыл! Апатқа түстім, емделуге ақша керек. {amount} аудар, қазір маған қоңырау шалма."),
    ("code_request", "{bank} қауіпсіздік қызметі: {amount} аударымды болдырмау үшін SMS-тегі кодты операторға айтыңыз."),
    ("code_request", "Сізге растау коды келді. Шотты тексеруді аяқтау үшін оны біздің қызметкерге хабарлаңыз."),
    ("code_request", "Байланыс операторы: нөміріңізді ұзарту үшін SMS арқылы келетін кодты айтыңыз."),
    ("code_request", "Қате жіберілген кодты {phone} нөміріне қайта жіберіңізші, бұл шұғыл."),
    ("code_request", "{bank} қызметкері: ақшаңызды қорғау үшін хабарламадағы кодты айтыңыз. Бұл қауіпсіз."),
    ("gov", "eGov: сіздің атыңызға {mid} айыппұл салынды. {days} күн ішінде төлемесеңіз, сома екі есе өседі. {pct}% жеңілдікпен төлеу: {url}"),
    ("gov", "Сізге {amount} әлеуметтік төлем тиесілі. Алу үшін сауалнаманы толтырыңыз: {url}"),
    ("gov", "Салық қызметі: сізде {mid} қарыз бар. Шетелге шығуға шектеу қойылмас үшін бүгін төлеңіз: {url}"),
    ("gov", "ЭЦҚ жойылады. Ұзарту үшін деректеріңізді сілтеме арқылы енгізіңіз: {url}"),
    ("gov", "Жол ережесін бұзғаныңыз үшін {mid} айыппұл. Фото және төлем: {url}"),
    ("invest", "Тәжірибесіз аптасына {amount} табыс! ЖИ-трейдинг платформасы, тек {places_left} орын қалды: {url}"),
    ("invest", "Айына {pct}% кепілді пайда. {mid} салып, табыс таба бастаңыз: {url}"),
    ("invest", "Құпия инвестициялық бағдарлама. Ақшаны бір аптада екі есе көбейту жолын біліңіз: {url}"),
    ("invest", "Сізге инвестициялық шот мақұлданды. Белсендіру үшін алғашқы {mid} депозит салыңыз: {url}"),
    ("invest", "Тәуекелсіз пассивті табыс! {phone} нөміріне жазыңыз, орын саны шектеулі."),
]

KK_SAFE = [
    ("personal", "Сәлем! Ертең сағат {time}-де {place} кездесеміз."),
    ("personal", "Мен жолдамын, {days}0 минуттан кейін жетемін."),
    ("personal", "{name}, үйге келе жатып нан мен сүт ала келші."),
    ("personal", "Сабақ {time}-ге ауыстырылды, {room} аудитория."),
    ("personal", "{subject} бойынша конспектіні жіберші."),
    ("personal", "{days}0 минутқа кешігемін, менсіз бастай беріңдер."),
    ("personal", "{relative}, жеттім, бәрі жақсы. Кешке қоңырау шаламын."),
    ("personal", "Жұмада киноға барамыз ба? Сеанс {time}-де."),
    ("personal", "Курстық жұмысты қорғау {date}, титул парағын басып шығаруды ұмытпа."),
    ("personal", "Түскі асқа {mid} жібердім, тексеріп көр."),
    ("personal", "Топ жиналысы ертең {time}-де {room} аудиторияда."),
    ("personal", "{name}, туған күніңмен! Бақыт, денсаулық, табыс тілеймін!"),
    ("bank_legit", "{bank}: растау коды {code}. Бұл кодты ешкімге, тіпті банк қызметкерлеріне де айтпаңыз."),
    ("bank_legit", "{bank}: сатып алу {mid}, {shop}. Қолжетімді: {balance}. Карта *{last4}."),
    ("bank_legit", "{bank}: шотқа {amount} түсті. Жалақы. Карта *{last4}."),
    ("bank_legit", "{name} жіберген {mid} аударым *{last4} картасына сәтті түсті."),
    ("bank_legit", "{bank}: қосымшаға кіру коды {code}. Егер кодты сұрамаған болсаңыз, хабарламаны елемеңіз."),
    ("bank_legit", "{bank}: {date} дейін {mid} несие төлемін еске саламыз. Банк қосымшасында төлеуге болады."),
    ("bank_legit", "{bank}: жаңа картаңыз дайын. Оны бөлімшеден алыңыз, мекенжайы қосымшада көрсетілген."),
    ("bank_legit", "Банк ешқашан SMS-тегі кодты сұрамайды. Сақ болыңыз! {bank}"),
    ("bank_legit", "{bank}: {mid} кэшбэк есептелді. Карта *{last4}."),
    ("bank_legit", "{bank}: {shop} төлемі {mid}. Қалдық: {balance}."),
    ("bank_legit", "{bank}: шот {mid} сомасына толықтырылды. Карта *{last4}."),
    ("bank_legit", "{bank}: коммуналдық қызметтер үшін {mid} төлем сәтті өтті."),
    ("delivery_legit", "{delivery}: №{order} тапсырыс жеткізуге берілді. Күтілетін күні: {date}."),
    ("delivery_legit", "{delivery}: сәлемдеме беру пунктіне келді. Сақтау мерзімі {days} күн. Алу коды {code}."),
    ("delivery_legit", "Курьер {date} күні {time}-де келеді. Тапсырыс нөмірі {order}. Бақылау: {safe_url}"),
    ("delivery_legit", "{shop}: №{order} тапсырысыңыз жеткізілді. Сатып алғаныңыз үшін рахмет!"),
    ("delivery_legit", "{shop}: №{order} тапсырыс жиналды және курьерді күтуде."),
    ("delivery_legit", "{delivery} курьері: 20 минуттан кейін жетемін, телефоным {phone}."),
    ("service", "Дәрігерге жазылуыңыз расталды: {date}, сағат {time}, {room}-кабинет."),
    ("service", "Нөміріңіздегі баланс: {balance}. Оператор қосымшасында толтыруға болады."),
    ("service", "Тарифіңіз {date} дейін ұзартылды. Интернет қалдығы {gb} ГБ."),
    ("service", "{date} күні 02:00-ден 04:00-ге дейін желіде жоспарлы жұмыстар жүргізіледі. Кешірім сұраймыз."),
    ("service", "Университет порталына кіру кодыңыз: {code}."),
    ("service", "Кітапхана: кітапты қайтару мерзімі {date} аяқталады."),
    ("service", "Айлық коммуналдық қызметтер шоты дайын. Сомасы {mid}."),
    ("service", "Сіз {date} күні сағат {time}-де емтиханға жазылдыңыз. Жеке куәлігіңізді алыңыз."),
    ("promo_legit", "{shop}: демалыс күндері барлық аяқ киімге {pct}% жеңілдік. Дүкендерімізде күтеміз!"),
    ("promo_legit", "{shop}: жаңа коллекция сатылымда. Толығырақ: {safe_url}"),
    ("promo_legit", "Кофехана: адалдық картасы бойынша әр бесінші кофе сыйлыққа."),
    ("promo_legit", "{shop}: сатып алғаныңыз үшін рахмет! Электрондық чек: {safe_url}"),
    ("promo_legit", "{shop} қызмет сапасын мобильді қосымшада бағалаңыз. Сіздің пікіріңіз біз үшін маңызды."),
    ("promo_legit", "Кітап дүкенінде ай соңына дейін оқу әдебиетіне {pct}% дейін жеңілдік."),
]

TEMPLATES = {
    ("ru", 1): RU_FRAUD, ("ru", 0): RU_SAFE,
    ("kk", 1): KK_FRAUD, ("kk", 0): KK_SAFE,
}

# Вариации оформления, чтобы сообщения одного шаблона различались не только слотами
_FRAUD_PREFIX = {"ru": ["", "", "", "ВНИМАНИЕ! ", "Уважаемый клиент! ", "Важно! "],
                 "kk": ["", "", "", "НАЗАР АУДАРЫҢЫЗ! ", "Құрметті клиент! ", "Маңызды! "]}
_FRAUD_SUFFIX = {"ru": ["", "", "", " Не игнорируйте это сообщение.", " ⚠️", " Время ограничено!"],
                 "kk": ["", "", "", " Бұл хабарламаны елемеңіз.", " ⚠️", " Уақыт шектеулі!"]}
_PERSONAL_PREFIX = {"ru": ["", "", "Привет! ", "Слушай, ", "Ок. "],
                    "kk": ["", "", "Сәлем! ", "Тыңда, ", "Жарайды. "]}
_PERSONAL_SUFFIX = ["", "", "", " 😊", " )", " Спасибо!"]
_SAFE_SUFFIX = {
    "ru": {
        "bank_legit": ["", "", " Подробнее в приложении.", " Спасибо, что вы с нами.", " Хорошего дня!",
                       " Это автоматическое сообщение."],
        "delivery_legit": ["", "", " Спасибо!", " Хорошего дня!", " Ждём вас."],
        "service": ["", "", " Спасибо.", " Хорошего дня!", " Подробности в личном кабинете.",
                    " Это автоматическое сообщение."],
        "promo_legit": ["", "", " Ждём вас!", " Акция действует до конца недели.", " Хороших покупок!"],
    },
    "kk": {
        "bank_legit": ["", "", " Толығырақ қосымшада.", " Бізбен бірге болғаныңызға рахмет.",
                       " Күніңіз сәтті өтсін!", " Бұл автоматты хабарлама."],
        "delivery_legit": ["", "", " Рахмет!", " Күніңіз сәтті өтсін!", " Сізді күтеміз."],
        "service": ["", "", " Рахмет.", " Күніңіз сәтті өтсін!", " Толығырақ жеке кабинетте.",
                    " Бұл автоматты хабарлама."],
        "promo_legit": ["", "", " Сізді күтеміз!", " Акция апта соңына дейін жарамды.", " Сәтті сауда!"],
    },
}
_PERSONAL_SUFFIX_KK = ["", "", "", " 😊", " )", " Рахмет!"]


def _decorate(text: str, lang: str, label: int, category: str, rnd: random.Random) -> str:
    if label == 1:
        text = rnd.choice(_FRAUD_PREFIX[lang]) + text + rnd.choice(_FRAUD_SUFFIX[lang])
        if rnd.random() < 0.15:
            text = text.replace("!", "!!", 1)
    elif category == "personal":
        prefix = rnd.choice(_PERSONAL_PREFIX[lang])
        if prefix.endswith(", ") and not text.split()[0].isupper():
            text = text[0].lower() + text[1:]
        text = prefix + text + rnd.choice(_PERSONAL_SUFFIX if lang == "ru" else _PERSONAL_SUFFIX_KK)
    else:
        text = text + rnd.choice(_SAFE_SUFFIX[lang][category])
    return " ".join(text.split())


def generate(seed: int = SEED, per_class: int = PER_CLASS) -> pd.DataFrame:
    rnd = random.Random(seed)
    custom_norm = set()
    if CUSTOM_PATH.exists():
        custom_norm = set(pd.read_csv(CUSTOM_PATH)["text"].map(normalize_sms))

    rows, short = [], []
    for (lang, label), templates in TEMPLATES.items():
        seen_norm: set[str] = set()
        produced = {i: 0 for i in range(len(templates))}
        quota = {i: per_class // len(templates) + (1 if i < per_class % len(templates) else 0)
                 for i in range(len(templates))}
        for i, (category, template) in enumerate(templates):
            attempts = 0
            while produced[i] < quota[i] and attempts < 5000:
                attempts += 1
                text = _decorate(template.format(**make_slots(lang, rnd)), lang, label, category, rnd)
                norm = normalize_sms(text)
                if norm in seen_norm or norm in custom_norm:
                    continue  # без дублей внутри набора и с ручным тестом
                seen_norm.add(norm)
                produced[i] += 1
                rows.append({"lang": lang, "text": text, "label": label,
                             "label_name": "fraud" if label else "safe", "category": category,
                             "template_id": f"{lang}-{'f' if label else 's'}-{i:02d}"})
            if produced[i] < quota[i]:
                short.append((lang, label, i, quota[i] - produced[i]))
        # недобор у «бедных» шаблонов добираем остальными шаблонами того же класса
        order = list(range(len(templates)))
        while sum(produced.values()) < per_class:
            i = rnd.choice(order)
            category, template = templates[i]
            text = _decorate(template.format(**make_slots(lang, rnd)), lang, label, category, rnd)
            norm = normalize_sms(text)
            if norm in seen_norm or norm in custom_norm:
                continue
            seen_norm.add(norm)
            produced[i] += 1
            rows.append({"lang": lang, "text": text, "label": label,
                         "label_name": "fraud" if label else "safe", "category": category,
                         "template_id": f"{lang}-{'f' if label else 's'}-{i:02d}"})

    if short:
        print("Шаблоны с малым числом вариантов (добраны другими):", short)
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    df["author"] = AUTHOR
    return df


def main() -> int:
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8", lineterminator="\n")
    print(df.groupby(["lang", "label_name"]).size().to_string())
    print(f"Шаблонов: {df['template_id'].nunique()}, сообщений: {len(df)}")
    print(f"Сохранено: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
