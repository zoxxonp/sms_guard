# Задание 14. Backend и REST API

**Автор:** Омаров Марлен Еркешович, SIS-2304

Django REST Framework. Код: `apps/analysis/api.py`, `apps/accounts/api.py`, `apps/analysis/apiutils.py`
(единый формат ошибок), маршруты — `apps/analysis/api_urls.py`. Тесты — `tests/test_api.py`.

## 1. Эндпоинты

| Метод | Эндпоинт | Назначение | Ответ |
| :--- | :--- | :--- | :--- |
| POST | `/api/v1/auth/register/` | Регистрация `{"username","email","password"}`, сразу вход | 201 / 400 |
| POST | `/api/v1/auth/login/` | Вход `{"username","password"}` (сессия) | 200 / 401 / 429 |
| POST | `/api/v1/auth/logout/` | Выход, сессия уничтожается | 204 |
| GET | `/api/v1/auth/csrf/` | Выдать CSRF-cookie (для curl и внешних клиентов) | 200 |
| POST | `/api/v1/check/sms/` | Анализ SMS `{"text": "..."}` | 200 / 400 / 401 / 429 / 503 |
| POST | `/api/v1/check/url/` | Анализ URL `{"url": "..."}` | 200 / 400 / 401 / 429 / 503 |
| GET | `/api/v1/checks/` | История: `date_from`, `date_to` (ГГГГ-ММ-ДД), `type`, `category`, `risk_level`, `page` | 200 / 400 / 404 |
| GET | `/api/v1/checks/{id}/` | Подробный результат (только своя запись) | 200 / 404 |
| DELETE | `/api/v1/checks/{id}/` | Удаление своей записи | 204 / 404 |
| GET | `/api/v1/stats/summary/?days=7\|30\|90` | Данные Dashboard; администратору — по всем пользователям | 200 |
| GET | `/api/v1/health/` | Статус приложения и моделей (без входа) | 200 |

Пример ответа `check/sms` — как в разделе 4.4 ТЗ, плюс `category_ru`, `risk_level_ru`, `p_url_max`, `language_note`,
`features_detail`, `processing_ms`, `input` (сохранённый или маскированный текст) и `text_stored`.

**Ошибки** — всегда `{"error": {"code": "...", "message": "..."}}`, сообщение на русском:

| HTTP | code | Когда |
| --: | :--- | :--- |
| 400 | `VALIDATION_ERROR` | пустой текст, > 1000 символов, URL > 2048, неверный формат URL или фильтра, не JSON |
| 401 | `NOT_AUTHENTICATED`, `INVALID_CREDENTIALS` | нет входа; неверный логин/пароль |
| 403 | `CSRF_FAILED` | POST без CSRF-токена |
| 404 | `NOT_FOUND` | запись не существует или принадлежит другому пользователю |
| 429 | `THROTTLED`, `TOO_MANY_ATTEMPTS` | > 30 проверок в минуту; 5 неудачных входов подряд (пауза 60 с, заголовок `Retry-After`) |
| 503 | `MODEL_UNAVAILABLE` | файл модели отсутствует — «Анализ временно недоступен: модель не загружена» |

## 2. Требования задания

| Требование | Реализация |
| :--- | :--- |
| Валидация ввода | Сериализаторы: текст 1–1000 символов после удаления управляющих символов (`\x00` и т. п.), URL до 2048; формат URL — `validate_url` (задание 9) |
| Регистрация / вход / выход | ФТ-1.1: логин 3–30 (латиница, цифры, `_`), e-mail, пароль ≥ 8 с буквами и цифрами + валидаторы Django; ФТ-1.2: уникальность без учёта регистра; ФТ-1.3: блокировка на 60 с после 5 неудач (кэш); ФТ-1.4: PBKDF2; ФТ-1.5: `logout` удаляет сессию |
| check/sms, check/url | сервис задания 12 → риск задания 13 → сохранение в транзакции (задание 16) → JSON |
| История | фильтры и пагинация по 20; запросы всегда `Check.objects.filter(user=request.user)` → чужая запись даёт 404 |
| Throttling | `ScopedRateThrottle`, scope `checks`, 30/min на пользователя → 429 |
| Повторная отправка | тот же ввод того же пользователя в течение 10 с возвращает уже сохранённую запись (`"duplicate": true`), дубликат не создаётся (НФТ-3) |
| Ошибки | `api_exception_handler`; отсутствие модели → 503; необработанное исключение → 500 с общим сообщением, детали только в логе |
| Логирование | `check id=… type=… category=… risk=… analysis_ms=… total_ms=…`; текст SMS и логин в лог не пишутся (НФТ-9) |
| Безопасность | только ORM (НФТ-4), CSRF для всех POST через сессию, в т. ч. login/register; cookie HttpOnly + SameSite=Lax; `X-Frame-Options: DENY`, `nosniff` |

## 3. Примеры curl

Сессионный API требует CSRF-токен. Cookie сохраняются в файл `cj.txt` (в PowerShell используйте `curl.exe`).

```bash
# 1. CSRF-cookie
curl -s -c cj.txt http://127.0.0.1:8000/api/v1/auth/csrf/
TOKEN=$(grep csrftoken cj.txt | awk '{print $7}')

# 2. Вход
curl -s -b cj.txt -c cj.txt -H "X-CSRFToken: $TOKEN" -H "Content-Type: application/json" \
     -d '{"username":"demo_user","password":"DemoPass2026"}' http://127.0.0.1:8000/api/v1/auth/login/
TOKEN=$(grep csrftoken cj.txt | awk '{print $7}')      # токен обновляется после входа

# 3. Проверка SMS
curl -s -b cj.txt -H "X-CSRFToken: $TOKEN" -H "Content-Type: application/json" \
     -d '{"text":"Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login"}' \
     http://127.0.0.1:8000/api/v1/check/sms/

# 4. Проверка URL
curl -s -b cj.txt -H "X-CSRFToken: $TOKEN" -H "Content-Type: application/json" \
     -d '{"url":"http://xn--exmple-cua.test/secure/login?verify=1"}' http://127.0.0.1:8000/api/v1/check/url/

# 5. История, подробный результат, удаление, статистика, health
curl -s -b cj.txt "http://127.0.0.1:8000/api/v1/checks/?type=SMS&category=FRAUDULENT&page=1"
curl -s -b cj.txt http://127.0.0.1:8000/api/v1/checks/1/
curl -s -b cj.txt -X DELETE -H "X-CSRFToken: $TOKEN" http://127.0.0.1:8000/api/v1/checks/1/
curl -s -b cj.txt "http://127.0.0.1:8000/api/v1/stats/summary/?days=30"
curl -s http://127.0.0.1:8000/api/v1/health/
```

## 4. Тесты (`tests/test_api.py`, 41 тест с параметрами — фактически пройдены)

Регистрация (успех, 5 вариантов неверных данных, повтор логина и e-mail), хеш пароля, вход/выход, неверный пароль → 401,
блокировка после 5 попыток → 429, все защищённые эндпоинты без входа → 401, CSRF без токена → 403 и с токеном → 200,
check/sms (полный ответ, сохранение ссылок и признаков), SMS без URL (`p_url_max = null`), check/url (Punycode → ≥ 60),
пустой/пробельный/отсутствующий/1001 символ → 400, ровно 1000 символов → 200, управляющие символы, 5 вариантов URL,
не-JSON → 400, повторная отправка без дубликата, 31-й запрос в минуту → 429, отсутствие модели → 503 и `health: degraded`,
фильтры и пагинация, просмотр и удаление своей записи (каскад), чужая запись → 404 (GET и DELETE),
SQL-инъекция в тексте и фильтре, XSS-строка хранится как данные и экранируется на странице, статистика пользователя
и администратора (без чужих текстов), отсутствие текста SMS в логах.
