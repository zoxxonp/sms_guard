# SMS Guard

Интеллектуальная система обнаружения мошеннических SMS-сообщений и подозрительных ссылок
на основе методов машинного обучения. Учебный проект бакалавриата.

**Автор:** Омаров Марлен Еркешович, SIS-2304

> Оценка системы носит вероятностный характер и не гарантирует безопасность сообщения или ссылки.
> Ссылки никогда не открываются: анализируется только их текст.

## Требования

* Python 3.11–3.13, Git
* ~1 ГБ свободного места (наборы данных и модели)

## Быстрый запуск (свежая установка)

Windows (PowerShell):

```powershell
git clone https://github.com/zoxxonp/sms_guard; cd sms_guard
py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # впишите SECRET_KEY (команда генерации — в самом файле)

# данные и модели (один раз; модели в git не хранятся)
python ml/data/download_sms.py
python ml/data/load_sms.py
python -m ml.data.clean_sms
python -m ml.data.download_urls
python -m ml.data.prepare_urls
python -m ml.train_sms          # ~2–4 мин → ml/artifacts/sms_model.joblib
python -m ml.train_url          # ~8–10 мин → ml/artifacts/url_model.joblib

# база данных и демо-данные
python manage.py migrate
python manage.py seed_demo      # пользователь demo_user / DemoPass2026 и 30 проверок
python manage.py createsuperuser   # (по желанию) администратор: статистика по всем пользователям

python manage.py runserver
```

Linux / macOS — те же команды, но `python3 -m venv .venv && source .venv/bin/activate` и `cp .env.example .env`.

Откройте <http://127.0.0.1:8000/> и войдите как `demo_user` / `DemoPass2026`.
Состояние приложения и моделей: <http://127.0.0.1:8000/api/v1/health/>.

Для демонстрационной сборки в `.env`: `DEBUG=False`; при работе по HTTPS — `SESSION_COOKIE_SECURE=True`,
`CSRF_COOKIE_SECURE=True`. Библиотеки интерфейса (Bootstrap, иконки, Plotly) лежат в `static/vendor/` —
интернет для работы приложения не нужен. При `DEBUG=False` статику раздаёт веб-сервер:
`python manage.py collectstatic` (папка `staticfiles/`).

## Тестирование

```bash
python -m pytest                                  # 239 автотестов
python -m pytest --cov=ml --cov=apps              # с покрытием (ml/ — 70%, модуль риска — 99%)
python scripts/qa_run.py                          # демо-сообщения, 5 сценариев, замер НФТ-1 → docs/qa/
python tests/e2e/ui_checks.py http://127.0.0.1:8000   # проверки интерфейса в браузере (нужен playwright)
```

Отчёт о тестировании — `docs/18_testing.md`.

## Страницы

| Страница | Адрес |
| :--- | :--- |
| Вход / регистрация | `/login/`, `/register/` |
| Dashboard | `/` |
| Проверка SMS / ссылки | `/check/sms/`, `/check/url/` |
| Результат проверки | `/checks/<id>/` |
| История | `/history/` |
| Статистика | `/stats/` |
| Профиль (пароль, согласие на хранение текстов, удаление истории) | `/profile/` |

## REST API

Аутентификация — сессия + CSRF-токен (заголовок `X-CSRFToken`). Ошибки — `{"error": {"code", "message"}}`.
Подробности и примеры curl — `docs/14_api.md`.

| Метод | Эндпоинт | Назначение | Ответ |
| :--- | :--- | :--- | :--- |
| POST | `/api/v1/auth/register/` | Регистрация `{"username","email","password"}` | 201 / 400 |
| POST | `/api/v1/auth/login/` | Вход | 200 / 401 / 429 |
| POST | `/api/v1/auth/logout/` | Выход | 204 |
| GET | `/api/v1/auth/csrf/` | Получить CSRF-cookie | 200 |
| POST | `/api/v1/check/sms/` | Анализ SMS `{"text": "..."}` | 200 / 400 / 429 / 503 |
| POST | `/api/v1/check/url/` | Анализ URL `{"url": "..."}` | 200 / 400 / 429 / 503 |
| GET | `/api/v1/checks/` | История: `date_from`, `date_to`, `type`, `category`, `risk_level`, `page` | 200 |
| GET / DELETE | `/api/v1/checks/{id}/` | Подробный результат / удаление своей записи | 200 / 204 / 404 |
| GET | `/api/v1/stats/summary/?days=7\|30\|90` | Данные Dashboard | 200 |
| GET | `/api/v1/health/` | Статус приложения и моделей | 200 |

```bash
curl -s -c cj.txt http://127.0.0.1:8000/api/v1/auth/csrf/
TOKEN=$(grep csrftoken cj.txt | awk '{print $7}')
curl -s -b cj.txt -c cj.txt -H "X-CSRFToken: $TOKEN" -H "Content-Type: application/json" \
     -d '{"username":"demo_user","password":"DemoPass2026"}' http://127.0.0.1:8000/api/v1/auth/login/
TOKEN=$(grep csrftoken cj.txt | awk '{print $7}')
curl -s -b cj.txt -H "X-CSRFToken: $TOKEN" -H "Content-Type: application/json" \
     -d '{"text":"Ваша карта заблокирована! Срочно подтвердите данные: http://secure-bank-verify.example/login"}' \
     http://127.0.0.1:8000/api/v1/check/sms/
```

## Данные и модели

| Команда | Задание | Результат |
| :--- | :--- | :--- |
| `python ml/data/download_sms.py`, `python ml/data/load_sms.py` | 3 | SMS Spam Collection (UCI) → `data/processed/sms_raw.csv` |
| `python -m ml.data.clean_sms` | 4 | `data/processed/sms_clean.csv` |
| `python -m ml.data.generate_ru_kk` | 7+ | синтетические 1000 ru + 1000 kk SMS (уже в репозитории) |
| `python -m ml.train_baseline [--multi]` | 7 | baseline-модели |
| `python -m ml.train_sms` | 8 | сравнение NB / LR / SVM / RF, итоговая `sms_model.joblib` |
| `python -m ml.data.download_urls`, `python -m ml.data.prepare_urls` | 10 | PhiUSIIL (UCI) → очищенная выборка 100 000 URL и признаки |
| `python -m ml.train_url` | 11 | сравнение LR / RF, `url_model.joblib`, пороги t1/t2 |
| ноутбук `08_risk_calibration` | 13 | веса и границы риска → `risk_config.yaml` |

Разбиения хранятся в `data/processed/*_split.json`, поэтому повторное обучение даёт те же выборки.
Параметры оценки риска (веса, границы уровней, правила усиления, доверенные домены, рекомендации) — в
`risk_config.yaml`; файл перечитывается при изменении, перезапуск не нужен.

Ноутбуки (`jupyter notebook notebooks/`) содержат сохранённые результаты:

| Ноутбук | Задание |
| :--- | :--- |
| `01_eda_sms.ipynb` | 5 — исследовательский анализ |
| `02_features_sms.ipynb` | 6 — разбиение и признаки |
| `03_baseline_sms.ipynb`, `04_multilang_sms.ipynb` | 7 — baseline (en; en + ru + kk) |
| `05_sms_models.ipynb` | 8 — сравнение SMS-моделей |
| `06_url_dataset.ipynb` | 10 — набор URL |
| `07_url_models.ipynb` | 11 — сравнение URL-моделей |
| `08_risk_calibration.ipynb` | 13 — калибровка риска |

## Структура

```text
config/          настройки Django, urls.py, wsgi.py
apps/accounts/   регистрация, вход (блокировка после 5 попыток), профиль, API авторизации
apps/analysis/   сервис анализа, REST API, модели БД, маскирование, история, seed_demo
apps/dashboard/  Dashboard и статистика
ml/              предобработка, признаки SMS и URL, словарь правил, обучение, Predictor, risk.py
ml/data/         загрузка и подготовка наборов данных
ml/artifacts/    модели (*.joblib, не в git) и метаданные (*.json)
risk_config.yaml параметры модуля оценки риска
notebooks/       Jupyter-ноутбуки
data/            исходные и обработанные данные
templates/, static/   интерфейс (Bootstrap 5, Plotly — локальные копии в static/vendor)
tests/           pytest; tests/e2e — проверки в браузере
scripts/         qa_run.py — прогон демо-сценариев для отчёта
docs/            отчёты по заданиям, диаграммы, скриншоты, результаты тестирования
logs/            app.log с ротацией (не в репозитории)
```

## Ход работы

| Задание | Отчёт и основные файлы |
| :--- | :--- |
| 1 | `docs/01_domain_analysis.md`, `docs/diagrams/use_case.*` |
| 2 | каркас Django, health, логирование |
| 3 | `docs/03_dataset_passport.md`, `data/raw/custom_ru_kk.csv` |
| 4 | `ml/preprocessing.py`, `docs/04_preprocessing_report.md` |
| 5 | `notebooks/01_eda_sms.ipynb`, `docs/05_eda_report.md` |
| 6 | `ml/features.py`, `ml/rules.py`, `docs/06_features.md` |
| 7 / 7+ | `ml/train_baseline.py`, `docs/07_baseline_report.md`, `docs/03b_synthetic_dataset.md`, `docs/07b_multilang_baseline.md` |
| 8 | `ml/train_sms.py`, `docs/08_sms_models.md` |
| 9 | `ml/url_features.py`, `docs/09_url_features.md` |
| 10 | `ml/data/prepare_urls.py`, `docs/10_url_dataset.md` |
| 11 | `ml/train_url.py`, `docs/11_url_models.md` |
| 12 | `ml/predictor.py`, `apps/analysis/services.py`, `docs/12_analysis_service.md` |
| 13 | `ml/risk.py`, `risk_config.yaml`, `docs/13_risk.md` |
| 14 | REST API, `docs/14_api.md` |
| 15 | интерфейс, `docs/15_ui.md` (скриншоты 360 / 768 / 1366) |
| 16 | модели БД, история, профиль, `seed_demo`, `docs/16_database.md` (ER-диаграмма) |
| 17 | Dashboard и статистика, `docs/17_dashboard.md` |
| 18 | `docs/18_testing.md`, `docs/qa/` |
