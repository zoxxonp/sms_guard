# SMS Guard

Интеллектуальная система обнаружения мошеннических SMS-сообщений и подозрительных ссылок
на основе методов машинного обучения. Учебный проект бакалавриата.

**Автор:** Омаров Марлен Еркешович, SIS-2304

> Оценка системы носит вероятностный характер и не гарантирует безопасность сообщения или ссылки.

## Требования

* Python 3.11–3.13
* Git

## Запуск

Linux / macOS:

```bash
git clone <адрес-репозитория> sms_guard && cd sms_guard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # затем впишите SECRET_KEY (команда генерации — в самом файле)
python manage.py migrate
python manage.py runserver
```

Windows (PowerShell):

```powershell
git clone <адрес-репозитория> sms_guard; cd sms_guard
py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # затем впишите SECRET_KEY
python manage.py migrate
python manage.py runserver
```

После запуска:

* главная страница — <http://127.0.0.1:8000/>
* проверка состояния — <http://127.0.0.1:8000/api/v1/health/> → `{"status": "ok"}`

Проверка конфигурации и тесты:

```bash
python manage.py check
pytest
```

## Данные

Набор UCI SMS Spam Collection (задание 3):

```bash
python ml/data/download_sms.py   # скачивает архив с UCI в data/raw/ и проверяет файл
python ml/data/load_sms.py       # приводит к виду id, text, label → data/processed/sms_raw.csv
```

Паспорт набора — `docs/03_dataset_passport.md`.

## Структура

```text
config/          настройки Django, urls.py, wsgi.py
apps/accounts/   регистрация, вход, профиль
apps/analysis/   сервис анализа, API, модели БД проверок
apps/dashboard/  статистика и графики
ml/              предобработка, признаки URL, правила, риск, предсказание
ml/data/         скрипты загрузки наборов данных
artifacts/       сохранённые модели (*.joblib) и метаданные
notebooks/       Jupyter-ноутбуки исследований
data/raw/        исходные наборы данных (без изменений)
data/processed/  обработанные данные и индексы разбиения
templates/       HTML-шаблоны (Bootstrap 5)
static/          CSS, JS
tests/           pytest
docs/            отчёты и диаграммы
logs/            app.log с ротацией (не в репозитории)
```

## Ход работы

| Задание | Результат |
| :--- | :--- |
| 1 | `docs/01_domain_analysis.md`, `docs/diagrams/use_case.puml/.png` |
| 2 | Каркас Django, health-эндпоинт, логирование, README |
| 3 | `ml/data/load_sms.py`, `data/processed/sms_raw.csv`, паспорт набора, `data/raw/custom_ru_kk.csv` |
