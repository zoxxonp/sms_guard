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

## Данные и модель

Команды выполняются из корня проекта по порядку:

```bash
python ml/data/download_sms.py   # задание 3: скачать набор с UCI в data/raw/
python ml/data/load_sms.py       # задание 3: → data/processed/sms_raw.csv (id, text, label)
python -m ml.data.clean_sms      # задание 4: очистка → data/processed/sms_clean.csv + отчёт
python -m ml.train_baseline      # задание 7: обучение baseline → ml/artifacts/sms_baseline.joblib
python -m ml.data.generate_ru_kk         # синтетический набор: 1000 ru + 1000 kk SMS
python -m ml.train_baseline --multi      # модель en + ru + kk → ml/artifacts/sms_baseline_multi.joblib
```

`predict_sms` использует многоязычную модель, если она обучена, иначе английскую.

Разбиение train / val / test хранится в `data/processed/sms_split.json` (задание 6); если файла нет,
`ml.train_baseline` создаёт его заново с `random_state=42`.

Ноутбуки исследований (выводы и графики уже сохранены внутри):

```bash
jupyter notebook notebooks/
```

| Ноутбук | Задание |
| :--- | :--- |
| `01_eda_sms.ipynb` | 5 — исследовательский анализ |
| `02_features_sms.ipynb` | 6 — разбиение и признаки |
| `03_baseline_sms.ipynb` | 7 — baseline-модель |
| `04_multilang_sms.ipynb` | 7 (дополнение) — модель на английском, русском и казахском |

Паспорт набора — `docs/03_dataset_passport.md`.

## Структура

```text
config/          настройки Django, urls.py, wsgi.py
apps/accounts/   регистрация, вход, профиль
apps/analysis/   сервис анализа, API, модели БД проверок
apps/dashboard/  статистика и графики
ml/              предобработка, признаки, словарь правил, модели, предсказание
ml/data/         скрипты загрузки и очистки наборов данных
ml/artifacts/    сохранённые модели (*.joblib, не в git) и метаданные (*.json)
notebooks/       Jupyter-ноутбуки исследований
data/raw/        исходные наборы данных (без изменений)
data/processed/  обработанные данные и индексы разбиения
docs/figures/    графики для отчётов
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
| 4 | `ml/preprocessing.py` (`normalize_sms`), `data/processed/sms_clean.csv`, `docs/04_preprocessing_report.md`, `tests/test_preprocessing.py` |
| 5 | `notebooks/01_eda_sms.ipynb`, `docs/figures/05_*.png`, `docs/05_eda_report.md` |
| 6 | `ml/features.py`, `ml/rules.py`, `data/processed/sms_split.json`, `notebooks/02_features_sms.ipynb`, `docs/06_features.md` |
| 7 | `ml/predictor.py` (`predict_sms`), `ml/train_baseline.py`, `notebooks/03_baseline_sms.ipynb`, `docs/07_baseline_report.md` |
| 7+ | `ml/data/generate_ru_kk.py`, `data/raw/synthetic_ru_kk.csv` (1000 ru + 1000 kk), `docs/03b_synthetic_dataset.md`, `notebooks/04_multilang_sms.ipynb`, `docs/07b_multilang_baseline.md` |
