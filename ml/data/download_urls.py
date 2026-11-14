"""Загрузка набора PhiUSIIL Phishing URL Dataset (UCI, id 967) — задание 10.

Запуск из корня проекта:
    python -m ml.data.download_urls

Архив (~ 22 МБ) скачивается с сайта UCI в data/raw/url/ и распаковывается;
файл PhiUSIIL_Phishing_URL_Dataset.csv (~ 54 МБ) в git не добавляется (.gitignore).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URL_DIR = PROJECT_ROOT / "data" / "raw" / "url"
ZIP_PATH = URL_DIR / "phiusiil_phishing_url_dataset.zip"
CSV_PATH = URL_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
EXPECTED_ROWS = 235795


def main() -> int:
    URL_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        print(f"Загрузка: {SOURCE_URL}")
        try:
            urllib.request.urlretrieve(SOURCE_URL, ZIP_PATH)
        except OSError as exc:
            print(f"Ошибка загрузки: {exc}\nСкачайте архив вручную со страницы "
                  "https://archive.ics.uci.edu/dataset/967 и распакуйте CSV в " + str(URL_DIR), file=sys.stderr)
            return 1
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(URL_DIR)
    md5 = hashlib.md5(CSV_PATH.read_bytes()).hexdigest()
    rows = sum(1 for _ in CSV_PATH.open(encoding="utf-8-sig")) - 1
    print(f"Дата проверки: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{CSV_PATH.name}: {CSV_PATH.stat().st_size} байт, MD5 {md5}, строк {rows} (ожидается {EXPECTED_ROWS})")
    return 0 if rows == EXPECTED_ROWS else 2


if __name__ == "__main__":
    sys.exit(main())
