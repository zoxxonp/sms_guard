"""Загрузка UCI SMS Spam Collection из официального источника (задание 3, п. 1).

Запуск из корня проекта:
    python ml/data/download_sms.py

Архив сохраняется в data/raw/ без изменений, файл набора распаковывается в
data/raw/sms_spam_collection/. Скрипт выводит размер и MD5, чтобы дату загрузки
и контрольную сумму можно было записать в паспорт набора.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_DIR = RAW_DIR / "sms_spam_collection"
ZIP_PATH = RAW_DIR / "sms_spam_collection.zip"

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
EXPECTED_ROWS = 5574  # по описанию набора на сайте UCI


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка: {SOURCE_URL}")
    try:
        urllib.request.urlretrieve(SOURCE_URL, ZIP_PATH)
    except OSError as exc:
        print(f"Ошибка загрузки: {exc}", file=sys.stderr)
        print("Скачайте архив вручную со страницы "
              "https://archive.ics.uci.edu/dataset/228/sms+spam+collection "
              f"и положите его в {ZIP_PATH}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(TARGET_DIR)
        names = zf.namelist()

    data_file = TARGET_DIR / "SMSSpamCollection"
    rows = sum(1 for line in data_file.read_bytes().split(b"\n") if line.strip())

    print(f"Дата загрузки: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Файлы в архиве: {names}")
    print(f"Архив: {ZIP_PATH.name}, {ZIP_PATH.stat().st_size} байт, MD5 {md5(ZIP_PATH)}")
    print(f"Набор: {data_file.name}, {data_file.stat().st_size} байт, MD5 {md5(data_file)}")
    print(f"Строк: {rows} (ожидается {EXPECTED_ROWS})")
    if rows != EXPECTED_ROWS:
        print("ВНИМАНИЕ: число строк не совпадает с описанием набора", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
