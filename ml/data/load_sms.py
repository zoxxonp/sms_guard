"""Загрузка набора UCI SMS Spam Collection и приведение к единой структуре (задание 3).

Запуск из корня проекта:
    python ml/data/load_sms.py

Результат: data/processed/sms_raw.csv (UTF-8) со столбцами
    id    — порядковый номер строки исходного файла (с 1);
    text  — текст SMS без изменений;
    label — 0 = ham (безопасное), 1 = spam (мошенничество).

Формат исходного файла: одна строка = одно сообщение, «метка<TAB>текст».
Файл читается построчно, а не через pandas.read_csv: в текстах SMS встречаются
кавычки, и CSV-парсер с настройками по умолчанию склеивает соседние строки.
Так гарантируется, что число строк результата совпадает с исходным файлом.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sms_spam_collection" / "SMSSpamCollection"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "sms_raw.csv"

LABEL_MAP = {"ham": 0, "spam": 1}
LABEL_NAMES = {0: "ham (безопасное)", 1: "spam (мошенничество)"}


def detect_encoding(raw: bytes) -> str:
    """Явно проверяет кодировку: UTF-8 (строго), иначе cp1252/latin-1."""
    for encoding in ("utf-8", "cp1252"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"  # декодирует любые байты


def load_sms(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Читает исходный файл и возвращает DataFrame со столбцами id, text, label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл набора: {path}\n"
            "Сначала выполните: python ml/data/download_sms.py"
        )

    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    lines = raw.decode(encoding).splitlines()

    records, bad_lines = [], []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            bad_lines.append((line_no, "пустая строка"))
            continue
        label_str, sep, text = line.partition("\t")
        label_str = label_str.strip().lower()
        if not sep or label_str not in LABEL_MAP:
            bad_lines.append((line_no, line[:60]))
            continue
        records.append({"id": line_no, "text": text, "label": LABEL_MAP[label_str]})

    df = pd.DataFrame(records, columns=["id", "text", "label"])
    df = df.astype({"id": "int64", "text": "string", "label": "int8"})
    df.attrs.update(
        source_rows=len([ln for ln in lines if ln.strip()]),
        encoding=encoding,
        bad_lines=bad_lines,
    )
    return df


def print_report(df: pd.DataFrame, examples: int = 5) -> None:
    """Выводит сводку по набору (задание 3, п. 3)."""
    print(f"Кодировка исходного файла: {df.attrs.get('encoding')}")
    print(f"Непустых строк в исходном файле: {df.attrs.get('source_rows')}")
    print(f"Строк в DataFrame: {len(df)}")
    bad = df.attrs.get("bad_lines") or []
    print(f"Некорректных строк: {len(bad)}" + (f" — {bad[:5]}" if bad else ""))
    print(f"Пропуски по столбцам: {df.isna().sum().to_dict()}")
    print(f"Пустые тексты: {int((df['text'].str.strip() == '').sum())}")
    print(f"Точных дубликатов текста: {int(df['text'].duplicated().sum())} "
          "(удаляются в задании 4)")

    print("\nТипы столбцов:")
    print(df.dtypes.to_string())

    print("\nРаспределение классов:")
    counts = df["label"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label} — {LABEL_NAMES[label]:<22} {count:>5}  ({count / len(df):6.2%})")

    with pd.option_context("display.max_colwidth", 110, "display.width", 160):
        for label in (0, 1):
            print(f"\nПримеры класса {label} — {LABEL_NAMES[label]}:")
            sample = df[df["label"] == label].head(examples)
            for _, row in sample.iterrows():
                text = row["text"] if len(row["text"]) <= 110 else row["text"][:107] + "..."
                print(f"  [{row['id']:>4}] {text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=RAW_PATH, help="путь к SMSSpamCollection")
    parser.add_argument("--output", type=Path, default=OUT_PATH, help="путь к sms_raw.csv")
    args = parser.parse_args(argv)

    df = load_sms(args.input)
    print_report(df)

    if len(df) != df.attrs["source_rows"]:
        print("ОШИБКА: число строк не совпадает с исходным файлом", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    print(f"\nСохранено: {args.output.relative_to(PROJECT_ROOT) if args.output.is_relative_to(PROJECT_ROOT) else args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
