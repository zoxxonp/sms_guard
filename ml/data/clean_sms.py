"""Очистка набора SMS и отчёт о преобразованиях (задание 4).

Запуск из корня проекта:
    python -m ml.data.clean_sms

Вход:  data/processed/sms_raw.csv   (результат ml/data/load_sms.py)
Выход: data/processed/sms_clean.csv (id, text_raw, text_clean, label)
       docs/04_preprocessing_report.md (таблица «шаг — число затронутых записей»)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

from ml.preprocessing import normalize_sms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "processed" / "sms_raw.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "sms_clean.csv"
REPORT_MD = PROJECT_ROOT / "docs" / "04_preprocessing_report.md"

CLASS_NAMES = {0: "ham (0)", 1: "spam (1)"}


def _has_control(text: str) -> bool:
    return any(unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\n\t " for ch in text)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    """Возвращает очищенный DataFrame, строки отчёта и сводку по дубликатам."""
    df = df.rename(columns={"text": "text_raw"}).copy()
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    n_start = len(df)

    df["text_clean"] = df["text_raw"].map(normalize_sms)
    clean = df["text_clean"]

    steps = [
        ("Исходные записи (sms_raw.csv)", n_start, ""),
        ("Unicode-нормализация NFKC изменила текст",
         int((df["text_raw"].map(lambda t: unicodedata.normalize("NFKC", t)) != df["text_raw"]).sum()),
         "полноширинные символы, лигатуры, неразрывные пробелы"),
        ("Раскодированы HTML-сущности (&lt; &gt; &amp;)",
         int(df["text_raw"].str.contains(r"&(?:lt|gt|amp|quot|#\d+);", regex=True).sum()),
         "метки &lt;#&gt;, &lt;DECIMAL&gt;, &lt;TIME&gt; → <num>, &lt;URL&gt; → <url>"),
        ("Исправлены / удалены управляющие и невидимые символы",
         int(df["text_raw"].map(_has_control).sum()), "C1-символы cp1252 (\\x92 и др.) → апострофы и кавычки; прочие Cc, Cf → пробел"),
        ("Приведение к нижнему регистру (были заглавные буквы)",
         int(df["text_raw"].map(lambda t: t != t.lower()).sum()), ""),
        ("URL → <url>", int(clean.str.contains("<url>", regex=False).sum()), ""),
        ("E-mail → <email>", int(clean.str.contains("<email>", regex=False).sum()), ""),
        ("Сумма с валютой → <money>", int(clean.str.contains("<money>", regex=False).sum()), "£, $, €, ₸, тг, руб, pence"),
        ("Телефон → <phone>", int(clean.str.contains("<phone>", regex=False).sum()), "7–15 цифр"),
        ("Прочие числа → <num>", int(clean.str.contains("<num>", regex=False).sum()), ""),
        ("«!» / «?» выделены как признаки давления",
         int(clean.str.contains(r"[!?]", regex=True).sum()), "сохраняются отдельными токенами"),
        ("Схлопнуты повторяющиеся пробелы",
         int(df["text_raw"].str.contains(r"\s{2,}", regex=True).sum()), ""),
    ]

    empty_mask = df["text_clean"].str.strip() == ""
    n_empty = int(empty_mask.sum())
    df = df[~empty_mask]
    steps.append(("Удалены строки с пустым текстом", n_empty, "после очистки"))

    # Конфликт меток: один и тот же text_clean встречается и как ham, и как spam
    labels_per_text = df.groupby("text_clean")["label"].nunique()
    conflicts = labels_per_text[labels_per_text > 1].index
    n_conflict_rows = int(df["text_clean"].isin(conflicts).sum())

    dup_mask = df.duplicated(subset="text_clean", keep="first")
    dups_by_class = df.loc[dup_mask, "label"].value_counts().reindex([0, 1], fill_value=0)
    raw_dups_by_class = (
        df.loc[df.duplicated(subset="text_raw", keep="first"), "label"]
        .value_counts().reindex([0, 1], fill_value=0)
    )
    df = df[~dup_mask]
    steps.append(("Удалены дубликаты по text_clean", int(dup_mask.sum()),
                  f"ham — {dups_by_class[0]}, spam — {dups_by_class[1]}"))
    steps.append(("Итог (sms_clean.csv)", len(df), ""))

    dup_summary = {
        "raw_dups": raw_dups_by_class.to_dict(),
        "clean_dups": dups_by_class.to_dict(),
        "conflict_texts": len(conflicts),
        "conflict_rows": n_conflict_rows,
    }
    df = df[["id", "text_raw", "text_clean", "label"]].reset_index(drop=True)
    return df, [dict(step=s, count=c, note=n) for s, c, n in steps], dup_summary


def class_table(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({
        "До очистки": before.value_counts().reindex([0, 1], fill_value=0),
        "После очистки": after.value_counts().reindex([0, 1], fill_value=0),
    })
    table["Удалено"] = table["До очистки"] - table["После очистки"]
    table["Доля после, %"] = (table["После очистки"] / table["После очистки"].sum() * 100).round(2)
    table.index = [CLASS_NAMES[i] for i in table.index]
    return table


_MD_TOKEN_RE = re.compile(r"(<\w+>|&lt;\S*?&gt;|&\w+;)")


def _md_code(text: str) -> str:
    """Оборачивает токены (<url>) и HTML-сущности в `код`, иначе GitHub примет их за теги."""
    return _MD_TOKEN_RE.sub(lambda m: f"`{m.group(1)}`", text)


def write_report(steps: list[dict], classes: pd.DataFrame, dup: dict, examples: pd.DataFrame) -> str:
    lines = [
        "# Задание 4. Отчёт о предварительной обработке SMS",
        "",
        f"*Сгенерирован скриптом `python -m ml.data.clean_sms`, {date.today():%d.%m.%Y}.*",
        "",
        "Функция обработки — `ml/preprocessing.py::normalize_sms`. Она же используется "
        "при обучении моделей и в веб-приложении.",
        "",
        "## 1. Шаги обработки",
        "",
        "Шаги 2–12 не удаляют записи, а изменяют текст; в столбце указано, у скольких "
        "сообщений шаг что-то изменил (одна запись может попасть в несколько шагов).",
        "",
        "| № | Шаг | Затронуто записей | Примечание |",
        "| :-: | :--- | --: | :--- |",
    ]
    for i, s in enumerate(steps, 1):
        lines.append(f"| {i} | {_md_code(s['step'])} | {s['count']} | {_md_code(s['note'])} |")

    lines += [
        "",
        "## 2. Классы до и после очистки",
        "",
        "| Класс | До очистки | После очистки | Удалено | Доля после, % |",
        "| :--- | --: | --: | --: | --: |",
    ]
    for name, row in classes.iterrows():
        lines.append(f"| {name} | {row['До очистки']} | {row['После очистки']} | "
                     f"{row['Удалено']} | {row['Доля после, %']:.2f} |")

    lines += [
        "",
        "## 3. Дубликаты по классам",
        "",
        "| Класс | Точных дубликатов исходного текста | Удалено дубликатов по `text_clean` |",
        "| :--- | --: | --: |",
        f"| ham (0) | {dup['raw_dups'][0]} | {dup['clean_dups'][0]} |",
        f"| spam (1) | {dup['raw_dups'][1]} | {dup['clean_dups'][1]} |",
        f"| **Всего** | **{dup['raw_dups'][0] + dup['raw_dups'][1]}** | "
        f"**{dup['clean_dups'][0] + dup['clean_dups'][1]}** |",
        "",
        "Дубликатов по `text_clean` больше, чем по исходному тексту: после замены чисел, "
        "телефонов и сумм токенами сообщения-шаблоны, отличающиеся только номером или "
        "суммой, становятся одинаковыми. Для модели это один и тот же пример, поэтому "
        "он оставлен один раз (первое вхождение).",
        "",
        f"Текстов, встречающихся с разными метками (ham и spam одновременно): "
        f"**{dup['conflict_texts']}**. "
        + ("Для них оставлено первое вхождение." if dup["conflict_texts"] else ""),
        "",
        "Дубликаты удалены **до** разбиения на выборки (задание 6), чтобы одно и то же "
        "сообщение не попало одновременно в обучающую и тестовую выборки.",
        "",
        "## 4. Примеры преобразования",
        "",
        "| id | Класс | text_raw | text_clean |",
        "| --: | :-: | :--- | :--- |",
    ]
    for _, r in examples.iterrows():
        raw = r["text_raw"].replace("|", "\\|")
        cln = r["text_clean"].replace("|", "\\|")
        lines.append(f"| {r['id']} | {r['label']} | {raw} | `{cln}` |")

    lines += [
        "",
        "## 5. Стоп-слова",
        "",
        "Удаление стоп-слов реализовано как опция: `normalize_sms(text, remove_stop_words=True)` "
        "или `remove_stopwords(text)`. В `sms_clean.csv` стоп-слова **не удалены**: их влияние "
        "на качество сравнивается в задании 8. Списки: английский — scikit-learn, русский и "
        "казахский — служебные слова, заданные в модуле (без загрузки корпусов NLTK).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if not RAW_CSV.exists():
        print(f"Нет файла {RAW_CSV}. Сначала выполните: python ml/data/load_sms.py", file=sys.stderr)
        return 1
    raw = pd.read_csv(RAW_CSV, keep_default_na=False)
    clean, steps, dup = clean_dataset(raw)
    classes = class_table(raw["label"], clean["label"])

    pick_ids = [3, 9, 12, 16, 1, 7]  # иллюстративные примеры: деньги, телефоны, числа, URL
    examples = clean[clean["id"].isin(pick_ids)]
    url_example = clean[clean["text_clean"].str.contains("<url>", regex=False)].head(2)
    examples = pd.concat([examples, url_example]).drop_duplicates("id")

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    REPORT_MD.write_text(write_report(steps, classes, dup, examples), encoding="utf-8")

    print(pd.DataFrame(steps).to_string(index=False))
    print()
    print(classes.to_string())
    print(f"\nСохранено: {CLEAN_CSV.relative_to(PROJECT_ROOT)}, {REPORT_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
