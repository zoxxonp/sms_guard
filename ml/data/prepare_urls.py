"""Подготовка набора URL: очистка, устранение артефактов, разбиение и признаки (задание 10).

Запуск из корня проекта (после python -m ml.data.download_urls):
    python -m ml.data.prepare_urls

Шаги:
1. Чтение data/raw/url/PhiUSIIL_Phishing_URL_Dataset.csv (только столбцы URL и label:
   остальные 54 признака набора получены при загрузке страниц, а система страницы не загружает).
2. Метка приводится к виду проекта: 1 — фишинг, 0 — безопасный (в исходнике наоборот).
3. Удаляются пустые и некорректные URL (validate_url), точные дубликаты.
4. Артефакт разметки: все легитимные URL имеют вид https://www.домен без пути. URL приводятся
   к канонической форме http://домен (canonical_url); дубликаты доменов удаляются, домены с
   противоречивыми метками исключаются.
5. Стратифицированная подвыборка 100 000 записей с сохранением баланса классов.
6. Разбиение 70 / 15 / 15 по группам регистрируемого домена (GroupShuffleSplit, random_state=42):
   один домен не попадает одновременно в обучение и тест.
7. Признаки — ml.url_features.model_features (та же функция, что в приложении).

Выход:
    data/processed/url_split.json        — id записей по выборкам (в git)
    data/processed/url_dataset.csv.gz    — id, url, host, registered_domain, label, split (не в git)
    data/processed/url_features.parquet  — признаки + label + split (не в git; без pyarrow —
                                           url_features.csv.gz)
    docs/10_url_dataset_steps.md         — таблица шагов очистки
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.url_features import MODEL_FEATURES, canonical_url, model_features, registered_domain, validate_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "url" / "PhiUSIIL_Phishing_URL_Dataset.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLIT_PATH = PROCESSED / "url_split.json"
DATASET_PATH = PROCESSED / "url_dataset.csv.gz"
FEATURES_PARQUET = PROCESSED / "url_features.parquet"
FEATURES_CSV = PROCESSED / "url_features.csv.gz"
STEPS_MD = PROJECT_ROOT / "docs" / "10_url_dataset_steps.md"

RANDOM_STATE = 42
SAMPLE_SIZE = 100_000


def _safe_canonical(url: str) -> str | None:
    try:
        return canonical_url(url)
    except ValueError:
        return None


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, str]]]:
    steps = [("Исходные записи", len(raw), "PhiUSIIL, столбцы URL и label")]
    df = raw.rename(columns={"URL": "url"})[["url", "label"]].copy()
    df["label"] = 1 - df["label"].astype(int)          # 1 — фишинг, 0 — безопасный
    df["url"] = df["url"].astype(str).str.strip()

    empty = df["url"].isin(["", "nan"])
    df = df[~empty]
    steps.append(("Удалены пустые URL", int(empty.sum()), ""))

    valid = df["url"].map(lambda u: validate_url(u).ok)
    df = df[valid]
    steps.append(("Удалены некорректные URL", int((~valid).sum()), "validate_url: пробелы, нет домена и т. п."))

    dup = df.duplicated("url")
    df = df[~dup]
    steps.append(("Удалены точные дубликаты URL", int(dup.sum()), ""))

    df["host_url"] = df["url"].map(_safe_canonical)
    bad = df["host_url"].isna()
    df = df[~bad]
    steps.append(("Каноническая форма http://домен", len(df), "схема, www., путь и запрос отброшены"))

    labels_per_host = df.groupby("host_url")["label"].nunique()
    conflict = labels_per_host[labels_per_host > 1].index
    n_conflict = int(df["host_url"].isin(conflict).sum())
    df = df[~df["host_url"].isin(conflict)]
    steps.append(("Удалены домены с противоречивыми метками", n_conflict, f"{len(conflict)} доменов"))

    dup_host = df.duplicated("host_url")
    df = df[~dup_host]
    steps.append(("Удалены дубликаты доменов", int(dup_host.sum()),
                  "в основном фишинговые URL с разными путями на одном домене"))
    df = df.reset_index(drop=True)
    return df, steps


def subsample(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Стратифицированная подвыборка n записей с сохранением доли классов."""
    if len(df) <= n:
        return df
    share = df["label"].value_counts(normalize=True)
    parts = [df[df["label"] == c].sample(int(round(n * share[c])), random_state=RANDOM_STATE) for c in share.index]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def group_split(df: pd.DataFrame) -> pd.Series:
    """70 / 15 / 15 по группам регистрируемого домена."""
    groups = df["registered_domain"].to_numpy()
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel, test_rel = next(gss2.split(temp_idx, groups=groups[temp_idx]))
    split = pd.Series("train", index=df.index)
    split.iloc[temp_idx[val_rel]] = "val"
    split.iloc[temp_idx[test_rel]] = "test"
    return split


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame([model_features(u) for u in df["host_url"]], index=df.index)[MODEL_FEATURES]
    return pd.concat([df[["id"]], feats, df[["label", "split"]]], axis=1)


def save_features(features: pd.DataFrame) -> Path:
    try:
        features.to_parquet(FEATURES_PARQUET, index=False)
        return FEATURES_PARQUET
    except ImportError:
        features.to_csv(FEATURES_CSV, index=False, compression="gzip")
        return FEATURES_CSV


def load_url_features() -> pd.DataFrame:
    """Матрица признаков URL (parquet, а при его отсутствии — csv.gz)."""
    if FEATURES_PARQUET.exists():
        try:
            return pd.read_parquet(FEATURES_PARQUET)
        except ImportError:
            pass
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV)
    raise FileNotFoundError("Нет матрицы признаков URL. Выполните: python -m ml.data.prepare_urls")


def write_steps(steps, df: pd.DataFrame) -> None:
    lines = ["# Задание 10. Шаги очистки набора URL", "",
             f"*Сгенерировано `python -m ml.data.prepare_urls`, {date.today():%d.%m.%Y}.*", "",
             "| № | Шаг | Записей | Примечание |", "| :-: | :--- | --: | :--- |"]
    lines += [f"| {i} | {s} | {n} | {note} |" for i, (s, n, note) in enumerate(steps, 1)]
    lines += ["", "| Выборка | Записей | Безопасных | Фишинг | Доля фишинга | Регистрируемых доменов |",
              "| :--- | --: | --: | --: | --: | --: |"]
    for part in ("train", "val", "test"):
        p = df[df["split"] == part]
        lines.append(f"| {part} | {len(p)} | {(p['label'] == 0).sum()} | {(p['label'] == 1).sum()} | "
                     f"{p['label'].mean():.2%} | {p['registered_domain'].nunique()} |")
    tr = set(df.loc[df["split"] == "train", "registered_domain"])
    te = set(df.loc[df["split"] == "test", "registered_domain"])
    va = set(df.loc[df["split"] == "val", "registered_domain"])
    lines += ["", f"Общих регистрируемых доменов: train ∩ test — {len(tr & te)}, train ∩ val — {len(tr & va)}, "
                  f"val ∩ test — {len(va & te)}.", ""]
    STEPS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not RAW_PATH.exists():
        print(f"Нет файла {RAW_PATH}. Сначала: python -m ml.data.download_urls", file=sys.stderr)
        return 1
    raw = pd.read_csv(RAW_PATH, usecols=["URL", "label"], encoding="utf-8-sig")
    df, steps = clean(raw)
    full_counts = df["label"].value_counts().to_dict()
    df = subsample(df)
    steps.append(("Стратифицированная подвыборка", len(df),
                  f"до {SAMPLE_SIZE} с сохранением доли фишинга ({df['label'].mean():.1%})"))
    df["registered_domain"] = df["host_url"].map(registered_domain)
    df["host"] = df["host_url"].str.replace("http://", "", regex=False)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["split"] = group_split(df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({
        "meta": {"source": "PhiUSIIL_Phishing_URL_Dataset.csv", "method": "GroupShuffleSplit by registered domain, 70/15/15",
                 "random_state": RANDOM_STATE, "sizes": df["split"].value_counts().to_dict(),
                 "classes_before_subsample": {str(k): int(v) for k, v in full_counts.items()}},
        **{part: df.loc[df["split"] == part, "id"].tolist() for part in ("train", "val", "test")},
    }), encoding="utf-8")
    df[["id", "url", "host", "registered_domain", "label", "split"]].to_csv(DATASET_PATH, index=False, compression="gzip")
    path = save_features(build_features(df))
    write_steps(steps, df)

    for s, n, note in steps:
        print(f"{s:45s} {n:>8}  {note}")
    print(df.groupby("split")["label"].agg(["size", "mean"]).round(4))
    print(f"Сохранено: {path.relative_to(PROJECT_ROOT)}, {SPLIT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
