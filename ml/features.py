"""Разбиение данных и векторизация SMS (задание 6).

Признаки строятся одним объектом ``ColumnTransformer``, который входит в Pipeline
модели и сохраняется вместе с ней (ФТ-2.4). На вход Pipeline подаётся DataFrame со
столбцами ``text_raw`` (исходный текст) и ``text_clean`` (результат ``normalize_sms``),
его собирает функция ``make_input``.

Блоки признаков:
    word   — TF-IDF по словам и биграммам ``text_clean``;
    char   — TF-IDF по символьным 3–5-граммам ``text_clean`` без спецтокенов (analyzer='char_wb');
    manual — 8 ручных признаков (длина, заглавные, «!», токены, словарь правил),
             масштабированных MaxAbsScaler в диапазон [0, 1].
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from ml.preprocessing import STOP_WORDS, TOKEN_PATTERN, normalize_sms
from ml.rules import count_rule_groups

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "sms_split.json"
SYNTH_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_ru_kk.csv"
SYNTH_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_split.json"

MANUAL_FEATURES = [
    "len_chars_log",    # log(1 + длина исходного текста в символах)
    "n_words",          # число слов в исходном тексте
    "upper_ratio",      # доля заглавных среди букв исходного текста
    "n_exclaim",        # число «!» в исходном тексте
    "has_url",          # есть <url>
    "has_money",        # есть <money>
    "has_phone",        # есть <phone>
    "n_rule_groups",    # число сработавших групп словаря социальной инженерии (0–6)
]


# --- Разбиение ---------------------------------------------------------------------

def split_dataset(df: pd.DataFrame, label_col: str = "label",
                  random_state: int = RANDOM_STATE) -> dict[str, list[int]]:
    """Стратифицированное разбиение 70 / 15 / 15. Возвращает списки ``id`` записей."""
    ids = df["id"].to_numpy()
    y = df[label_col].to_numpy()
    train_ids, temp_ids, _, y_temp = train_test_split(
        ids, y, test_size=0.30, stratify=y, random_state=random_state)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, stratify=y_temp, random_state=random_state)
    return {"train": sorted(map(int, train_ids)),
            "val": sorted(map(int, val_ids)),
            "test": sorted(map(int, test_ids))}


def split_by_template(df: pd.DataFrame, random_state: int = RANDOM_STATE,
                      share: float = 0.15) -> dict[str, list[int]]:
    """Разбиение синтетического набора ПО ШАБЛОНАМ (template_id), примерно 70 / 15 / 15.

    Все сообщения одного шаблона попадают в одну выборку, поэтому на validation и test
    модель видит формулировки, которых не было при обучении. Шаблоны делятся внутри
    каждой тройки (язык, класс, категория): в validation и test идёт по ~15% шаблонов
    категории (минимум по одному), остальные — в train. Так каждая категория каждого
    языка представлена и в обучении, и в проверке.
    """
    rng = np.random.default_rng(random_state)
    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for _, group in df.groupby(["lang", "label", "category"], sort=True):
        templates = np.array(sorted(group["template_id"].unique()))
        rng.shuffle(templates)
        n_hold = max(1, int(round(len(templates) * share)))
        assign = {t: "val" for t in templates[:n_hold]}
        assign.update({t: "test" for t in templates[n_hold:2 * n_hold]})
        assign.update({t: "train" for t in templates[2 * n_hold:]})
        for part in parts:
            parts[part] += group.loc[group["template_id"].map(assign) == part, "id"].astype(int).tolist()
    return {k: sorted(v) for k, v in parts.items()}


def save_synthetic_split(split: dict[str, list[int]], df: pd.DataFrame,
                         path: Path = SYNTH_SPLIT_PATH) -> None:
    by_id = df.set_index("id")
    meta = {
        "source": "data/raw/synthetic_ru_kk.csv",
        "method": "split by template_id within (lang, label, category), ~70/15/15",
        "random_state": RANDOM_STATE,
        "sizes": {k: len(v) for k, v in split.items()},
        "templates": {k: int(by_id.loc[v, "template_id"].nunique()) for k, v in split.items()},
        "by_lang_label": {k: {f"{lang}_{lab}": int(n) for (lang, lab), n in
                              by_id.loc[v].groupby(["lang", "label"]).size().items()}
                          for k, v in split.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, **split}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_synthetic(path: Path = SYNTH_PATH, split_path: Path = SYNTH_SPLIT_PATH) -> dict[str, pd.DataFrame]:
    """Синтетический ru/kk-набор, разделённый по шаблонам: {'train', 'val', 'test'}.

    К каждой части добавлены столбцы text_raw / text_clean (вход Pipeline).
    """
    df = pd.read_csv(path, keep_default_na=False)
    df["text_raw"] = df["text"]
    df["text_clean"] = df["text"].map(normalize_sms)
    if not split_path.exists():
        save_synthetic_split(split_by_template(df), df, split_path)
    data = json.loads(split_path.read_text(encoding="utf-8"))
    by_id = df.set_index("id", drop=False)
    return {part: by_id.loc[data[part]].reset_index(drop=True) for part in ("train", "val", "test")}


def save_split(split: dict[str, list[int]], df: pd.DataFrame, path: Path = SPLIT_PATH) -> None:
    labels = df.set_index("id")["label"]
    meta = {
        "source": "data/processed/sms_clean.csv",
        "method": "train_test_split, stratify=label, 70/15/15",
        "random_state": RANDOM_STATE,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sizes": {k: len(v) for k, v in split.items()},
        "class_counts": {k: {str(c): int(n) for c, n in labels.loc[v].value_counts().sort_index().items()}
                         for k, v in split.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, **split}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_split(df: pd.DataFrame, path: Path = SPLIT_PATH) -> dict[str, pd.DataFrame]:
    """Возвращает {'train': df, 'val': df, 'test': df} по сохранённым индексам."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = df.set_index("id", drop=False)
    return {part: by_id.loc[data[part]].reset_index(drop=True) for part in ("train", "val", "test")}


# --- Вход Pipeline -------------------------------------------------------------------

def make_input(texts) -> pd.DataFrame:
    """Собирает вход Pipeline из исходных текстов: text_raw + text_clean."""
    if isinstance(texts, str):
        texts = [texts]
    raw = pd.Series(list(texts), dtype=object).fillna("").astype(str)
    return pd.DataFrame({"text_raw": raw, "text_clean": raw.map(normalize_sms)})


# --- Ручные признаки ---------------------------------------------------------------

def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def manual_features(X: pd.DataFrame) -> np.ndarray:
    """8 ручных признаков; X — DataFrame со столбцами text_raw и text_clean."""
    raw = X["text_raw"].astype(str)
    clean = X["text_clean"].astype(str)
    return np.column_stack([
        np.log1p(raw.str.len().to_numpy()),
        raw.str.split().str.len().fillna(0).to_numpy(),
        raw.map(_upper_ratio).to_numpy(),
        raw.str.count("!").to_numpy(),
        clean.str.contains("<url>", regex=False).to_numpy(),
        clean.str.contains("<money>", regex=False).to_numpy(),
        clean.str.contains("<phone>", regex=False).to_numpy(),
        clean.map(lambda t: count_rule_groups(t, normalized=True)).to_numpy(),
    ]).astype(float)


_SPECIAL_TOKEN_RE = re.compile(r"<(?:url|email|phone|money|num)>")


def strip_special_tokens(text: str) -> str:
    """Убирает спецтокены перед символьным TF-IDF: иначе `<num>` распадается на n-граммы
    `<nu`, `num`, `um>` и дублирует информацию словного блока и ручных признаков."""
    return " ".join(_SPECIAL_TOKEN_RE.sub(" ", text).split())


def _manual_names(_transformer, _input_features=None):
    return np.array(MANUAL_FEATURES, dtype=object)


# --- Векторизатор ------------------------------------------------------------------

def build_vectorizer(use_word: bool = True, use_char: bool = True, use_manual: bool = True,
                     remove_stop_words: bool = False) -> ColumnTransformer:
    """Возвращает ColumnTransformer с выбранными блоками признаков.

    Параметры word-TF-IDF заданы ТЗ: ngram_range=(1, 2), min_df=2, max_df=0.95,
    sublinear_tf=True. Текст уже нормализован, поэтому lowercase=False.
    """
    blocks = []
    if use_word:
        blocks.append(("word", TfidfVectorizer(
            lowercase=False, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True,
            stop_words=sorted(STOP_WORDS) if remove_stop_words else None,
        ), "text_clean"))
    if use_char:
        blocks.append(("char", TfidfVectorizer(
            lowercase=False, analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_df=0.95, sublinear_tf=True, preprocessor=strip_special_tokens,
        ), "text_clean"))
    if use_manual:
        blocks.append(("manual", Pipeline([
            ("extract", FunctionTransformer(manual_features, validate=False,
                                            feature_names_out=_manual_names)),
            ("scale", MaxAbsScaler()),
        ]), ["text_raw", "text_clean"]))
    if not blocks:
        raise ValueError("Нужен хотя бы один блок признаков")
    return ColumnTransformer(blocks, sparse_threshold=1.0)


def top_tfidf_features(vectorizer: TfidfVectorizer, X, y, n: int = 15) -> dict[int, pd.Series]:
    """Топ-n признаков с наибольшим средним TF-IDF для каждого класса."""
    names = vectorizer.get_feature_names_out()
    y = np.asarray(y)
    result = {}
    for cls in np.unique(y):
        mean = np.asarray(X[y == cls].mean(axis=0)).ravel()
        order = mean.argsort()[::-1][:n]
        result[int(cls)] = pd.Series(mean[order], index=names[order]).round(4)
    return result
