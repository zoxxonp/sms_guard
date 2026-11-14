"""Загрузка сохранённой модели SMS и предсказание (задание 7).

Модель — это Pipeline «признаки (ml.features.build_vectorizer) → классификатор»,
сохранённый в ml/artifacts/*.joblib. Текст перед подачей в модель нормализуется той же
функцией ``normalize_sms``, что и при обучении (через ``make_input``).

В задании 12 этот модуль станет классом-синглтоном, который загружает обе модели
(SMS и URL) один раз за время жизни процесса.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib

from ml.features import make_input

logger = logging.getLogger("sms_guard")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BASELINE_PATH = ARTIFACTS_DIR / "sms_baseline.joblib"
BASELINE_META_PATH = ARTIFACTS_DIR / "sms_baseline_meta.json"
MULTI_PATH = ARTIFACTS_DIR / "sms_baseline_multi.joblib"
MULTI_META_PATH = ARTIFACTS_DIR / "sms_baseline_multi_meta.json"


SMS_MODEL_PATH = ARTIFACTS_DIR / "sms_model.joblib"
SMS_MODEL_META_PATH = ARTIFACTS_DIR / "sms_model_meta.json"


def default_model_paths() -> tuple[Path, Path]:
    """Итоговая модель задания 8, если она обучена; иначе многоязычная baseline; иначе английская."""
    for model, meta in ((SMS_MODEL_PATH, SMS_MODEL_META_PATH), (MULTI_PATH, MULTI_META_PATH)):
        if model.exists():
            return model, meta
    return BASELINE_PATH, BASELINE_META_PATH

LABELS = {0: "safe", 1: "fraud"}


@lru_cache(maxsize=4)
def load_model(path: str | Path = BASELINE_PATH):
    """Загружает Pipeline из файла (один раз на путь за время жизни процесса)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла модели {path}. Обучите её: python -m ml.train_baseline [--multi] "
            "или выполните notebooks/03_baseline_sms.ipynb"
        )
    model = joblib.load(path)
    logger.info("Модель загружена: %s", path.name)
    return model


def load_threshold(meta_path: str | Path = BASELINE_META_PATH, default: float = 0.5) -> float:
    meta_path = Path(meta_path)
    if meta_path.exists():
        return float(json.loads(meta_path.read_text(encoding="utf-8")).get("threshold", default))
    return default


def predict_sms(text: str, model=None, threshold: float | None = None) -> dict:
    """Классифицирует одно SMS.

    Возвращает ``{"label": "fraud" | "safe", "label_id": 1 | 0, "probability": P_sms}``,
    где probability — вероятность класса «мошенничество» от 0 до 1.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Текст SMS не должен быть пустым")
    model_path, meta_path = default_model_paths()
    model = model if model is not None else load_model(model_path)
    threshold = load_threshold(meta_path) if threshold is None else threshold
    proba = float(model.predict_proba(make_input([text]))[0, 1])
    label_id = int(proba >= threshold)
    return {"label": LABELS[label_id], "label_id": label_id, "probability": round(proba, 4)}
