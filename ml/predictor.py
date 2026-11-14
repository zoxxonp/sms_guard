"""Загрузка моделей и предсказание (задания 7 и 12).

Predictor — синглтон: SMS- и URL-модели загружаются ОДИН раз за жизнь процесса
(в логе одна запись «Модели загружены»), повторные вызовы используют те же объекты.
Если файла модели нет, приложение всё равно стартует: в лог пишется понятная ошибка,
а методы предсказания бросают ModelUnavailableError (API отвечает 503) — НФТ-14.
"""
from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.features import make_input
from ml.url_features import MODEL_FEATURES, MODEL_TEXT_COLUMN, model_input

logger = logging.getLogger("sms_guard")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BASELINE_PATH = ARTIFACTS_DIR / "sms_baseline.joblib"
BASELINE_META_PATH = ARTIFACTS_DIR / "sms_baseline_meta.json"
MULTI_PATH = ARTIFACTS_DIR / "sms_baseline_multi.joblib"
MULTI_META_PATH = ARTIFACTS_DIR / "sms_baseline_multi_meta.json"
SMS_MODEL_PATH = ARTIFACTS_DIR / "sms_model.joblib"
SMS_MODEL_META_PATH = ARTIFACTS_DIR / "sms_model_meta.json"
URL_MODEL_PATH = ARTIFACTS_DIR / "url_model.joblib"
URL_MODEL_META_PATH = ARTIFACTS_DIR / "url_model_meta.json"

LABELS = {0: "safe", 1: "fraud"}


class ModelUnavailableError(RuntimeError):
    """Файл модели отсутствует или не загружается."""


def default_model_paths() -> tuple[Path, Path]:
    """Итоговая модель задания 8, если она обучена; иначе многоязычная baseline; иначе английская."""
    for model, meta in ((SMS_MODEL_PATH, SMS_MODEL_META_PATH), (MULTI_PATH, MULTI_META_PATH)):
        if model.exists():
            if model != SMS_MODEL_PATH:
                logger.warning("Итоговая SMS-модель %s не найдена — используется базовая %s. "
                               "Обучите итоговую: python -m ml.train_sms", SMS_MODEL_PATH.name, model.name)
            return model, meta
    if not SMS_MODEL_PATH.exists():
        logger.warning("Итоговая SMS-модель %s не найдена — пробуем базовую %s", SMS_MODEL_PATH.name,
                       BASELINE_PATH.name)
    return BASELINE_PATH, BASELINE_META_PATH


def _read_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class Predictor:
    """Синглтон с SMS- и URL-моделями."""

    _instance: "Predictor | None" = None
    _lock = threading.Lock()

    def __init__(self, sms_path: Path | None = None, sms_meta: Path | None = None,
                 url_path: Path = URL_MODEL_PATH, url_meta: Path = URL_MODEL_META_PATH):
        default_sms, default_meta = default_model_paths()
        self.sms_path, self.sms_meta_path = sms_path or default_sms, sms_meta or default_meta
        self.url_path, self.url_meta_path = url_path, url_meta
        self.sms_model = self.url_model = None
        self.sms_meta: dict = {}
        self.url_meta: dict = {}
        self.errors: dict[str, str] = {}
        self._load()

    @classmethod
    def instance(cls) -> "Predictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Сбросить синглтон (тесты, повторная загрузка после переобучения)."""
        with cls._lock:
            cls._instance = None

    def _load(self) -> None:
        for kind, path, meta_path in (("sms", self.sms_path, self.sms_meta_path),
                                      ("url", self.url_path, self.url_meta_path)):
            try:
                if not path.exists():
                    raise FileNotFoundError(f"нет файла {path.name}")
                setattr(self, f"{kind}_model", joblib.load(path))
                setattr(self, f"{kind}_meta", _read_meta(meta_path))
            except Exception as exc:                       # приложение стартует и без модели
                self.errors[kind] = str(exc)
                logger.error("Модель %s не загружена: %s. Анализ %s недоступен. "
                             "Обучите модель: python -m ml.train_%s", kind.upper(), exc, kind.upper(), kind)
        if not self.errors:
            logger.info("Модели загружены: SMS %s (%s), URL %s (%s)", self.sms_meta.get("version", "?"),
                        self.sms_path.name, self.url_meta.get("version", "?"), self.url_path.name)

    # --- состояние -------------------------------------------------------------------
    @property
    def sms_ready(self) -> bool:
        return self.sms_model is not None

    @property
    def url_ready(self) -> bool:
        return self.url_model is not None

    @property
    def url_thresholds(self) -> tuple[float, float]:
        t = self.url_meta.get("thresholds", {})
        return float(t.get("t1", 0.5)), float(t.get("t2", 0.8))

    @property
    def sms_threshold(self) -> float:
        return float(self.sms_meta.get("threshold", 0.5))

    def versions(self) -> dict:
        return {"sms": self.sms_meta.get("version"), "url": self.url_meta.get("version")}

    def status(self) -> dict:
        return {kind: {"loaded": getattr(self, f"{kind}_ready"),
                       "version": getattr(self, f"{kind}_meta").get("version"),
                       "algorithm": getattr(self, f"{kind}_meta").get("algorithm"),
                       "error": self.errors.get(kind)} for kind in ("sms", "url")}

    # --- предсказания ----------------------------------------------------------------
    def predict_sms_proba(self, text: str) -> float:
        if not self.sms_ready:
            raise ModelUnavailableError("SMS-модель не загружена")
        return float(self.sms_model.predict_proba(make_input([text]))[0, 1])

    def predict_url_proba(self, url: str) -> float:
        if not self.url_ready:
            raise ModelUnavailableError("URL-модель не загружена")
        row = pd.DataFrame([model_input(url)])[MODEL_FEATURES + [MODEL_TEXT_COLUMN]]
        return float(self.url_model.predict_proba(row)[0, 1])

    def url_category(self, p_url: float) -> str:
        t1, t2 = self.url_thresholds
        return "DANGEROUS" if p_url >= t2 else "SUSPICIOUS" if p_url >= t1 else "SAFE"


def get_predictor() -> Predictor:
    return Predictor.instance()


# --- Функции задания 7 (используются ноутбуками и тестами) ------------------------------
@lru_cache(maxsize=4)
def load_model(path: str | Path = BASELINE_PATH):
    """Загружает Pipeline из файла (один раз на путь за время жизни процесса)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла модели {path}. Обучите её: python -m ml.train_baseline [--multi] "
            "или python -m ml.train_sms"
        )
    model = joblib.load(path)
    logger.info("Модель загружена: %s", path.name)
    return model


def load_threshold(meta_path: str | Path = BASELINE_META_PATH, default: float = 0.5) -> float:
    meta = _read_meta(Path(meta_path))
    return float(meta.get("threshold", default))


def predict_sms(text: str, model=None, threshold: float | None = None) -> dict:
    """Классифицирует одно SMS: {"label": "fraud" | "safe", "label_id": 1 | 0, "probability": P_sms}."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Текст SMS не должен быть пустым")
    model_path, meta_path = default_model_paths()
    model = model if model is not None else load_model(model_path)
    threshold = load_threshold(meta_path) if threshold is None else threshold
    proba = float(model.predict_proba(make_input([text]))[0, 1])
    label_id = int(proba >= threshold)
    return {"label": LABELS[label_id], "label_id": label_id, "probability": round(proba, 4)}
