"""Калибровка весов и границ уровней модуля риска (задание 13, п. 4).

Цель (ТЗ): на validation SMS и собственном наборе
    Recall мошеннических SMS по уровню «средний и выше» ≥ 0,95,
    доля безопасных SMS с уровнем «высокий» < 3%.

P_sms считается моделью, обученной ТОЛЬКО на train (та же конфигурация, что итоговая модель задания 8):
итоговая модель переобучена на train + val и на validation дала бы завышенные вероятности.
URL-модель обучена только на train.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

from ml.risk import base_score, compute_risk, level_for, load_config


def train_only_sms_model(data):
    """Итоговая конфигурация задания 8 (из sms_model_meta.json), обученная только на train."""
    import json

    from ml import train_sms as ts
    meta = json.loads(ts.META_PATH.read_text(encoding="utf-8"))
    name = {v: k for k, v in ts.ALGO_TITLES.items()}[meta["algorithm"]]
    pipe, _ = ts.make_pipeline(name)
    pipe.set_params(**{f"clf__{k}": v for k, v in meta["params"].items()})
    return pipe.fit(data.train[ts.X_COLS], data.train["label"])


def collect_components(frame: pd.DataFrame, sms_model, predictor) -> pd.DataFrame:
    """Для каждого SMS: P_sms, ссылки и P_url,max, группы социальной инженерии, флаги ссылок."""
    from apps.analysis.services import detect_social_engineering, extract_urls, is_trusted
    from ml.url_features import extract_url_features

    p_sms = sms_model.predict_proba(frame[["text_raw", "text_clean"]])[:, 1]
    rows = []
    for text, p in zip(frame["text_raw"], p_sms):
        urls = extract_urls(text)
        p_urls, has_ip, has_puny = [], 0, 0
        for u in urls:
            p_u = 0.0 if is_trusted(u) else predictor.predict_url_proba(u)
            p_urls.append(p_u)
            f = extract_url_features(u)
            has_ip |= f["has_ip"]
            has_puny |= f["has_punycode"]
        rows.append({"p_sms": float(p), "n_urls": len(urls), "p_url_max": max(p_urls) if p_urls else None,
                     "groups": detect_social_engineering(text, urls), "has_ip": has_ip, "has_punycode": has_puny})
    out = pd.DataFrame(rows, index=frame.index)
    out["k"] = out["groups"].map(len)
    return pd.concat([frame[["label", "lang"]].reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def scores_for(comp: pd.DataFrame, cfg: dict, url_t2: float) -> np.ndarray:
    return np.array([
        compute_risk(r.p_sms, None if pd.isna(r.p_url_max) else r.p_url_max, r.groups, {"has_ip": r.has_ip, "has_punycode": r.has_punycode},
                     url_t2=url_t2, config=cfg).score
        for r in comp.itertuples()
    ])


def evaluate(comp: pd.DataFrame, scores: np.ndarray, cfg: dict) -> dict:
    levels = np.array([level_for(s, cfg) for s in scores])
    fraud, safe = comp["label"].to_numpy() == 1, comp["label"].to_numpy() == 0
    return {
        "recall_medium_plus": round(float(np.isin(levels[fraud], ["MEDIUM", "HIGH"]).mean()), 4),
        "recall_high": round(float((levels[fraud] == "HIGH").mean()), 4),
        "safe_high_share": round(float((levels[safe] == "HIGH").mean()), 4),
        "safe_medium_plus_share": round(float(np.isin(levels[safe], ["MEDIUM", "HIGH"]).mean()), 4),
    }


WEIGHT_VARIANTS = [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.4, 0.3, 0.3), (0.5, 0.2, 0.3), (0.6, 0.3, 0.1),
                   (0.4, 0.4, 0.2), (0.7, 0.2, 0.1)]
LEVEL_VARIANTS = [(34, 69), (29, 69), (24, 69), (34, 64), (29, 59)]


def run_grid(comp: pd.DataFrame, url_t2: float, base_cfg: dict | None = None) -> pd.DataFrame:
    base_cfg = base_cfg or load_config()
    rows = []
    for (w_sms, w_url, w_se), (low_max, medium_max) in itertools.product(WEIGHT_VARIANTS, LEVEL_VARIANTS):
        cfg = copy.deepcopy(base_cfg)
        cfg["weights"] = {"w_sms": w_sms, "w_url": w_url, "w_se": w_se}
        cfg["levels"] = {"low_max": low_max, "medium_max": medium_max}
        metrics = evaluate(comp, scores_for(comp, cfg, url_t2), cfg)
        rows.append({"w_sms": w_sms, "w_url": w_url, "w_se": w_se, "low_max": low_max,
                     "medium_max": medium_max, **metrics})
    table = pd.DataFrame(rows)
    table["meets_goal"] = (table["recall_medium_plus"] >= 0.95) & (table["safe_high_share"] < 0.03)
    return table


def choose(table: pd.DataFrame) -> pd.Series:
    """Правило выбора (зафиксировано до просмотра test):
    1) вариант достигает цели ТЗ (Recall «средний+» ≥ 0,95 и доля безопасных на «высоком» < 3%);
    2) границы уровней ТЗ 34/69 сохраняются, если с ними цель достижима (их видит пользователь);
    3) вес словаря w_se ≥ 0,2: словарь — единственный сигнал, не зависящий от SMS-модели, а для ru/kk модель
       проверена только на искусственных сообщениях;
    4) среди оставшихся — наименьшая доля безопасных SMS на уровне «средний+», затем наибольший Recall уровня «высокий».
    """
    ok = table[table["meets_goal"]]
    pool = ok if len(ok) else table
    default = pool[(pool["low_max"] == 34) & (pool["medium_max"] == 69)]
    pool = default if len(default) else pool
    se = pool[pool["w_se"] >= 0.2]
    pool = se if len(se) else pool
    return pool.sort_values(["safe_medium_plus_share", "recall_high", "recall_medium_plus"],
                            ascending=[True, False, False]).iloc[0]


def save_choice(row: pd.Series, path=None) -> None:
    """Записывает выбранные веса и границы в risk_config.yaml, сохраняя комментарии файла."""
    import re
    from pathlib import Path

    path = Path(path) if path else Path(__file__).resolve().parent.parent / "risk_config.yaml"
    text = path.read_text(encoding="utf-8")
    for key in ("w_sms", "w_url", "w_se", "low_max", "medium_max"):
        value = row[key]
        value = int(value) if key.endswith("_max") else float(value)
        text = re.sub(rf"(?m)^(\s+{key}:\s*)[0-9.]+", rf"\g<1>{value}", text, count=1)
    path.write_text(text, encoding="utf-8")


__all__ = ["save_choice", "train_only_sms_model", "collect_components", "scores_for", "evaluate", "run_grid", "choose",
           "base_score"]
