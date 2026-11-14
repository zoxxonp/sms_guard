"""Модульные тесты ml/risk.py (задание 13). Конфигурация задаётся в тестах явно."""
import copy
import time

import pytest
import yaml

from ml.risk import base_score, compute_risk, level_for, load_config, url_probability_to_score

CFG = {
    "weights": {"w_sms": 0.5, "w_url": 0.3, "w_se": 0.2},
    "se_saturation": 3,
    "levels": {"low_max": 34, "medium_max": 69},
    "overrides": {"code_request_min": 70, "sms_and_url_high_min": 85, "sms_high": 0.8, "ip_or_punycode_min": 60},
    "reason_min_probability": 0.5,
    "max_reasons": 6,
    "recommendations": {"low": "R-low", "medium": "R-medium", "high": "R-high"},
    "url_recommendations": {"low": "U-low", "medium": "U-medium", "high": "U-high"},
}


def risk(p_sms, p_url=None, groups=None, flags=None, **kw):
    return compute_risk(p_sms, p_url, groups or {}, flags or {}, config=CFG, url_t2=0.8, **kw)


@pytest.mark.parametrize("score, level", [(0, "LOW"), (34, "LOW"), (35, "MEDIUM"), (69, "MEDIUM"),
                                          (70, "HIGH"), (100, "HIGH")])
def test_level_boundaries(score, level):
    assert level_for(score, CFG) == level


@pytest.mark.parametrize("score, category", [(34, "SAFE"), (35, "SUSPICIOUS"), (69, "SUSPICIOUS"), (70, "FRAUDULENT")])
def test_boundaries_through_compute_risk(score, category):
    r = compute_risk(None, None, {}, {}, url_only_score=score, config=CFG)
    assert r.score == score and r.category == category


def test_formula_without_url_excludes_url_weight():
    # 100 * (0.5*0.6 + 0.2*min(1/3,1)) / (0.5 + 0.2)
    expected = 100 * (0.5 * 0.6 + 0.2 / 3) / 0.7
    assert base_score(0.6, None, 1, CFG) == pytest.approx(expected)
    r = risk(0.6, None, {"URGENCY": ["срочно"]})
    assert r.score == round(expected) and r.applied_overrides == []


def test_formula_with_url():
    expected = 100 * (0.5 * 0.4 + 0.3 * 0.5 + 0.2 * (2 / 3)) / 1.0
    assert base_score(0.4, 0.5, 2, CFG) == pytest.approx(expected)


def test_se_saturation():
    assert base_score(0.0, None, 3, CFG) == base_score(0.0, None, 6, CFG) == pytest.approx(100 * 0.2 / 0.7)


def test_code_request_override_at_least_70():
    r = risk(0.05, None, {"CODE_REQUEST": ["назовите код"]})
    assert r.base_score < 70 and r.score == 70 and r.level == "HIGH" and "CODE_REQUEST" in r.applied_overrides


def test_sms_and_url_high_override_at_least_85():
    r = risk(0.85, 0.82)
    assert r.score >= 85 and "SMS_AND_URL_HIGH" in r.applied_overrides
    assert "SMS_AND_URL_HIGH" not in risk(0.79, 0.82).applied_overrides     # P_sms ниже 0,8
    assert "SMS_AND_URL_HIGH" not in risk(0.85, 0.79).applied_overrides     # P_url ниже t2


@pytest.mark.parametrize("flag", ["has_ip", "has_punycode"])
def test_ip_or_punycode_override_at_least_60(flag):
    r = risk(0.05, 0.1, {}, {flag: 1})
    assert r.score == 60 and r.level == "MEDIUM" and "IP_OR_PUNYCODE" in r.applied_overrides
    assert any("IP-адрес или Punycode" in reason for reason in r.reasons)


def test_all_signals_at_once():
    groups = {g: ["x"] for g in ("URGENCY", "MONEY_TRANSFER", "BANK_CARD", "ACCOUNT_BLOCK", "PRIZE", "CODE_REQUEST")}
    r = risk(0.99, 0.99, groups, {"has_ip": 1, "has_punycode": 1},
             url_reasons=["a", "b", "c", "d"])
    assert r.score >= 95 and r.category == "FRAUDULENT"
    assert 1 <= len(r.reasons) <= 6
    assert r.reasons[0] == "CODE_REQUEST: «x»"          # просьба назвать код — самый весомый признак
    assert r.applied_overrides == []                      # формула уже выше всех минимумов правил


def test_reasons_sorted_by_contribution_and_at_least_one():
    r = risk(0.95, None, {"URGENCY": ["срочно"]}, se_titles={"URGENCY": "Срочность"})
    assert r.reasons[0].startswith("Текст похож на мошеннические")
    assert "Срочность: «срочно»" in r.reasons
    safe = risk(0.01)
    assert safe.reasons == ["Явных признаков мошенничества не обнаружено"] and safe.recommendation == "R-low"


@pytest.mark.parametrize("p, level, rec", [(0.01, "LOW", "R-low"), (0.6, "MEDIUM", "R-medium"), (0.99, "HIGH", "R-high")])
def test_recommendation_for_each_level(p, level, rec):
    r = risk(p, None, {"URGENCY": ["x"], "PRIZE": ["y"], "BANK_CARD": ["z"]} if level == "HIGH" else {})
    assert r.level == level and r.recommendation == rec
    assert r.disclaimer.startswith("Оценка носит вероятностный характер")


def test_url_probability_mapping_matches_thresholds():
    assert url_probability_to_score(0.0, 0.2, 0.7, CFG) == 0
    assert url_probability_to_score(0.2, 0.2, 0.7, CFG) == pytest.approx(35)
    assert url_probability_to_score(0.7, 0.2, 0.7, CFG) == pytest.approx(70)
    assert url_probability_to_score(1.0, 0.2, 0.7, CFG) == pytest.approx(100)
    assert 35 < url_probability_to_score(0.45, 0.2, 0.7, CFG) < 70


def test_config_file_change_changes_result_without_code(tmp_path):
    path = tmp_path / "risk_config.yaml"
    path.write_text(yaml.safe_dump(CFG, allow_unicode=True), encoding="utf-8")
    first = compute_risk(0.5, None, {}, {}, config=load_config(path)).score
    changed = copy.deepcopy(CFG)
    changed["weights"]["w_se"] = 0.0                     # без вклада словаря: 100*0.5 = 50
    time.sleep(0.01)
    path.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
    import os
    os.utime(path, (time.time() + 5, time.time() + 5))  # гарантированно новое время изменения
    second = compute_risk(0.5, None, {}, {}, config=load_config(path)).score
    assert first == round(100 * 0.25 / 0.7) and second == 50
