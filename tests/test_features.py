"""Тесты разбиения, словаря правил и признаков (задание 6)."""
import numpy as np
import pandas as pd
import pytest

from ml.features import (MANUAL_FEATURES, build_vectorizer, make_input, manual_features,
                         split_dataset)
from ml.rules import RULES, count_rule_groups, match_rules


@pytest.fixture(scope="module")
def toy_df():
    spam = ["WINNER! Claim your £900 prize now, call 09061701461",
            "URGENT! Your account is blocked, verify your card at http://bank.example",
            "Free entry to win cash! Txt WIN to 87121",
            "You have won a bonus reward, claim at www.prize.test today only"] * 10
    ham = ["Ok see you at home later", "Are we meeting for lunch tomorrow?",
           "Thanks for the notes, see you in class", "I'll call you when I get home"] * 30
    texts = spam + ham
    labels = [1] * len(spam) + [0] * len(ham)
    df = make_input(texts)
    df["label"] = labels
    df["id"] = range(1, len(df) + 1)
    return df


def test_split_is_stratified_and_disjoint(toy_df):
    split = split_dataset(toy_df)
    sizes = {k: len(v) for k, v in split.items()}
    assert sum(sizes.values()) == len(toy_df)
    assert sizes["train"] == pytest.approx(0.70 * len(toy_df), abs=2)
    assert not set(split["train"]) & set(split["val"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["val"]) & set(split["test"])
    spam_ids = set(toy_df.loc[toy_df["label"] == 1, "id"])
    for part in split.values():
        share = len(spam_ids & set(part)) / len(part)
        assert share == pytest.approx(0.25, abs=0.05)
    assert split_dataset(toy_df) == split  # random_state=42 → воспроизводимо


def test_rules_have_six_groups_with_ten_expressions_per_language():
    required = {"URGENCY", "MONEY_TRANSFER", "BANK_CARD", "ACCOUNT_BLOCK", "PRIZE", "CODE_REQUEST"}
    assert required <= set(RULES)
    for group in required:
        for lang in ("ru", "kk", "en"):
            assert len(RULES[group][lang]) >= 10, (group, lang)


@pytest.mark.parametrize("text, group", [
    ("Назовите код из SMS оператору", "CODE_REQUEST"),
    ("Кодты операторға айтыңыз", "CODE_REQUEST"),
    ("Please tell me the code you received", "CODE_REQUEST"),
    ("Ваша карта заблокирована", "ACCOUNT_BLOCK"),
    ("Картаңыз бұғатталды", "ACCOUNT_BLOCK"),
    ("Срочно переведи деньги", "URGENCY"),
    ("Тез арада жауап беріңіз", "URGENCY"),
    ("Вы выиграли приз", "PRIZE"),
    ("Сізге сыйлық берілді", "PRIZE"),
    ("You won a FREE prize", "PRIZE"),
])
def test_rules_detect_groups(text, group):
    assert group in match_rules(text)


@pytest.mark.parametrize("text", [
    "Ваш код подтверждения: 4821. Никому не сообщайте этот код.",
    "Растау коды: 4821. Бұл кодты ешкімге айтпаңыз.",
    "Never share your code with anyone",
])
def test_legit_otp_is_not_code_request(text):
    assert "CODE_REQUEST" not in match_rules(text)


def test_manual_features_values():
    X = make_input(["WIN £900 NOW!! call +7 000 123 45 67 http://x.example", "ok"])
    feats = manual_features(X)
    assert feats.shape == (2, len(MANUAL_FEATURES)) == (2, 8)
    row = dict(zip(MANUAL_FEATURES, feats[0]))
    assert row["n_exclaim"] == 2
    assert row["has_url"] == row["has_money"] == row["has_phone"] == 1
    assert row["upper_ratio"] == pytest.approx(6 / 22)  # WIN NOW = 6 заглавных из 22 букв
    assert row["n_rule_groups"] >= 1
    assert feats[1, 4:7].sum() == 0


def test_vectorizer_fit_on_train_only_and_shapes(toy_df):
    split = split_dataset(toy_df)
    by_id = toy_df.set_index("id")
    train = by_id.loc[split["train"]]
    val = by_id.loc[split["val"]]
    vec = build_vectorizer()
    X_train = vec.fit_transform(train[["text_raw", "text_clean"]])
    vocab = vec.named_transformers_["word"].vocabulary_.copy()
    X_val = vec.transform(val[["text_raw", "text_clean"]])
    assert vec.named_transformers_["word"].vocabulary_ == vocab  # transform не меняет словарь
    assert X_train.shape[0] == len(train) and X_val.shape[0] == len(val)
    assert X_train.shape[1] == X_val.shape[1] == len(vec.get_feature_names_out())


def test_stop_words_option_reduces_vocabulary(toy_df):
    with_sw = build_vectorizer(use_char=False, use_manual=False).fit(toy_df)
    without_sw = build_vectorizer(use_char=False, use_manual=False, remove_stop_words=True).fit(toy_df)
    assert len(without_sw.get_feature_names_out()) < len(with_sw.get_feature_names_out())

