"""Тесты синтетического ru/kk-набора и разбиения по шаблонам."""
import re

import pandas as pd
import pytest

from ml.data.generate_ru_kk import CUSTOM_PATH, OUT_PATH, generate
from ml.features import split_by_template
from ml.preprocessing import normalize_sms


@pytest.fixture(scope="module")
def synth():
    return pd.read_csv(OUT_PATH, keep_default_na=False)


def test_sizes_and_balance(synth):
    counts = synth.groupby(["lang", "label"]).size().to_dict()
    assert counts == {("kk", 0): 500, ("kk", 1): 500, ("ru", 0): 500, ("ru", 1): 500}


def test_file_matches_generator(synth):
    regenerated = generate()
    assert regenerated["text"].tolist() == synth["text"].tolist()  # seed=42 → воспроизводимо


def test_no_duplicates_and_no_overlap_with_handwritten(synth):
    norm = synth["text"].map(normalize_sms)
    assert norm.nunique() == len(synth)
    custom = set(pd.read_csv(CUSTOM_PATH)["text"].map(normalize_sms))
    assert not set(norm) & custom


def test_no_real_personal_data(synth):
    text = " ".join(synth["text"])
    # телефон: +7 / 8, код оператора, затем ещё три цифры (суммы вроде «8 200 тг» не подходят)
    phones = re.findall(r"(?<![\d*№])(?:\+7 ?|8 \(?)(\d{3})\)?[ -]?\d{3}", text)
    assert phones
    assert set(phones) == {"000"}  # только несуществующий код оператора 000
    for host in re.findall(r"https?://([^/\s]+)", text):
        assert host.endswith((".example", ".test", "example.com")), host
    assert not re.search(r"\b\d{16}\b|\b\d{4} \d{4} \d{4} \d{4}\b", text)  # нет полных номеров карт


def test_kazakh_texts_contain_kazakh_letters(synth):
    kk = synth[synth["lang"] == "kk"]["text"]
    share = kk.str.contains(r"[әғқңөұүһі]", regex=True).mean()
    assert share > 0.9


def test_split_by_template_has_no_shared_templates(synth):
    split = split_by_template(synth)
    tpl = {k: set(synth.loc[synth["id"].isin(v), "template_id"]) for k, v in split.items()}
    assert not tpl["train"] & tpl["val"] and not tpl["train"] & tpl["test"] and not tpl["val"] & tpl["test"]
    assert sum(len(v) for v in split.values()) == len(synth)
    # каждая категория каждого языка есть в обучении
    train = synth[synth["id"].isin(split["train"])]
    assert set(map(tuple, train[["lang", "category"]].drop_duplicates().values)) == \
        set(map(tuple, synth[["lang", "category"]].drop_duplicates().values))
