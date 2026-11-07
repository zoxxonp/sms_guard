"""Тесты baseline-модели и predict_sms (задание 7)."""
import pytest

from ml.features import make_input
from ml.predictor import BASELINE_PATH, load_model, predict_sms
from ml.train_baseline import build_baseline


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



def test_predict_sms_returns_label_and_probability(toy_df):
    model = build_baseline().fit(toy_df[["text_raw", "text_clean"]], toy_df["label"])
    spam = predict_sms("WINNER! Claim your £500 prize, call 09061701461 now!", model=model, threshold=0.5)
    ham = predict_sms("See you at home for lunch tomorrow", model=model, threshold=0.5)
    assert set(spam) == {"label", "label_id", "probability"}
    assert 0.0 <= spam["probability"] <= 1.0
    assert spam["label"] == "fraud" and ham["label"] == "safe"
    with pytest.raises(ValueError):
        predict_sms("   ", model=model)


@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="модель не обучена: python -m ml.train_baseline")
def test_saved_baseline_loads_and_predicts():
    model = load_model()
    fraud = predict_sms("URGENT! You have won a £1000 prize. Call 09061701461 to claim now!", model=model)
    safe = predict_sms("Ok, see you at home tonight", model=model)
    assert fraud["label"] == "fraud" and fraud["probability"] > 0.5
    assert safe["label"] == "safe" and safe["probability"] < 0.5
