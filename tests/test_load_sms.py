"""Тесты загрузчика SMS Spam Collection (задание 3)."""
from ml.data.load_sms import RAW_PATH, load_sms


def test_parses_labels_and_quotes(tmp_path):
    raw = tmp_path / "SMSSpamCollection"
    raw.write_text(
        'ham\tHi "friend", see you\nspam\tWIN £100 now! Call 0800\n\nham\tOk\n',
        encoding="utf-8",
    )
    df = load_sms(raw)
    assert list(df.columns) == ["id", "text", "label"]
    assert len(df) == 3 == df.attrs["source_rows"]
    assert df["label"].tolist() == [0, 1, 0]
    assert df.loc[0, "text"] == 'Hi "friend", see you'


def test_bad_lines_are_reported(tmp_path):
    raw = tmp_path / "SMSSpamCollection"
    raw.write_text("ham\tok\nunknown\ttext\nno tab here\n", encoding="utf-8")
    df = load_sms(raw)
    assert len(df) == 1
    assert len(df.attrs["bad_lines"]) == 2


def test_real_dataset_shape():
    df = load_sms(RAW_PATH)
    assert len(df) == 5574
    assert df["label"].value_counts().to_dict() == {0: 4827, 1: 747}
