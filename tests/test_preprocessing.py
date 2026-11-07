"""Модульные тесты ml/preprocessing.py (задание 4)."""
import pytest

from ml.preprocessing import STOP_WORDS, normalize_sms, remove_stopwords


@pytest.mark.parametrize("text", ["", "   ", "\n\t  ", None])
def test_empty_input_gives_empty_string(text):
    assert normalize_sms(text) == ""


def test_only_url():
    assert normalize_sms("http://secure-bank-verify.example/login") == "<url>"


@pytest.mark.parametrize("url", [
    "https://kaspi-login.test/verify?id=1",
    "www.getzed.co.uk",
    "bit.ly/prize-claim-test",
    "egov-fines.test/pay",
])
def test_url_variants(url):
    assert normalize_sms(f"go {url} now") == "go <url> now"


def test_mixed_case():
    assert normalize_sms("ВаШа КаРтА BlOcKeD") == "ваша карта blocked"


def test_cyrillic_preserved():
    assert normalize_sms("Ваша карта заблокирована") == "ваша карта заблокирована"


def test_kazakh_letters_preserved():
    letters = "ӘҒҚҢӨҰҮҺІ әғқңөұүһі"
    result = normalize_sms(f"Картаңыз бұғатталды {letters}")
    for ch in "әғқңөұүһі":
        assert ch in result
    assert result.startswith("картаңыз бұғатталды")


@pytest.mark.parametrize("text", [
    "оплатите 990 тг", "оплатите 5 000 тенге", "win £900", "only $50", "cost 150p",
    "перевод 1 000 000 тг", "приз 20,000 pounds", "штраф 15 000 руб", "50 €",
])
def test_money(text):
    assert "<money>" in normalize_sms(text)
    assert "<num>" not in normalize_sms(text)


@pytest.mark.parametrize("phone", [
    "+7 000 123 45 67", "8 (700) 123-45-67", "09061701461", "+44 7700 900123",
])
def test_phones(phone):
    assert normalize_sms(f"call {phone} now") == "call <phone> now"


def test_short_code_is_number_not_phone():
    assert normalize_sms("text WIN to 87121") == "text win to <num>"


def test_other_numbers():
    assert normalize_sms("встреча в 18:00, ауд. 305") == "встреча в <num> , ауд. <num>"
    assert normalize_sms("12 hours") == "<num> hours"


def test_email():
    assert normalize_sms("write to John.Doe@mail.com") == "write to <email>"


def test_emoji_kept_and_no_crash():
    assert normalize_sms("Привет 😊 как дела") == "привет 😊 как дела"


def test_exclamation_and_question_are_separate_tokens():
    assert normalize_sms("СРОЧНО!!! Что?") == "срочно ! ! ! что ?"


def test_control_characters_and_spaces():
    assert normalize_sms("a\x00b​  c\r\n\td") == "a b c d"
    assert normalize_sms("I\x92m ok") == "i'm ok"


def test_nfkc_normalization():
    # полноширинные латинские буквы и цифры → обычные
    assert normalize_sms("ＦＲＥＥ ｃａｓｈ") == "free cash"


def test_obfuscation_is_not_destroyed():
    # символьные n-граммы (задание 6) должны видеть искажения
    assert normalize_sms("b@nk fr33") == "b@nk fr33"


def test_pure_function():
    text = "Ваша карта *4417 заблокирована! http://unblock.example"
    copy = str(text)
    first = normalize_sms(text)
    second = normalize_sms(text)
    assert text == copy
    assert first == second == "ваша карта * <num> заблокирована ! <url>"


def test_stop_words_option():
    text = "You have won a prize and это твой приз"
    result = normalize_sms(text, remove_stop_words=True)
    assert result == "won prize твой приз"
    assert remove_stopwords("<url> and <money>") == "<url> <money>"
    assert "и" in STOP_WORDS and "және" in STOP_WORDS and "the" in STOP_WORDS


def test_html_entities_and_corpus_placeholders():
    assert normalize_sms("I am  &lt;#&gt;  inches") == "i am <num> inches"
    assert normalize_sms("at &lt;TIME&gt; see &lt;URL&gt;") == "at <num> see <url>"
    assert normalize_sms("T&amp;C apply") == "t&c apply"
