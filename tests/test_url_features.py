"""Модульные тесты ml/url_features.py (задание 9). Только зарезервированные домены."""
import socket

import pytest

from ml.url_features import (FEATURE_NAMES, InvalidURLError, explain_url, extract_url_features,
                             registered_domain, validate_url)


def f(url):
    return extract_url_features(url)


def test_fixed_keys_and_order():
    a, b = f("https://www.example.com/about"), f("http://192.0.2.15/x")
    assert list(a) == list(b) == FEATURE_NAMES
    assert len(FEATURE_NAMES) >= 20
    assert all(isinstance(v, (int, float)) for v in a.values())


def test_safe_url_has_no_red_flags():
    x = f("https://www.example.com/about")
    assert x["is_https"] == 1 and x["has_ip"] == 0 and x["n_subdomains"] == 0
    assert x["n_suspicious_words"] == 0 and x["is_shortener"] == 0
    assert explain_url(x, "https://www.example.com/about") == []


@pytest.mark.parametrize("url", ["http://192.0.2.15/account", "http://[2001:db8::1]/login", "http://3232235777/"])
def test_ip_instead_of_domain(url):
    assert f(url)["has_ip"] == 1


def test_scheme_is_added_only_for_parsing():
    res = validate_url("example.org/path")
    assert res.ok and res.normalized == "http://example.org/path"
    assert f("example.org/path")["is_https"] == 0
    assert f("https://example.org")["is_https"] == 1


def test_counts_subdomains_dots_hyphens_digits_special():
    x = f("https://a1.b-2.c.secure-pay.example.com/p_a?x=1&y=%20")
    assert x["n_subdomains"] == 4               # a1, b-2, c, secure-pay
    assert x["n_hyphens_host"] == 2
    assert x["n_dots"] == 5
    assert x["n_digits"] == 5
    assert x["n_special"] >= 6                   # - - _ ? = & = %
    assert x["n_query_params"] == 2


def test_port_at_and_double_slash():
    x = f("http://user@login.example.test:8080/a//b")
    assert x["nonstandard_port"] == 1 and x["has_at"] == 1 and x["double_slash_path"] == 1
    assert f("https://example.com:443/")["nonstandard_port"] == 0


@pytest.mark.parametrize("url", ["bit.ly/prize-claim-test", "https://tinyurl.com/abc-test", "clck.ru/xyz",
                                 "https://cutt.ly/t", "is.gd/q"])
def test_shorteners(url):
    assert f(url)["is_shortener"] == 1


def test_suspicious_words_and_brand_in_url():
    x = f("http://kaspi-secure-login.example/verify")
    assert x["n_suspicious_words"] == 4          # kaspi, secure, login, verify
    assert x["brand_in_url"] == 1
    official = f("https://kaspi.kz/shop")
    assert official["n_suspicious_words"] == 0 and official["brand_in_url"] == 0


def test_punycode_and_mixed_scripts():
    assert f("http://xn--exmple-cua.test/secure/login?verify=1")["has_punycode"] == 1
    assert f("http://pаypal.example/login")["mixed_scripts"] == 1      # «а» — кириллица
    assert f("http://каспи.test")["mixed_scripts"] == 0                # одна метка — один алфавит


def test_brand_similarity_levenshtein():
    assert f("http://kaspl.kz")["brand_similarity"] == 1               # kaspl ↔ kaspi
    assert f("http://paypa1.com/login")["brand_similarity"] == 1
    assert f("https://kaspi.kz")["brand_similarity"] == 0              # официальный домен
    assert f("https://www.example.com")["brand_similarity"] == 0


def test_suspicious_tld_and_entropy():
    assert f("http://prize-winner.xyz")["suspicious_tld"] == 1
    assert f("https://example.com")["suspicious_tld"] == 0
    assert f("http://qx7zk9w2vj.example.com")["host_entropy"] > f("http://aaaa.example.com")["host_entropy"]


@pytest.mark.parametrize("bad, message_part", [
    ("", "Введите"), ("   ", "Введите"), ("ftp://example.com/file", "http"), ("http://exa mple.com", "пробел"),
    ("http://example.com:99999", "порт"), ("http://", "доменн"), ("localhost", "точку"),
    ("http://" + "a" * 2050 + ".com", "длиннее"), (None, "строкой"), ("http://ex\x00ample.com", "управляющие"),
])
def test_invalid_input_is_handled(bad, message_part):
    res = validate_url(bad)
    assert not res.ok and message_part in res.error
    with pytest.raises(InvalidURLError):
        extract_url_features(bad)


def test_explain_url_reasons_sorted_and_limited():
    url = "http://user@192.0.2.15:8080//login-verify-account-update?x=1"
    reasons = explain_url(f(url), url)
    assert reasons[0] == "Используется IP-адрес вместо доменного имени"
    assert 1 <= len(reasons) <= 6
    assert any("Подозрительные слова" in r for r in reasons)


def test_registered_domain_for_grouping():
    assert registered_domain("https://login.secure.example.co.uk/a") == "example.co.uk"
    assert registered_domain("http://a.b.example.test") == "example.test"
    assert registered_domain("192.0.2.15") == "192.0.2.15"


def test_no_network_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("сетевой вызов запрещён")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    x = f("https://brand-new-domain-never-seen.example.org/login")
    assert x["n_suspicious_words"] == 1


def test_module_does_not_import_network_libraries():
    import inspect

    import ml.url_features as mod
    source = inspect.getsource(mod)
    for banned in ("import requests", "import socket", "urlopen", "gethostbyname", "http.client"):
        assert banned not in source
