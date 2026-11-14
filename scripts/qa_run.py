"""Прогон демонстрационных сценариев и замеров для отчёта о тестировании (задание 18).

Запуск (нужны обученные модели):  python scripts/qa_run.py
Работает на отдельной временной SQLite-базе, основную db.sqlite3 не трогает. Результаты считают настоящие модели
в момент запуска; итог пишется в docs/qa/qa_results.json и выводится на экран.
"""
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP_DB = Path(tempfile.gettempdir()) / "sms_guard_qa.sqlite3"
if TMP_DB.exists():
    TMP_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TMP_DB.as_posix()}"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("SECRET_KEY", "qa-run-only")

import django  # noqa: E402

django.setup()
from django.conf import settings  # noqa: E402
from django.core.management import call_command  # noqa: E402

settings.ALLOWED_HOSTS = ["*"]
call_command("migrate", verbosity=0)

from django.contrib.auth import get_user_model  # noqa: E402
from django.test import Client  # noqa: E402

from ml import train_sms as ts  # noqa: E402
from ml.predictor import get_predictor  # noqa: E402

DEMO = [  # раздел 8.3 ТЗ: (№, язык, текст, ожидание автора набора)
    (1, "ru", "Привет! Встречаемся завтра в 18:00 у входа в библиотеку.", "Безопасное"),
    (2, "en", "Your order #4821 has been shipped and will arrive on Monday.", "Безопасное"),
    (3, "kk", "Сәлем! Ертең сағат 10:00-де кездесеміз бе?", "Безопасное (с предупреждением о языке)"),
    (4, "ru", "Ваш код подтверждения: 4821. Никому не сообщайте этот код.", "Безопасное или подозрительное"),
    (5, "ru", "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login",
     "Мошенническое"),
    (6, "en", "URGENT: Your account will be suspended in 24 hours. Verify now at http://192.0.2.15/account-update",
     "Мошенническое"),
    (7, "ru", "Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори.",
     "Мошенническое (без ссылки)"),
    (8, "en", "Congratulations! You won a $1000 gift card. Claim your prize: bit.ly/prize-claim-test", "Мошенническое"),
    (9, "ru", "Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции.",
     "Мошенническое (CODE_REQUEST)"),
    (10, "kk", "Картаңыз бұғатталды. Жедел түрде мына сілтеме арқылы растаңыз: http://bank-kz-verify.test",
     "Подозрительное или мошенническое"),
    (11, "ru", "Ваша посылка ожидает оплаты таможенного сбора 350 T: http://post-delivery-pay.example",
     "Подозрительное или мошенническое"),
]
DEMO_URLS = [(12, "https://www.example.com/about", "Безопасная ссылка"),
             (13, "http://xn--exmple-cua.test/secure/login?verify=1", "Опасная ссылка")]
EXTRA = [  # сообщения, которых нет в обучающих данных и демо-наборе (проверка «в реальном времени»)
    (14, "ru", "Добрый день! Напоминаем о записи к стоматологу в четверг в 15:30. Клиника «Улыбка».", "Безопасное"),
    (15, "ru", "Вам начислен кешбэк 25 000 тг! Чтобы получить, введите данные карты: http://cashback-kz.test/get",
     "Мошенническое"),
    (16, "kk", "Құрметті клиент, сіздің шотыңыз бұғатталды. Кодты бізге жіберіңіз.", "Мошенническое"),
    (17, "en", "Hi Sam, the meeting moved to 3pm. Bring the slides please.", "Безопасное"),
]


def short(d):
    return {"category": d["category"], "risk_level": d["risk_level"], "risk_score": d["risk_score"],
            "p_sms": d.get("p_sms"), "p_url_max": d.get("p_url_max"), "language": d.get("language"),
            "features": d["features"], "urls": [(u["url"], u["category"], u["p_url"]) for u in d["urls"]],
            "overrides_hint": d["explanation"][:2], "language_note": d.get("language_note")}


def main():
    predictor = get_predictor()
    if not (predictor.sms_ready and predictor.url_ready):
        sys.exit("Модели не загружены: python -m ml.train_sms и python -m ml.train_url")
    user = get_user_model().objects.create_user("qa_user", "qa@example.com", "QaPass2026")
    client = Client(HTTP_HOST="127.0.0.1")
    client.force_login(user)
    out = {"demo": [], "extra": [], "scenarios": {}}

    for num, lang, text, expected in DEMO + EXTRA:
        r = client.post("/api/v1/check/sms/", {"text": text}, content_type="application/json")
        key = "demo" if num <= 11 else "extra"
        out[key].append({"n": num, "lang": lang, "text": text, "expected": expected, "status": r.status_code,
                         **short(r.json())})
    for num, url, expected in DEMO_URLS:
        r = client.post("/api/v1/check/url/", {"url": url}, content_type="application/json")
        out["demo"].append({"n": num, "lang": "URL", "text": url, "expected": expected, "status": r.status_code,
                            **short(r.json())})

    # пять обязательных сценариев (раздел 18.8 ТЗ)
    by_n = {d["n"]: d for d in out["demo"]}
    out["scenarios"] = {"a": by_n[1], "b": by_n[7], "c": by_n[5], "d": by_n[13], "e": by_n[9]}
    history = client.get("/api/v1/checks/").json()
    stats = client.get("/api/v1/stats/summary/?days=7").json()
    out["after_checks"] = {"history_count": history["count"], "stats_total": stats["totals"]["total"]}

    # НФТ-1: 100 запросов анализа SMS (до 3 URL), время на сервере
    texts = (ts.load_data().test["text_raw"].sample(100, random_state=1).tolist())
    texts[0] = "Срочно! http://a.example/1 http://b.example/2 http://c.example/3 http://d.example/4"
    wall, analysis = [], []
    client.force_login(get_user_model().objects.create_user("qa_perf", "p@example.com", "QaPass2026"))
    from django.core.cache import cache
    for i, text in enumerate(texts):
        if i and i % 25 == 0:
            cache.clear()                         # сброс лимита 30/мин — замеряем время, а не throttling
        t = time.perf_counter()
        r = client.post("/api/v1/check/sms/", {"text": text}, content_type="application/json")
        wall.append((time.perf_counter() - t) * 1000)
        analysis.append(r.json()["processing_ms"])
    wall.sort()
    analysis.sort()
    out["nft1"] = {"n": len(wall), "request_ms_mean": round(statistics.mean(wall), 1),
                   "request_ms_p95": round(wall[94], 1), "request_ms_max": round(wall[-1], 1),
                   "analysis_ms_mean": round(statistics.mean(analysis), 1), "analysis_ms_max": round(analysis[-1], 1)}

    # Ошибки SMS-модели на test (ложные срабатывания и пропуски) — для анализа
    data = ts.load_data()
    proba = predictor.sms_model.predict_proba(data.test[ts.X_COLS])[:, 1]
    thr = predictor.sms_threshold
    test = data.test.assign(p=proba)
    fp = test[(test.label == 0) & (test.p >= thr)].sort_values("p", ascending=False)
    fn = test[(test.label == 1) & (test.p < thr)].sort_values("p")
    pick = lambda d: [{"lang": r.lang, "text": r.text_raw, "p": round(float(r.p), 4)}  # noqa: E731
                      for r in d.groupby("lang", group_keys=False).head(4).itertuples()]
    out["sms_errors"] = {"threshold": thr, "fp_total": int(len(fp)), "fn_total": int(len(fn)),
                         "fp_by_lang": fp.lang.value_counts().to_dict(), "fn_by_lang": fn.lang.value_counts().to_dict(),
                         "fp": pick(fp), "fn": pick(fn)}

    target = ROOT / "docs" / "qa" / "qa_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in out["demo"] + out["extra"]:
        print(f"{row['n']:>2} {row['lang']:3} {row['category']:11} {row['risk_level']:6} {row['risk_score']:>3}  "
              f"ожидание: {row['expected']}")
    print("НФТ-1:", out["nft1"])
    print("После проверок:", out["after_checks"])
    print("Ошибки SMS-модели на test:", {k: out["sms_errors"][k] for k in ("fp_total", "fn_total", "fp_by_lang",
                                                                             "fn_by_lang")})
    print("Сохранено:", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
