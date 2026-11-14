"""Сквозные проверки интерфейса в браузере (задание 18). Не входит в pytest: нужен запущенный сервер и Chrome.

    python manage.py runserver            # в другом терминале, после seed_demo
    pip install playwright && playwright install chromium
    python tests/e2e/ui_checks.py [http://127.0.0.1:8000] [путь к chrome]

Результат: таблица «проверка — пройдено/не пройдено» и docs/qa/ui_results.json.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CHROME = sys.argv[2] if len(sys.argv) > 2 else None
USER, PASSWORD = "demo_user", "DemoPass2026"
results = []


def check(case_id, title, ok, actual):
    results.append({"id": case_id, "title": title, "passed": bool(ok), "actual": actual})
    print(f"{'ПРОЙДЕН ' if ok else 'НЕ ПРОЙДЕН'} {case_id} {title}: {actual}")


def login(page):
    page.goto(BASE + "/login/")
    page.fill("#id_username", USER)
    page.fill("#id_password", PASSWORD)
    page.click("button[type=submit]")
    page.wait_for_load_state("networkidle")


def history_count(page):
    return page.evaluate("fetch('/api/v1/checks/').then(r => r.json()).then(d => d.count)")


with sync_playwright() as p:
    kwargs = {"executable_path": CHROME, "args": ["--no-sandbox"]} if CHROME else {}
    browser = p.chromium.launch(**kwargs)
    page = browser.new_page(viewport={"width": 1366, "height": 900})
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))

    page.goto(BASE + "/history/")
    check("UI-01", "Без входа страница истории перенаправляет на вход", "/login/" in page.url, page.url.replace(BASE, ""))
    login(page)

    # Проверка SMS без перезагрузки страницы
    page.goto(BASE + "/check/sms/")
    page.evaluate("window.__marker = 42")
    page.fill("#sms-text", "Ваша карта заблокирована! Срочно подтвердите данные: http://secure-bank-verify.example/login")
    page.click("#sms-submit")
    page.wait_for_selector("#result .result-head")
    marker = page.evaluate("window.__marker")
    check("UI-02", "Проверка работает без перезагрузки (fetch)", marker == 42,
          f"переменная страницы сохранилась: {marker}; результат: {page.inner_text('.result-title')}")
    links = page.locator("#result a[href*='secure-bank-verify']").count()
    check("UI-03", "Ссылки в результате не кликабельны", links == 0,
          f"элементов <a> со ссылкой: {links}; домен выделен: {page.locator('#result mark.url-domain').count() > 0}")
    badge = page.locator("#result .risk-badge").first
    check("UI-04", "Бейдж: цвет + иконка + текст", badge.locator("i.bi").count() == 1 and badge.inner_text().strip() != "",
          f"текст бейджа «{badge.inner_text().strip()}», иконка есть")

    # Ошибки рядом с полем
    page.click("#sms-clear")
    page.click("#sms-submit")
    err = page.inner_text("#sms-error")
    check("UI-05", "Пустой ввод — ошибка рядом с полем", "не может быть пустым" in err, err)
    page.fill("#sms-text", "а" * 1001)
    counter = page.inner_text("#sms-counter")
    page.click("#sms-submit")
    err = page.inner_text("#sms-error")
    check("UI-06", "1001 символ — счётчик и ошибка", counter == "1001/1000" and "длиннее 1000" in err,
          f"счётчик {counter}; ошибка «{err}»")

    # XSS
    before = history_count(page)
    page.fill("#sms-text", "<script>alert(1)</script>")
    page.click("#sms-submit")
    page.wait_for_selector("#result .result-head")
    detail = page.get_attribute("#result a[href^='/checks/']", "href")
    page.goto(BASE + detail)
    page.wait_for_selector("#result .input-text")
    shown = page.inner_text("#result .input-text")
    check("UI-07", "<script>alert(1)</script> отображается как текст", shown == "<script>alert(1)</script>" and not dialogs,
          f"на странице: «{shown}», диалогов alert: {len(dialogs)}")

    # Двойной клик
    page.goto(BASE + "/check/sms/")
    page.fill("#sms-text", "Проверка двойного клика: встреча перенесена на пятницу")
    before = history_count(page)
    page.dblclick("#sms-submit")
    page.wait_for_selector("#result .result-head")
    page.wait_for_timeout(500)
    after = history_count(page)
    check("UI-08", "Двойной клик создаёт одну запись (НФТ-3)", after - before == 1, f"записей добавлено: {after - before}")

    # URL
    page.goto(BASE + "/check/url/")
    hint = page.inner_text("#url-hint")
    page.fill("#url-input", "http://xn--exmple-cua.test/secure/login?verify=1")
    page.click("#url-submit")
    page.wait_for_selector("#result .url-features")
    check("UI-09", "Проверка URL: подсказка и таблица признаков", "не открывается" in hint,
          f"{page.inner_text('.result-title')}; строк признаков: {page.locator('.url-features tr').count()}")

    # Dashboard и период
    page.goto(BASE + "/")
    page.wait_for_selector("#cards .stat-card")
    page.click("[data-days='7']")
    page.wait_for_timeout(800)
    check("UI-10", "Переключатель периода меняет адресную строку", page.url.endswith("?days=7"), page.url.replace(BASE, ""))

    # История: фильтры в адресной строке
    page.goto(BASE + "/history/")
    page.select_option("#f-type", "URL")
    page.click("#filters button[type=submit]")
    page.wait_for_load_state("networkidle")
    types = set(page.locator("tbody tr td:nth-child(2)").all_inner_texts())
    check("UI-11", "Фильтр истории по типу сохраняется в URL", "type=URL" in page.url and types == {"URL"},
          f"{page.url.replace(BASE, '')}; типы в таблице: {sorted(types)}")

    # Адаптивность: нет горизонтальной прокрутки страницы
    for width in (360, 768, 1366):
        page.set_viewport_size({"width": width, "height": 800})
        overflow = []
        for path in ("/", "/check/sms/", "/check/url/", "/history/", "/stats/", "/profile/"):
            page.goto(BASE + path)
            page.wait_for_load_state("networkidle")
            sw = page.evaluate("document.documentElement.scrollWidth")
            if sw > width:
                overflow.append(f"{path}: {sw}px")
        check(f"UI-12-{width}", f"Нет горизонтальной прокрутки страницы при {width} px", not overflow,
              "; ".join(overflow) or "все 6 страниц помещаются по ширине")

    # Выход
    page.set_viewport_size({"width": 1366, "height": 900})
    page.goto(BASE + "/")
    page.click("form[action='/logout/'] button")
    page.wait_for_load_state("networkidle")
    page.goto(BASE + "/history/")
    check("UI-13", "После выхода защищённые страницы недоступны", "/login/" in page.url, page.url.replace(BASE, ""))
    browser.close()

out = Path(__file__).resolve().parents[2] / "docs" / "qa" / "ui_results.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Итого: {sum(r['passed'] for r in results)} из {len(results)} пройдено")
