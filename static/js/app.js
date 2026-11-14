/* SMS Guard — общие функции интерфейса.
 * Пользовательский текст выводится ТОЛЬКО через textContent (НФТ-5): innerHTML для данных не используется.
 * Ссылки из сообщений показываются неактивным текстом (без <a>), домен выделяется (ФТ-3.10).
 */
"use strict";

const Guard = (() => {
  const CATEGORY = {
    SAFE: { cls: "safe", icon: "bi-shield-check", sms: "Безопасное сообщение", url: "Безопасная ссылка", short: "Безопасное" },
    SUSPICIOUS: { cls: "suspicious", icon: "bi-exclamation-triangle", sms: "Подозрительное сообщение", url: "Подозрительная ссылка", short: "Подозрительное" },
    FRAUDULENT: { cls: "fraud", icon: "bi-x-octagon", sms: "Мошенническое сообщение", url: "Мошенническое", short: "Мошенническое" },
    DANGEROUS: { cls: "fraud", icon: "bi-x-octagon", sms: "Опасная ссылка", url: "Опасная ссылка", short: "Опасная" },
  };
  const LEVEL = { LOW: "низкий", MEDIUM: "средний", HIGH: "высокий" };

  function getCookie(name) {
    const match = document.cookie.split("; ").find((c) => c.startsWith(name + "="));
    return match ? decodeURIComponent(match.split("=")[1]) : "";
  }

  /** fetch к API с CSRF-токеном; возвращает {ok, status, data}. */
  async function api(url, options = {}) {
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    if (options.body !== undefined) headers["Content-Type"] = "application/json";
    if (options.method && options.method !== "GET") headers["X-CSRFToken"] = getCookie("csrftoken");
    let response;
    try {
      response = await fetch(url, { credentials: "same-origin", ...options, headers,
        body: options.body !== undefined ? JSON.stringify(options.body) : undefined });
    } catch (e) {
      return { ok: false, status: 0, data: { error: { code: "NETWORK", message: "Нет связи с сервером. Проверьте подключение и повторите." } } };
    }
    if (response.status === 401) {
      window.location.href = "/login/?next=" + encodeURIComponent(window.location.pathname);
      return { ok: false, status: 401, data: null };
    }
    let data = null;
    try { data = await response.json(); } catch (e) { data = null; }
    if (!response.ok && !(data && data.error)) {
      data = { error: { code: "HTTP_" + response.status, message: "Ошибка сервера (" + response.status + "). Попробуйте позже." } };
    }
    return { ok: response.ok, status: response.status, data };
  }

  /** Создание элемента: el("span", {class: "x"}, ["текст", childNode]) — строки становятся текстовыми узлами. */
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined) continue;
      node.appendChild(typeof child === "string" || typeof child === "number" ? document.createTextNode(String(child)) : child);
    }
    return node;
  }

  function icon(name) { return el("i", { class: "bi " + name, "aria-hidden": "true" }); }

  /** Бейдж категории: цвет + иконка + текст (цвет — не единственный носитель смысла). */
  function badge(category, kind = "short") {
    const meta = CATEGORY[category] || CATEGORY.SUSPICIOUS;
    return el("span", { class: "badge risk-badge risk-" + meta.cls }, [icon(meta.icon), " ", meta[kind] || meta.short]);
  }

  /** Ссылка неактивным текстом: схема и путь обычным шрифтом, домен выделен. */
  function urlText(url, domain) {
    const wrap = el("span", { class: "url-text", title: "Ссылка не открывается: показан только текст" });
    const idx = domain ? url.toLowerCase().indexOf(domain.toLowerCase()) : -1;
    if (idx >= 0) {
      wrap.append(el("span", { text: url.slice(0, idx) }), el("mark", { class: "url-domain", text: url.slice(idx, idx + domain.length) }),
        el("span", { text: url.slice(idx + domain.length) }));
    } else {
      wrap.textContent = url;
    }
    return wrap;
  }

  function fmtP(p) { return p === null || p === undefined ? "—" : Number(p).toFixed(2).replace(".", ","); }

  function fmtDate(iso) {
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function scoreBar(score, category) {
    const meta = CATEGORY[category] || CATEGORY.SUSPICIOUS;
    return el("div", { class: "score-wrap" }, [
      el("div", { class: "d-flex justify-content-between small mb-1" }, [el("span", { text: "Risk Score" }), el("strong", { text: score + "/100" })]),
      el("div", { class: "progress score-bar", role: "progressbar", "aria-label": "Risk Score", "aria-valuenow": score, "aria-valuemin": 0, "aria-valuemax": 100 }, [
        el("div", { class: "progress-bar bg-risk-" + meta.cls, style: "width: " + Math.max(2, score) + "%" }),
      ]),
      el("div", { class: "score-scale small text-body-secondary" }, [el("span", { text: "0" }), el("span", { text: "35" }), el("span", { text: "70" }), el("span", { text: "100" })]),
    ]);
  }

  const URL_FEATURE_ROWS = [
    ["url_length", "Длина", (v) => v],
    ["is_https", "HTTPS", (v) => (v ? "да" : "нет")],
    ["has_ip", "IP-адрес вместо домена", (v) => (v ? "да" : "нет")],
    ["is_shortener", "Сокращатель ссылок", (v) => (v ? "да" : "нет")],
    ["n_subdomains", "Поддомены", (v) => v],
    ["n_suspicious_words", "Подозрительные слова", (v) => v],
    ["has_punycode", "Punycode (xn--)", (v) => (v ? "да" : "нет")],
    ["mixed_scripts", "Смешение алфавитов", (v) => (v ? "да" : "нет")],
    ["nonstandard_port", "Нестандартный порт", (v) => (v ? "да" : "нет")],
    ["has_at", "Символ @ в адресе", (v) => (v ? "да" : "нет")],
    ["brand_similarity", "Сходство с брендом", (v) => (v ? "да" : "нет")],
  ];

  function urlFeatureTable(u) {
    const tbody = el("tbody");
    for (const [key, title, fmt] of URL_FEATURE_ROWS) {
      if (!u.features || !(key in u.features)) continue;
      let value = String(fmt(u.features[key]));
      if (key === "n_suspicious_words" && u.suspicious_words && u.suspicious_words.length) value += " (" + u.suspicious_words.join(", ") + ")";
      tbody.append(el("tr", {}, [el("th", { scope: "row", class: "fw-normal", text: title }), el("td", { text: value })]));
    }
    return el("table", { class: "table table-sm mb-0 url-features" }, [
      el("caption", { class: "visually-hidden", text: "Признаки ссылки" }), tbody]);
  }

  /** Полный блок результата (страницы проверки и подробного результата). */
  function renderResult(container, d, opts = {}) {
    container.replaceChildren();
    const isUrl = d.type === "URL";
    const meta = CATEGORY[d.category] || CATEGORY.SUSPICIOUS;

    const head = el("div", { class: "result-head risk-border-" + meta.cls }, [
      el("div", { class: "result-title" }, [icon(meta.icon), " ", (isUrl ? meta.url : meta.sms).toUpperCase()]),
      el("div", { class: "result-sub" }, [
        badge(d.category), " ",
        el("span", { class: "ms-1", text: "Риск: " + (d.risk_level_ru || LEVEL[d.risk_level] || d.risk_level) }),
        el("span", { class: "text-body-secondary ms-2", text: "· " + fmtDate(d.created_at) }),
      ]),
    ]);
    container.append(head);

    const body = el("div", { class: "result-body" });
    body.append(scoreBar(d.risk_score, d.category));

    // Вероятности моделей
    const probs = el("dl", { class: "row probs small mb-0" });
    if (!isUrl) probs.append(el("dt", { class: "col-sm-8 fw-normal", text: "Вероятность мошенничества по тексту (SMS-модель)" }), el("dd", { class: "col-sm-4 fw-semibold mb-1", text: fmtP(d.p_sms) }));
    probs.append(el("dt", { class: "col-sm-8 fw-normal", text: isUrl ? "Вероятность опасности ссылки (URL-модель)" : "Максимальная вероятность опасности ссылки (URL-модель)" }),
      el("dd", { class: "col-sm-4 fw-semibold mb-1", text: d.p_url_max === null ? "ссылок нет" : fmtP(d.p_url_max) }));
    if (d.language) probs.append(el("dt", { class: "col-sm-8 fw-normal", text: "Язык сообщения" }), el("dd", { class: "col-sm-4 mb-1", text: d.language }));
    body.append(el("section", { class: "result-section" }, [el("h3", { class: "h6", text: "Вероятности моделей" }), probs]));

    if (d.language_note) body.append(el("div", { class: "alert alert-secondary small py-2", role: "note" }, [icon("bi-translate"), " ", d.language_note]));

    // Исходный текст
    if (opts.showInput && d.input) {
      const inputBlock = isUrl ? el("p", { class: "mb-0" }, [urlText(d.input, d.urls[0] && d.urls[0].domain)]) : el("p", { class: "input-text mb-0", text: d.input });
      const note = d.text_stored === false ? el("div", { class: "form-text", text: "Сохранён маскированный текст: номера карт, телефонов и суммы заменены на ***." }) : null;
      body.append(el("section", { class: "result-section" }, [el("h3", { class: "h6", text: isUrl ? "Проверенная ссылка" : "Текст сообщения" }), inputBlock, note]));
    }

    // Найденные ссылки
    if (d.urls && d.urls.length) {
      const list = el("ul", { class: "list-unstyled url-list mb-0" });
      for (const u of d.urls) {
        const li = el("li", { class: "url-item" }, [
          el("div", { class: "d-flex flex-wrap align-items-center gap-2" }, [urlText(u.url, u.domain), badge(u.category, "url"),
            el("span", { class: "small text-body-secondary", text: "P_url = " + fmtP(u.p_url) })]),
        ]);
        if (u.reasons && u.reasons.length) li.append(el("ul", { class: "small mb-0 mt-1" }, u.reasons.slice(0, 5).map((r) => el("li", { text: r }))));
        if (isUrl) li.append(el("div", { class: "mt-2" }, [urlFeatureTable(u)]));
        list.append(li);
      }
      body.append(el("section", { class: "result-section" }, [
        el("h3", { class: "h6", text: isUrl ? "Детали ссылки" : "Найденные ссылки (" + d.urls.length + ")" }),
        el("p", { class: "form-text mt-0", text: "Ссылки не открываются: анализируется только их структура." }), list]));
    }

    // Признаки социальной инженерии
    if (d.features_detail && d.features_detail.length) {
      body.append(el("section", { class: "result-section" }, [el("h3", { class: "h6", text: "Обнаруженные признаки" }),
        el("ul", { class: "feature-list mb-0" }, d.features_detail.map((f) => el("li", {}, [icon("bi-flag"), " ", f.title])))]));
    }

    // Объяснение
    body.append(el("section", { class: "result-section" }, [el("h3", { class: "h6", text: "Почему такой результат" }),
      el("ol", { class: "mb-0" }, (d.explanation || []).map((r) => el("li", { text: r })))]));

    body.append(el("div", { class: "alert alert-recommend mb-2", role: "note" }, [icon("bi-lightbulb"), " ", el("strong", { text: "Рекомендация: " }), d.recommendation]));
    body.append(el("p", { class: "disclaimer small mb-1" }, [icon("bi-info-circle"), " ", d.disclaimer]));
    const versions = d.model_versions || {};
    body.append(el("p", { class: "small text-body-secondary mb-0", text: "Модели: SMS " + (versions.sms || "—") + ", URL " + (versions.url || "—") + (d.processing_ms ? " · анализ " + Math.round(d.processing_ms) + " мс" : "") }));

    if (opts.detailLink && d.id) {
      body.append(el("div", { class: "mt-3" }, [el("a", { class: "btn btn-outline-guard btn-sm", href: "/checks/" + encodeURIComponent(d.id) + "/" }, [icon("bi-box-arrow-up-right"), " Подробный результат №" + d.id])]));
    }
    container.append(body);
    container.hidden = false;
  }

  /** Отправка формы проверки: блокировка кнопки, индикатор, ошибка рядом с полем. */
  function bindCheckForm({ form, input, button, errorBox, resultBox, endpoint, field, validate }) {
    let busy = false;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (busy) return;                                   // защита от двойной отправки (НФТ-3)
      const value = input.value;
      const clientError = validate ? validate(value) : null;
      showError(input, errorBox, clientError);
      if (clientError) return;
      busy = true;
      button.disabled = true;
      const label = button.querySelector(".btn-label");
      const spinner = button.querySelector(".spinner-border");
      const original = label.textContent;
      label.textContent = "Анализ…";
      spinner.hidden = false;
      resultBox.setAttribute("aria-busy", "true");
      try {
        const { ok, data } = await api(endpoint, { method: "POST", body: { [field]: value } });
        if (ok) {
          renderResult(resultBox, data, { detailLink: true });
          resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
        } else if (data && data.error) {
          showError(input, errorBox, data.error.message);
        }
      } finally {
        busy = false;
        button.disabled = false;
        label.textContent = original;
        spinner.hidden = true;
        resultBox.removeAttribute("aria-busy");
      }
    });
  }

  function showError(input, box, message) {
    if (message) {
      box.textContent = message;
      box.hidden = false;
      input.classList.add("is-invalid");
      input.setAttribute("aria-invalid", "true");
    } else {
      box.textContent = "";
      box.hidden = true;
      input.classList.remove("is-invalid");
      input.removeAttribute("aria-invalid");
    }
  }

  return { api, el, icon, badge, urlText, renderResult, bindCheckForm, showError, fmtDate, fmtP, CATEGORY, LEVEL };
})();
