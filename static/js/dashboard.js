/* Dashboard и статистика: данные из /api/v1/stats/summary/, графики — Plotly.js (локальная копия). */
"use strict";

const GuardCharts = (() => {
  const COLORS = { SAFE: "#198754", SUSPICIOUS: "#d69e00", FRAUD: "#c62828" };
  const NAMES = { SAFE: "Безопасные", SUSPICIOUS: "Подозрительные", FRAUD: "Мошеннические / опасные" };
  const LEVEL_COLORS = { LOW: "#198754", MEDIUM: "#d69e00", HIGH: "#c62828" };
  const LEVEL_NAMES = { LOW: "Низкий", MEDIUM: "Средний", HIGH: "Высокий" };
  const CONFIG = { displaylogo: false, responsive: true, locale: "ru", modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  const { el, icon, badge, fmtDate } = Guard;

  function layout(title, extra = {}) {
    return {
      title: { text: title, font: { size: 14 }, x: 0, xanchor: "left" },
      margin: { l: 50, r: 10, t: 40, b: 50 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size: 12, color: "#212529" },
      legend: { orientation: "h", y: -0.3 }, separators: ", ", ...extra,
    };
  }

  function doughnut(node, data) {
    const keys = ["SAFE", "SUSPICIOUS", "FRAUD"];
    Plotly.react(node, [{
      type: "pie", hole: 0.55, sort: false, direction: "clockwise",
      values: keys.map((k) => data.categories[k]), labels: keys.map((k) => NAMES[k]),
      marker: { colors: keys.map((k) => COLORS[k]) }, textinfo: "percent", hovertemplate: "%{label}: %{value}<extra></extra>",
    }], layout("Распределение категорий", { margin: { l: 10, r: 10, t: 40, b: 10 }, legend: { orientation: "h", y: -0.1 } }), CONFIG);
  }

  /** Целые деления оси: не более ~6 подписей. */
  function intAxis(max) { return { tick0: 0, dtick: Math.max(1, Math.ceil(max / 6)), rangemode: "tozero" }; }

  function daily(node, data, stacked = false) {
    const keys = ["SAFE", "SUSPICIOUS", "FRAUD"];
    const max = stacked ? Math.max(1, ...data.daily.dates.map((_, i) => keys.reduce((s, k) => s + data.daily[k][i], 0)))
      : Math.max(1, ...keys.flatMap((k) => data.daily[k]));
    const traces = ["SAFE", "SUSPICIOUS", "FRAUD"].map((k) => ({
      type: stacked ? "bar" : "scatter", mode: "lines+markers", name: NAMES[k], x: data.daily.dates, y: data.daily[k],
      line: { color: COLORS[k], width: 2 }, marker: { color: COLORS[k], size: 5 },
      hovertemplate: "%{x|%d.%m}: %{y}<extra>" + NAMES[k] + "</extra>",
    }));
    Plotly.react(node, traces, layout("Проверки по дням за " + data.period_days + " дн.", {
      barmode: "stack",
      xaxis: { title: { text: "Дата" }, type: "date", tickformat: "%d.%m" },
      yaxis: { title: { text: "Число проверок" }, ...intAxis(max) },
    }), CONFIG);
  }

  function topFeatures(node, data) {
    const items = data.top_features.slice().reverse();
    Plotly.react(node, [{
      type: "bar", orientation: "h", x: items.map((f) => f.count), name: "Срабатываний",
      y: items.map((f) => (f.title.length > 26 ? f.title.slice(0, 25) + "…" : f.title)), customdata: items.map((f) => f.title),
      marker: { color: "#1f3a5f" }, hovertemplate: "%{customdata}: %{x}<extra></extra>",
    }], layout("Топ-5 признаков социальной инженерии", {
      margin: { l: 10, r: 10, t: 60, b: 50 }, showlegend: true,
      legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12 },
      xaxis: { title: { text: "Число SMS с признаком" }, ...intAxis(Math.max(1, ...items.map((f) => f.count))) },
      yaxis: { title: { text: "" }, automargin: true },
    }), CONFIG);
  }

  function riskLevels(node, data) {
    const keys = ["LOW", "MEDIUM", "HIGH"];
    Plotly.react(node, [{
      type: "bar", x: keys.map((k) => LEVEL_NAMES[k]), y: keys.map((k) => data.risk_levels[k]), name: "Проверок",
      marker: { color: keys.map((k) => LEVEL_COLORS[k]) }, text: keys.map((k) => data.risk_levels[k]), textposition: "auto",
      hovertemplate: "%{x}: %{y}<extra></extra>",
    }], layout("Распределение уровней риска", {
      showlegend: true, legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12 },
      xaxis: { title: { text: "Уровень риска" } },
      yaxis: { title: { text: "Число проверок" }, ...intAxis(Math.max(1, ...keys.map((k) => data.risk_levels[k]))) },
    }), CONFIG);
  }

  function types(node, data) {
    Plotly.react(node, [{
      type: "pie", hole: 0.55, values: [data.totals.sms, data.totals.url], labels: ["SMS", "Ссылки (URL)"],
      marker: { colors: ["#1f3a5f", "#7c96b8"] }, textinfo: "percent+value", hovertemplate: "%{label}: %{value}<extra></extra>",
    }], layout("Типы проверок", { margin: { l: 10, r: 10, t: 40, b: 10 }, legend: { orientation: "h", y: -0.1 } }), CONFIG);
  }

  function cards(node, data) {
    const t = data.totals;
    const items = [
      ["Всего SMS", t.sms, "bi-chat-left-text", ""],
      ["Всего URL", t.url, "bi-link-45deg", ""],
      ["Безопасные", t.safe, "bi-shield-check", "safe"],
      ["Подозрительные", t.suspicious, "bi-exclamation-triangle", "suspicious"],
      ["Мошеннические", t.fraud, "bi-x-octagon", "fraud"],
      ["Средний Risk Score", t.avg_score === null ? "—" : String(t.avg_score).replace(".", ","), "bi-speedometer", ""],
    ];
    node.replaceChildren(...items.map(([title, value, ic, cls]) => el("div", { class: "col-6 col-md-4 col-xxl-2" }, [
      el("div", { class: "stat-card" + (cls ? " stat-" + cls : "") }, [
        el("div", { class: "stat-title" }, [icon(ic), " ", title]), el("div", { class: "stat-value", text: String(value) })]),
    ])));
  }

  function recent(node, data) {
    const tbody = el("tbody");
    for (const r of data.recent) {
      const fragment = r.fragment === null ? el("span", { class: "text-body-secondary", text: "скрыто (" + (r.username || "другой пользователь") + ")" }) : (
        r.own ? el("a", { href: "/checks/" + r.id + "/", class: "link-body-emphasis text-decoration-none", text: r.fragment }) : el("span", { text: r.fragment }));
      tbody.append(el("tr", {}, [
        el("td", { class: "small text-nowrap", text: fmtDate(r.created_at) }),
        el("td", {}, [el("span", { class: "badge text-bg-light border", text: r.type })]),
        el("td", { class: "fragment" }, [fragment]),
        el("td", { class: "text-end fw-semibold", text: String(r.risk_score) }),
        el("td", {}, [badge(r.category)]),
      ]));
    }
    node.replaceChildren(...tbody.children);
  }

  function models(node, data) {
    if (!data.models) { node.hidden = true; return; }
    const pct = (v) => (v === undefined || v === null ? "—" : Number(v).toFixed(3).replace(".", ","));
    const rows = data.models.map((m) => el("tr", {}, [
      el("td", { text: m.type.toUpperCase() }), el("td", { text: m.version }), el("td", { text: m.algorithm }),
      el("td", { text: m.threshold === null ? "—" : String(m.threshold).replace(".", ",") }),
      el("td", { text: pct(m.metrics.precision) }), el("td", { text: pct(m.metrics.recall) }),
      el("td", { text: pct(m.metrics.f1) }), el("td", { text: pct(m.metrics.roc_auc) }), el("td", { class: "small", text: m.trained_at }),
    ]));
    node.querySelector("tbody").replaceChildren(...rows);
    node.hidden = false;
  }

  async function load(days) {
    const { ok, data } = await Guard.api("/api/v1/stats/summary/?days=" + days);
    if (!ok) throw new Error(data && data.error ? data.error.message : "Не удалось загрузить статистику");
    return data;
  }

  /** Переключатель периода 7/30/90: состояние хранится в адресной строке (?days=). */
  function bindPeriod(group, onChange) {
    group.querySelectorAll("[data-days]").forEach((btn) => btn.addEventListener("click", () => {
      group.querySelectorAll("[data-days]").forEach((b) => { b.classList.toggle("active", b === btn); b.setAttribute("aria-pressed", b === btn); });
      const url = new URL(window.location.href);
      url.searchParams.set("days", btn.dataset.days);
      window.history.replaceState(null, "", url);
      onChange(Number(btn.dataset.days));
    }));
  }

  return { load, cards, doughnut, daily, topFeatures, riskLevels, types, recent, models, bindPeriod };
})();
