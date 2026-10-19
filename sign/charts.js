// =============================================================================
//  sign/charts.js: small dependency-free charts (SVG + HTML)
// =============================================================================
//  - lineChart: training curves (accuracy, loss). One y-axis per chart:
//    accuracy and loss have different scales, so they are TWO charts, never
//    one chart with two axes. 2px lines, legend + direct end labels, hover
//    crosshair with a tooltip.
//  - confusionHeatmap: rows = true letter, columns = predicted letter.
//    Single-hue (blue) scale: brighter = larger share of that row.
//  Colours come from CSS variables (--series-1..3, --heat) defined in
//  style.css; text always uses the ink colours, not the series colour.
// =============================================================================

const SVGNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};

// series: [{ name, color (CSS value), values: [number|null per epoch] }]
// opts: { title, yMin, yMax, format(v), width, height }
export function lineChart(container, series, opts = {}) {
  const W = opts.width || 270, H = opts.height || 130;
  const m = { l: 34, r: 40, t: 22, b: 20 };
  const n = Math.max(1, ...series.map((s) => s.values.length));
  const all = series.flatMap((s) => s.values).filter((v) => v != null && Number.isFinite(v));
  const yMin = opts.yMin ?? Math.min(0, ...all);
  const yMax = opts.yMax ?? (Math.max(...all, 1e-6) * 1.08);
  const fmt = opts.format || ((v) => v.toFixed(2));
  const x = (i) => m.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
  const y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);

  container.innerHTML = "";
  container.classList.add("chart");
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": opts.title || "chart" }, container);

  // Title + legend (identity is never colour-alone: names + swatches)
  const head = document.createElement("div");
  head.className = "chart-head";
  head.innerHTML = `<span class="chart-title">${opts.title || ""}</span>` +
    series.map((s) => `<span class="chart-key"><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  container.prepend(head);

  // Recessive grid: 3 horizontal lines + labels
  for (let g = 0; g <= 2; g++) {
    const v = yMin + ((yMax - yMin) * g) / 2;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }, svg);
    el("text", { x: m.l - 6, y: y(v) + 3, class: "tick", "text-anchor": "end" }, svg).textContent = fmt(v);
  }
  el("text", { x: m.l, y: H - 4, class: "tick" }, svg).textContent = "epoch 1";
  el("text", { x: W - m.r, y: H - 4, class: "tick", "text-anchor": "end" }, svg).textContent = `${n}`;

  // Lines + direct label at the last point
  const ends = [];
  for (const s of series) {
    const pts = s.values.map((v, i) => (v == null ? null : `${x(i)},${y(v)}`)).filter(Boolean);
    if (!pts.length) continue;
    el("polyline", { points: pts.join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const li = s.values.length - 1;
    const lv = s.values[li];
    if (lv != null) {
      el("circle", { cx: x(li), cy: y(lv), r: 3, fill: s.color, class: "end-dot" }, svg);
      ends.push({ x: x(li) + 6, y: y(lv) + 3, text: fmt(lv) });
    }
  }
  // End labels: when lines finish close together, push the labels apart so
  // they don't print on top of each other (min 10px between baselines).
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 10);
  for (const e of ends) el("text", { x: e.x, y: e.y, class: "end-label" }, svg).textContent = e.text;

  // Hover: crosshair + tooltip for the nearest epoch
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" }, svg);
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  container.appendChild(tip);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>Epoch ${i + 1}</b>` + series.map((s) =>
      `<div><i style="background:${s.color}"></i>${s.name}<span>${s.values[i] == null ? "–" : fmt(s.values[i])}</span></div>`).join("");
    tip.hidden = false;
    const left = (x(i) / W) * r.width;
    tip.style.left = `${Math.min(left + 8, r.width - 130)}px`;
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
}

// matrix: rows = true class, cols = predicted (last column = "none").
// labels: class names; showNone: include the "none" column.
export function confusionHeatmap(container, matrix, labels, { showNone = true } = {}) {
  const cols = showNone ? labels.length + 1 : labels.length;
  const colName = (c) => (c === labels.length ? "∅" : labels[c]);
  let html = `<div class="cm" style="grid-template-columns: 22px repeat(${cols}, minmax(14px, 1fr))">`;
  html += `<div class="cm-corner" title="rows: true letter · columns: predicted">↓→</div>`;
  for (let c = 0; c < cols; c++) html += `<div class="cm-col">${colName(c)}</div>`;
  for (let r = 0; r < labels.length; r++) {
    const row = matrix[r];
    const total = row.reduce((a, b) => a + b, 0);
    html += `<div class="cm-row">${labels[r]}</div>`;
    for (let c = 0; c < cols; c++) {
      const n = row[c] || 0;
      const share = total ? n / total : 0;
      const diag = c === r;
      const tipText = total
        ? `True ${labels[r]} → predicted ${c === labels.length ? "none" : labels[c]}: ${n} of ${total} (${Math.round(share * 100)}%)`
        : `No test samples of ${labels[r]}`;
      html += `<div class="cm-cell${diag ? " diag" : ""}${n && !diag ? " off" : ""}" style="--v:${share.toFixed(3)}" title="${tipText}"></div>`;
    }
  }
  html += "</div>";
  container.innerHTML = html;
}
