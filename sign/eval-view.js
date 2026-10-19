// =============================================================================
//  sign/eval-view.js: the "Evaluate" window (see sign/evaluate.js for the method)
// =============================================================================

import { runEvaluation } from "./evaluate.js";
import { confusionHeatmap } from "./charts.js";
import { KIND_NAMES, KINDS, knnSettings } from "./classifiers.js";
import { MLP_DEFAULTS } from "./mlp.js";
import * as dataset from "./dataset.js";

// Categorical colours, fixed order: Rules = slot 1, k-NN = slot 2, MLP = slot 3.
const COLOR = { rules: "var(--series-1)", knn: "var(--series-2)", mlp: "var(--series-3)" };
const pct = (v) => `${Math.round(v * 100)}%`;

let modal, results, statusEl, runBtn, splitSel, kInput, metricSel, epochsInput, dataEl;
let last = null;
let cmKind = "mlp";
let running = false;

export function initEvalView() {
  const $ = (id) => document.getElementById(id);
  modal = $("evalModal");
  results = $("evalResults");
  statusEl = $("evalStatus");
  runBtn = $("evalRun");
  splitSel = $("evalSplit");
  kInput = $("evalK");
  metricSel = $("evalMetric");
  epochsInput = $("evalEpochs");
  dataEl = $("evalData");
  $("evalClose").addEventListener("click", closeEvalView);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeEvalView(); });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) closeEvalView();
  });
  runBtn.addEventListener("click", run);
  results.addEventListener("click", (e) => {
    const tab = e.target.closest("[data-cm]");
    if (tab && last) { cmKind = tab.dataset.cm; renderConfusion(); }
  });
}

export function openEvalView() {
  kInput.value = knnSettings.k;
  metricSel.value = knnSettings.metric;
  if (!epochsInput.value) epochsInput.value = MLP_DEFAULTS.epochs;
  const c = dataset.counts();
  const signers = Object.keys(c.bySigner).length;
  dataEl.textContent = `${c.total} samples · ${signers} signer${signers === 1 ? "" : "s"}` +
    (signers < 2 ? " · leave-one-signer-out needs 2+ signers (import a friend's export)" : "");
  modal.hidden = false;
}

export function closeEvalView() {
  modal.hidden = true;
}

export const isEvalOpen = () => modal && !modal.hidden;

async function run() {
  if (running) return;
  running = true;
  runBtn.disabled = true;
  results.classList.add("stale");
  try {
    last = await runEvaluation(dataset.allSamples(), {
      split: splitSel.value,
      k: Number(kInput.value) || 5,
      metric: metricSel.value,
      epochs: Math.max(5, Math.min(200, Number(epochsInput.value) || MLP_DEFAULTS.epochs)),
      onProgress: (t) => (statusEl.textContent = t),
    });
    render(last);
    statusEl.textContent = `Done: ${last.nTrain} train / ${last.nTest} test samples.`;
  } catch (err) {
    statusEl.textContent = `Evaluation failed: ${err.message}`;
    console.error(err);
  } finally {
    running = false;
    runBtn.disabled = false;
    results.classList.remove("stale");
  }
}

function render(r) {
  const R = r.results;
  const splitName = r.split === "time" ? "time-blocked 80/20" : "random 80/20";

  // --- Summary tiles ---
  let html = `<div class="ev-tiles">` + KINDS.map((k) => `
    <div class="ev-tile">
      <div class="ev-tile-name"><i style="background:${COLOR[k]}"></i>${KIND_NAMES[k]}</div>
      <div class="ev-tile-value">${pct(R[k].acc)}</div>
      <div class="ev-tile-sub">${R[k].correct} / ${R[k].total} correct · 19 rule letters: ${pct(R[k].ruleAcc)}</div>
    </div>`).join("") + `</div>`;
  html += `<p class="ev-caption">Accuracy on the held-out test set (${splitName}, k = ${r.k}, ${r.metric}, MLP ${r.epochs} epochs).
    “19 rule letters” leaves out E M N S T, which the rules can't recognise.</p>`;

  // --- Leave-one-signer-out ---
  html += `<h3>Leave-one-signer-out</h3>`;
  if (r.loso && r.loso.rows.length) {
    html += `<p class="ev-caption">Each row: trained on everyone else, tested on this person. How well it works for someone it has never seen.</p>`;
    html += `<table class="ev-table"><thead><tr><th>Tested on</th><th>Samples</th>${KINDS.map((k) => `<th><i style="background:${COLOR[k]}"></i>${KIND_NAMES[k]}</th>`).join("")}</tr></thead><tbody>`;
    for (const row of r.loso.rows) {
      html += `<tr><td>${escapeHtml(row.signer)}</td><td>${row.n}</td>${KINDS.map((k) => barCell(row[k], k)).join("")}</tr>`;
    }
    html += `<tr class="ev-mean"><td>Mean</td><td></td>${KINDS.map((k) => barCell(r.loso.mean[k], k)).join("")}</tr></tbody></table>`;
  } else {
    html += `<p class="ev-caption">Needs samples from 2+ signers. Ask a friend to record and send you their export, then use Import JSON in the Train panel.</p>`;
  }

  // --- Per-letter accuracy ---
  html += `<h3>Per-letter accuracy</h3><div class="ev-scroll"><table class="ev-table ev-letters"><thead><tr><th>Letter</th><th>Test n</th>` +
    KINDS.map((k) => `<th><i style="background:${COLOR[k]}"></i>${KIND_NAMES[k]}</th>`).join("") + `</tr></thead><tbody>`;
  r.letters.forEach((L, i) => {
    const n = R.mlp.perLetter[i].total;
    if (!n) { html += `<tr class="ev-empty"><td>${L}</td><td>0</td><td colspan="3">no test samples</td></tr>`; return; }
    html += `<tr><td>${L}</td><td>${n}</td>` + KINDS.map((k) => barCell(R[k].perLetter[i].correct / n, k)).join("") + `</tr>`;
  });
  html += `</tbody></table></div>`;

  // --- Confusion matrix ---
  html += `<h3>Confusion matrix</h3>
    <div class="segmented ev-tabs">${KINDS.map((k) => `<button class="seg${k === cmKind ? " active" : ""}" data-cm="${k}">${KIND_NAMES[k]}</button>`).join("")}</div>
    <p class="ev-caption">Rows: the true letter. Columns: what was predicted (∅ = no confident answer). Brighter = larger share of that row. Hover a cell for counts.</p>
    <div class="ev-cm-wrap"><div id="evCm"></div><div id="evTopConf" class="ev-topconf"></div></div>`;

  results.innerHTML = html;
  renderConfusion();
}

function renderConfusion() {
  const R = last.results[cmKind];
  results.querySelectorAll("[data-cm]").forEach((b) => b.classList.toggle("active", b.dataset.cm === cmKind));
  confusionHeatmap(document.getElementById("evCm"), R.confusion, last.letters, { showNone: true });
  // Text alternative: the most common mistakes.
  const errs = [];
  R.confusion.forEach((row, t) => row.forEach((n, p) => { if (n && p !== t) errs.push({ t, p, n, share: n / row.reduce((a, b) => a + b, 0) }); }));
  errs.sort((a, b) => b.n - a.n);
  const name = (p) => (p === last.letters.length ? "no answer" : last.letters[p]);
  document.getElementById("evTopConf").innerHTML = `<div class="ev-topconf-title">Most common mistakes (${KIND_NAMES[cmKind]})</div>` +
    (errs.length
      ? errs.slice(0, 8).map((e) => `<div><b>${last.letters[e.t]}</b> → ${name(e.p)}<span>${e.n} (${pct(e.share)})</span></div>`).join("")
      : `<div class="sp-muted">None: every test sample was correct.</div>`);
}

// A table cell with a mini bar (series colour) and the value in ink.
function barCell(v, kind) {
  return `<td class="ev-bar-cell"><span class="ev-bar"><span style="width:${(v * 100).toFixed(1)}%;background:${COLOR[kind]}"></span></span><span class="ev-val">${pct(v)}</span></td>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}
